"""Evaluator-private, read-only initial state evidence before policy launch.

Never call reset/step/render/get_obs/reward/end checks. IsaacLab's scene state
alone omits RoboDojo's separately managed task objects, so both are captured.
This is an observation of audited state, not a resumable PhysX checkpoint.
The caller must mask evaluator_private in every actor filesystem namespace.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import random
import sys

from agp_rl.io import canonical, digest, read_json, sha256, utcnow, write_json


SCHEMA = "robodojo-evaluator-initial-state-v1"
OBJECT_STORES = {
    "rigid_or_dynamic": "_rigid_and_dynamic_objects", "articulation": "_articulation_objects",
    "garment": "_garment_objects", "geometry": "_geometry_objects", "fluid": "_fluid_objects",
}
REWARD_FIELDS = (
    "num_envs", "check_list", "final_check_list", "query_list", "trigger_check_list", "trigger_query_list",
    "score_list", "score_achieved", "score_completed_count", "score_meta", "score_trigger_meta",
    "final_score_list", "final_score_achieved", "final_score_completed_count", "final_score_meta", "_gated_score_lst",
)
PARSER_FIELDS = ("num_envs", "pre_state", "robot_origin_endpose", "joint_ratio_transition_state")
ENV_FIELDS = ("task_name", "num_envs", "step_lim", "take_action_cnt", "end_flag", "success")


def plain(value):
    """Lossless JSON numbers with array dtype/shape; refuse opaque repr fallbacks."""
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite state value")
        return float(value)
    module = type(value).__module__
    if module.startswith("torch") and hasattr(value, "detach"):
        data = value.detach().cpu()
        return {"dtype": str(data.dtype), "shape": list(data.shape), "values": plain(data.tolist())}
    if module.startswith("numpy"):
        if hasattr(value, "shape") and value.shape != ():
            return {"dtype": str(value.dtype), "shape": list(value.shape), "values": plain(value.tolist())}
        return plain(value.item())
    if isinstance(value, Mapping):
        keys = [str(key) for key in value]
        if len(set(keys)) != len(keys):
            raise ValueError("State mapping has ambiguous JSON keys")
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return {"set": sorted((plain(item) for item in value), key=canonical)}
    raise TypeError("Unsupported state type: " + module + "." + type(value).__name__)


def _read_fields(obj, names, section, issues):
    result = {}
    for name in names:
        try:
            result[name] = plain(getattr(obj, name))
        except Exception as exc:
            issues.append({"section": section, "field": name, "error": type(exc).__name__})
    return result


def _clock(env):
    context = env.sim.sim
    return {"current_time": float(context.current_time),
            "current_time_step_index": int(context.current_time_step_index),
            "take_action_cnt": plain(env.take_action_cnt)}


def _rng(env):
    state = {"python": plain(random.getstate())}
    numpy = sys.modules.get("numpy")
    torch = sys.modules.get("torch")
    if numpy is not None:
        state["numpy_global"] = plain(numpy.random.get_state())
    if torch is not None:
        state["torch_cpu"] = plain(torch.random.get_rng_state())
        if torch.cuda.is_initialized():
            state["torch_cuda"] = plain(torch.cuda.get_rng_state_all())
    generator = vars(env).get("_np_random")
    if generator is not None:
        state["gym_numpy_generator"] = plain(generator.bit_generator.state)
    return state


def _native_object(obj, category, issues, label):
    """Use existing native getters; Dynamic/cloth/fluid remain explicitly partial."""
    result = {"class": type(obj).__module__ + "." + type(obj).__name__, "category": category}
    result.update(_read_fields(obj, ("instance_name", "usd_path", "instance_config"), "object_metadata", []))
    is_static = category == "geometry" or (category == "fixture" and getattr(obj, "is_static", True))
    result["motion_class"] = "static" if is_static else "dynamic"
    getters = {"world_pose": "get_world_pose"}
    if not is_static:
        getters.update(linear_velocity="get_linear_velocity", angular_velocity="get_angular_velocity")
    if category == "articulation":
        getters.update(joint_position="get_joint_positions", joint_velocity="get_joint_velocities")
    for key, method in getters.items():
        try:
            result[key] = plain(getattr(obj, method)())
        except Exception as exc:
            issues.append({"section": "physical", "field": label + "." + key, "error": type(exc).__name__})
    if category in {"garment", "fluid"} or type(obj).__name__ == "DynamicObject":
        issues.append({"section": "physical", "field": label,
                       "error": "Native wrapper does not expose complete deformable/child-body state"})
        # A garment get_state() only returns a root pose; do not mislabel it as
        # nodal positions/velocities or introduce a renderer/physics update.
    return result


def _source_recipe(env, job):
    files = {}
    objects = [type(env), type(env.scene_manager), type(env.robot_manager), type(env.reward_manager),
               type(env.reward_manager.func_parser), type(env.sim.scene)]
    objects.extend(type(env).__mro__)
    for cls in objects:
        module = sys.modules.get(cls.__module__)
        path = getattr(module, "__file__", None)
        if path and Path(path).is_file() and str(Path(path).suffix) == ".py":
            files[str(Path(path).resolve())] = sha256(path)
    return {"schema": SCHEMA, "reader_sha256": sha256(__file__), "source_hash": job["source_hash"],
            "native_source_files": files, "coordinate_frame": "world", "quaternion_order": "wxyz",
            "native_scene_api": "InteractiveScene.get_state(is_relative=False)",
            "object_apis": "native world pose, linear/angular velocity, articulation joint getters",
            "logical_scope": "insert_key evaluator flags + RewardManager + Func_Parser + robot control state",
            "rng_scope": "existing Python/NumPy/Torch/gym generators; no generator initialization",
            "excluded_opaque_state": ["PhysX contact/solver caches", "Warp RNG internals", "renderer temporal history"]}


def _logical(env, issues):
    logical = {"evaluator": _read_fields(env, ENV_FIELDS, "logical", issues)}
    if getattr(env, "task_name", None) != "insert_key":
        issues.append({"section": "logical", "field": "task_schema", "error": "Task-specific hidden state not audited"})
    reward = env.reward_manager
    parser = reward.func_parser
    reward_fields = set(REWARD_FIELDS) | (set(vars(reward)) - {"env", "func_parser"})
    parser_fields = set(PARSER_FIELDS) | (set(vars(parser)) - {"env", "layout_manager", "robot_manager"})
    logical["reward_manager"] = _read_fields(reward, sorted(reward_fields), "logical", issues)
    logical["func_parser"] = _read_fields(parser, sorted(parser_fields), "logical", issues)
    manager = env.robot_manager
    logical["robot_manager"] = _read_fields(manager, ("robot_origin_endpose", "robot_init_joint"), "logical", issues)
    control = manager.control_manager
    logical["control_previous"] = _read_fields(control, ("prev_control",), "logical", issues)
    logical["control_queues"] = []
    for index, sequence in enumerate(control.control_queue):
        queue = sequence.control_seq
        with queue.mutex:
            # Observe the queue without pop/get/get_empty (get_empty can drain
            # queues for other environments in the native implementation).
            logical["control_queues"].append([plain(item.control_info_dict) for item in list(queue.queue)])
    return logical


def build_snapshot(env, job):
    """Capture in memory; no environment mutation and no actor-visible GT."""
    issues = []
    before, rng_before = _clock(env), _rng(env)
    if any(value != 0 for value in before["take_action_cnt"]):
        raise ValueError("Initial state must be captured before the first policy action")
    recipe = _source_recipe(env, job)
    physical = {"scene": plain(env.sim.scene.get_state(is_relative=False)),
                "env_origins": plain(env.sim.scene.env_origins), "objects": {}, "fixtures": {}}
    if not physical["scene"].get("articulation"):
        issues.append({"section": "physical", "field": "scene.articulation", "error": "No robot articulation state"})
    for category, fields in {
        "articulation": ("root_pose", "root_velocity", "joint_position", "joint_velocity"),
        "rigid_object": ("root_pose", "root_velocity"),
        "deformable_object": ("nodal_position", "nodal_velocity"),
    }.items():
        if category not in physical["scene"]:
            issues.append({"section": "physical", "field": "scene." + category, "error": "Missing native asset inventory"})
        for name, state in physical["scene"].get(category, {}).items():
            for field in fields:
                if field not in state:
                    issues.append({"section": "physical", "field": f"scene.{category}.{name}.{field}",
                                   "error": "Missing native asset state"})
    scene_manager = env.scene_manager
    for category, store in OBJECT_STORES.items():
        physical["objects"][category] = []
        for index, objects in enumerate(getattr(scene_manager, store)):
            physical["objects"][category].append({
                str(name): _native_object(obj, category, issues, f"{store}[{index}].{name}")
                for name, obj in sorted(objects.items())})
    if not any(objects for groups in physical["objects"].values() for objects in groups):
        issues.append({"section": "physical", "field": "objects", "error": "No native task objects captured"})
    for store in ("_tables", "_rooms", "_grounds"):
        physical["fixtures"][store] = []
        for index, entry in enumerate(getattr(scene_manager, store)):
            entries = entry if isinstance(entry, list) else [entry]
            states = []
            for number, obj in enumerate(entries):
                if obj is None:
                    continue
                target = getattr(obj, "geometry_prim", None) or obj
                states.append(_native_object(target, "fixture", issues, f"{store}[{index}][{number}]"))
            physical["fixtures"][store].append(states)
    logical = _logical(env, issues)
    layout = scene_manager.layout_manager
    layout_data = _read_fields(layout, ("saved_layouts", "instance_type_by_env", "table_info", "ground_info"),
                               "static", issues)
    configuration = _read_fields(env, ("sim_config", "robot_config", "camera_config"), "static", issues)
    metadata = _read_fields(env, ("env_seeds", "eval_seed", "_global_seed", "_seed_list", "current_env_seed_map"),
                            "seed_metadata", issues)
    after, rng_after = _clock(env), _rng(env)
    if before != after or digest(rng_before) != digest(rng_after):
        raise ValueError("Snapshot capture changed physics/actions or consumed RNG")
    completeness = {"native_physical_state": not any(i["section"] == "physical" for i in issues),
                    "audited_task_logical_state": not any(i["section"] == "logical" for i in issues),
                    "static_recipe": not any(i["section"] == "static" for i in issues),
                    "available_rng_state": True, "full_simulator_checkpoint": False,
                    "all_hidden_state": False}
    body = {"physical": physical, "logical": logical, "static": {"layout": layout_data, "configuration": configuration},
            "rng": rng_before, "seed_metadata": metadata}
    return {"schema": SCHEMA, "schema_version": 1, "created_utc": utcnow(), "recipe": recipe,
            "recipe_hash": digest(recipe), "identity": job["key"], "state": body,
            "initial_state_hash": digest(body), "physical_state_hash": digest(physical),
            "logical_state_hash": digest(logical), "static_state_hash": digest(body["static"]),
            "rng_state_hash": digest(rng_before), "completeness": completeness, "issues": issues,
            "capture_proof": {"before": before, "after": after, "physics_unchanged": before == after,
                              "rng_unchanged": True, "policy_actions": 0, "mutating_environment_calls": 0},
            "privacy": {"contains_ground_truth": True, "actor_namespace_mask_required": True,
                        "filesystem_mode_alone_does_not_isolate_same_uid_actor": True}}


def public_provenance(snapshot):
    """Actor-visible recipe/coverage only; no GT fingerprint or private-file hash."""
    keys = ("schema", "schema_version", "recipe_hash", "completeness")
    return {**{key: snapshot[key] for key in keys},
            "contains_ground_truth": False, "qualification": "audited_initial_state_observation_only"}


def capture_initial_state(env, job, *, job_path=None):
    """Write immutable jobdir/evaluator_private/initial_state.json; return recipe/flags.

    Call before original eval_one_episode or PolicyProcess. Mask this private
    directory with an empty tmpfs in all actor subprocess namespaces.
    """
    job_path = Path(job_path).resolve() if job_path else Path(job["workdir"]).resolve().parent / "job.json"
    if read_json(job_path) != job or Path(job["workdir"]).resolve().parent != job_path.parent:
        raise ValueError("Initial-state evidence must belong to the exact evaluator job")
    private = job_path.parent / "evaluator_private"
    if private.is_symlink():
        raise ValueError("Evaluator private directory must not be a symlink")
    private.mkdir(mode=0o700, exist_ok=True)
    private.chmod(0o700)
    path = private / "initial_state.json"
    if path.exists():
        raise FileExistsError(path)
    snapshot = build_snapshot(env, job)
    write_json(path, snapshot)
    path.chmod(0o600)
    return public_provenance(snapshot)


def compare_initial_states(left, right):
    """Exact equality within one recipe; never certify unavailable hidden state."""
    def valid_hash(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

    same_schema = left.get("schema") == right.get("schema") == SCHEMA
    same_recipe = (same_schema and valid_hash(left.get("recipe_hash"))
                   and left.get("recipe_hash") == right.get("recipe_hash"))
    parts = {name: same_recipe and valid_hash(left.get(name)) and left.get(name) == right.get(name) for name in
             ("physical_state_hash", "logical_state_hash", "static_state_hash", "rng_state_hash")}
    audited_complete = all(value.get("completeness", {}).get(key) is True
                           for value in (left, right) for key in
                           ("native_physical_state", "audited_task_logical_state", "static_recipe", "available_rng_state"))
    return {"same_schema": same_schema, "same_recipe": same_recipe, "parts": parts,
            "audited_state_equivalent": same_recipe and audited_complete and all(parts.values()),
            "full_hidden_state_equivalent": False,
            "exact_physical_duplicate": (same_schema and valid_hash(left.get("physical_state_hash"))
                                         and left.get("physical_state_hash") == right.get("physical_state_hash")),
            "use": "Pairing evidence and conservative train/final duplicate exclusion; not a simulator restore guarantee"}
