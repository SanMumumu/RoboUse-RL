"""Replace only agent launch; official controls, episode loop and scoring stay intact."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from robouse.data.datasets import write_json
from robouse.environments.robodojo.native_job import source_root, verify_sources
from agp_rl.io import digest


def eval_one_episode(TASK_ENV, model_client=None):
    path = Path(os.environ["ROBOUSE_NATIVE_JOB"])
    job = json.loads(path.read_text())
    source_hash = verify_sources()
    policy = source_root()
    runtime = policy / "agp_runtime"
    sys.path[:0] = [str(runtime), str(runtime / "source/agent")]
    from agp_rl.adapters.initial_state import build_snapshot

    native_job = {
        "key": {
            "task": job["identity"]["task"],
            "seed": job["identity"]["seed"],
            "layout": job["identity"]["layout"],
        },
        "source_hash": source_hash,
    }
    snapshot = build_snapshot(TASK_ENV, native_job)
    private = path.parent / "evaluator_private"
    private.mkdir(exist_ok=True)
    write_json(private / "initial-state.json", snapshot)
    # IDs prove comparability without transmitting GT. Exclude RNG/process metadata.
    write_json(
        path.parent / "initial-proof.json",
        {
            "initial_state_hash": digest(
                {
                    k: snapshot["state"][k]
                    for k in ("physical", "static", "seed_metadata")
                }
            ),
            "layout_available": True,
            "native_source_hash": source_hash,
        },
    )
    spec = importlib.util.spec_from_file_location(
        "robouse_official_deploy", runtime / "agp_core/official_deploy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if job.get("actor_kind") == "astra_teacher":
        original_config = module.load_config

        def load_config(policy_dir, overlay=None):
            if overlay is not None:
                raise ValueError("Teacher overlays are not authorized")
            value = original_config(policy_dir, overlay)
            teacher = job["teacher"]
            value["model_settings"] = {
                "OURS_MODEL": teacher["model"],
                "OURS_EFFORT": teacher["effort"],
                "OURS_SERVICE_TIER": teacher["service_tier"],
            }
            value["codex_binary"] = teacher["cli_binary"]
            return value

        module.load_config = load_config
        original_pins = module.frozen_policy_pins

        def teacher_pins(config, record, state):
            pins = original_pins(config, record, state)
            pins.update(
                codex_binary=job["teacher"]["cli_binary"],
                codex_sha256=job["teacher"]["cli_sha256"],
                codex_version=job["teacher"]["cli_version"],
                teacher_binary_pin_source="explicit teacher configuration; verified again before execution",
            )
            return pins

        module.frozen_policy_pins = teacher_pins
    original_env = module.agent_environment

    def agent_env(*args, **kwargs):
        env = original_env(*args, **kwargs)
        env["PYTHONPATH"] = (
            str(Path(__file__).resolve().parents[3])
            + os.pathsep
            + str(runtime)
            + os.pathsep
            + str(runtime / "source/agent")
        )
        env["ROBOUSE_NATIVE_JOB"] = str(path)
        if job.get("actor_kind") == "astra_teacher":
            for key in ("ASTRA_AUTH_HOME", "ASTRA_ACCOUNT_SHA256"):
                if key in os.environ:
                    env[key] = os.environ[key]
        return env

    def agent_command(controls, session):
        if controls["scripted"] or controls["fault"]:
            raise ValueError("Scripted/teacher controls forbidden")
        write_json(path.parent / "native-session.json", {"path": str(session)})
        actor = (
            "robouse.teacher.actor"
            if job.get("actor_kind") == "astra_teacher"
            else "robouse.environments.robodojo.native_actor"
        )
        return [
            job["transport_python"],
            "-B",
            "-m",
            actor,
            "--session",
            str(session),
            "--job",
            str(path),
        ]

    module.agent_environment = agent_env
    module.agent_command = agent_command
    try:
        return module.eval_one_episode(TASK_ENV, model_client, policy_dir=policy)
    finally:
        # Finalize only already-recorded streams, including failure streams.
        directory = path.parent / "videos"
        directory.mkdir(exist_ok=True)
        for index in list(TASK_ENV.video_writers):
            TASK_ENV.save_video(
                index, str(directory / f"episode-{index}.mp4"), "native"
            )
