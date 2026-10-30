"""Durable rounds: collection → qualification → update → policy-load handshake."""
import json
import math
from pathlib import Path
import subprocess
import sys
import os
from robouse.data.datasets import write_json
from agp_rl.io import digest, sha256
from agp_rl.rl.grpo import group_advantages


def rank_plan(size, world=8):
    slots = math.ceil(size / world) * world
    return [
        [i if i < size else None for i in range(rank, slots, world)]
        for rank in range(world)
    ]


def validate_group(episodes, policy):
    if len(episodes) < 2:
        raise ValueError("Independent episodes required")
    if len({(ep["identity_id"], ep["initial_state_hash"]) for ep in episodes}) != 1:
        raise ValueError("Initial-state mismatch")
    if len({ep["episode_id"] for ep in episodes}) != len(episodes):
        raise ValueError("Repeated episode")
    for ep in episodes:
        if (
            ep["seed"] != 0
            or ep["origin"] != "official_online"
            or ep["policy_version"] != policy
        ):
            raise ValueError("Not Seed0 on-policy student")
        if ep["outcome"] not in ("completed", "policy_failure", "timeout") or ep[
            "reward"
        ] not in (0, 1):
            raise ValueError("Unscored infrastructure/interruption")
        from agp_rl.reward.official import read_official

        native = ep["native_reward"]
        actual = read_official(native["evidence_path"], native["layout_id"]).record()
        if actual != native or float(actual["success"]) != ep["reward"]:
            raise ValueError("Native reward evidence changed")
        if not ep["decisions"]:
            raise ValueError("No student decisions")
        for decision in ep["decisions"]:
            if (
                decision["policy_version"] != policy
                or sha256(decision["file"]) != decision["sha256"]
            ):
                raise ValueError("Policy/sample changed")
            if not decision["input_images"] or not len(
                decision["generated_ids"]
            ) == len(decision["behavior_logprobs"]) == len(decision["loss_mask"]):
                raise ValueError("Incomplete behavior evidence")
    return group_advantages([ep["reward"] for ep in episodes])


def launch_update(config, request_path, phase):
    import time
    from tqdm import tqdm

    output = Path(config["output"])
    argv = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        "-m",
        "robouse.training.rl.update",
        "--request",
        str(request_path),
        "--phase",
        phase,
    ]
    with (output / f"{phase}.log").open("a") as log:
        child = subprocess.Popen(
            argv,
            env={
                **os.environ,
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "OMP_NUM_THREADS": "2",
            },
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        with tqdm(
            total=None,
            desc=f"round {config['round']} {phase}",
            dynamic_ncols=True,
            unit="step",
        ) as bar:
            while child.poll() is None:
                progress = output / "update" / "progress.json"
                if progress.exists():
                    value = json.loads(progress.read_text())
                    bar.total = value["total"]
                    bar.n = value["step"]
                    bar.set_postfix(
                        loss=value.get("loss", "Unknown"),
                        tokens_s=round(value.get("tokens_per_second", 0), 2),
                        ETA="Unknown"
                        if value.get("eta") is None
                        else round(value["eta"]),
                    )
                time.sleep(0.5)
        code = child.wait()
    if code:
        print(
            "\n".join(
                (output / f"{phase}.log").read_text(errors="replace").splitlines()[-30:]
            ),
            file=sys.stderr,
        )
        raise subprocess.CalledProcessError(code, argv)


def run(config, *, collector=None, updater=None):
    from robouse.models.qwen import policy_version

    for key in (
        "group_size",
        "groups_per_round",
        "update_epochs",
        "max_rounds",
        "max_env_episodes",
    ):
        if config[key] < 1:
            raise ValueError("Positive round/group/budget controls required")
    if config["group_size"] < 2:
        raise ValueError("GRPO group_size must be at least two")
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    from robouse.utils.manifests import write_run_manifest

    write_run_manifest(root, config, stage="grpo", datasets={"identities_sha256": sha256(config["identities"])})
    path = root / "round-state.json"
    if path.exists():
        if not config.get("resume"):
            raise ValueError("Existing run requires --resume")
        state = json.loads(path.read_text())
    else:
        policy = policy_version(config["checkpoint"])
        state = {
            "round": 0,
            "consumed_groups": [],
            "env_episodes": 0,
            "policy": policy,
            "checkpoint": config["checkpoint"],
            "reference": config["checkpoint"],
            "optimizer_checkpoint": None,
            "phase": "ready",
        }
    contract = {
        k: config[k]
        for k in (
            "groups_per_round",
            "group_size",
            "update_epochs",
            "identities",
            "env_urls",
        )
    }
    contract["identities_sha256"] = sha256(config["identities"])
    from robouse.models.qwen import training_code_hash

    contract["code_sha256"] = training_code_hash()
    contract["loss_options"] = {
        k: config.get(k)
        for k in (
            "learning_rate",
            "beta",
            "clip_epsilon",
            "temperature",
            "max_decisions",
            "max_new_tokens",
            "max_context_tokens",
        )
    }
    fingerprint = digest(contract)
    if state.get("contract_hash", fingerprint) != fingerprint:
        raise ValueError("Resume algorithm/identity contract changed")
    state["contract_hash"] = fingerprint
    write_json(path, state)
    while state["round"] < config["max_rounds"]:
        if (root / "STOP").exists():
            break
        count = config["groups_per_round"] * config["group_size"]
        if (
            state["phase"] == "ready"
            and state["env_episodes"] + count > config["max_env_episodes"]
        ):
            break
        folder = root / f"round-{state['round']:06d}"
        folder.mkdir(exist_ok=True)
        request = {
            **config,
            "output": str(folder),
            "policy_version": state["policy"],
            "checkpoint": state["checkpoint"],
            "round": state["round"],
            "stop_file": str(root / "STOP"),
            "seed": 0,
            "reference_checkpoint": state["reference"],
            "optimizer_checkpoint": state["optimizer_checkpoint"],
        }
        if state["phase"] == "ready":
            # Reserve exactly once before touching the environment; crashes do
            # not silently reset the budget or create replacement episodes.
            state.update(phase="collecting", env_episodes=state["env_episodes"] + count)
            write_json(path, state)
        if state["phase"] == "collecting":
            if (folder / "collection.json").exists():
                response = json.loads((folder / "collection.json").read_text())
            elif collector is not None:
                response = collector(request)
                write_json(folder / "collection.json", response)
            else:
                from robouse.collect import collect

                response = collect(request)
            if (
                len(response["groups"]) != config["groups_per_round"]
                or response["loaded_policy_version"] != state["policy"]
            ):
                raise ValueError("Policy-load/count handshake failed")
            effective = []
            consumed = []
            for i, episodes in enumerate(response["groups"]):
                if len(episodes) != config["group_size"]:
                    raise ValueError("Incomplete group")
                unscored = any(
                    ep.get("reward") is None
                    or ep.get("outcome") in ("infrastructure_error", "interrupted")
                    for ep in episodes
                )
                if unscored:
                    import torch

                    advantages, usable = (
                        torch.zeros(len(episodes), dtype=torch.float64),
                        False,
                    )
                    write_json(
                        folder / f"group-{i}-unscored.json",
                        {
                            "outcomes": [ep["outcome"] for ep in episodes],
                            "updates": 0,
                            "rewards": [ep.get("reward") for ep in episodes],
                        },
                    )
                else:
                    advantages, usable = validate_group(episodes, state["policy"])
                group_id = digest([ep["episode_id"] for ep in episodes])
                if group_id in state["consumed_groups"]:
                    raise ValueError("Previously consumed group")
                consumed.append(group_id)
                p = folder / f"group-{i}.json"
                write_json(
                    p,
                    {
                        "episodes": episodes,
                        "advantages": advantages.tolist(),
                        "usable": usable,
                    },
                )
                if usable:
                    effective.append(str(p))
            write_json(
                folder / "groups.json", {"effective": effective, "consumed": consumed}
            )
            state["phase"] = "qualified_pending"
            write_json(path, state)
        grouping = json.loads((folder / "groups.json").read_text())
        request.update(groups=grouping["effective"])
        request_path = folder / "update-request.json"
        write_json(request_path, request)
        if grouping["effective"]:
            if state["phase"] == "qualified_pending":
                if updater is None:
                    launch_update(request, request_path, "qualify")
                state["phase"] = "updating"
                write_json(path, state)
            if state["phase"] == "updating":
                response_path = folder / "update-response.json"
                if response_path.exists():
                    result = json.loads(response_path.read_text())
                elif updater is not None:
                    result = updater(request)
                    write_json(response_path, result)
                else:
                    launch_update(request, request_path, "train")
                    result = json.loads(response_path.read_text())
                if (
                    result["policy_version"] == state["policy"]
                    or not result["optimizer_rng_checkpoint"]
                ):
                    raise ValueError("No committed new policy/optimizer/RNG")
                if policy_version(result["checkpoint"]) != result["policy_version"]:
                    raise ValueError("New checkpoint identity failed")
                state.update(
                    policy=result["policy_version"],
                    checkpoint=result["checkpoint"],
                    optimizer_checkpoint=result["optimizer_rng_checkpoint"],
                )
        else:
            write_json(
                folder / "no-signal.json",
                {"updates": 0, "reason": "NO_USABLE_GROUPS", "extra_sampling": False},
            )
        state["consumed_groups"].extend(grouping["consumed"])
        state.update(round=state["round"] + 1, phase="ready")
        write_json(path, state)
    return state
