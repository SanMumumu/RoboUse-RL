"""Native on-policy collection and identity-complete evaluation."""
import asyncio
import json
from pathlib import Path
import random
import os
import urllib.parse
import torch
from agp_rl.io import digest
from robouse.data.datasets import read_jsonl, write_json
from robouse.models.qwen import load, policy_version
from robouse.environments.robodojo.client import RemoteEnvironment
from robouse.scheduler import HfEngine, swift_scheduler_class


def collect(config, *, evaluation=False):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    source = config["checkpoint"]
    version = policy_version(source)
    if config.get("policy_version") and version != config["policy_version"]:
        raise ValueError("Actual loaded weight identity differs")
    identities = read_jsonl(config["identities"])
    expected_seed = (
        {"val": 1, "test": 2}[config.get("split", "val")] if evaluation else 0
    )
    if not identities or any(r["seed"] != expected_seed for r in identities):
        raise ValueError("Environment identity role mismatch")
    from robouse.data.datasets import check_roles, role

    registry = Path(source) / "role-ownership.json"
    ownership = json.loads(registry.read_text()) if registry.exists() else []
    if registry.exists():
        if (Path(source) / "CHECKPOINT_COMMITTED.json").exists():
            from robouse.checkpoints import verify

            verify(source)
        else:
            complete = Path(source) / "training-complete.json"
            if (
                not complete.exists()
                or json.loads(complete.read_text()).get("status") != "COMPLETED"
            ):
                raise ValueError("Uncommitted or interrupted checkpoint")
    if not ownership and not (Path(source) / "model-manifest.json").exists():
        raise ValueError("Checkpoint role provenance missing")
    checks = [
        {
            **r,
            "identity_id": f"{r['task']}/seed-{r['seed']}/layout-{r['layout']:03d}",
            "role": role(r["seed"]),
        }
        for r in identities
    ]
    check_roles(ownership + checks)
    endpoints = config["env_urls"]
    if not endpoints or len(os.environ.get("ROBOUSE_ENV_TOKEN", "")) < 24:
        raise ValueError("Explicit loopback environment URLs and ROBOUSE_ENV_TOKEN (24+ characters) required before CUDA")
    for endpoint in endpoints:
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Use an authenticated SSH tunnel to a loopback environment")
    model, processor = load(source, device="cuda:0")
    torch.manual_seed(config.get("round", 0))
    engine = HfEngine(model, processor, version, config)
    scheduler = swift_scheduler_class()(engine, max_turns=config.get("max_decisions", 512))
    groups = []
    all_episodes = []
    selected = (
        identities
        if evaluation
        else [
            identities[
                (config.get("round", 0) * config["groups_per_round"] + i)
                % len(identities)
            ]
            for i in range(config["groups_per_round"])
        ]
    )
    size = 1 if evaluation else config["group_size"]
    for group_index, identity in enumerate(selected):
        group_id = digest(
            {
                "round": config.get("round", 0),
                "index": group_index,
                "policy": version,
                "identity": identity,
            }
        )
        group = []
        for index in range(size):
            eid = digest({"group": group_id, "sample": index})
            folder = output / "episodes" / eid
            env = RemoteEnvironment(
                endpoints[(group_index * size + index) % len(endpoints)],
                eid,
                folder / "images",
            )
            try:
                from types import SimpleNamespace

                request = SimpleNamespace(
                    data_dict={
                        "robouse": {
                            "environment": env,
                            "identity": identity,
                            "group_id": group_id,
                            "directory": folder,
                            "max_turns": config.get("max_decisions", 512),
                            "stop_path": config.get("stop_file", output / "STOP"),
                        }
                    }
                )
                result = asyncio.run(scheduler.run(request, None)).rollout_infos
            except BaseException as exc:
                # Preserve unscored failure and stop; never substitute reward=0.
                failure = {
                    "episode_id": eid, "identity": identity, "seed": identity["seed"],
                    "origin": "official_online", "policy_version": version,
                    "classification": "interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "infrastructure_error",
                    "error": str(exc), "reward": None, "decisions": [], "timeline": [],
                    "termination_reason": "collection_exception",
                }
                failure["outcome"] = failure["classification"]
                write_json(
                    folder / "failure.json",
                    failure,
                )
                write_json(folder / "episode.json", failure)
                all_episodes.append(failure)
                (output / "episodes.jsonl").write_text("".join(json.dumps(ep) + "\n" for ep in all_episodes))
                raise
            videos = {}
            for camera, name in result.get("videos", {}).items():
                public_camera = "head" if camera == "top" else camera
                local = output / "videos" / eid / f"{public_camera}.mp4"
                env.download(name, local)
                videos[public_camera] = str(local.relative_to(output))
            # Reward evidence is transferred verbatim; updater can audit its SHA.
            native = result.get("native_reward")
            if native:
                original = native["evidence_path"]
                # Worker exposes a safe relative artifact name alongside reward.
                if result.get("reward_artifact"):
                    local = folder / "reward" / "_result.json"
                    env.download(result["reward_artifact"], local)
                    native["evidence_path"] = str(local.resolve())
            result["videos"] = videos
            result["policy_version"] = version
            write_json(folder / "episode.json", result)
            group.append(result)
            all_episodes.append(result)
            (output / "episodes.jsonl").write_text(
                "".join(
                    json.dumps(ep, ensure_ascii=False) + "\n" for ep in all_episodes
                )
            )
        groups.append(group)
    (output / "episodes.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    **ep,
                    "score": ep.get("score"),
                    "termination_reason": ep.get("termination_reason", "unknown"),
                },
                ensure_ascii=False,
            )
            + "\n"
            for ep in all_episodes
        )
    )
    result = {"loaded_policy_version": version, "groups": groups}
    write_json(output / "collection.json", result)
    return result
