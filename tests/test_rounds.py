import json
from pathlib import Path
import pytest
from agp_rl.io import sha256
from agp_rl.reward.official import read_official
from robouse.data.datasets import write_json
from robouse.training.rl.grpo import run
from robouse.models.qwen import policy_version


def checkpoint(root, name, content):
    p = root / name
    p.mkdir(exist_ok=True)
    (p / "model.safetensors").write_bytes(content)
    return str(p)


def config(tmp_path):
    ids = tmp_path / "identities.jsonl"
    ids.write_text('{"task":"task","seed":0,"layout":1}\n')
    cp = checkpoint(tmp_path, "sft", b"original")
    return {
        "output": str(tmp_path / "run"),
        "checkpoint": cp,
        "groups_per_round": 1,
        "group_size": 2,
        "update_epochs": 1,
        "identities": str(ids),
        "env_urls": ["http://127.0.0.1:8765"],
        "max_rounds": 2,
        "max_env_episodes": 4,
    }


def collector(request, zero=False):
    folder = Path(request["output"])
    episodes = []
    for i in range(2):
        reward = 0 if zero else i
        directory = folder / f"ep{i}"
        directory.mkdir(exist_ok=True)
        path = directory / "_result.json"
        write_json(
            path,
            {
                "eval_time": 1,
                "success_rate": reward,
                "score": reward * 100,
                "details": {
                    "0": {"layout_id": 1, "success": bool(reward), "score": reward}
                },
            },
        )
        sample = directory / "sample.pt"
        sample.write_bytes(b"synthetic-test-only")
        episodes.append(
            {
                "episode_id": f"round-{request['round']}-sample-{i}",
                "identity_id": "task/seed-0/layout-001",
                "initial_state_hash": "same",
                "seed": 0,
                "origin": "official_online",
                "policy_version": request["policy_version"],
                "outcome": "completed",
                "reward": reward,
                "native_reward": read_official(path, 1).record(),
                "decisions": [
                    {
                        "policy_version": request["policy_version"],
                        "file": str(sample),
                        "sha256": sha256(sample),
                        "input_images": {"x": "x"},
                        "generated_ids": [1],
                        "behavior_logprobs": [-0.3],
                        "loss_mask": [True],
                    }
                ],
            }
        )
    return {"groups": [episodes], "loaded_policy_version": request["policy_version"]}


def test_multiround_resume_after_committed_update_no_duplicate_sampling(tmp_path):
    cfg = config(tmp_path)
    calls = []

    def update(req):
        calls.append(req["round"])
        cp = checkpoint(tmp_path, f"cp{req['round']}", str(req["round"]).encode())
        result = {
            "checkpoint": cp,
            "policy_version": policy_version(cp),
            "optimizer_rng_checkpoint": cp,
        }
        write_json(Path(req["output"]) / "update-response.json", result)
        if len(calls) == 1:
            raise RuntimeError("crash after durable checkpoint response")
        return result

    with pytest.raises(RuntimeError):
        run(cfg, collector=collector, updater=update)
    result = run({**cfg, "resume": True}, collector=collector, updater=update)
    assert calls == [0, 1]
    assert (
        result["round"] == 2
        and result["env_episodes"] == 4
        and len(result["consumed_groups"]) == 2
    )
    assert result["reference"] == cfg["checkpoint"]


def test_zero_variance_stops_at_budget_without_fake_update(tmp_path):
    cfg = config(tmp_path)

    def no_update(_):
        raise AssertionError("Zero variance must not call updater")

    result = run(
        cfg, collector=lambda req: collector(req, zero=True), updater=no_update
    )
    assert result["round"] == 2 and result["policy"] == policy_version(
        cfg["checkpoint"]
    )
    assert result["env_episodes"] == 4
