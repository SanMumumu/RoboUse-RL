"""Public release contracts; synthetic inputs are never native robot experiments."""
import asyncio
import copy
import json
from pathlib import Path
import subprocess
import sys
import pytest
from robouse.data.trajectory import load_trajectory, decision_windows
from robouse.data.schemas import Trajectory, Identity
from robouse.data.datasets import image_bytes
from robouse.data.sampling import load_sources, sample_indices
from robouse.harness.executor import HarnessExecutor
from robouse.harness.tool_registry import ToolRegistry
from robouse.tools.base import Tool
from robouse.evaluation.metrics import evaluate_metrics
from agp_rl.reward.official import read_official

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/trajectory_sft/fixture"


def test_public_rgb_trajectory_and_local_decision_survive_partial_history():
    trajectory = load_trajectory(FIXTURE / "trajectory-seed0.json")
    assert Trajectory.from_dict(trajectory.to_dict()) == trajectory
    windows = list(decision_windows(trajectory))
    assert len(trajectory.decisions) == 2 and len(windows) == 1
    assert windows[0]["scope"] == "qualified_local_decisions"
    assert windows[0]["origin"] == "synthetic_fixture"
    for ref in windows[0]["images"].values():
        assert image_bytes(ref, FIXTURE).startswith(b"P3")
    with pytest.raises(ValueError, match="Seed role"):
        Identity("e", "a", "s", "t", 2, "l", "train")


def test_multi_source_weights_do_not_open_registered_test_targets(tmp_path):
    train = list(decision_windows(load_trajectory(FIXTURE / "trajectory-seed0.json")))
    paths = []
    for index in range(2):
        path = tmp_path / f"source-{index}.jsonl"
        path.write_text(json.dumps({**train[0], "window_id": f"window-{index}"}) + "\n")
        paths.append(path)
    config = {"datasets": {"train": [{"path": str(paths[0]), "weight": 9}, {"path": str(paths[1]), "weight": 1}], "test": [{"path": "/path/to/unavailable-sealed-test-targets"}]}}
    rows, weights, manifests = load_sources(config, "train")
    selected = sample_indices(2, "weighted_replacement", weights=weights, count=1000, seed=7)
    assert selected == sample_indices(2, "weighted_replacement", weights=weights, count=1000, seed=7)
    assert selected.count(0) > 850
    assert len(rows) == len(manifests) == 2
    assert sample_indices(2, "coverage", weights=weights) == [0, 1]
    with pytest.raises(ValueError, match="explicit samples_per_epoch"):
        sample_indices(2, "weighted_replacement", weights=weights)


def test_harness_schema_dispatch_errors_and_physical_success_boundary():
    calls = []
    async def observe(arguments):
        calls.append(arguments)
        return {"isError": False, "visible_feedback": "synthetic"}
    registry = ToolRegistry([Tool("observe", {"type": "object", "properties": {}, "additionalProperties": False}, observe)])
    harness = HarnessExecutor(registry)
    text = '{"name":"observe","arguments":{}}'
    first = asyncio.run(harness.execute("call-1", text))
    assert first["dispatched"] and first["physical_success"] is None
    assert asyncio.run(harness.execute("call-1", text)) == first and len(calls) == 1
    with pytest.raises(ValueError, match="different arguments"):
        asyncio.run(harness.execute("call-1", text + " "))
    invalid = asyncio.run(harness.execute("call-2", '{"name":"observe","arguments":{"extra":1}}'))
    assert invalid["is_error"] and not invalid["dispatched"] and len(calls) == 1
    with pytest.raises(ValueError, match="idempotent"):
        asyncio.run(harness.execute("call-3", text, retries=1))


def test_harness_propagates_native_transport_failure():
    async def broken(arguments):
        raise ConnectionError("synthetic controller connection failure")
    harness = HarnessExecutor(ToolRegistry([Tool("move", {"type": "object"}, broken)]))
    receipt = asyncio.run(harness.execute("call", '{"name":"move","arguments":{}}'))
    assert receipt["is_error"] and receipt["physical_success"] is None
    assert "controller connection failure" in receipt["error"]


def test_metrics_verify_native_evidence_and_keep_unscored_separate(tmp_path):
    path = tmp_path / "_result.json"
    path.write_text(json.dumps({"eval_time": 1, "success_rate": 1.0, "score": 100.0, "details": {"0": {"layout_id": 1, "success": True, "score": 1.0}}}))
    native = read_official(path, 1).record()
    episodes = [{"episode_id": "synthetic-score-fixture", "reward": 1.0, "native_reward": native, "outcome": "completed", "timeline": [{"native_tool_calls": 2, "native_tool_seconds": 0.25, "decision_seconds": 0.5, "tool_seconds": 0.3}], "episode_seconds": 1.0}, {"episode_id": "synthetic-infra", "reward": None, "outcome": "infrastructure_error"}]
    result = evaluate_metrics(episodes, requested_count=3)
    assert result["native_task_success_rate"] == 1.0
    assert result["scored_episodes"] == result["unscored_episodes"] == 1
    assert result["requested_episodes"] == 3
    assert result["tool_execution_latency_seconds"] == 0.25
    assert result["episodes"][1]["native_success"] is None
    with pytest.raises(ValueError, match="without native"):
        evaluate_metrics([{"episode_id": "false-agent-claim", "reward": 1.0}])
    changed = copy.deepcopy(episodes)
    changed[0]["native_reward"]["evidence_sha256"] = "bad"
    with pytest.raises(ValueError, match="evidence differs"):
        evaluate_metrics(changed)


@pytest.mark.parametrize("name", ["train_sft", "train_rl", "evaluate"])
def test_public_training_cli_help(name):
    result = subprocess.run(["bash", str(ROOT / "scripts" / f"{name}.sh"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--output-dir" in result.stdout


def test_target_sft_cli_is_connected_and_aliases_override_config():
    result = subprocess.run([sys.executable, "-m", "robouse.cli", "train_sft", "--config", str(ROOT / "configs/sft/qwen35_2b.yaml"), "--train-data", "train.jsonl", "--val-data", "val.jsonl", "--output-dir", "outputs/sft_exp01", "--num-gpus", "8", "--dry-run"], capture_output=True, text=True, check=True)
    config = json.loads(result.stdout)["config"]
    assert config["output"] == "outputs/sft_exp01" and config["train_data"] == "train.jsonl"
    assert config["finetuning_mode"] == "full" and len(config["revision"]) == 40


def test_experimental_rl_fails_before_cuda_or_data_access():
    result = subprocess.run([sys.executable, "-m", "robouse.cli", "train_rl", "--init-checkpoint", "/path/to/missing-checkpoint", "--execute"], capture_output=True, text=True)
    assert result.returncode != 0 and "--allow-experimental" in result.stderr
    assert "Traceback" not in result.stderr


def test_student_rollout_exports_shared_schema_with_relocated_original_rgb(tmp_path):
    import hashlib
    import torch
    from robouse.scheduler import episode

    images = tmp_path / "images"
    images.mkdir()
    raw = (FIXTURE / "rgb-seed0.ppm").read_bytes()
    (images / "original.ppm").write_bytes(raw)
    key = hashlib.sha256(raw).hexdigest()
    class Environment:
        episode_id = "synthetic-student-contract"
        root = images
        async def reset(self, *args):
            return {"system": "synthetic", "initial_state_hash": "synthetic-state", "blocks": []}
        def observation(self, value):
            return [{"type": "text", "text": "Observe this synthetic RGB."}, {"type": "image", "image": "art-image://" + key}], {key: {"path": "original.ppm", "sha256": key, "bytes": len(raw)}}
        async def step(self, *args, **kwargs):
            return {"done": True, "reward": None, "outcome": "infrastructure_error", "termination_reason": "synthetic-unscored", "native_tool_calls": 1, "native_tool_seconds": 0.1}
    class Engine:
        policy = "synthetic-untrained"
        async def infer(self, *args):
            return {"inputs": {"input_ids": torch.tensor([[1]])}, "tokens": torch.tensor([2]), "behavior_logprobs": torch.tensor([-1.0]), "mask": torch.tensor([True]), "text": '{"name":"observe","arguments":{}}', "policy_version": self.policy, "input_hash": "synthetic-input"}
    asyncio.run(episode(Engine(), Environment(), {"task": "synthetic", "seed": 0, "layout": 1}, "group", tmp_path))
    trajectory = load_trajectory(tmp_path / "trajectory.json")
    assert trajectory.origin == "student_on_policy" and trajectory.success is None
    ref = trajectory.decisions[0].observation.rgb[key]
    assert image_bytes(ref, tmp_path) == raw
    assert ref["path"] == "images/original.ppm"
    assert list(decision_windows(trajectory)) == []
