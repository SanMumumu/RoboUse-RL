import json
from pathlib import Path
import pytest
from robouse.environments.robodojo.native_job import verify_sources
from robouse.environments.robodojo.worker import NativeWorker


def test_official_control_snapshot_is_byte_pinned():
    assert len(verify_sources()) == 64


def test_worker_rejects_traversal_before_starting_simulator(tmp_path):
    worker = NativeWorker({"output": str(tmp_path)})
    for name in ("../old-run", "bad/path", "x\n"):
        with pytest.raises(ValueError):
            worker.handle({"op": "reset", "episode_id": name})


def test_restored_owner_never_signals_reused_pid(tmp_path):
    worker = NativeWorker({"output": str(tmp_path)})
    p = tmp_path / "episode"
    p.mkdir()
    (p / "owner.json").write_text(
        json.dumps({"pid": 1, "start_ticks": "wrong-start-time"})
    )
    assert worker.alive("episode") is False


def test_remote_artifact_endpoint_cannot_download_private_ground_truth(tmp_path):
    worker = NativeWorker({"output": str(tmp_path)})
    private = tmp_path / "episode/evaluator_private"
    private.mkdir(parents=True)
    (private / "initial-state.json").write_text('{"private":"must not reach actor"}')
    with pytest.raises(ValueError, match="public RGB"):
        worker.handle(
            {
                "op": "blob",
                "episode_id": "episode",
                "path": "evaluator_private/initial-state.json",
            }
        )


@pytest.mark.parametrize("interrupted", [False, True])
def test_published_native_score_does_not_hide_infrastructure_or_operator_failure(tmp_path, interrupted):
    worker = NativeWorker({"output": str(tmp_path)})
    folder = tmp_path / "synthetic-episode"
    work = folder / "work"
    work.mkdir(parents=True)
    (work / "_result.json").write_text(json.dumps({"eval_time": 1, "success_rate": 0.0, "score": 0.0, "details": {"0": {"layout_id": 1, "success": False, "score": 0.0}}}))
    (folder / "job.json").write_text(json.dumps({"identity": {"layout": 1}}))
    session = folder / "session"
    session.mkdir()
    (folder / "native-session.json").write_text(json.dumps({"path": str(session)}))
    (session / "episode_metrics.json").write_text(json.dumps({"infrastructure_failure": not interrupted, "official_ended_before_finalize": False}))
    if interrupted:
        (folder / "CANCEL").touch()
    result = worker.result("synthetic-episode")
    assert result["reward"] is None and "native_reward" not in result
    assert result["outcome"] == ("interrupted" if interrupted else "infrastructure_error")
