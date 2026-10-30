"""All collector extension tests use CPU fixtures; no real Astra/Isaac calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import pytest
from agp_rl.io import sha256
from robouse.data.datasets import write_json
from robouse.teacher.admission import FrontierBlocked, hashed
from robouse.teacher.provider import IsolatedCodex
from robouse.teacher.recovery import metadata_with_recovery
from robouse.teacher.recovery import close_admission
from robouse.teacher.failures import emit, flush
from robouse.teacher.guardian import supervise
from robouse.teacher.guardian import resume_stopped
from robouse.teacher.observer import observe, ProcessorAcceptanceError
from robouse.teacher.placement import verify_renderer
from robouse.teacher.service import TeacherService
from robouse.teacher.remote import run as remote_run, TeacherClient
from robouse.environments.robodojo.remote import make_server
from test_teacher import cfg, native_fixture
from robouse.teacher.seal import seal
from robouse.teacher.state import excluded_identities
from robouse.teacher.state import file_lock


def wire_runner(tmp_path, method, http=0):
    runner = object.__new__(IsolatedCodex)
    runner.directory = tmp_path / "isolated"
    runner.profile = runner.directory / "profile"
    runner.profile.mkdir(parents=True)
    runner.admitted = False
    runner.config = {"model": "gpt-6-astra", "effort": "high", "service_tier": "default", "metadata_timeout_s": 5}
    runner.identity = {"account_sha256": "cpu-fixture-account", "user_sha256": "cpu-fixture-user", "email_sha256": hashed("cpu-fixture@invalid"), "credential_file_sha256": "cpu-fixture-credential-hash"}
    runner.verify_identity = lambda *args, **kwargs: runner.identity
    runner.last_verified_metadata = {"account_sha256": runner.identity["account_sha256"], "credential_file_sha256": runner.identity["credential_file_sha256"], "account_metadata_admitted": True, "observed_at": time.time(), "remaining_percent": 99, "catalog": {"advertised": True}, "model": "gpt-6-astra", "effort": "high"}
    fixture = Path(__file__).parent / "fixtures/rpc_metadata_server.py"
    runner.command = lambda *args, **kwargs: [sys.executable, "-B", str(fixture), str(tmp_path / "counter"), method, str(http)]
    return runner


@pytest.mark.parametrize("method", ["account/read", "model/list"])
def test_metadata_real_subprocess_wire_has_three_bounded_same_attempt_checks(tmp_path, method):
    runner = wire_runner(tmp_path, method)
    sleeps = []
    proof = metadata_with_recovery(runner, runner._metadata_once, "fixture", sleep=sleeps.append)
    assert (tmp_path / "counter").read_text() == "3"
    assert sleeps == [2, 5] and proof["inference_requests"] == 0
    assert len(list((runner.directory / "failure-spool").glob("*.json"))) == 3
    assert flush(runner.directory) == 3
    assert flush(runner.directory) == 0


@pytest.mark.parametrize("status", [401, 403, 429])
def test_real_subprocess_denials_are_not_retried(tmp_path, status):
    runner = wire_runner(tmp_path, "account/read", status)
    with pytest.raises(FrontierBlocked, match="HTTP_" + str(status)):
        metadata_with_recovery(runner, runner._metadata_once, "fixture", sleep=lambda seconds: pytest.fail("Denied RPC retried"))
    assert (tmp_path / "counter").read_text() == "1"


@pytest.mark.parametrize("unsafe", ["stale", "different_account", "started"])
def test_unknown_internal_fault_needs_fresh_same_identity_prior_and_no_model_start(tmp_path, unsafe):
    runner = wire_runner(tmp_path, "account/read")
    if unsafe == "stale":
        runner.last_verified_metadata["observed_at"] = 0
    elif unsafe == "different_account":
        runner.last_verified_metadata["account_sha256"] = "another-fixture-account"
    else:
        workspace = tmp_path / "session"
        workspace.mkdir()
        (workspace / "model-process-start-intent.json").write_text("{}")
        runner.config["workspace"] = str(workspace)
    with pytest.raises(FrontierBlocked):
        metadata_with_recovery(runner, runner._metadata_once, "fixture", sleep=lambda seconds: pytest.fail("Unsafe retry"))
    assert (tmp_path / "counter").read_text() == "1"


def test_failure_journal_does_not_retain_uncontrolled_provider_text(tmp_path):
    emit(tmp_path, classification="provider-error\nprivate-body-do-not-retain", stage="metadata")
    flush(tmp_path)
    text = "\n".join(p.read_text() for p in tmp_path.rglob("*") if p.is_file())
    assert "private-body-do-not-retain" not in text
    assert "UNCLASSIFIED" in text and "Unknown" in text


@pytest.mark.parametrize("message", ["permission denied despite network availability", "quota exceeded", "read-only filesystem", "workspace requirements denied"])
def test_permission_quota_and_io_classifications_do_not_retry(tmp_path, message):
    runner = wire_runner(tmp_path, "account/read")
    original = runner.command
    runner.command = lambda *args, **kwargs: original(*args, **kwargs) + [message]
    with pytest.raises(FrontierBlocked):
        metadata_with_recovery(runner, runner._metadata_once, "fixture", sleep=lambda seconds: pytest.fail("Denied condition retried"))
    assert (tmp_path / "counter").read_text() == "1"


def test_internal_rechecks_exhaust_exactly_three_attempts(tmp_path):
    runner = wire_runner(tmp_path, "account/read", 503)
    with pytest.raises(FrontierBlocked):
        metadata_with_recovery(runner, runner._metadata_once, "fixture", sleep=lambda seconds: None)
    assert (tmp_path / "counter").read_text() == "3"


def test_global_quota_denial_blocks_a_prepared_lane_before_model_start(tmp_path):
    runner = wire_runner(tmp_path / "denied", "account/read", 429)
    root = tmp_path / "run"
    runner.config["failure_root"] = str(root)
    with pytest.raises(FrontierBlocked, match="HTTP_429"):
        metadata_with_recovery(runner, runner._metadata_once, "fixture", sleep=lambda seconds: pytest.fail("429 retried"))
    assert (root / "ADMISSION_CLOSED.json").exists()
    prepared = wire_runner(tmp_path / "prepared", "account/read")
    prepared.config["failure_root"] = str(root)
    prepared.admitted = True
    with pytest.raises(FrontierBlocked, match="GLOBAL_MODEL_ADMISSION_CLOSED"):
        IsolatedCodex.command(prepared, ["exec"])


def test_global_hold_publication_serializes_with_model_start_intent(tmp_path):
    complete = threading.Event()
    def close():
        close_admission(tmp_path, "FIXTURE_QUOTA_HOLD")
        complete.set()
    with file_lock(tmp_path / "model-start.lock"):
        thread = threading.Thread(target=close)
        thread.start()
        assert not complete.wait(0.1)
    thread.join(timeout=5)
    assert complete.is_set() and (tmp_path / "ADMISSION_CLOSED.json").exists()


def test_explicit_resume_preserves_deadline_and_consumed_failed_attempt(tmp_path):
    old = {"status": "STOPPED", "deadline": time.time() + 100, "terminal": {"failed": {"outcome": "infrastructure_error"}}, "reserved": {"failed": {"episode_id": "original-failed"}}}
    write_json(tmp_path / "teacher-state.json", old)
    (tmp_path / "STOP").touch()
    resume_stopped(tmp_path)
    current = json.loads((tmp_path / "teacher-state.json").read_text())
    assert current == {**old, "status": "READY"} and not (tmp_path / "STOP").exists()
    write_json(tmp_path / "teacher-state.json", old)
    (tmp_path / "ADMISSION_CLOSED.json").write_text("{}")
    with pytest.raises(FrontierBlocked, match="NO_SAFETY_HOLD"):
        resume_stopped(tmp_path)


def test_guardian_only_starts_canonical_collector_and_never_restarts_admission_hold(tmp_path):
    config = {"output": str(tmp_path / "run"), "guardian_max_starts": 2}
    root = Path(config["output"])
    calls = []
    def fake_controller(command, **kwargs):
        assert command[3:6] == ["robouse.cli", "collect_astra", "--config"]
        calls.append(command)
        script = "import json,time; from pathlib import Path; p=Path(" + repr(str(root)) + "); (p/'ADMISSION_CLOSED.json').write_text('{}'); (p/'teacher-state.json').write_text(json.dumps({'status':'ADMISSION_CLOSED','valid_scores':0})); time.sleep(.05)"
        return subprocess.Popen([sys.executable, "-c", script], **kwargs)
    result = supervise(config, interval=0.02, popen=fake_controller)
    assert result["status"] == "SAFETY_HOLD" and len(calls) == 1
    assert result["model_calls_by_guardian"] == result["episode_retries"] == 0
    result = supervise(config, interval=0.02, popen=lambda *args, **kwargs: pytest.fail("Safety hold bypassed"))
    assert result["status"] == "SAFETY_HOLD"


def test_guardian_restart_budget_is_durable(tmp_path):
    root = tmp_path / "run"
    config = {"output": str(root), "guardian_max_starts": 2}
    calls = []
    def fake_exit(command, **kwargs):
        calls.append(command)
        return subprocess.Popen([sys.executable, "-c", "raise SystemExit(1)"], **kwargs)
    result = supervise(config, interval=0.02, popen=fake_exit)
    assert len(calls) == 2 and result["status"] == "CONTROLLER_START_LIMIT_REACHED"
    result = supervise(config, popen=lambda *args, **kwargs: pytest.fail("Restart budget reset"))
    assert result["controller_starts"] == 2


def test_observer_does_not_claim_terminal_or_processor_from_files_alone(tmp_path):
    source, job = native_fixture(tmp_path)
    data, output = tmp_path / "sealed", tmp_path / "observer"
    seal(source, data, job, 0)
    result = observe({"data_root": str(data), "output": str(output)})
    assert result["raw_artifact_integrity"] == "PASS"
    assert result["first_complete_terminal"] == "PENDING" and result["processor"] == "NOT_RUN"
    image = next(data.rglob("*.png"))
    image.write_bytes(b"synthetic-corruption")
    with pytest.raises(ValueError, match="artifact changed"):
        observe({"data_root": str(data), "output": str(output), "full_verify": True})
    assert (output / "RAW_INTEGRITY_HOLD.json").exists()


def test_processor_acceptance_failure_does_not_mislabel_raw_integrity(tmp_path):
    source, job = native_fixture(tmp_path)
    data, output = tmp_path / "sealed", tmp_path / "observer"
    seal(source, data, job, 0)
    with pytest.raises(ProcessorAcceptanceError):
        observe({"data_root": str(data), "output": str(output), "processor_model": "/path/to/not-configured-processor"})
    report = json.loads((output / "teacher-acceptance.json").read_text())
    assert report["raw_artifact_integrity"] == "PASS" and report["processor"] == "FAIL"
    assert not (output / "RAW_INTEGRITY_HOLD.json").exists()


def test_sealed_first_terminal_and_real_local_processor_are_separate_checks(tmp_path):
    model = os.environ.get("ROBOUSE_TEST_PROCESSOR")
    if not model:
        pytest.skip("Optional original local Qwen processor; no weights loaded")
    from agp_rl.adapters.tool_codec import TOOLS
    source, job = native_fixture(tmp_path)
    session = source / "agp_home/session"
    write_json(session / "policy_finished.json", {"account_identity_verified": True, "inference_process_started": True})
    data, output = tmp_path / "sealed", tmp_path / "observer"
    seal(source, data, job, 0)
    config = {"data_root": str(data), "output": str(output)}
    report = observe(config)
    assert report["first_complete_terminal"] == "PASS" and report["processor"] == "NOT_RUN"
    schemas = tmp_path / "schemas.json"
    write_json(schemas, {"schemas": {name: {"type": "object"} for name in TOOLS}})
    report = observe({**config, "processor_model": model, "processor_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc", "tool_schema": str(schemas)})
    assert report["processor"] == "PASS" and report["model_weights_loaded"] is False
    assert report["target_tokens"] > 0 and len(report["processor_signature"]) == 64


def test_a800_cuda_name_alone_is_not_render_acceptance():
    with pytest.raises(FrontierBlocked, match="A800_ISAAC_RGB"):
        verify_renderer({}, {"name": "NVIDIA A800", "uuid": "fixture-only-uuid"})
    assert verify_renderer({}, {"name": "NVIDIA RTX fixture", "uuid": "fixture-only-uuid"}) is None


def test_a800_renderer_rejects_synthetic_or_unbound_acceptance(tmp_path):
    from robouse.environments.robodojo.native_job import verify_sources
    path = tmp_path / "synthetic-render-proof.json"
    write_json(path, {"status": "PASS", "real_native_execution": False, "synthetic_fixture": True, "gpu_uuid": "fixture", "native_source_sha256": verify_sources(), "observed_at": time.time()})
    with pytest.raises(FrontierBlocked, match="STALE_OR_MISMATCHED"):
        verify_renderer({"render_acceptance": {"path": str(path), "sha256": sha256(path)}}, {"name": "A800", "uuid": "fixture"})


def test_sealed_failure_exclusions_are_frozen_metadata_and_keep_role_ownership(tmp_path):
    config, identity = cfg(tmp_path)
    path = tmp_path / "sealed-failures.jsonl"
    path.write_text(json.dumps({**identity, "outcome": "infrastructure_error", "physical_aliases": ["known-physical-fixture"]}) + "\n")
    config["excluded_identity_files"] = [{"path": str(path), "sha256": sha256(path)}]
    assert excluded_identities(config, [identity]) == {(identity["task"], 0, 1)}
    with pytest.raises(ValueError, match="Cross-role"):
        excluded_identities(config, [{**identity, "seed": 1, "physical_aliases": ["known-physical-fixture"]}])
    path.write_text(path.read_text() + "\n")
    with pytest.raises(FrontierBlocked, match="MANIFEST_CHANGED"):
        excluded_identities(config, [identity])


def test_metadata_lane_fault_preserves_failure_and_other_lane_dispatch(tmp_path, monkeypatch):
    from robouse.teacher import collect as collector

    config, identity = cfg(tmp_path)
    identities = [{**identity, "layout": layout} for layout in (1, 2, 3)]
    Path(config["identities"]).write_text("".join(json.dumps(row) + "\n" for row in identities))
    config.update(max_episodes=3, max_concurrency=2, account_lock_file=str(tmp_path / "account.lock"), min_free_gib=0, dataset_version="synthetic-lane-fixture-v1", cli_binary=sys.executable, bwrap_binary=sys.executable)
    config["workers"] = [{"renderer_gpu": gpu, "resource_lock_file": str(tmp_path / f"gpu-{gpu}.lock")} for gpu in (0, 1)]
    starts = []
    class Worker:
        def __init__(self, spec):
            self.config, self.children = spec, {}
            self.root = Path(spec["output"])
            self.root.mkdir(parents=True, exist_ok=True)
        def directory(self, episode):
            return self.root / episode
        def start_episode(self, episode, identity, group_id):
            folder = self.directory(episode)
            folder.mkdir()
            write_json(folder / "job.json", {**self.config, "episode_id": episode, "identity": identity})
            self.children[episode] = SimpleNamespace(poll=lambda: 0, returncode=0)
            starts.append(self.config["renderer_gpu"])
    class Runner:
        def __init__(self, *args, **kwargs): pass
        def metadata(self, *args): return {"inference_requests": 0}
    def idle(spec):
        if spec["renderer_gpu"] == 0:
            raise FrontierBlocked("METADATA_TIMEOUT")
    monkeypatch.setattr(collector, "NativeWorker", Worker)
    monkeypatch.setattr(collector, "IsolatedCodex", Runner)
    monkeypatch.setattr(collector, "worker_idle", idle)
    monkeypatch.setattr(collector, "resource_admission", lambda spec: {"expires_at": time.time() + 100})
    monkeypatch.setattr(collector, "seal", lambda *args: {"valid_score": False, "outcome": "infrastructure_error", "native_success": None})
    state = collector.run(config)
    assert starts == [1, 1] and len(state["terminal"]) == 3
    assert "0" in state["lane_holds"] and not (Path(config["output"]) / "ADMISSION_CLOSED.json").exists()
    collector.run(config, resume=True)
    assert starts == [1, 1]


def test_remote_a800_control_and_sealed_pull_over_actual_loopback_http(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOUSE_ENV_TOKEN", "fixture-only-loopback-secret-" + "a" * 24)
    config, identity = cfg(tmp_path)
    config["max_wall_seconds"] = 30
    template = tmp_path / "private-template.json"
    write_json(template, config)
    starts = []
    def fake_guardian(command, **kwargs):
        assert command[3] == "robouse.teacher.guardian"
        cfg = json.loads(Path(command[5]).read_text())
        starts.append(cfg)
        script = "import json,time; from pathlib import Path; r=Path(" + repr(cfg["output"]) + "); d=Path(" + repr(cfg["data_root"]) + "); d.mkdir(parents=True); (d/'teacher-dataset.json').write_text('{}'); (d/'original.rgb').write_bytes(b'original-synthetic-RGB-not-video-frames'); (r/'teacher-state.json').write_text(json.dumps({'status':'FINISHED_BOUNDED','valid_scores':0,'reserved':{},'terminal':{}})); time.sleep(.05)"
        return subprocess.Popen([sys.executable, "-c", script], **kwargs)
    service_config = {"teacher_config": str(template), "output": str(tmp_path / "service"), "port": 0}
    service = TeacherService(service_config, popen=fake_guardian)
    server = make_server(service_config, service)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = {"endpoint": f"http://127.0.0.1:{server.server_address[1]}", "campaign_id": "synthetic-campaign", "plan_sha256": service.plan, "output": str(tmp_path / "a800-control"), "data_root": str(tmp_path / "a800-data"), "max_episodes": 1, "max_wall_seconds": 20, "poll_seconds": 0.02}
    try:
        result = remote_run(client)
        assert result["terminal"] and len(starts) == 1
        assert (Path(client["data_root"]) / "original.rgb").read_bytes() == b"original-synthetic-RGB-not-video-frames"
        remote_run({**client, "resume": client["output"]})
        assert len(starts) == 1
        with pytest.raises(FrontierBlocked, match="IDEMPOTENCY"):
            service.handle({"op": "teacher_start", "episode_id": client["campaign_id"], "plan_sha256": service.plan, "max_episodes": 1, "max_wall_seconds": 19})
        transport = TeacherClient(client["endpoint"], client["campaign_id"], tmp_path / "transport-proof")
        with pytest.raises(FrontierBlocked, match="IDEMPOTENCY"):
            transport.request("teacher_start", plan_sha256=service.plan, max_episodes=1, max_wall_seconds=19)
        with pytest.raises(ValueError, match="manifest-bound"):
            service.handle({"op": "blob", "episode_id": client["campaign_id"], "path": "../../auth.json", "offset": 0})
        with pytest.raises(FrontierBlocked, match="ALREADY_CLAIMED"):
            service.handle({"op": "teacher_start", "episode_id": "different-id-same-plan", "plan_sha256": service.plan, "max_episodes": 1, "max_wall_seconds": 20})
    finally:
        server.shutdown()
        server.server_close()


def test_uncertain_remote_process_start_cannot_replay_on_resume(tmp_path):
    config, identity = cfg(tmp_path)
    template = tmp_path / "template.json"
    write_json(template, config)
    starts = []
    def failed_start(*args, **kwargs):
        starts.append(1)
        raise OSError("synthetic spawn boundary failure")
    service = TeacherService({"teacher_config": str(template), "output": str(tmp_path / "service")}, popen=failed_start)
    request = {"op": "teacher_start", "episode_id": "uncertain-fixture", "plan_sha256": service.plan, "max_episodes": 1, "max_wall_seconds": 50}
    with pytest.raises(OSError):
        service.handle(request)
    assert service.status(request["episode_id"])["blocked_without_terminal_proof"]
    with pytest.raises(FrontierBlocked, match="UNCERTAIN_NO_AUTOMATIC_REPLAY"):
        service.handle({**request, "resume": True})
    assert len(starts) == 1
