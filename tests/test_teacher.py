"""Astra teacher tests use synthetic identities, fake RPC and synthetic native evidence only."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import pytest
from agp_rl.io import sha256
from agp_rl.adapters.tool_codec import TOOLS
from robouse.teacher.admission import FrontierBlocked, hashed, normalize_quota
from robouse.teacher.provider import IsolatedCodex, model_permission, isolated_environment
from robouse.teacher.state import Budget, validate, file_lock, resource_admission
from robouse.teacher.seal import seal, snapshot


def quota():
    return {
        "rateLimits": {
            "primary": {
                "usedPercent": 10,
                "windowDurationMins": 300,
                "resetsAt": time.time() + 9000,
            },
            "secondary": {
                "usedPercent": 15,
                "windowDurationMins": 10080,
                "resetsAt": time.time() + 9000,
            },
        }
    }


def cfg(tmp_path):
    layout = tmp_path / "layout.json"
    layout.write_text('{"synthetic_layout":true}')
    identity = {
        "task": "task",
        "seed": 0,
        "layout": 1,
        "layout_path": str(layout),
        "layout_sha256": sha256(layout),
    }
    ids = tmp_path / "identities.jsonl"
    ids.write_text(json.dumps(identity) + "\n")
    return {
        "model": "gpt-6-astra",
        "effort": "high",
        "service_tier": "default",
        "max_episodes": 1,
        "max_wall_seconds": 100,
        "episode_wall_seconds": 30,
        "max_extra_cost_usd": 0,
        "paid_credits_authorized": False,
        "automatic_retries": 0,
        "identities": str(ids),
        "workers": [{"renderer_gpu": 0}],
        "output": str(tmp_path / "run"),
        "data_root": str(tmp_path / "data"),
        "cli_sha256": "fake-cli-pin",
        "cli_version": "fake-cli",
        "bwrap_sha256": "fake-bwrap-pin",
    }, identity


@pytest.mark.parametrize("effort", ["high", "medium", "max"])
def test_configured_effort_requires_exact_catalog_support(effort):
    unsupported = {
        "data": [{"id": "gpt-6-astra", "supportedReasoningEfforts": ["low"]}]
    }
    with pytest.raises(FrontierBlocked, match="NO_DOWNGRADE"):
        model_permission(unsupported, "gpt-6-astra", effort)
    catalog = {
        "data": [{
            "id": "gpt-6-astra",
            "supportedReasoningEfforts": [{"reasoningEffort": effort}],
        }]
    }
    result = model_permission(catalog, "gpt-6-astra", effort)
    assert result["effort"] == effort
    assert result["inference_entitlement_verified"] is False


def test_quota_unknown_low_or_paid_fallback_fails_closed():
    identity = {"account_sha256": "hash", "user_sha256": "user"}
    assert normalize_quota(quota(), identity, time.time())["allowed"]
    for value in (
        {},
        {"rateLimits": {}},
        {"rateLimits": {"credits": {"unlimited": True}, **quota()["rateLimits"]}},
    ):
        try:
            assert not normalize_quota(value, identity, time.time())["allowed"]
        except FrontierBlocked:
            pass
    low = quota()
    low["rateLimits"]["secondary"]["usedPercent"] = 80
    assert not normalize_quota(low, identity, time.time())["allowed"]


def test_budget_resume_does_not_reset_wall_or_repeat_reservation(tmp_path):
    config, identity = cfg(tmp_path)
    validate(config)
    budget = Budget(config, [identity])
    reservation = budget.reserve(identity, 0)
    resumed = Budget(config, [identity], resume=True)
    assert (
        resumed.value["deadline"] == budget.value["deadline"]
        and resumed.reserve(identity, 0) == reservation
    )
    other = {**identity, "layout": 2}
    assert resumed.reserve(other, 0) is None
    resumed.complete(
        identity, {"outcome": "infrastructure_error", "valid_score": False}
    )
    assert resumed.reserve(identity, 0) is None
    with pytest.raises(FrontierBlocked):
        Budget({**config, "effort": "medium"}, [identity], resume=True)


def test_model_and_effort_support_comes_from_live_catalog(tmp_path):
    config, identity = cfg(tmp_path)
    config.update(model="synthetic-configured-model", effort="synthetic-new-effort")
    assert validate(config) == [identity]
    catalog = {"data": [{"id": config["model"], "supportedReasoningEfforts": [config["effort"]]}]}
    assert model_permission(catalog, config["model"], config["effort"])["advertised"]
    for value in (None, "", " high ", 1):
        with pytest.raises(FrontierBlocked, match="EXPLICIT_MODEL"):
            validate({**config, "effort": value})


def test_seed2_requires_final_flag_and_no_identity_alias_leak(tmp_path):
    config, identity = cfg(tmp_path)
    path = Path(config["identities"])
    path.write_text(json.dumps({**identity, "seed": 2}) + "\n")
    with pytest.raises(FrontierBlocked, match="FINAL_TEST"):
        validate(config)
    validate(config, final_test=True)
    path.write_text(
        json.dumps(identity) + "\n" + json.dumps({**identity, "seed": 1}) + "\n"
    )
    with pytest.raises(ValueError, match="Cross-role"):
        validate(config)


def test_owned_locks_and_resource_lease_no_preemption(tmp_path):
    with file_lock(tmp_path / "controller.lock"):
        with pytest.raises(FrontierBlocked):
            with file_lock(tmp_path / "controller.lock"):
                pass
    lease = tmp_path / "lease.json"
    lease.write_text(
        json.dumps(
            {
                "allocated": True,
                "owner": "this-fixture",
                "gpu": 0,
                "expires_at": time.time() + 100,
            }
        )
    )
    worker = {
        "renderer_gpu": 0,
        "resource_owner": "this-fixture",
        "resource_lease": {"path": str(lease), "sha256": sha256(lease)},
    }
    resource_admission(worker)
    with pytest.raises(FrontierBlocked):
        resource_admission(worker, now=time.time() + 200)


def jwt(value):
    return (
        "header."
        + base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")
        + ".synthetic-signature"
    )


def test_fake_metadata_has_no_model_turns_or_auth_copy(tmp_path, monkeypatch):
    import robouse.teacher.provider as provider

    auth = tmp_path / "selected-auth"
    auth.mkdir()
    ns = "https://api.openai.com/auth"
    payload = {
        "tokens": {
            "id_token": jwt(
                {
                    "sub": "fake-user",
                    "email": "fake@example.invalid",
                    ns: {"chatgpt_account_id": "fake-account"},
                }
            ),
            "access_token": jwt(
                {"exp": time.time() + 10000, ns: {"chatgpt_account_id": "fake-account"}}
            ),
            "account_id": "fake-account",
        }
    }
    (auth / "auth.json").write_text(json.dumps(payload))
    before = (auth / "auth.json").read_bytes()
    monkeypatch.setenv("ASTRA_AUTH_HOME", str(auth))
    monkeypatch.setenv("ASTRA_ACCOUNT_SHA256", hashed("fake-account"))
    monkeypatch.setenv("OPENAI_API_KEY", "not-to-be-forwarded")
    config = {
        "cli_binary": sys.executable,
        "cli_sha256": sha256(sys.executable),
        "bwrap_binary": sys.executable,
        "bwrap_sha256": sha256(sys.executable),
        "cli_version": "fake-codex",
        "model": "gpt-6-astra",
        "effort": "high",
        "service_tier": "default",
    }
    monkeypatch.setattr(
        provider.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="fake-codex\n"),
    )
    writes = []

    class Input(io.StringIO):
        def write(self, value):
            writes.append(json.loads(value))
            return super().write(value)

    class Process:
        def __init__(self, *args, **kwargs):
            assert "OPENAI_API_KEY" not in kwargs["env"]
            results = [
                {},
                {"account": {"type": "chatgpt", "email": "fake@example.invalid"}},
                quota(),
                {
                    "data": [
                        {"id": "gpt-6-astra", "supportedReasoningEfforts": ["high"]}
                    ]
                },
            ]
            self.stdin = Input()
            self.stdout = io.StringIO(
                "".join(
                    json.dumps({"id": i + 1, "result": v}) + "\n"
                    for i, v in enumerate(results)
                )
            )

        def poll(self):
            return 0

    monkeypatch.setattr(provider.subprocess, "Popen", Process)
    runner = IsolatedCodex(config, tmp_path / "isolated")
    with pytest.raises(FrontierBlocked):
        runner.command(["exec"])
    proof = runner.metadata("fixture")
    assert proof["account_metadata_admitted"] and proof["inference_requests"] == 0
    assert {r["method"] for r in writes} == {
        "initialize",
        "initialized",
        "account/read",
        "account/rateLimits/read",
        "model/list",
    }
    assert (runner.profile / "auth.json").read_bytes() == b"" and (
        auth / "auth.json"
    ).read_bytes() == before
    assert isolated_environment(runner.profile).get("HOME") == os.environ.get("HOME")
    config.update(max_extra_cost_usd=0, paid_credits_authorized=False)
    runner.admit(max_wall_seconds=30)
    command = runner.command(["exec"])
    assert "--ro-bind" in command
    with pytest.raises(FrontierBlocked, match="SINGLE_USE"):
        runner.command(["exec"])


def native_fixture(tmp_path, *, infra=False):
    from PIL import Image

    source = tmp_path / "worker/episode"
    session = source / "agp_home/session"
    session.mkdir(parents=True)
    image = session / "observation.png"
    Image.new("RGB", (4, 4), "red").save(image)
    config, identity = cfg(tmp_path)
    job = {"episode_id": "fixture-teacher", "identity": identity, "teacher": config}
    (source / "job.json").write_text(json.dumps(job))
    (source / "native-session.json").write_text(json.dumps({"path": str(session)}))
    metrics = {
        "infrastructure_failure": "storage_failure" if infra else None,
        "official_ended_before_finalize": not infra,
        "counts_match": True,
        "marked_failed": False,
    }
    (session / "episode_metrics.json").write_text(json.dumps(metrics))
    (session / "policy_finished.json").write_text(
        json.dumps({"account_identity_verified": True})
    )
    profile = source / "private-teacher/profile/sessions"
    profile.mkdir(parents=True)
    records = [
        {"type": "session_meta", "payload": {"cwd": str(session.resolve())}},
        {
            "type": "turn_context",
            "payload": {"model": "gpt-6-astra", "effort": "high"},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Synthetic fixture task"},
                    {"type": "input_image", "image_url": str(image.resolve())},
                ],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "observe",
                "arguments": "{}",
                "call_id": "c",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "c",
                "output": {"content": [{"type": "text", "text": '{"ok":true}'}]},
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done"}],
            },
        },
    ]
    (profile / "rollout-fixture.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    (source / "work").mkdir()
    (source / "work/_result.json").write_text(
        json.dumps(
            {
                "eval_time": 1,
                "success_rate": 1.0,
                "score": 100.0,
                "details": {"0": {"layout_id": 1, "success": True, "score": 1.0}},
            }
        )
    )
    return source, job


def test_native_seal_snapshot_resume_and_relocated_prepare_adapter(
    tmp_path, monkeypatch
):
    source, job = native_fixture(tmp_path)
    root = tmp_path / "dataset"
    terminal = seal(source, root, job, 0)
    assert terminal["valid_score"] and terminal["effort"] == "high"
    snapshot_path = root / "seed-0/snapshots/frozen.jsonl"
    proof = snapshot(root, 0, snapshot_path)
    # Simulate crash after terminal but before publication: next seal recovers index.
    (root / "seed-0/indexes/episodes.jsonl").unlink()
    assert seal(source, root, job, None) == terminal
    assert snapshot(root, 0, snapshot_path) == proof
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    from agp_rl.pipeline2b.fullft import source_adapters as adapters

    monkeypatch.setattr(adapters, "ROOT", relocated.resolve())
    refs = [json.loads(s) for s in snapshot_path.read_text().splitlines()]
    schemas = {name: {"type": "object"} for name in TOOLS}
    results = list(
        adapters.native_candidates(
            {"root": str(relocated / "seed-0"), "seed": 0, "name": "astra-high"},
            adapters.Verifier(),
            schemas,
            refs,
        )
    )
    assert results[0][1] is not None, results[0][0]
    candidate = results[0][1]
    assert len(candidate["bundle"]["decisions"]) == 2
    assert all(
        Path(r["path"]).is_relative_to(relocated) for r in candidate["images"].values()
    )


def test_infrastructure_native_failure_is_not_reward_zero(tmp_path):
    source, job = native_fixture(tmp_path, infra=True)
    terminal = seal(source, tmp_path / "dataset", job, 1)
    assert terminal["valid_score"] is False and terminal["native_success"] is None
    assert (
        terminal["outcome"] == "infrastructure_error"
        and "NATIVE_INFRASTRUCTURE_FAILURE" in terminal["unscored_reason"]
    )


def test_bounded_collector_resume_without_real_gpu_or_teacher_calls(
    tmp_path, monkeypatch
):
    import robouse.teacher.collect as collector

    config, identity = cfg(tmp_path)
    config.update(
        dataset_version="synthetic-teacher-test",
        account_lock_file=str(tmp_path / "account.lock"),
        min_free_gib=0,
    )
    config.update(cli_binary="/fake/pinned/codex", bwrap_binary="/fake/pinned/bwrap")
    config["workers"] = [
        {
            "renderer_gpu": 0,
            "resource_lock_file": str(tmp_path / "gpu.lock"),
            "simulator_python": "not-executed",
            "transport_python": "not-executed",
        }
    ]
    calls = []

    class Runner:
        def __init__(self, *a, **k):
            pass

        def metadata(self, label):
            calls.append(("metadata", label))
            return {"inference_requests": 0}

    class Worker:
        def __init__(self, spec):
            self.config = spec
            self.root = Path(spec["output"])
            self.root.mkdir(parents=True, exist_ok=True)
            self.children = {}

        def directory(self, episode):
            return self.root / episode

        def start_episode(self, episode, identity, group):
            calls.append(("synthetic_episode", episode))
            temp = tmp_path / "fixture-native"
            temp.mkdir(exist_ok=True)
            source, job = native_fixture(temp)
            folder = self.directory(episode)
            shutil.copytree(source, folder)
            for path in folder.rglob("*"):
                if path.suffix in (".json", ".jsonl"):
                    path.write_text(path.read_text().replace(str(source), str(folder)))
            job = {**self.config, "episode_id": episode, "identity": identity}
            (folder / "job.json").write_text(json.dumps(job))
            self.children[episode] = SimpleNamespace(poll=lambda: 0, returncode=0)
            return folder

        def alive(self, *args):
            return False

    monkeypatch.setattr(collector, "NativeWorker", Worker)
    monkeypatch.setattr(collector, "IsolatedCodex", Runner)
    monkeypatch.setattr(collector, "worker_idle", lambda spec: None)
    monkeypatch.setattr(
        collector, "resource_admission", lambda spec: {"expires_at": time.time() + 100}
    )
    result = collector.run(config)
    assert len(result["reserved"]) == 1 and len(result["terminal"]) == 1
    assert list(result["terminal"].values())[0]["valid_score"] is True
    before = list(calls)
    resumed = collector.run(config, resume=True)
    assert calls == before and resumed["deadline"] == result["deadline"]
    assert (Path(config["output"]) / "snapshots.json").exists()


def test_observed_effort_mismatch_is_not_published(tmp_path):
    source, job = native_fixture(tmp_path)
    trace = next((source / "private-teacher/profile/sessions").glob("*.jsonl"))
    trace.write_text(trace.read_text().replace('"effort": "high"', '"effort": "medium"'))
    terminal = seal(source, tmp_path / "dataset", job, 0)
    assert terminal["valid_score"] is False and terminal["native_success"] is None
    assert "NATIVE_TEACHER_TRANSCRIPT_UNVERIFIED" in terminal["unscored_reason"]
    assert list(
        (
            tmp_path
            / "dataset/seed-0/episodes/fixture-teacher/attempt-01/codex_transcripts"
        ).glob("*.jsonl")
    )


def test_teacher_exec_settings_use_only_original_robot_tools(tmp_path):
    from robouse.teacher.actor import exec_args

    config = {"model": "gpt-6-astra", "effort": "high", "service_tier": "default"}
    args = exec_args(
        config,
        tmp_path,
        Path(sys.executable),
        8765,
        {"run": True},
        [tmp_path / "real-input.png"],
        "skills.config=[]",
    )
    assert args.count("-m") == 1 and args[args.index("-m") + 1] == "gpt-6-astra"
    assert 'model_reasoning_effort="high"' in args
    assert 'service_tier="default"' in args
    assert "features.shell_tool=false" in args and "features.multi_agent=false" in args
    assert any(s.startswith("mcp_servers.robot_js.url=") for s in args)
    assert "--dangerously-bypass-approvals-and-sandbox" not in args
