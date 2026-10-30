"""Immutable raw teacher evidence and hash-bound snapshots for the existing native adapter."""
import json
import os
from pathlib import Path
import shutil
import time
from agp_rl.io import sha256, digest
from agp_rl.reward.official import read_official
from robouse.data.datasets import write_json, read_jsonl


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("Immutable teacher artifact differs")
    else:
        write_json(path, value)


def copy_once(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(source) != sha256(target):
            raise ValueError("Archived teacher source changed")
    else:
        before = source.stat()
        temp = target.with_suffix(target.suffix + ".partial")
        shutil.copyfile(source, temp)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ) or sha256(temp) != sha256(source):
            raise ValueError("Teacher artifact still changing; no partial seal")
        with temp.open("rb") as stream:
            os.fsync(stream.fileno())
        temp.replace(target)
    return target


def transcript_evidence(paths, session, model, effort, tier):
    observed = []
    completed = 0
    usage = {}
    provider_error = False
    if len(paths) != 1:
        raise ValueError(
            "Exactly one native teacher transcript required; no concatenated or synthetic trace"
        )
    path = Path(paths[0])
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("Teacher transcript is still incomplete")
    records = [json.loads(line) for line in raw.splitlines()]
    if not any(
        r.get("type") == "session_meta"
        and r.get("payload", {}).get("cwd") == str(Path(session).resolve())
        for r in records
    ):
        raise ValueError("Native transcript is not this teacher session")
    for rec in records:
        payload = rec.get("payload", {})
        if rec.get("type") == "turn_context":
            actual = (
                payload.get("model"),
                payload.get("effort", payload.get("reasoning_effort")),
            )
            if actual != (model, effort):
                raise ValueError("Observed teacher model/effort differs; no relabeling")
            if payload.get("service_tier") not in (None, tier):
                raise ValueError("Observed service tier differs")
            observed.append(
                {
                    "model": actual[0],
                    "effort": actual[1],
                    "response_service_tier": payload.get("service_tier"),
                }
            )
        if rec.get("type") == "event_msg" and payload.get("type") == "token_count":
            total = (payload.get("info") or {}).get("total_token_usage")
            if isinstance(total, dict):
                usage = {k: v for k, v in total.items() if type(v) is int and v >= 0}
    if not observed:
        raise ValueError("No native turn_context evidence")
    return {
        "observed_settings": observed,
        "token_usage": usage or None,
        "requested_service_tier": tier,
        "response_model_snapshot": None,
        "hidden_reasoning_supervised": False,
    }


def atomic_text(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def publish(root, dest, terminal):
    from robouse.teacher.state import file_lock

    index = json.loads((dest / "trajectory-index.json").read_text())
    if index["terminal"] != terminal:
        raise ValueError("Sealed terminal/index mismatch")
    manifest = root / index["artifacts"]["path"]
    if sha256(manifest) != index["artifacts"]["sha256"]:
        raise ValueError("Sealed artifact manifest changed")
    for row in read_jsonl(manifest):
        p = (root / row["path"]).resolve()
        if (
            not p.is_relative_to(dest.resolve())
            or p.stat().st_size != row["bytes"]
            or sha256(p) != row["sha256"]
        ):
            raise ValueError("Sealed teacher evidence changed")
    index_path = root / f"seed-{terminal['seed']}" / "indexes/episodes.jsonl"
    entry = {
        "path": str((dest / "trajectory-index.json").relative_to(root)),
        "sha256": sha256(dest / "trajectory-index.json"),
        "episode_id": terminal["episode_id"],
        "valid_score": terminal["valid_score"],
        "model": terminal["model"],
        "effort": terminal["effort"],
    }
    with file_lock(index_path.with_suffix(".lock"), wait_seconds=10):
        rows = read_jsonl(index_path) if index_path.exists() else []
        match = [r for r in rows if r["episode_id"] == entry["episode_id"]]
        if match and match != [entry]:
            raise ValueError("Conflicting sealed episode index")
        if not match:
            atomic_text(
                index_path, "".join(json.dumps(r) + "\n" for r in rows + [entry])
            )


def seal(source, data_root, job, returncode):
    source = Path(source).resolve()
    root = Path(data_root).resolve()
    identity = job["identity"]
    seed = identity["seed"]
    dest = root / f"seed-{seed}" / "episodes" / job["episode_id"] / "attempt-01"
    if (dest / "trajectory-index.json").exists():
        index = json.loads((dest / "trajectory-index.json").read_text())
        if (
            index["identity"]["seed"] != seed
            or index["terminal"]["episode_id"] != job["episode_id"]
        ):
            raise ValueError("Partial seal belongs to another identity")
        immutable_json(dest / "terminal.json", index["terminal"])
        publish(root, dest, index["terminal"])
        return index["terminal"]
    dest.mkdir(parents=True, exist_ok=True)
    session_ref = source / "native-session.json"
    session = (
        Path(json.loads(session_ref.read_text())["path"])
        if session_ref.exists()
        else None
    )
    terminals = {
        "episode_id": job["episode_id"],
        "seed": seed,
        "layout_id": identity["layout"],
        "layout_sha256": identity["layout_sha256"],
        "role": {0: "train", 1: "validation", 2: "test"}[seed],
        "valid_score": False,
        "native_success": None,
        "native_score": None,
        "session": str(session) if session else None,
        "returncode": returncode,
        "model": job["teacher"]["model"],
        "effort": job["teacher"]["effort"],
    }
    evidence = {}
    infra = []
    if session is not None:
        for path in session.rglob("*.png"):
            if path.is_symlink():
                raise ValueError("Teacher image symlink refused")
            copy_once(path, dest / "native-session" / path.relative_to(session))
        for name in (
            "episode_metrics.json",
            "policy_finished.json",
            "teacher-infrastructure.json",
            "teacher-request.json",
            "teacher-admission.json",
            "model-process-start-intent.json",
            "agent_events.jsonl",
        ):
            if (session / name).is_file():
                copy_once(session / name, dest / "native-session" / name)
        metrics = (
            json.loads((session / "episode_metrics.json").read_text())
            if (session / "episode_metrics.json").exists()
            else {}
        )
        finished = (
            json.loads((session / "policy_finished.json").read_text())
            if (session / "policy_finished.json").exists()
            else {}
        )
        if metrics.get("infrastructure_failure") not in (None, False):
            infra.append("NATIVE_INFRASTRUCTURE_FAILURE")
        if (session / "teacher-infrastructure.json").exists() and not metrics.get(
            "official_ended_before_finalize"
        ):
            infra.append("TEACHER_INFRASTRUCTURE_FAILURE")
        terminals["model_start_reserved"] = (
            session / "model-process-start-intent.json"
        ).exists()
        terminals["model_process_started"] = finished.get("inference_process_started")
        if finished.get("account_identity_verified") is not True:
            infra.append("TEACHER_ACCOUNT_UNVERIFIED")
        if (source / "CANCEL").exists() and not metrics.get(
            "official_ended_before_finalize"
        ):
            infra.append("OPERATOR_INTERRUPTED")
        traces = list(
            (source / "private-teacher/profile/sessions").rglob("rollout-*.jsonl")
        )
        for trace in traces:
            copy_once(trace, dest / "codex_transcripts" / trace.name)
        try:
            evidence = transcript_evidence(
                traces,
                session,
                job["teacher"]["model"],
                job["teacher"]["effort"],
                job["teacher"]["service_tier"],
            )
        except (ValueError, OSError, KeyError):
            infra.append("NATIVE_TEACHER_TRANSCRIPT_UNVERIFIED")
        events = session / "agent_events.jsonl"
        if (
            events.exists()
            and any(
                r.get("type") in ("error", "turn.failed") for r in read_jsonl(events)
            )
            and not metrics.get("official_ended_before_finalize")
        ):
            infra.append("PROVIDER_OR_TURN_FAILURE")
    else:
        infra.append("NATIVE_SESSION_NOT_STARTED")

    def ref(path):
        return {"path": str(path.relative_to(root)), "sha256": sha256(path)}

    results = list((source / "work").rglob("_result.json"))
    if len(results) == 1:
        archived = copy_once(results[0], dest / "native-result" / "_result.json")
        terminals["native_result"] = ref(archived)
        if not infra:
            native = read_official(archived, identity["layout"])
            terminals.update(
                valid_score=True,
                native_success=native.success,
                native_score=native.score,
            )
    else:
        infra.append("NATIVE_SCORE_NOT_PUBLISHED")
    metrics_path = dest / "native-session/episode_metrics.json"
    if metrics_path.exists():
        terminals["native_metrics"] = ref(metrics_path)
    for path in source.rglob("*.mp4"):
        if path.stat().st_size > 48:
            copy_once(path, dest / "videos" / path.name)
    copy_once(source / "job.json", dest / "job.json")
    immutable_json(dest / "teacher-provenance.json", evidence)
    terminals.update(
        unscored_reason=infra or None,
        outcome="interrupted"
        if "OPERATOR_INTERRUPTED" in infra
        else "infrastructure_error"
        if not terminals["valid_score"]
        else "completed"
        if terminals["native_success"]
        else "policy_failure",
    )
    artifacts = []
    for path in sorted(dest.rglob("*")):
        if (
            not path.is_file()
            or path.name
            in ("artifacts.jsonl", "terminal.json", "trajectory-index.json")
            or path.name.endswith(".partial")
        ):
            continue
        row = {**ref(path), "bytes": path.stat().st_size}
        if (
            session
            and path.suffix == ".png"
            and path.is_relative_to(dest / "native-session")
        ):
            row["original_path"] = str(
                session / path.relative_to(dest / "native-session")
            )
        artifacts.append(row)
    manifest = dest / "artifacts.jsonl"
    payload = "".join(json.dumps(r) + "\n" for r in artifacts)
    if manifest.exists() and manifest.read_text() != payload:
        raise ValueError("Partial sealed manifest differs")
    if not manifest.exists():
        atomic_text(manifest, payload)
    index = {
        "source_id": "local:astra-teacher-seedroles-v1",
        "teacher_model": job["teacher"]["model"],
        "teacher_effort": job["teacher"]["effort"],
        "identity": {
            **identity,
            "base_task": identity.get("base_task", identity["task"]),
        },
        "terminal": terminals,
        "role": terminals["role"],
        "artifacts": ref(manifest),
        "teacher_provenance": ref(dest / "teacher-provenance.json"),
    }
    immutable_json(dest / "trajectory-index.json", index)
    immutable_json(dest / "terminal.json", terminals)
    publish(root, dest, terminals)
    return terminals


def snapshot(data_root, seed, destination):
    root = Path(data_root).resolve()
    index = root / f"seed-{seed}" / "indexes/episodes.jsonl"
    from robouse.teacher.state import file_lock

    with file_lock(index.with_suffix(".lock"), wait_seconds=10):
        rows = read_jsonl(index) if index.exists() else []
        for row in rows:
            path = root / row["path"]
            if (
                sha256(path) != row["sha256"]
                or not (path.parent / "terminal.json").exists()
            ):
                raise ValueError("Snapshot references incomplete/changed seal")
        payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.read_text() != payload:
            raise ValueError("Immutable snapshot already exists with another cutoff")
        if not dest.exists():
            with dest.open("x") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
    return {
        "path": str(dest),
        "sha256": sha256(dest),
        "episodes": len(rows),
        "seed": seed,
        "model_inference_requests": 0,
    }
