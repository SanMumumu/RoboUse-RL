"""At most three pre-model metadata checks; no inference or episode retries."""
import os
from pathlib import Path
import time
from robouse.teacher.admission import FrontierBlocked
from robouse.teacher.rpc_policy import retry_kind
from robouse.teacher.failures import emit


def close_admission(root, reason):
    from robouse.teacher.state import file_lock
    from robouse.data.datasets import write_json

    root = Path(root)
    with file_lock(root / "model-start.lock", wait_seconds=30):
        write_json(root / "ADMISSION_CLOSED.json", {"reason": reason, "automatic_account_switch": False})


def writable_probe(directory):
    directory = Path(directory)
    name = directory / (".admission-probe-" + str(time.time_ns()))
    moved = name.with_suffix(".committed")
    try:
        with name.open("x") as stream:
            stream.write("admission-storage-probe")
            stream.flush()
            os.fsync(stream.fileno())
        name.replace(moved)
        if moved.read_text() != "admission-storage-probe":
            raise OSError("Storage readback mismatch")
    finally:
        name.unlink(missing_ok=True)
        moved.unlink(missing_ok=True)


def prestart(runner):
    if runner.admitted or getattr(runner, "model_command_issued", False):
        return False
    session = runner.config.get("workspace")
    if session:
        session = Path(session)
        if any((session / name).exists() for name in ("model-process-start-intent.json", "agent.pid")):
            return False
        events = session / "agent_events.jsonl"
        if events.exists() and events.stat().st_size:
            return False
    runner.verify_identity()
    writable_probe(runner.directory)
    writable_probe(runner.profile)
    return True


def positive_prior(runner, now=None):
    now = time.time() if now is None else now
    proof = getattr(runner, "last_verified_metadata", None)
    return bool(proof and proof.get("account_metadata_admitted") is True
                and proof.get("account_sha256") == runner.identity["account_sha256"]
                and proof.get("model") == runner.config["model"]
                and proof.get("effort") == runner.config["effort"]
                and proof.get("credential_file_sha256") == runner.identity["credential_file_sha256"]
                and 0 <= now - proof.get("observed_at", 0) <= runner.config.get("retry_prior_max_age_s", 900)
                and proof.get("remaining_percent", 0) > 20
                and proof.get("catalog", {}).get("advertised") is True)


def metadata_with_recovery(runner, once, label, *, sleep=time.sleep):
    attempts = runner.config.get("metadata_max_attempts", 3)
    if type(attempts) is not int or not 1 <= attempts <= 3:
        raise FrontierBlocked("METADATA_ATTEMPT_LIMIT_MUST_BE_1_TO_3")
    for attempt in range(1, attempts + 1):
        try:
            result = once(label)
            runner.last_verified_metadata = result
            if attempt > 1:
                emit(runner.config.get("failure_root", runner.directory), classification="METADATA_RECOVERED", stage=label, attempt=attempt,
                     kind="RECOVERED", decision="SAME_ACCOUNT_MODEL_QUOTA_VERIFIED")
            return result
        except Exception as exc:
            kind = retry_kind(exc)
            if kind == "HOLD" and runner.config.get("failure_root"):
                close_admission(runner.config["failure_root"], str(exc) if isinstance(exc, FrontierBlocked) else type(exc).__name__)
            safe = False
            safety_error = None
            if kind in ("NETWORK", "INTERNAL_CANDIDATE") and attempt < attempts:
                try:
                    safe = prestart(runner) and (kind == "NETWORK" or positive_prior(runner))
                except Exception as failure:
                    safety_error = failure
            emit(runner.config.get("failure_root", runner.directory), classification=str(exc).split(":", 1)[0] if isinstance(exc, FrontierBlocked) else type(exc).__name__,
                 stage=label, attempt=attempt, kind=kind,
                 decision="SAME_ATTEMPT_METADATA_RECHECK" if safe else "HOLD_NO_MODEL_START",
                 rpc_method=getattr(exc, "rpc_method", None), rpc_code=getattr(exc, "rpc_code", None),
                 http_status=getattr(exc, "http_status", None))
            if safety_error is not None:
                raise safety_error from exc
            if not safe:
                raise
            sleep((2, 5)[attempt - 1])


def lane_local_hold(exc):
    """Unknown transient bootstrap/resource faults do not disable healthy lanes."""
    code = str(exc).upper()
    if any(word in code for word in ("HTTP_401", "HTTP_403", "HTTP_429", "PERMISSION", "QUOTA", "LOGIN", "CREDENTIAL", "IDENTITY", "SETTINGS", "STORAGE", "LOCAL_IO")):
        return False
    if retry_kind(exc) in ("NETWORK", "INTERNAL_CANDIDATE"):
        return True
    return code.startswith(("RESOURCE_", "NO_CURRENT_EXPLICIT_RTX", "RTX_HAS_EXISTING", "A800_"))
