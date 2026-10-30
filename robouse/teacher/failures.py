"""Credential-free failure spool and idempotent, locked Markdown journal.

Derived from the research collector's failure_journal; runtime records stay in
the run directory, never in the checked-in knowledge-base template.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from agp_rl.io import digest, sha256
from robouse.data.datasets import write_json
from robouse.teacher.state import file_lock


def emit(root, *, classification, stage, attempt=None, decision="HOLD_NO_MODEL_START", kind="HOLD", rpc_method=None, rpc_code=None, http_status=None):
    def controlled(value):
        return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_:/.-]{1,180}", value) else "UNCLASSIFIED"

    event = {
        "utc": datetime.now(timezone.utc).isoformat(), "time_ns": time.time_ns(),
        "classification": controlled(classification), "stage": controlled(stage),
        "attempt": attempt, "decision": controlled(decision), "kind": controlled(kind),
        "rpc_method": controlled(rpc_method) if rpc_method else None,
        "rpc_code": rpc_code if type(rpc_code) is int else None,
        "http_status": http_status if type(http_status) is int else None,
        "root_cause": "Unknown", "model_calls_by_recorder": 0,
    }
    event["event_id"] = digest(event)
    path = Path(root) / "failure-spool" / (event["event_id"] + ".json")
    write_json(path, event)
    return path


def flush(root):
    root = Path(root)
    document = root / "SEED1_ROLLOUT_FAILURES.md"
    count = 0
    with file_lock(root / "failure-journal.lock", wait_seconds=2):
        text = document.read_text() if document.exists() else "# Collection failure journal\n\nRoot causes remain Unknown unless independently verified. No credentials or provider bodies are recorded.\n"
        for path in sorted((root / "failure-spool").glob("*.json")):
            event = json.loads(path.read_text())
            identifier = event["event_id"]
            if digest({k: v for k, v in event.items() if k != "event_id"}) != identifier:
                raise ValueError("Failure event identity changed")
            marker = "<!-- failure-event:" + identifier + " -->"
            if marker not in text:
                text += f"\n{marker}\n- {event['utc']} | {event['classification']} | stage={event['stage']} | attempt={event['attempt']} | {event['decision']} | root_cause=Unknown | evidence_sha256={sha256(path)}\n"
                count += 1
        temporary = document.with_suffix(".md.partial")
        with temporary.open("w") as stream:
            import os
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(document)
    return count
