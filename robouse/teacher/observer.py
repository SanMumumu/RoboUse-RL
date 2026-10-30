"""Read only published teacher records; artifact and processor acceptance are separate."""
from copy import deepcopy
import json
from pathlib import Path
import time
from agp_rl.io import sha256
from robouse.data.datasets import read_jsonl, write_json


class ProcessorAcceptanceError(RuntimeError):
    pass


def watch(config):
    from robouse.teacher.state import file_lock
    from robouse.teacher.guardian import TERMINAL, read

    root = Path(config["output"])
    interval = config.get("observer_interval_seconds", 30)
    if interval < 5:
        raise ValueError("Observer interval must be at least five seconds")
    deadline = time.monotonic() + config.get("max_wall_seconds", 86400)
    with file_lock(root / "observer.lock"):
        while True:
            result = observe(config)
            state = read(root / "teacher-state.json", {})
            if state.get("status") in TERMINAL | {"ADMISSION_CLOSED", "RAW_INTEGRITY_HOLD", "LANES_HELD"} or (root / "STOP").exists():
                return result
            if time.monotonic() >= deadline:
                return {**result, "observer_status": "DEADLINE_REACHED_ACCEPTANCE_UNCHANGED"}
            time.sleep(interval)


def observe(config):
    root, output = Path(config["data_root"]).resolve(), Path(config["output"]).resolve()
    previous_path = output / "teacher-acceptance.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    if previous.get("data_root") != str(root):
        previous = {}
    verified = previous.get("verified_indexes", {})
    records, valid, full = [], 0, []
    try:
        for listing in sorted(root.glob("seed-*/indexes/episodes.jsonl")):
            for ref in read_jsonl(listing):
                path = (root / ref["path"]).resolve()
                if not path.is_relative_to(root) or sha256(path) != ref["sha256"]:
                    raise ValueError("Published trajectory index changed")
                index = json.loads(path.read_text())
                terminal = json.loads((path.parent / "terminal.json").read_text())
                if terminal != index["terminal"]:
                    raise ValueError("Terminal and index differ")
                if config.get("full_verify") or verified.get(ref["path"]) != ref["sha256"]:
                    manifest = root / index["artifacts"]["path"]
                    if sha256(manifest) != index["artifacts"]["sha256"]:
                        raise ValueError("Artifact inventory changed")
                    for asset in read_jsonl(manifest):
                        file = (root / asset["path"]).resolve()
                        if not file.is_relative_to(path.parent) or file.stat().st_size != asset["bytes"] or sha256(file) != asset["sha256"]:
                            raise ValueError("Sealed original artifact changed")
                    verified[ref["path"]] = ref["sha256"]
                records.append(ref)
                if terminal["valid_score"]:
                    from agp_rl.reward.official import read_official

                    native_ref = terminal["native_result"]
                    native_path = (root / native_ref["path"]).resolve()
                    if not native_path.is_relative_to(root) or sha256(native_path) != native_ref["sha256"]:
                        raise ValueError("Native result evidence changed")
                    native = read_official(native_path, terminal["layout_id"])
                    if native.success != terminal["native_success"] or native.score != terminal["native_score"]:
                        raise ValueError("Native score and terminal differ")
                valid += bool(terminal["valid_score"])
                if terminal["valid_score"] and terminal.get("native_result") and terminal.get("model_process_started") is True:
                    full.append(ref)
        result = {"status": "OBSERVING", "data_root": str(root), "observed_at": time.time(), "sealed_records": len(records),
                  "verified_artifact_records": len(records), "native_scored": valid,
                  "raw_artifact_integrity": "PASS" if records else "PENDING",
                  "first_complete_terminal": "PASS" if full else "PENDING",
                  "first_terminal_index": full[0] if full else None,
                  "processor": "NOT_RUN", "causal_window_validation": "NOT_RUN",
                  "verified_indexes": verified, "model_calls": 0}
        if config.get("processor_model"):
            try:
                schema_sha = sha256(config["tool_schema"]) if config.get("tool_schema") else None
                if previous.get("processor") == "PASS" and previous.get("processor_model") == config["processor_model"] and previous.get("processor_revision") == config.get("processor_revision") and previous.get("tool_schema_sha256") == schema_sha:
                    result.update({k: previous[k] for k in ("processor", "causal_window_validation", "processor_model", "processor_revision", "processor_signature", "target_tokens", "tested_window", "model_weights_loaded", "tool_schema_sha256")})
                else:
                    result.update(processor_probe(config, root, records))
                    result["tool_schema_sha256"] = schema_sha
            except (OSError, ValueError, KeyError, ImportError) as exc:
                from robouse.teacher.failures import emit

                emit(output, classification="PROCESSOR_ACCEPTANCE_FAILED", stage="processor", kind="HOLD", decision="KEEP_NATIVE_INTEGRITY_SEPARATE")
                result.update(processor="FAIL", processor_error_type=type(exc).__name__)
                write_json(previous_path, result)
                raise ProcessorAcceptanceError("Processor acceptance failed; collection artifacts are not relabeled as corrupt") from exc
        write_json(previous_path, result)
        return result
    except (OSError, ValueError, KeyError) as exc:
        from robouse.teacher.state import file_lock

        with file_lock(output / "model-start.lock", wait_seconds=30):
            write_json(output / "RAW_INTEGRITY_HOLD.json", {"reason": type(exc).__name__, "new_dispatch": False, "existing_episodes_terminated": False})
        from robouse.teacher.failures import emit

        emit(output, classification="RAW_ARTIFACT_INTEGRITY_UNVERIFIED", stage="observer", kind="HOLD")
        raise


def processor_probe(config, root, records):
    if not config.get("tool_schema") or not config.get("processor_revision"):
        raise ValueError("Explicit frozen tool schema and processor revision required")
    if not records:
        return {"processor": "PENDING_QUALIFIED_SUCCESS_WINDOW", "causal_window_validation": "PENDING"}
    from transformers import AutoProcessor
    from agp_rl.pipeline2b.fullft import source_adapters as adapters
    from robouse.training.sft.train import encode
    from robouse.models.qwen import processor_signature

    processor = AutoProcessor.from_pretrained(config["processor_model"], revision=config["processor_revision"], local_files_only=True, trust_remote_code=False)
    schemas = json.loads(Path(config["tool_schema"]).read_text())["schemas"]
    previous_root = adapters.ROOT
    adapters.ROOT = root
    try:
        for seed in (0, 1):  # Never tokenize Seed2 targets.
            refs = [r for r in records if Path(r["path"]).parts[0] == f"seed-{seed}"]
            source = {"root": str(root / f"seed-{seed}"), "seed": seed, "name": "acceptance-probe"}
            for audit, candidate in adapters.native_candidates(source, adapters.Verifier(), schemas, refs):
                if candidate is None:
                    continue
                for decision in candidate["bundle"]["decisions"]:
                    if not decision["data_quality"]["sft_candidate"]:
                        continue
                    end = decision["visible_history"]["message_end_exclusive"]
                    row = {"window_id": decision["decision_id"], "messages": [{"role": "system", "content": [{"type": "text", "text": candidate["system_prompt"]}]}] + deepcopy(candidate["bundle"]["messages"][:end]) + [{"role": "assistant", "content": [{"type": "text", "text": decision["model_output"]["text"]}]}],
                           "images": {key: {**ref, "path": str(Path(ref["path"]).relative_to(root))} for key, ref in candidate["images"].items()}}
                    value = encode(row, processor, root)
                    if len(value["target"]) > config.get("max_new_tokens", 2048) or value["inputs"]["input_ids"].numel() > config.get("max_context_tokens", 131072):
                        continue
                    return {"processor": "PASS", "causal_window_validation": "PASS", "processor_signature": processor_signature(processor), "processor_model": config["processor_model"], "processor_revision": config["processor_revision"], "target_tokens": len(value["target"]), "tested_window": row["window_id"], "model_weights_loaded": False}
        return {"processor": "PENDING_QUALIFIED_SUCCESS_WINDOW", "causal_window_validation": "PENDING"}
    finally:
        adapters.ROOT = previous_root
