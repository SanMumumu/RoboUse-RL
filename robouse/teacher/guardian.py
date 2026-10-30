"""Independent bounded supervisor; only the canonical collector dispatches jobs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from agp_rl.io import digest
from robouse.data.datasets import write_json
from robouse.teacher.admission import FrontierBlocked
from robouse.teacher.failures import flush, emit
from robouse.teacher.ownership import alive, receipt
from robouse.teacher.recovery import writable_probe
from robouse.teacher.state import file_lock

TERMINAL = {"FINISHED_BOUNDED", "BUDGET_LIMITED", "TARGET_REACHED", "IDENTITIES_EXHAUSTED_BELOW_TARGET", "STOPPED"}


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def resume_stopped(root):
    state = read(root / "teacher-state.json", {})
    if state.get("status") != "STOPPED":
        return
    if any((root / name).exists() for name in ("ADMISSION_CLOSED.json", "RAW_INTEGRITY_HOLD.json")) or time.time() >= state["deadline"]:
        raise FrontierBlocked("STOPPED_CAMPAIGN_REQUIRES_UNEXPIRED_BUDGET_AND_NO_SAFETY_HOLD")
    state["status"] = "READY"
    write_json(root / "teacher-state.json", state)
    (root / "STOP").unlink(missing_ok=True)


def supervise(config, *, interval=5, popen=subprocess.Popen, sleep=time.sleep):
    root = Path(config["output"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    maximum = config.get("guardian_max_starts", 4)
    if type(maximum) is not int or not 1 <= maximum <= 4:
        raise FrontierBlocked("GUARDIAN_START_LIMIT_MUST_BE_1_TO_4")
    frozen = root / "guardian-config.json"
    normalized = {k: v for k, v in config.items() if k not in ("resume", "execute", "dry_run")}
    contract = digest(normalized)
    with file_lock(root / "guardian.lock"):
        if config.get("resume"):
            resume_stopped(root)
        if frozen.exists() and read(frozen) != normalized:
            raise FrontierBlocked("FROZEN_GUARDIAN_CONFIGURATION_CHANGED")
        if not frozen.exists():
            write_json(frozen, normalized)
        saved = read(root / "guardian-state.json", {})
        if saved.get("contract", contract) != contract:
            raise FrontierBlocked("GUARDIAN_CONTRACT_CHANGED")
        starts = saved.get("controller_starts", 0)
        child = None
        while True:
            state = read(root / "teacher-state.json", {})
            owner = read(root / "teacher-controller-owner.json", {})
            if read(root / "teacher-controller-exit.json", {}).get("owner") == owner:
                owner = {}
            result = {"owner": receipt(), "contract": contract, "controller_starts": starts,
                      "model_calls_by_guardian": 0, "episode_retries": 0,
                      "observed_at": time.time(), "valid_scores": state.get("valid_scores", 0),
                      "acceptance": read(root / "teacher-acceptance.json", {"first_complete_terminal": "NOT_RUN", "processor": "NOT_RUN"})}
            result["acceptance"].pop("verified_indexes", None)
            try:
                flush(root)
            except (OSError, ValueError, FrontierBlocked):
                result["status"] = "JOURNAL_IO_HOLD"
                write_json(root / "guardian-state.json", result)
                return result
            if child is not None:
                child.poll()  # Reap only the child this guardian started.
            pending = read(root / "guardian-controller-start.json", {}).get("owner", {})
            if alive(owner) or alive(pending):
                result.update(status="MONITORING", controller=owner if alive(owner) else pending)
                write_json(root / "guardian-state.json", result)
                sleep(interval)
                continue
            if (root / "STOP").exists() or state.get("status") in TERMINAL:
                result.update(status=state.get("status", "STOPPED"), terminal=True)
            elif state.get("status") == "LANES_HELD" or (root / "ADMISSION_CLOSED.json").exists() or (root / "RAW_INTEGRITY_HOLD.json").exists():
                result.update(status="SAFETY_HOLD", terminal=True)
            elif starts >= maximum:
                result.update(status="CONTROLLER_START_LIMIT_REACHED", terminal=True)
            else:
                writable_probe(root)
                # A direct/manual collector still owns the same canonical mutex.
                # Probe then release before spawn; the child also acquires it.
                try:
                    with file_lock(root / "teacher-controller.lock"):
                        pass
                except FrontierBlocked:
                    result["status"] = "WAITING_FOR_CANONICAL_CONTROLLER"
                    write_json(root / "guardian-state.json", result)
                    sleep(interval)
                    continue
                starts += 1
                result.update(status="STARTING_CANONICAL_CONTROLLER", controller_starts=starts)
                write_json(root / "guardian-state.json", result)
                command = [sys.executable, "-B", "-m", "robouse.cli", "collect_astra", "--config", str(frozen), "--execute"]
                if (root / "teacher-state.json").exists():
                    command += ["--resume", str(root)]
                if config.get("final_test"):
                    command.append("--final-test")
                with (root / "guardian-controller.log").open("a") as log:
                    child = popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                write_json(root / "guardian-controller-start.json", {"owner": receipt(child.pid), "starts": starts})
                sleep(interval)
                continue
            if result.get("status") in ("SAFETY_HOLD", "CONTROLLER_START_LIMIT_REACHED"):
                emit(root, classification=result["status"], stage="guardian", kind="HOLD")
                flush(root)
            write_json(root / "guardian-state.json", result)
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.dry_run or not args.execute:
        print(json.dumps({"status": "DRY_RUN", "model_calls_by_guardian": 0, "output": config["output"]}))
        return
    print(json.dumps(supervise(config)))


if __name__ == "__main__":
    main()
