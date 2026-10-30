"""A800 control and verified sealed-data pull; Isaac and selected login stay remote."""
import json
from pathlib import Path
import time
import re
import urllib.error
from agp_rl.io import digest
from robouse.data.datasets import verify, write_json
from robouse.environments.robodojo.client import RemoteEnvironment
from robouse.teacher.admission import FrontierBlocked
from robouse.teacher.state import file_lock


class TeacherClient(RemoteEnvironment):
    def request(self, op, **values):
        try:
            return super().request(op, **values)
        except urllib.error.HTTPError as exc:
            try:
                value = json.loads(exc.read(65536))
            except (ValueError, UnicodeError):
                raise exc
            code = value.get("error_code")
            if isinstance(code, str) and re.fullmatch(r"[A-Z0-9_]{1,180}", code):
                raise FrontierBlocked(code) from None
            raise exc


def run(config):
    root, data = Path(config["output"]).resolve(), Path(config["data_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / "remote-collection.lock"):
        client = TeacherClient(config["endpoint"], config["campaign_id"], root / "transport", timeout=config.get("rpc_timeout", 120))
        description = client.request("teacher_describe")
        write_json(root / "remote-description.json", description)
        if config.get("check_only"):
            return description
        if config.get("plan_sha256") != description["plan_sha256"]:
            raise FrontierBlocked("REVIEW_AND_PIN_REMOTE_PLAN_SHA256_BEFORE_COLLECTION")
        if config.get("stop_remote"):
            saved = json.loads((root / "remote-campaign.json").read_text())
            if saved["endpoint"] != config["endpoint"] or saved["campaign_id"] != config["campaign_id"]:
                raise FrontierBlocked("REMOTE_STOP_OWNERSHIP_MISMATCH")
            return client.request("teacher_stop")
        request = {"plan_sha256": config["plan_sha256"], "max_episodes": config["max_episodes"],
                   "max_wall_seconds": config["max_wall_seconds"], "final_test": config.get("final_test", False)}
        contract_path = root / "remote-campaign.json"
        contract = {"request": request, "endpoint": config["endpoint"], "campaign_id": config["campaign_id"]}
        if contract_path.exists():
            saved = json.loads(contract_path.read_text())
            if any(saved.get(key) != value for key, value in contract.items()) or not config.get("resume"):
                raise FrontierBlocked("EXACT_REMOTE_RESUME_REQUIRED")
        else:
            write_json(contract_path, {**contract, "created_at": time.time()})
        status = client.request("teacher_start", **request, resume=bool(config.get("resume")))
        start = json.loads(contract_path.read_text())["created_at"]
        deadline = start + request["max_wall_seconds"] + config.get("finalize_grace_seconds", 300)
        while not status["terminal"]:
            write_json(root / "remote-status.json", status)
            if status.get("blocked_without_terminal_proof"):
                raise FrontierBlocked("REMOTE_CONTROL_EXITED_WITHOUT_TERMINAL_PROOF_RESUME_SAME_CAMPAIGN")
            if (root / "STOP").exists():
                client.request("teacher_stop")
            if time.time() >= deadline:
                client.request("teacher_stop")
                raise FrontierBlocked("REMOTE_DEADLINE_REACHED_STATUS_UNCONFIRMED_RESUME_SAME_CAMPAIGN")
            time.sleep(config.get("poll_seconds", 5))
            status = client.request("teacher_status")
        write_json(root / "remote-status.json", status)
        if config.get("pull_data", True):
            marker = data / "remote-campaign.json"
            binding = {"campaign_id": config["campaign_id"], "request_sha256": digest(request)}
            if data.exists() and any(data.iterdir()) and (not marker.exists() or json.loads(marker.read_text()) != binding):
                raise FrontierBlocked("REFUSE_UNOWNED_LOCAL_DATA_MIRROR")
            data.mkdir(parents=True, exist_ok=True)
            write_json(marker, binding)
            manifest = client.request("teacher_manifest")
            for row in manifest["files"]:
                relative = Path(row["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Remote manifest path escapes DATA_ROOT")
                client.download(row["path"], data / relative)
            manifest_path = root / "remote-transfer.json"
            write_json(manifest_path, manifest)
            verify(data, manifest_path)
        failures = client.request("teacher_failures")
        (root / "SEED1_ROLLOUT_FAILURES.md").write_text(failures["markdown"])
        status = client.request("teacher_status")
        write_json(root / "remote-status.json", status)
        return status
