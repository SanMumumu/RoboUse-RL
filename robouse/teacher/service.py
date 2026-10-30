"""Pinned teacher campaigns behind the existing authenticated loopback server.

Only server-side configuration selects code, accounts, identities and renderers.
The A800 caller submits a bounded campaign ID, never shell commands or credentials.
"""
import base64
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from agp_rl.io import digest, sha256
from robouse.data.datasets import export, write_json
from robouse.environments.robodojo.remote import make_server
from robouse.teacher.admission import FrontierBlocked
from robouse.teacher.guardian import TERMINAL, read
from robouse.teacher.ownership import alive, receipt
from robouse.teacher.state import file_lock, validate, excluded_identities


class TeacherService:
    def __init__(self, config, *, popen=subprocess.Popen):
        self.config = config
        self.root = Path(config["output"]).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.root / "service-host.lock"):
            binding = {"host_sha256": receipt()["host_sha256"]}
            marker = self.root / "service-host.json"
            if marker.exists() and read(marker) != binding:
                raise FrontierBlocked("REMOTE_SERVICE_ROOT_BELONGS_TO_ANOTHER_HOST")
            write_json(marker, binding)
        self.template_path = Path(config["teacher_config"]).resolve()
        self.template = read(self.template_path)
        self.template_sha = sha256(self.template_path)
        self.identity_sha = sha256(self.template["identities"])
        self.code = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        self.plan = digest({"template": self.template_sha, "identities": self.identity_sha,
                            "teacher_code": self.code,
                            "selected_account_sha256": os.environ.get("ASTRA_ACCOUNT_SHA256")})
        self.popen = popen
        self.lock = threading.Lock()
        self.children = {}
        self.manifests = {}
        self.file_hashes = {}

    def public_error(self, exc):
        if isinstance(exc, FrontierBlocked) and re.fullmatch(r"[A-Z0-9_]{1,180}", str(exc)):
            return 409, {"error_code": str(exc), "automatic_retry": False}
        return 500, {"error_code": "TEACHER_SERVICE_INTERNAL_ERROR", "automatic_retry": False}

    def directory(self, campaign):
        if not isinstance(campaign, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", campaign):
            raise ValueError("Canonical campaign ID required")
        return self.root / "campaigns" / campaign

    def status(self, campaign):
        root = self.directory(campaign)
        if campaign in self.children:
            self.children[campaign].poll()
        state = read(root / "teacher-state.json", {})
        guardian = read(root / "guardian-state.json", {})
        controller = read(root / "teacher-controller-owner.json", {})
        if read(root / "teacher-controller-exit.json", {}).get("owner") == controller:
            controller = {}
        owners = [read(root / "service-owner.json", {}), controller,
                  read(root / "guardian-controller-start.json", {}).get("owner", {})]
        active = any(alive(owner) for owner in owners)
        for path in root.glob("workers/*/*/owner.json"):
            owner = read(path, {})
            active |= alive({**owner, "host_sha256": receipt()["host_sha256"]})
        terminal = state.get("status") in TERMINAL or state.get("status") in ("ADMISSION_CLOSED", "RAW_INTEGRITY_HOLD") or guardian.get("terminal", False)
        blocked = bool((root / "service-start-intent.json").exists() and not active and not terminal)
        acceptance = read(root / "teacher-acceptance.json", {"first_complete_terminal": "NOT_RUN", "processor": "NOT_RUN"})
        return {"campaign_id": campaign, "plan_sha256": self.plan,
                "status": state.get("status", guardian.get("status", "NOT_STARTED")),
                "active": active, "terminal": bool(terminal and not active),
                "blocked_without_terminal_proof": blocked,
                "valid_scores": state.get("valid_scores", 0), "reserved": len(state.get("reserved", {})),
                "sealed_attempts": len(state.get("terminal", {})),
                "acceptance": {k: v for k, v in acceptance.items() if k != "verified_indexes"}}

    def start(self, request):
        campaign = request["episode_id"]
        root = self.directory(campaign)
        current_code = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        if request.get("plan_sha256") != self.plan or sha256(self.template_path) != self.template_sha or sha256(self.template["identities"]) != self.identity_sha or current_code != self.code:
            raise FrontierBlocked("REMOTE_PINNED_COLLECTION_PLAN_CHANGED")
        budget = {key: request[key] for key in ("max_episodes", "max_wall_seconds")}
        if any(type(value) is not int or not 0 < value <= self.template[key] for key, value in budget.items()):
            raise FrontierBlocked("REMOTE_CAMPAIGN_BUDGET_EXCEEDS_SERVER_AUTHORIZATION")
        final_test = request.get("final_test") is True
        if final_test and self.config.get("allow_final_test") is not True:
            raise FrontierBlocked("REMOTE_FINAL_TEST_NOT_AUTHORIZED")
        contract = {"campaign_id": campaign, "plan_sha256": self.plan, **budget, "final_test": final_test}
        if root.exists() and any(root.iterdir()) and not (root / "request.json").exists():
            raise FrontierBlocked("REFUSE_UNOWNED_REMOTE_CAMPAIGN_DIRECTORY")
        root.mkdir(parents=True, exist_ok=True)
        saved = read(root / "request.json")
        if saved is not None and saved != contract:
            raise FrontierBlocked("REMOTE_CAMPAIGN_IDEMPOTENCY_CONFLICT")
        if saved:
            status = self.status(campaign)
            stopped = read(root / "teacher-state.json", {}).get("status") == "STOPPED"
            if status["active"] or (status["terminal"] and not (stopped and request.get("resume"))) or not request.get("resume"):
                return status
            if not (root / "service-owner.json").exists():
                raise FrontierBlocked("REMOTE_START_INTENT_UNCERTAIN_NO_AUTOMATIC_REPLAY")
        cfg = {**self.template, **budget, "output": str(root),
               "data_root": str(Path(self.template["data_root"]).resolve() / campaign), "final_test": final_test}
        if saved and request.get("resume"):
            cfg["resume"] = str(root)
            self.manifests.pop(campaign, None)
        excluded_identities(cfg, validate(cfg, final_test=final_test))
        with file_lock(self.root / "service-campaign.lock"):
            claim = self.root / ("plan-claim-" + self.plan + ".json")
            existing = read(claim, {})
            if existing and existing.get("campaign_id") != campaign:
                raise FrontierBlocked("FROZEN_PLAN_ALREADY_CLAIMED_RESUME_ORIGINAL_CAMPAIGN_NO_REPLAY")
            for other in (self.root / "campaigns").iterdir():
                if other.name != campaign and self.status(other.name)["active"]:
                    raise FrontierBlocked("SERVER_ALREADY_HAS_AN_ACTIVE_CAMPAIGN")
            write_json(root / "request.json", contract)
            write_json(claim, {"campaign_id": campaign, "plan_sha256": self.plan})
            write_json(root / "campaign-config.json", cfg)
            write_json(root / "service-start-intent.json", {"contract": digest(contract)})
            with (root / "guardian.log").open("a") as log:
                child = self.popen([sys.executable, "-B", "-m", "robouse.teacher.guardian", "--config", str(root / "campaign-config.json"), "--execute"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.children[campaign] = child
            write_json(root / "service-owner.json", receipt(child.pid))
        return self.status(campaign)

    def manifest(self, campaign):
        if campaign in self.manifests:
            return self.manifests[campaign]
        status = self.status(campaign)
        if not status["terminal"]:
            raise FrontierBlocked("REMOTE_WRITERS_NOT_TERMINAL_NO_LIVE_DATA_EXPORT")
        root = self.directory(campaign)
        cfg = read(root / "campaign-config.json")
        data = Path(cfg["data_root"]).resolve()
        # Only this collector's sealed data tree is exposed, never profiles/logins.
        if not (data / "teacher-dataset.json").exists():
            raise FrontierBlocked("REMOTE_DATASET_NOT_INITIALIZED")
        from robouse.teacher.observer import observe

        observe({"data_root": str(data), "output": str(root), "full_verify": True})
        names = [str(p.relative_to(data)) for p in sorted(data.rglob("*")) if p.is_file() and p.suffix not in (".lock", ".partial")]
        value = export(data, names, root / "transfer.json", min_age=0)
        self.manifests[campaign] = value
        return value

    def handle(self, request):
        with self.lock:
            op = request["op"]
            campaign = request["episode_id"]
            if op == "teacher_describe":
                return {"plan_sha256": self.plan, "model": self.template["model"], "effort": self.template["effort"],
                        "renderer_mode": self.template.get("renderer_mode", "rtx"), "maximum_concurrency": self.template.get("max_concurrency", 1),
                        "inference_calls": 0, "acceptance": "NOT_RUN", "scope": "REMOTE_CONFIGURATION_ONLY", "live_admission_verified": False}
            if op == "teacher_start":
                return self.start(request)
            if op == "teacher_status":
                return self.status(campaign)
            if op == "teacher_stop":
                root = self.directory(campaign)
                if not (root / "request.json").exists():
                    raise ValueError("Unknown remote campaign")
                with file_lock(root / "model-start.lock", wait_seconds=30):
                    (root / "STOP").touch()
                return {"cooperative_stop_requested": True}
            if op == "teacher_manifest":
                return self.manifest(campaign)
            if op == "teacher_failures":
                journal = self.directory(campaign) / "SEED1_ROLLOUT_FAILURES.md"
                return {"markdown": journal.read_text()[-65536:] if journal.exists() else "", "provider_bodies_included": False}
            if op == "blob":
                manifest = self.manifest(campaign)
                row = next((r for r in manifest["files"] if r["path"] == request["path"]), None)
                if row is None:
                    raise ValueError("Only manifest-bound sealed data may be transferred")
                cfg = read(self.directory(campaign) / "campaign-config.json")
                path = Path(cfg["data_root"]) / row["path"]
                stat = path.stat()
                key = (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns)
                if key not in self.file_hashes:
                    if stat.st_size != row["bytes"] or sha256(path) != row["sha256"]:
                        raise ValueError("Sealed remote artifact changed")
                    self.file_hashes[key] = row["sha256"]
                offset, size = request["offset"], request.get("chunk_bytes", 4 * 1024 * 1024)
                if type(offset) is not int or type(size) is not int or not 0 <= offset <= row["bytes"] or not 0 < size <= 4 * 1024 * 1024:
                    raise ValueError("Invalid sealed artifact range")
                with path.open("rb") as stream:
                    stream.seek(offset)
                    raw = stream.read(size)
                after = path.stat()
                if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                    raise ValueError("Remote artifact still changing")
                return {"offset": offset, "data": base64.b64encode(raw).decode(), "total_bytes": row["bytes"], "sha256": row["sha256"]}
            raise ValueError("Unknown teacher service operation")


def serve(config):
    server = make_server(config, TeacherService(config))
    try:
        server.serve_forever()
    finally:
        server.server_close()
