"""Pinned Codex subprocesses; one selected read-only login, no model probes or fallback."""
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import threading
import time
from agp_rl.io import sha256
from robouse.data.datasets import write_json
from robouse.teacher.admission import FrontierBlocked, hashed, login_identity, normalize_quota


def model_permission(raw, model, effort):
    records = raw.get("data")
    if not isinstance(records, list):
        raise FrontierBlocked("MODEL_CATALOG_INVALID")
    matches = [r for r in records if r.get("model", r.get("id")) == model]
    if len(matches) != 1:
        raise FrontierBlocked("REQUESTED_MODEL_NOT_ADVERTISED")
    supported = [
        r if isinstance(r, str) else r.get("reasoningEffort")
        for r in matches[0].get("supportedReasoningEfforts", [])
    ]
    if effort not in supported:
        raise FrontierBlocked("REQUESTED_EFFORT_NOT_ADVERTISED_NO_DOWNGRADE")
    return {
        "model": model,
        "effort": effort,
        "advertised": True,
        "inference_entitlement_verified": False,
    }


def isolated_environment(profile, inherited=None):
    inherited = os.environ if inherited is None else inherited
    keys = (
        "PATH",
        "HOME",
        "LANG",
        "TZ",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    )
    env = {k: inherited[k] for k in keys if k in inherited}
    profile = Path(profile)
    for key in ("NO_PROXY", "no_proxy"):
        current = env.get(key, "")
        env[key] = ",".join(filter(None, [current, "127.0.0.1", "localhost", "::1"]))
    env.update(
        CODEX_HOME=str(profile),
        CODEX_SQLITE_HOME=str(profile / "sqlite"),
        TMPDIR=str(profile / "tmp"),
        XDG_CACHE_HOME=str(profile / "cache"),
        XDG_CONFIG_HOME=str(profile / "config"),
        XDG_DATA_HOME=str(profile / "data"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


class IsolatedCodex:
    def __init__(self, config, directory, *, minimum_lifetime=90):
        self.config = config
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.binary = Path(config["cli_binary"]).resolve(strict=True)
        self.bwrap = Path(config["bwrap_binary"]).resolve(strict=True)
        if (
            sha256(self.binary) != config["cli_sha256"]
            or sha256(self.bwrap) != config["bwrap_sha256"]
        ):
            raise FrontierBlocked("PINNED_EXECUTABLE_CHANGED")
        source = os.environ.get("ASTRA_AUTH_HOME")
        fingerprint = os.environ.get("ASTRA_ACCOUNT_SHA256")
        if not source or not fingerprint:
            raise FrontierBlocked("EXPLICIT_SELECTED_ACCOUNT_REQUIRED")
        self.source = Path(source).resolve(strict=True)
        self.identity = login_identity(self.source, minimum_lifetime=minimum_lifetime)
        if self.identity["account_sha256"] != fingerprint:
            raise FrontierBlocked("SELECTED_ACCOUNT_FINGERPRINT_MISMATCH")
        self.profile = self.directory / "profile"
        if self.profile.exists():
            raise FrontierBlocked("FRESH_PROFILE_REQUIRED_NEVER_REPLAY_A_MODEL_START")
        for name in ("sqlite", "tmp", "cache", "config", "data", "logs", "sessions"):
            (self.profile / name).mkdir(parents=True, exist_ok=True, mode=0o700)
        (self.profile / "auth.json").touch(mode=0o600)
        lines = [
            'model_provider = "openai"',
            'forced_login_method = "chatgpt"',
            'cli_auth_credentials_store = "file"',
            "model = " + json.dumps(config["model"]),
            "model_reasoning_effort = " + json.dumps(config["effort"]),
            "service_tier = " + json.dumps(config["service_tier"]),
        ]
        (self.profile / "config.toml").write_text("\n".join(lines) + "\n")
        self.admitted = False
        result = subprocess.run(
            self.command(["--version"]),
            env=isolated_environment(self.profile),
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        if result.stdout.strip() != config["cli_version"]:
            raise FrontierBlocked("PINNED_CLI_VERSION_CHANGED")

    def verify_identity(self, minimum_lifetime=90):
        value = login_identity(self.source, minimum_lifetime=minimum_lifetime)
        if value != self.identity:
            raise FrontierBlocked("SELECTED_LOGIN_CHANGED_NO_ROTATION")
        return value

    def command(self, args, *, session=None, private_roots=()):
        if args[:1] == ["exec"]:
            self.assert_model_admission_open()
            if not self.admitted or getattr(self, "model_command_issued", False):
                raise FrontierBlocked("FRESH_SINGLE_USE_ADMISSION_REQUIRED_BEFORE_EXEC")
            self.model_command_issued = True
        if (
            args[:1] != ["exec"]
            and args != ["app-server", "--listen", "stdio://"]
            and args != ["--version"]
        ):
            raise FrontierBlocked("NONCANONICAL_CODEX_COMMAND")
        cmd = [
            str(self.bwrap),
            "--die-with-parent",
            "--new-session",
            "--unshare-user",
            "--uid",
            "0",
            "--gid",
            "0",
            "--unshare-pid",
            "--unshare-ipc",
            "--ro-bind",
            "/",
            "/",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
        ]
        if session:
            cmd += [
                "--bind",
                str(Path(session).resolve()),
                str(Path(session).resolve()),
            ]
        cmd += [
            "--bind",
            str(self.profile),
            str(self.profile),
            "--tmpfs",
            str(self.profile / "sqlite"),
            "--ro-bind",
            str(self.source / "auth.json"),
            str(self.profile / "auth.json"),
        ]
        for root in private_roots:
            cmd += ["--tmpfs", str(Path(root).resolve())]
        cmd += [
            "--chdir",
            str(self.profile),
            str(self.binary),
            "--no-daemon",
            "-c",
            "sqlite_home=" + json.dumps(str(self.profile / "sqlite")),
            "-c",
            "log_dir=" + json.dumps(str(self.profile / "logs")),
            *args,
        ]
        return cmd

    def assert_model_admission_open(self):
        root = self.config.get("failure_root")
        if root and any((Path(root) / name).exists() for name in ("ADMISSION_CLOSED.json", "RAW_INTEGRITY_HOLD.json", "STOP")):
            raise FrontierBlocked("GLOBAL_MODEL_ADMISSION_CLOSED")
        proof = getattr(self, "admission_proof", None)
        if proof and time.time() - proof["observed_at"] > self.config.get("admission_max_age_s", 60):
            raise FrontierBlocked("STALE_ADMISSION_BEFORE_MODEL_PROCESS_START")

    def metadata(self, label):
        from robouse.teacher.recovery import metadata_with_recovery

        return metadata_with_recovery(self, self._metadata_once, label)

    def _metadata_once(self, label):
        before = self.verify_identity()
        process = None
        try:
            process = subprocess.Popen(
                self.command(["app-server", "--listen", "stdio://"]),
                env=isolated_environment(self.profile),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
            )
            messages = queue.Queue()

            def receive():
                for line in process.stdout:
                    try:
                        messages.put(json.loads(line))
                    except ValueError:
                        pass
                messages.put(None)

            threading.Thread(target=receive, daemon=True).start()
            deadline = time.monotonic() + self.config.get("metadata_timeout_s", 45)

            def rpc(number, method, params):
                process.stdin.write(
                    json.dumps({"id": number, "method": method, "params": params})
                    + "\n"
                )
                process.stdin.flush()
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise FrontierBlocked("METADATA_TIMEOUT")
                    try:
                        value = messages.get(timeout=remaining)
                    except queue.Empty:
                        raise FrontierBlocked("METADATA_TIMEOUT") from None
                    if value is None:
                        raise FrontierBlocked("METADATA_PROCESS_ENDED")
                    if value.get("id") == number:
                        if "error" in value:
                            from robouse.teacher.rpc_policy import rpc_fault

                            error = value["error"] if isinstance(value["error"], dict) else {}
                            details = error.get("data") if isinstance(error.get("data"), dict) else {}
                            message = " ".join(str(value) for value in (error.get("message", ""), details.get("reason", ""), details.get("type", ""), details.get("code", ""))).lower()
                            category = "UNKNOWN"
                            for name, pattern in (
                                ("PERMISSION", r"permission|forbidden|unauthori[sz]ed|unauthenticated|access denied"),
                                ("QUOTA", r"quota|credits|rate.limit|usage.limit"),
                                ("WORKSPACE_REQUIREMENTS", r"workspace.require|organization.require"),
                                ("LOCAL_IO", r"read.only|disk.full|no.space|i/o|filesystem"),
                                ("NETWORK", r"connection|network|proxy|timed.out|timeout|reset.by.peer"),
                            ):
                                if re.search(pattern, message):
                                    category = name
                                    break
                            rpc_code = error.get("code") if type(error.get("code")) is int else "UNKNOWN"
                            base = "METADATA_" + method.replace("/", "_") + "_RPC_ERROR_" + str(rpc_code) + "_" + category
                            fault = rpc_fault(method, error, base)
                            exc = FrontierBlocked(fault["classification"])
                            for key in ("rpc_method", "rpc_code", "http_status"):
                                setattr(exc, key, fault[key])
                            raise exc
                        return value["result"]

            rpc(
                1,
                "initialize",
                {"clientInfo": {"name": "robouse-astra-admission", "version": "1"}},
            )
            process.stdin.write(
                json.dumps({"method": "initialized", "params": {}}) + "\n"
            )
            process.stdin.flush()
            account = (
                rpc(2, "account/read", {"refreshToken": False}).get("account") or {}
            )
            if (
                account.get("type") != "chatgpt"
                or hashed(account.get("email")) != before["email_sha256"]
            ):
                raise FrontierBlocked("LIVE_SELECTED_IDENTITY_MISMATCH")
            quota = normalize_quota(
                rpc(3, "account/rateLimits/read", {}), before, time.time()
            )
            catalog = rpc(4, "model/list", {"includeHidden": True})
            access = model_permission(
                catalog, self.config["model"], self.config["effort"]
            )
            if not quota["allowed"]:
                raise FrontierBlocked("QUOTA_CLOSED_OR_UNKNOWN")
            self.verify_identity()
            proof = {
                "account_sha256": before["account_sha256"],
                "credential_file_sha256": before["credential_file_sha256"],
                "observed_at": time.time(),
                "remaining_percent": quota["remaining_percent"],
                "quota": quota,
                "catalog": access,
                "model": self.config["model"],
                "effort": self.config["effort"],
                "service_tier": self.config["service_tier"],
                "inference_requests": 0,
                "account_metadata_admitted": True,
                "scope": "NON_GENERATIVE_METADATA_ONLY",
                "budget_checked": False,
            }
            write_json(self.directory / f"admission-{label}.json", proof)
            return proof
        except ConnectionError:
            raise FrontierBlocked("METADATA_TRANSPORT_FAILURE") from None
        except (OSError, KeyError, TypeError):
            raise FrontierBlocked("METADATA_UNAVAILABLE_FAIL_CLOSED") from None
        finally:
            if process is not None:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                process.stdin.close()
                process.stdout.close()

    def admit(self, *, max_wall_seconds):
        self.assert_model_admission_open()
        if (
            not isinstance(self.config.get("model"), str)
            or not self.config["model"].strip()
            or self.config.get("service_tier") != "default"
            or self.config.get("max_extra_cost_usd") != 0
            or self.config.get("paid_credits_authorized") is not False
        ):
            raise FrontierBlocked("EXPLICIT_INCLUDED_USAGE_ASTRA_SETTINGS_REQUIRED")
        if type(max_wall_seconds) is not int or not 0 < max_wall_seconds <= 14400:
            raise FrontierBlocked("EXPLICIT_EPISODE_WALL_BUDGET_REQUIRED")
        self.verify_identity(minimum_lifetime=max_wall_seconds + 90)
        proof = self.metadata("launch")
        if time.time() - proof["observed_at"] > self.config.get(
            "admission_max_age_s", 60
        ):
            raise FrontierBlocked("STALE_ADMISSION")
        self.admitted = True
        self.admission_proof = proof
        self.assert_model_admission_open()
        proof.update(launch_admitted=True, wall_admission_checked=True)
        return proof
