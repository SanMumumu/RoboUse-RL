"""Durable included-usage budgets and non-overlapping controller/resource ownership."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
from agp_rl.io import digest, sha256
from robouse.data.datasets import write_json, read_jsonl, check_roles, role
from robouse.teacher.admission import FrontierBlocked


@contextmanager
def file_lock(path, wait_seconds=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise FrontierBlocked(
                        "COLLECTOR_OR_RESOURCE_ALREADY_OWNED"
                    ) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def validate(config, *, final_test=False):
    # The selected CLI catalog determines supported names, not a local enum.
    for key in ("model", "effort"):
        value = config.get(key)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise FrontierBlocked("EXPLICIT_MODEL_AND_REASONING_EFFORT_REQUIRED")
    if (
        config.get("service_tier") != "default"
        or config.get("max_extra_cost_usd") != 0
        or config.get("paid_credits_authorized") is not False
    ):
        raise FrontierBlocked("INCLUDED_USAGE_ONLY_NO_PAID_FALLBACK")
    for key in ("max_episodes", "max_wall_seconds", "episode_wall_seconds"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise FrontierBlocked("EXPLICIT_POSITIVE_BUDGET_REQUIRED")
    if config["episode_wall_seconds"] > 14400:
        raise FrontierBlocked("NATIVE_EPISODE_WALL_CAP_EXCEEDED")
    if config.get("automatic_retries", 0) != 0:
        raise FrontierBlocked("NO_UNBUDGETED_AUTOMATIC_TEACHER_RETRIES")
    target = config.get("target_valid_scores")
    if target is not None and (type(target) is not int or target < 1 or target > config["max_episodes"]):
        raise FrontierBlocked("VALID_SCORE_TARGET_REQUIRES_EXPLICIT_SUFFICIENT_ATTEMPT_BUDGET")
    attempts = config.get("metadata_max_attempts", 3)
    if type(attempts) is not int or not 1 <= attempts <= 3:
        raise FrontierBlocked("METADATA_ATTEMPT_LIMIT_MUST_BE_1_TO_3")
    if len({str(w["renderer_gpu"]) for w in config["workers"]}) != len(
        config["workers"]
    ):
        raise FrontierBlocked("DISTINCT_EXPLICIT_RTX_WORKERS_REQUIRED")
    identities = read_jsonl(config["identities"])
    seen = set()
    rows = []
    if not identities:
        raise FrontierBlocked("NO_FROZEN_COLLECTION_IDENTITIES")
    for item in identities:
        seed = item["seed"]
        if seed == 2 and not final_test:
            raise FrontierBlocked("SEED2_REQUIRES_EXPLICIT_FINAL_TEST")
        key = (item["task"], seed, item["layout"])
        if key in seen:
            raise FrontierBlocked("DUPLICATE_ENVIRONMENT_IDENTITY")
        seen.add(key)
        if seed not in (0, 1, 2):
            raise FrontierBlocked("FIXED_SEED_ROLES_REQUIRED")
        path = Path(item["layout_path"])
        if sha256(path) != item["layout_sha256"]:
            raise FrontierBlocked("FROZEN_LAYOUT_CHANGED")
        rows.append(
            {
                **item,
                "identity_id": f"{item['task']}/seed-{seed}/layout-{item['layout']:03d}",
                "role": role(seed),
                "layout_hash": item["layout_sha256"],
            }
        )
    check_roles(rows)
    return identities


def excluded_identities(config, identities=()):
    """Frozen metadata-only tombstones prevent replay of sealed successes/failures."""
    keys = set()
    ownership = []
    def record(row):
        return {**row, "identity_id": f"{row['task']}/seed-{row['seed']}/layout-{row['layout']:03d}", "role": role(row["seed"]), "layout_hash": row.get("layout_hash", row.get("layout_sha256"))}
    for reference in config.get("excluded_identity_files", []):
        if sha256(reference["path"]) != reference["sha256"]:
            raise FrontierBlocked("FROZEN_EXCLUDED_IDENTITY_MANIFEST_CHANGED")
        for row in read_jsonl(reference["path"]):
            if any(name in row for name in ("messages", "targets", "assistant", "decisions")):
                raise FrontierBlocked("EXCLUSION_MANIFEST_MUST_CONTAIN_IDENTITIES_ONLY")
            keys.add((row["task"], row["seed"], row["layout"]))
            ownership.append(record(row))
    check_roles(ownership + [record(row) for row in identities])
    return keys


class Budget:
    def __init__(self, config, identities, *, resume=False):
        self.config = config
        self.root = Path(config["output"]).resolve()
        self.path = self.root / "teacher-state.json"
        self.root.mkdir(parents=True, exist_ok=True)
        options = {
            k: config[k]
            for k in (
                "model",
                "effort",
                "service_tier",
                "max_episodes",
                "max_wall_seconds",
                "episode_wall_seconds",
                "data_root",
                "workers",
            )
        }
        options.update(
            {k: config[k] for k in ("cli_sha256", "cli_version", "bwrap_sha256")}
        )
        options.update(target_valid_scores=config.get("target_valid_scores"), metadata_max_attempts=config.get("metadata_max_attempts", 3))
        options["selected_account_sha256"] = os.environ.get("ASTRA_ACCOUNT_SHA256")
        options["admission_options"] = {key: config.get(key) for key in ("metadata_timeout_s", "admission_max_age_s", "retry_prior_max_age_s")}
        options["excluded_identity_files"] = config.get("excluded_identity_files", [])
        options["teacher_code_sha256"] = digest(
            {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        )
        contract = digest(
            {"options": options, "identity_file_sha256": sha256(config["identities"])}
        )
        if self.path.exists():
            if not resume:
                raise FrontierBlocked("TEACHER_RUN_EXISTS_EXPLICIT_RESUME_REQUIRED")
            self.value = json.loads(self.path.read_text())
            if self.value["contract"] != contract:
                raise FrontierBlocked("FROZEN_TEACHER_CONTRACT_CHANGED")
        else:
            self.value = {
                "contract": contract,
                "started_at": time.time(),
                "deadline": time.time() + config["max_wall_seconds"],
                "reserved": {},
                "terminal": {},
                "lane_holds": {},
                "status": "READY",
            }
            write_json(self.path, self.value)

    def reserve(self, identity, worker):
        key = digest(identity)
        if key in self.value["terminal"]:
            return None
        if key in self.value["reserved"]:
            return self.value["reserved"][key]
        if time.time() >= self.value["deadline"]:
            raise FrontierBlocked("TEACHER_WALL_BUDGET_EXHAUSTED")
        if len(self.value["reserved"]) >= self.config["max_episodes"]:
            return None
        episode_id = (
            "astra-"
            + digest({"identity": identity, "run": self.value["contract"]})[:32]
        )
        value = {
            "episode_id": episode_id,
            "identity": identity,
            "worker": worker,
            "deadline": self.value["deadline"],
            "reserved_at": time.time(),
        }
        self.value["reserved"][key] = value
        write_json(self.path, self.value)
        return value

    def complete(self, identity, terminal):
        self.value["terminal"][digest(identity)] = terminal
        write_json(self.path, self.value)


def resource_admission(worker, *, now=None):
    now = time.time() if now is None else now
    reference = worker["resource_lease"]
    if sha256(reference["path"]) != reference["sha256"]:
        raise FrontierBlocked("RESOURCE_LEASE_CHANGED")
    lease = json.loads(Path(reference["path"]).read_text())
    if (
        lease.get("allocated") is not True
        or str(lease.get("gpu")) != str(worker["renderer_gpu"])
        or lease.get("expires_at", 0) <= now
        or lease.get("owner") != worker["resource_owner"]
    ):
        raise FrontierBlocked("NO_CURRENT_EXPLICIT_RTX_ALLOCATION")
    return lease
