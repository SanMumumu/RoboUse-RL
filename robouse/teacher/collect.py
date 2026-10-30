"""Bounded multi-worker native Astra queue; restart never respawns a reserved attempt."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import time
from agp_rl.io import sha256, digest
from robouse.data.datasets import write_json
from robouse.environments.robodojo.native_job import verify_sources
from robouse.environments.robodojo.worker import NativeWorker
from robouse.teacher.admission import FrontierBlocked
from robouse.teacher.provider import IsolatedCodex
from robouse.teacher.state import Budget, file_lock, resource_admission, validate, excluded_identities
from robouse.teacher.seal import seal, snapshot, immutable_json


def worker_idle(config):
    resource_admission(config)
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(config["renderer_gpu"]),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    if result.stdout.strip():
        raise FrontierBlocked("RTX_HAS_EXISTING_COMPUTE_JOB_NO_PREEMPTION")
    result = subprocess.run(["nvidia-smi", "-i", str(config["renderer_gpu"]), "--query-gpu=uuid,name", "--format=csv,noheader"], capture_output=True, text=True, check=True, timeout=10)
    uuid, name = result.stdout.strip().split(",", 1)
    from robouse.teacher.placement import verify_renderer

    verify_renderer(config, {"uuid": uuid.strip(), "name": name.strip()})


def run(config, *, resume=False, final_test=False, check_only=False):
    root = Path(config["output"]).resolve()
    identities = [] if check_only else validate(config, final_test=final_test)
    excluded = set() if check_only else excluded_identities(config, identities)
    root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(file_lock(root / "teacher-controller.lock"))
        from robouse.teacher.ownership import receipt

        controller_owner = receipt()
        previous_owner = root / "teacher-controller-owner.json"
        if previous_owner.exists() and json.loads(previous_owner.read_text()).get("host_sha256") != controller_owner["host_sha256"]:
            raise FrontierBlocked("FROZEN_COLLECTION_EXECUTION_HOST_CHANGED_USE_NEW_RUN")
        previous_exit = root / "teacher-controller-exit.json"
        if previous_exit.exists():
            previous_exit.rename(root / f"teacher-controller-exit-{time.time_ns()}.json")
        write_json(root / "teacher-controller-owner.json", controller_owner)
        stack.callback(write_json, root / "teacher-controller-exit.json", {"owner": controller_owner, "returned": True})
        if check_only:
            runner = IsolatedCodex(
                config, root / ("metadata-check-" + str(time.time_ns()))
            )
            result = runner.metadata("check-only")
            write_json(root / "teacher-check.json", result)
            return result
        data = Path(config["data_root"]).resolve()
        marker = data / "teacher-dataset.json"
        if data.exists() and any(data.iterdir()) and not marker.exists():
            raise FrontierBlocked("REFUSE_TO_APPEND_TO_UNOWNED_OR_ACTIVE_DATASET")
        data.mkdir(parents=True, exist_ok=True)
        stack.enter_context(file_lock(data / "teacher-dataset.lock"))
        immutable_json(
            marker,
            {
                "schema": "astra-teacher-seedroles-v1",
                "dataset_version": config["dataset_version"],
                "model": config["model"],
                "effort": config["effort"],
                "service_tier": config["service_tier"],
                "native_source": verify_sources(),
                "seed_roles": {"0": "train", "1": "validation", "2": "sealed_test"},
            },
        )
        from robouse.data.datasets import check_roles, role

        registry_path = data / "manifest/identity-roles.json"
        registry = (
            json.loads(registry_path.read_text()) if registry_path.exists() else []
        )
        rows = [
            {
                **i,
                "identity_id": f"{i['task']}/seed-{i['seed']}/layout-{i['layout']:03d}",
                "role": role(i["seed"]),
                "layout_hash": i["layout_sha256"],
            }
            for i in identities
        ]
        check_roles(registry + rows)
        merged = {r["identity_id"]: r for r in registry}
        for row in rows:
            if row["identity_id"] in merged and merged[row["identity_id"]] != row:
                raise FrontierBlocked("FROZEN_DATASET_IDENTITY_CHANGED")
            merged[row["identity_id"]] = row
        write_json(registry_path, list(merged.values()))
        frozen = root / "frozen-identities.json"
        immutable_json(frozen, identities)
        budget = Budget(config, identities, resume=resume)
        if not config.get("account_lock_file"):
            raise FrontierBlocked("EXPLICIT_ACCOUNT_CONCURRENCY_LOCK_REQUIRED")
        stack.enter_context(file_lock(config["account_lock_file"]))
        workers = []
        for i, spec in enumerate(config["workers"]):
            cfg = {
                **spec,
                "output": str(root / "workers" / f"worker-{i}"),
                "actor_kind": "astra_teacher",
                "teacher": {
                    k: config[k]
                    for k in (
                        "model",
                        "effort",
                        "service_tier",
                        "cli_binary",
                        "cli_sha256",
                        "cli_version",
                        "bwrap_binary",
                        "bwrap_sha256",
                        "episode_wall_seconds",
                        "max_extra_cost_usd",
                        "paid_credits_authorized",
                    )
                },
                "rpc_timeout": config["episode_wall_seconds"] + 300,
            }
            cfg["teacher"].update(
                local_control_journal=config.get("local_control_journal", True),
                private_runtime_dir=config.get("private_runtime_dir"),
                failure_root=str(root),
                metadata_max_attempts=config.get("metadata_max_attempts", 3),
            )
            cfg["teacher"].update({key: config[key] for key in ("metadata_timeout_s", "admission_max_age_s", "retry_prior_max_age_s") if key in config})
            workers.append(NativeWorker(cfg))
        if not workers:
            raise FrontierBlocked("EXPLICIT_RTX_WORKERS_REQUIRED")

        def execute(identity, reservation):
            number = reservation["worker"]
            worker = workers[number]
            folder = worker.directory(reservation["episode_id"])
            # A completed seal can be recovered without model/account/GPU work.
            if not folder.exists():
                with file_lock(Path(config["workers"][number]["resource_lock_file"])):
                    import shutil

                    if (
                        shutil.disk_usage(data).free
                        < config.get("min_free_gib", 100) * 1024**3
                    ):
                        raise FrontierBlocked("TEACHER_MINIMUM_FREE_STORAGE_NOT_MET")
                    if sha256(identity["layout_path"]) != identity["layout_sha256"]:
                        raise FrontierBlocked("FROZEN_LAYOUT_CHANGED_BEFORE_LAUNCH")
                    lease = resource_admission(config["workers"][number])
                    worker_idle(config["workers"][number])
                    reservation["deadline"] = min(
                        reservation["deadline"], lease["expires_at"]
                    )
                    # Fresh non-generative quota check before allocating Isaac.
                    runner = IsolatedCodex(
                        {**config, "failure_root": str(root)},
                        root / "admission" / reservation["episode_id"],
                        minimum_lifetime=config["episode_wall_seconds"] + 90,
                    )
                    scheduler_admission = runner.metadata("scheduler")
                    reservation["scheduler_admission"] = scheduler_admission
                    reservation_path = (
                        root / "reservations" / (reservation["episode_id"] + ".json")
                    )
                    write_json(reservation_path, reservation)
                    worker.config["teacher_reservation"] = {
                        "path": str(reservation_path),
                        "sha256": sha256(reservation_path),
                    }
                    worker.start_episode(
                        reservation["episode_id"], identity, reservation["episode_id"]
                    )
                    return finish(worker, folder, identity)
            return finish(worker, folder, identity)

        def finish(worker, folder, identity):
            job = json.loads((folder / "job.json").read_text())
            child = worker.children.get(job["episode_id"])
            deadline = min(
                budget.value["deadline"],
                json.loads(Path(job["teacher_reservation"]["path"]).read_text())[
                    "deadline"
                ],
            )
            while (
                child.poll() is None
                if child is not None
                else worker.alive(job["episode_id"])
            ):
                if (root / "STOP").exists() or time.time() > deadline:
                    worker.handle({"op": "close", "episode_id": job["episode_id"]})
                    break
                time.sleep(0.25)
            code = child.returncode if child is not None else None
            return seal(folder, data, job, code)

        # Assign each identity once; one future per explicitly allocated worker.
        concurrency = config.get("max_concurrency", 1)
        if type(concurrency) is not int or not 1 <= concurrency <= len(workers):
            raise FrontierBlocked("BOUNDED_ACCOUNT_CONCURRENCY_REQUIRED")
        pending = [identity for identity in identities if (identity["task"], identity["seed"], identity["layout"]) not in excluded]
        futures = {}
        held_lanes = budget.value.setdefault("lane_holds", {})
        free = [number for number in range(concurrency) if str(number) not in held_lanes]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            while pending or futures:
                if (root / "RAW_INTEGRITY_HOLD.json").exists():
                    pending = []
                if (root / "STOP").exists():
                    pending = []
                while pending and free:
                    target = config.get("target_valid_scores")
                    if target and sum(t.get("valid_score", False) for t in budget.value["terminal"].values()) >= target:
                        pending = []
                        break
                    identity = pending.pop(0)
                    key = digest(identity)
                    if key in budget.value["terminal"]:
                        continue
                    existing = budget.value["reserved"].get(key)
                    number = existing["worker"] if existing else free[0]
                    if number not in free:
                        pending.append(identity)
                        break
                    reservation = budget.reserve(identity, number)
                    if reservation is None:
                        pending = []
                        break
                    free.remove(number)
                    future = pool.submit(execute, identity, reservation)
                    futures[future] = (identity, number)
                if not futures:
                    break
                done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    identity, number = futures.pop(future)
                    try:
                        terminal = future.result()
                    except Exception as exc:
                        from robouse.teacher.failures import emit

                        emit(root, classification=str(exc) if isinstance(exc, FrontierBlocked) else type(exc).__name__, stage="controller", kind="HOLD")
                        terminal = {
                            "outcome": "infrastructure_error",
                            "valid_score": False,
                            "native_success": None,
                            "error_type": type(exc).__name__,
                            "reason": str(exc)
                            if isinstance(exc, FrontierBlocked)
                            else "EXECUTION_OR_SEAL_FAILED",
                        }
                        if isinstance(exc, FrontierBlocked):
                            from robouse.teacher.recovery import lane_local_hold

                            if lane_local_hold(exc):
                                held_lanes[str(number)] = {"reason": str(exc), "episode_retries": 0, "new_dispatch": False}
                            else:
                                pending = []
                                from robouse.teacher.recovery import close_admission

                                close_admission(root, str(exc))
                        # Failed attempts are charged and terminal, never retried automatically.
                    budget.complete(identity, terminal)
                    if str(number) not in held_lanes:
                        free.append(number)
                    write_json(
                        root / "teacher-progress.json",
                        {
                            "reserved": len(budget.value["reserved"]),
                            "terminal": len(budget.value["terminal"]),
                            "max_episodes": config["max_episodes"],
                            "valid_scores": sum(
                                t.get("valid_score", False)
                                for t in budget.value["terminal"].values()
                            ),
                            "elapsed": time.time() - budget.value["started_at"],
                        },
                    )
        result = []
        for seed in sorted({i["seed"] for i in identities}):
            path = (
                data / f"seed-{seed}" / "snapshots" / f"sealed-{time.time_ns()}.jsonl"
            )
            result.append(snapshot(data, seed, path))
        budget.value["requested_identities"] = len(identities)
        budget.value["not_started"] = sum(
            digest(i) not in budget.value["reserved"] and (i["task"], i["seed"], i["layout"]) not in excluded for i in identities
        )
        budget.value["excluded_sealed_identities"] = sum((i["task"], i["seed"], i["layout"]) in excluded for i in identities)
        budget.value["valid_scores"] = sum(
            t.get("valid_score", False) for t in budget.value["terminal"].values()
        )
        budget.value["status"] = (
            "RAW_INTEGRITY_HOLD"
            if (root / "RAW_INTEGRITY_HOLD.json").exists()
            else "ADMISSION_CLOSED"
            if (root / "ADMISSION_CLOSED.json").exists()
            else "STOPPED"
            if (root / "STOP").exists()
            else "LANES_HELD"
            if held_lanes and budget.value["not_started"] and len(held_lanes) >= concurrency
            else "BUDGET_LIMITED"
            if budget.value["not_started"]
            else "FINISHED_BOUNDED"
        )
        target = config.get("target_valid_scores")
        if target and budget.value["status"] not in ("ADMISSION_CLOSED", "STOPPED", "RAW_INTEGRITY_HOLD", "LANES_HELD"):
            budget.value["status"] = "TARGET_REACHED" if budget.value["valid_scores"] >= target else "IDENTITIES_EXHAUSTED_BELOW_TARGET" if not budget.value["not_started"] else "BUDGET_LIMITED"
        write_json(budget.path, budget.value)
        write_json(root / "snapshots.json", result)
        return budget.value
