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
from robouse.teacher.state import Budget, file_lock, resource_admission, validate
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


def run(config, *, resume=False, final_test=False, check_only=False):
    root = Path(config["output"]).resolve()
    identities = [] if check_only else validate(config, final_test=final_test)
    root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(file_lock(root / "teacher-controller.lock"))
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
            )
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
                        config,
                        root / "admission" / reservation["episode_id"],
                        minimum_lifetime=config["episode_wall_seconds"] + 90,
                    )
                    runner.metadata("scheduler")
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
        pending = list(identities)
        futures = {}
        free = list(range(concurrency))
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            while pending or futures:
                if (root / "STOP").exists():
                    pending = []
                while pending and free:
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
                            pending = []
                            write_json(
                                root / "ADMISSION_CLOSED.json",
                                {"reason": str(exc), "automatic_account_switch": False},
                            )
                        # Failed attempts are charged and terminal, never retried automatically.
                    budget.complete(identity, terminal)
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
            digest(i) not in budget.value["reserved"] for i in identities
        )
        budget.value["valid_scores"] = sum(
            t.get("valid_score", False) for t in budget.value["terminal"].values()
        )
        budget.value["status"] = (
            "ADMISSION_CLOSED"
            if (root / "ADMISSION_CLOSED.json").exists()
            else "STOPPED"
            if (root / "STOP").exists()
            else "BUDGET_LIMITED"
            if budget.value["not_started"]
            else "FINISHED_BOUNDED"
        )
        write_json(budget.path, budget.value)
        write_json(root / "snapshots.json", result)
        return budget.value
