"""One RTX evaluator worker. Only explicitly owned native episode groups are signalled."""
import base64
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from agp_rl.io import sha256, digest
from robouse.data.datasets import write_json


def close_state(a, b, tol=1e-6):
    if type(a) in (int, float) and type(b) in (int, float):
        return abs(a - b) <= tol
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(close_state(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close_state(x, y, tol) for x, y in zip(a, b))
    return type(a) == type(b) and a == b


class NativeWorker:
    def __init__(self, config):
        self.config = config
        self.root = Path(config["output"]).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.children = {}
        self.lock = threading.RLock()
        self.hashes = {}

    def directory(self, episode):
        if not episode or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for c in episode
        ):
            raise ValueError("Invalid episode ID")
        return self.root / episode

    def alive(self, episode):
        folder = self.directory(episode)
        owner = folder / "owner.json"
        if not owner.exists():
            return False
        info = json.loads(owner.read_text())
        try:
            stat = (
                Path(f"/proc/{info['pid']}/stat").read_text().rsplit(")", 1)[1].split()
            )
            return stat[19] == info["start_ticks"] and stat[0] != "Z"
        except FileNotFoundError:
            return False

    def wait(self, path, episode):
        deadline = time.monotonic() + self.config.get("rpc_timeout", 1800)
        while not path.exists():
            if (self.root / "STOP").exists():
                (path.parent / "CANCEL").touch()
                if self.alive(episode):
                    info = json.loads((path.parent / "owner.json").read_text())
                    os.killpg(info["pid"], signal.SIGTERM)
                return None
            child = self.children.get(episode)
            if child is not None and child.poll() is not None:
                return None
            if child is None and not self.alive(episode):
                return None
            if (path.parent / "CANCEL").exists():
                raise RuntimeError("INTERRUPTED")
            if time.monotonic() >= deadline:
                raise TimeoutError("Native RPC timeout")
            time.sleep(0.05)
        return json.loads(path.read_text())

    def blocks(self, blocks, folder):
        result = []
        for b in blocks:
            if b["type"] == "text":
                result.append(b)
                continue
            path = Path(b["image"]).resolve()
            if not path.is_relative_to(folder):
                raise ValueError("Image outside episode")
            result.append(
                {
                    "type": "image",
                    "sha256": sha256(path),
                    "data": base64.b64encode(path.read_bytes()).decode(),
                    "name": str(path.relative_to(folder)),
                }
            )
        return result

    def result(self, episode):
        folder = self.directory(episode)
        child = self.children.get(episode)
        if (folder / "result.json").exists():
            return json.loads((folder / "result.json").read_text())
        if child is None:
            deadline = time.monotonic() + self.config.get("rpc_timeout", 1800)
            while self.alive(episode):
                if time.monotonic() > deadline:
                    raise TimeoutError("Restored native evaluator timeout")
                time.sleep(0.1)
        if child is not None:
            try:
                code = child.wait(timeout=self.config.get("rpc_timeout", 1800))
            except subprocess.TimeoutExpired:
                raise TimeoutError("Native evaluator finalization timeout")
        candidates = list((folder / "work").rglob("*_result.json"))
        if not candidates:
            interrupted = (folder / "CANCEL").exists()
            return {
                "done": True,
                "outcome": "interrupted" if interrupted else "infrastructure_error",
                "reward": None,
                "termination_reason": "operator_stop"
                if interrupted
                else "native_unscored_exit",
            }
        from agp_rl.reward.official import read_official

        job = json.loads((folder / "job.json").read_text())
        candidates = list((folder / "work").rglob("*_result.json"))
        if len(candidates) != 1:
            raise RuntimeError(
                "Expected one unchanged official result; no synthetic reward"
            )
        native = read_official(candidates[0], job["identity"]["layout"])
        reference = folder / "native-session.json"
        session = Path(json.loads(reference.read_text())["path"]) if reference.exists() else None
        metrics_path = session / "episode_metrics.json" if session else None
        metrics = json.loads(metrics_path.read_text()) if metrics_path and metrics_path.exists() else {}
        finished_path = session / "policy_finished.json" if session else None
        finished = json.loads(finished_path.read_text()) if finished_path and finished_path.exists() else {}
        if not metrics.get("official_ended_before_finalize") and (metrics.get("infrastructure_failure") or (folder / "CANCEL").exists()):
            interrupted = (folder / "CANCEL").exists()
            result = {"done": True, "outcome": "interrupted" if interrupted else "infrastructure_error", "reward": None, "termination_reason": "operator_stop" if interrupted else "native_infrastructure_failure"}
            write_json(folder / "result.json", result)
            return result
        truncated = not native.success and finished.get("stopped_by") in ("decision_limit", "context_limit")
        videos = {}
        for path in folder.rglob("*.mp4"):
            for camera in ("top", "left_wrist", "right_wrist"):
                if camera in path.name and path.stat().st_size > 48:
                    videos[camera] = str(path.relative_to(folder))
        result = {
            "done": True,
            "outcome": "completed" if native.success else "timeout" if truncated else "policy_failure",
            "reward": float(native.success),
            "score": native.score,
            "truncated": truncated,
            "native_reward": native.record(),
            "reward_artifact": str(candidates[0].relative_to(folder)),
            "videos": videos,
            "versions": {
                "native_control_source": __import__("robouse.environments.robodojo.native_job", fromlist=["verify_sources"]).verify_sources(),
                "benchmark": __import__("robouse.utils.manifests", fromlist=["git_sha"]).git_sha(job["benchmark"]),
                "sensor": self.config.get("sensor_version"),
                "controller": self.config.get("controller_version"),
            },
            "termination_reason": "native_success"
            if native.success
            else finished.get("stopped_by") or "native_failure",
        }
        write_json(folder / "result.json", result)
        return result

    def start_episode(self, episode, identity, group_id):
        folder = self.directory(episode)
        if not folder.exists():
            if any(self.alive(p.name) for p in self.root.iterdir() if p.is_dir()):
                raise RuntimeError("Worker busy; allocate another explicit RTX worker")
            folder.mkdir()
            job = {k: v for k, v in self.config.items() if k not in ("token", "secret")}
            job.update(
                episode_id=episode,
                identity=identity,
                group_id=group_id,
            )
            write_json(folder / "job.json", job)
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(self.config["renderer_gpu"]),
                "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            with (folder / "native.log").open("a") as log:
                child = subprocess.Popen(
                    [
                        self.config["simulator_python"],
                        "-m",
                        "robouse.environments.robodojo.native_job",
                        "--job",
                        str(folder / "job.json"),
                    ],
                    env=env,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            self.children[episode] = child
            write_json(
                folder / "owner.json",
                {
                    "pid": child.pid,
                    "start_ticks": Path(f"/proc/{child.pid}/stat")
                    .read_text()
                    .rsplit(")", 1)[1]
                    .split()[19],
                },
            )
        return folder

    def handle(self, request):
        with self.lock:
            op = request["op"]
            episode = request["episode_id"]
            folder = self.directory(episode)
            if op == "blob":
                path = (folder / request["path"]).resolve()
                if not path.is_relative_to(folder) or path.suffix not in (
                    ".png",
                    ".jpg",
                    ".mp4",
                    ".json",
                ):
                    raise ValueError("Invalid artifact")
                published = (
                    path.suffix == ".mp4"
                    or (
                        path.suffix in (".png", ".jpg")
                        and path.is_relative_to(folder / "images")
                    )
                    or (
                        path.name == "_result.json"
                        and path.is_relative_to(folder / "work")
                    )
                )
                if not published:
                    raise ValueError(
                        "Only public RGB, native videos and official score artifacts may be downloaded"
                    )
                stat = path.stat()
                key = (str(path), stat.st_size, stat.st_mtime_ns)
                if key not in self.hashes:
                    self.hashes[key] = sha256(path)
                offset = int(request.get("offset", 0))
                size = int(request.get("chunk_bytes", 4 * 1024 * 1024))
                if (
                    offset < 0
                    or offset > stat.st_size
                    or not 0 < size <= 4 * 1024 * 1024
                ):
                    raise ValueError("Invalid artifact range")
                with path.open("rb") as stream:
                    stream.seek(offset)
                    raw = stream.read(size)
                after = path.stat()
                if (after.st_size, after.st_mtime_ns) != (
                    stat.st_size,
                    stat.st_mtime_ns,
                ):
                    raise ValueError("Artifact still changing")
                return {
                    "data": base64.b64encode(raw).decode(),
                    "sha256": self.hashes[key],
                    "total_bytes": stat.st_size,
                    "offset": offset,
                }
            if op == "reset":
                self.start_episode(episode, request["identity"], request["group_id"])
                observation = self.wait(folder / "observation-0.json", episode)
                if observation is None:
                    return self.result(episode)
                snapshot = json.loads(
                    (folder / "evaluator_private/initial-state.json").read_text()
                )
                state = {
                    k: snapshot["state"][k]
                    for k in ("physical", "static", "seed_metadata")
                }
                group_path = self.root / (
                    "group-" + digest(request["group_id"]) + ".json"
                )
                if group_path.exists():
                    baseline = json.loads(group_path.read_text())
                    if not close_state(state, baseline["state"]):
                        raise ValueError(
                            "Native initial-state comparability failed at 1e-6"
                        )
                    fingerprint = baseline["hash"]
                else:
                    fingerprint = digest(state)
                    write_json(group_path, {"state": state, "hash": fingerprint})
                return {
                    **observation,
                    "blocks": self.blocks(observation["blocks"], folder),
                    "initial_state_hash": fingerprint,
                }
            if op == "step":
                index = request["index"]
                path = folder / f"action-{index}.json"
                action = {
                    "text": request["text"],
                    "finish": request.get("finish", False),
                }
                if path.exists() and json.loads(path.read_text()) != action:
                    raise ValueError("Idempotency conflict")
                if not path.exists():
                    write_json(path, action)
                obs = self.wait(folder / f"observation-{index+1}.json", episode)
                if obs is None:
                    return self.result(episode)
                if obs["done"]:
                    return {
                        **obs,
                        **self.result(episode),
                        "blocks": self.blocks(obs["blocks"], folder),
                    }
                return {**obs, "blocks": self.blocks(obs["blocks"], folder)}
            if op == "result":
                return self.result(episode)
            if op == "close":
                (folder / "CANCEL").touch()
                child = self.children.get(episode)
                if child is None and self.alive(episode):
                    owner = json.loads((folder / "owner.json").read_text())
                    os.killpg(owner["pid"], signal.SIGTERM)
                if child is not None and child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                return {"closed": True}
            raise ValueError("Unknown operation")
