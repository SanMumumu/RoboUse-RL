"""Portable selection-only seam into the unchanged RoboDojo evaluator."""
import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import signal
import socket
import subprocess
import sys
import time
import types
from robouse.data.datasets import write_json
from agp_rl.io import sha256, digest


def source_root():
    return Path(__file__).resolve().parents[2] / "_native/AGP"


def verify_sources():
    root = source_root()
    manifest = json.loads((root.parent / "agp-source-manifest.json").read_text())
    for name, h in manifest.items():
        if sha256(root / name) != h:
            raise ValueError("Official control source changed: " + name)
    return digest(manifest)


def register(job):
    benchmark = Path(job["benchmark"])
    policy = source_root()
    sys.path[:0] = [str(benchmark), str(benchmark / "XPolicyLab")]
    importlib.import_module("XPolicyLab.policy")
    name = "XPolicyLab.policy.RoboUse"
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name + ".model", policy / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from robouse.environments.robodojo import native_deploy

    sys.modules[name + ".deploy"] = native_deploy


def restrict_identity(job):
    from env.seed_manager import seed_manager

    original = seed_manager.SeedManager

    class Selected(original):
        def init_eval(self, *args, **kwargs):
            super().init_eval(*args, **kwargs)
            identity = job["identity"]
            if self.task_name != identity["task"] or self.eval_seed != identity["seed"]:
                raise ValueError("Native task/seed mismatch")
            if identity["layout"] not in self.seed_list:
                raise ValueError("Requested native layout unavailable")
            self.seed_list = [identity["layout"]]
            self.st_idx, self.idx, self.ed_idx = 0, 0, 1

    seed_manager.SeedManager = Selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", required=True)
    p.add_argument("--stub", action="store_true")
    p.add_argument("--port", type=int)
    args = p.parse_args()
    path = Path(args.job).resolve()
    job = json.loads(path.read_text())
    if not args.stub:
        owner = path.parent / "owner.json"
        deadline = time.monotonic() + 30
        while not owner.exists():
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "Worker ownership was not committed; refusing simulator startup"
                )
            time.sleep(0.05)
        info = json.loads(owner.read_text())
        actual = (
            Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]
        )
        if info["pid"] != os.getpid() or info["start_ticks"] != actual:
            raise ValueError("Native process owner mismatch")

        def terminate(number, frame):
            raise SystemExit(128 + number)

        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
    verify_sources()
    if job.get("actor_kind") == "astra_teacher":
        identity = job["identity"]
        if sha256(identity["layout_path"]) != identity["layout_sha256"]:
            raise ValueError("Frozen teacher layout changed")
    register(job)
    benchmark = Path(job["benchmark"])
    policy = source_root()
    if args.stub:
        sys.argv = [
            str(benchmark / "XPolicyLab/setup_policy_server.py"),
            "--config_path",
            str(policy / "deploy.yml"),
            "--overrides",
            "policy_name=RoboUse",
            f"port={args.port}",
            "host=127.0.0.1",
            "bench_name=RoboDojo",
            f"task_name={job['identity']['task']}",
            "env_cfg_type=arx_x5",
            "action_type=joint",
            f"seed={job['identity']['seed']}",
            "ckpt_name=no-checkpoint",
        ]
        runpy.run_path(
            str(benchmark / "XPolicyLab/setup_policy_server.py"), run_name="__main__"
        )
        return
    work = path.parent / "work"
    work.mkdir(exist_ok=True)
    os.chdir(work)
    home = path.parent / "agp_home"
    home.mkdir(exist_ok=True)
    for name in ("config", "cache/kit", "logs/kit"):
        (home / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(
        ROBOUSE_NATIVE_JOB=str(path),
        AGP_HOME=str(home),
        ROBODOJO_ROOT=str(benchmark),
        ROBODOJO_RUN_ID=job["episode_id"],
        EVAL_NUM="1",
    )
    restrict_identity(job)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [
                str(Path(__file__).resolve().parents[3]),
                str(benchmark),
                str(benchmark / "XPolicyLab"),
            ]
        ),
    }
    with (path.parent / "stub.log").open("a") as log:
        stub = subprocess.Popen(
            [
                job["transport_python"],
                "-m",
                "robouse.environments.robodojo.native_job",
                "--job",
                str(path),
                "--stub",
                "--port",
                str(port),
            ],
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if stub.poll() is not None:
                    raise RuntimeError("Native policy server exited")
                if any(
                    row.split()[3] == "0A"
                    and int(row.split()[1].split(":")[-1], 16) == port
                    for table in ("/proc/net/tcp", "/proc/net/tcp6")
                    for row in Path(table).read_text().splitlines()[1:]
                ):
                    break
                time.sleep(0.1)
            else:
                raise TimeoutError("Native policy server startup timeout")
            sys.argv = [
                str(benchmark / "src/eval_client/main.py"),
                "--task_name",
                job["identity"]["task"],
                "--num_envs",
                "1",
                "--env_cfg_type",
                "arx_x5",
                "--device_id",
                "0",
                "--policy_name",
                "RoboUse",
                "--port",
                str(port),
                "--host",
                "127.0.0.1",
                "--additional_info",
                job["episode_id"],
                "--seed",
                str(job["identity"]["seed"]),
                "--headless",
                "--enable_cameras",
                "--kit_args",
                " ".join(
                    [
                        "--enable isaacsim.replicator.behavior",
                        "--enable isaacsim.sensors.camera",
                        "--/renderer/multiGpu/enabled=false",
                        "--/renderer/activeGpu=0",
                        "--/physics/cudaDevice=0",
                        f"--/app/userConfigPath={home}/config/user.config.json",
                        f"--/app/cachePath={home}/cache/kit",
                        f"--/app/logPath={home}/logs/kit",
                    ]
                ),
            ]
            runpy.run_path(
                str(benchmark / "src/eval_client/main.py"), run_name="__main__"
            )
        finally:
            if stub.poll() is None:
                stub.terminate()
                try:
                    stub.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    stub.kill()
                    stub.wait()


if __name__ == "__main__":
    main()
