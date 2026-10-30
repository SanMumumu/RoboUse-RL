"""Portable execution provenance without environment dumps or credentials."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
import time
from robouse.data.datasets import write_json


def git_sha(root=None):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root or Path(__file__).resolve().parents[2], stderr=subprocess.DEVNULL, text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def write_run_manifest(output, config, *, stage, datasets=None, versions=None):
    packages = {}
    for name in ("robouse-rl", "torch", "transformers", "accelerate", "deepspeed"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    value = {"schema": "robouse-run-v1", "stage": stage, "created_at": time.time(), "git_sha": git_sha(), "seed": config.get("seed", 0), "config": config, "datasets": datasets, "packages": packages, "versions": versions or {}, "test_targets_read": False if stage == "sft" else None}
    write_json(Path(output) / "run_manifest.json", value)
    return value
