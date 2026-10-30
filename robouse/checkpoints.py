"""Publish completed HF/ZeRO checkpoints only after all ranks finish saving."""
import json
from pathlib import Path
from agp_rl.io import sha256
from robouse.data.datasets import write_json

MARKER = "CHECKPOINT_COMMITTED.json"


def commit(path, world_size):
    path = Path(path)
    if not (path / "trainer_state.json").is_file() or not list(
        path.glob("*.safetensors")
    ):
        raise ValueError("Full HF checkpoint incomplete")
    zero = list(path.glob("global_step*"))
    if world_size == 8:
        if (
            not zero
            or len(list(path.glob("global_step*/*optim_states.pt"))) < 8
            or len(list(path.glob("rng_state*.pth"))) < 8
        ):
            raise ValueError(
                "Complete eight-rank ZeRO optimizer/RNG checkpoint required"
            )
    files = {}
    for p in sorted(path.rglob("*")):
        if not p.is_file() or p.name == MARKER:
            continue
        # Full optimizer shards are inventoried by size; weights and small
        # metadata/RNG files have hashes. Avoid re-reading 8 Adam shards on NFS.
        row = {"bytes": p.stat().st_size}
        if p.suffix == ".safetensors" or p.stat().st_size < 1024 * 1024:
            row["sha256"] = sha256(p)
        files[str(p.relative_to(path))] = row
    write_json(
        path / MARKER,
        {
            "world_size": world_size,
            "files": files,
            "optimizer_integrity": "file sizes plus actual framework reload; not a full-shard checksum",
        },
    )


def verify(path):
    path = Path(path)
    value = json.loads((path / MARKER).read_text())
    for name, row in value["files"].items():
        p = path / name
        if p.stat().st_size != row["bytes"] or (
            "sha256" in row and sha256(p) != row["sha256"]
        ):
            raise ValueError("Committed checkpoint changed: " + name)
    return value
