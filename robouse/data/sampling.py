"""Deterministic multi-source mixtures, separate from model tokenization."""
import math
import random
from pathlib import Path
from robouse.data.datasets import read_jsonl, check_roles


def load_sources(config, split):
    specs = config.get("datasets", {}).get(split)
    if specs is None:
        specs = [{"path": config[split + "_data"], "name": split, "weight": 1.0}]
    if not isinstance(specs, list) or not specs:
        raise ValueError("Nonempty dataset source list required")
    rows, weights, manifests = [], [], []
    from agp_rl.io import sha256
    for source in specs:
        path = Path(source["path"])
        weight = float(source.get("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Positive finite dataset weights required")
        samples = read_jsonl(path)
        rows.extend({**row, "sampling_source": source.get("name", path.stem)} for row in samples)
        # A source's total sampling mass equals its configured weight.
        weights.extend([weight / len(samples)] * len(samples) if samples else [])
        manifests.append({"path": str(path), "sha256": sha256(path), "weight": weight, "rows": len(samples)})
    check_roles(rows)
    if len({r["window_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate windows across dataset sources")
    return rows, weights, manifests


def sample_indices(size, strategy, *, weights, count=None, seed=0):
    if size < 1:
        raise ValueError("Empty training data")
    if strategy == "coverage":
        if count is not None and count != size:
            raise ValueError("Coverage sampling visits every window exactly once per epoch")
        return list(range(size))
    if strategy != "weighted_replacement":
        raise ValueError("Unknown sampling strategy")
    if count is None or type(count) is not int or count < 1:
        raise ValueError("Weighted sampling requires explicit samples_per_epoch")
    return random.Random(seed).choices(range(size), weights=weights, k=count)
