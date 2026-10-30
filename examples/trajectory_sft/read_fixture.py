"""Read public synthetic RGB/history and export qualified windows; no model calls."""
import argparse
import json
from pathlib import Path
from robouse.data.trajectory import load_trajectory, decision_windows
from robouse.data.datasets import image_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "fixture"
    counts = {}
    for seed, role in ((0, "train"), (1, "val")):
        trajectory = load_trajectory(root / f"trajectory-seed{seed}.json")
        rows = list(decision_windows(trajectory))
        for row in rows:
            for ref in row["images"].values():
                image_bytes(ref, root)
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / f"seed{seed}-{role}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        counts[role] = len(rows)
    print(json.dumps({"origin": "synthetic_fixture", "qualified_windows": counts, "data_root": str(root), "native_experiment": False}))


if __name__ == "__main__":
    main()
