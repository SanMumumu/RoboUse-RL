"""Verify local raw weights against pinned HF LFS identities, separately from datasets."""
import argparse
import json
from pathlib import Path
from agp_rl.io import digest, sha256
from robouse.data.datasets import export, write_json


def inventory(model, repo, revision):
    if len(revision) != 40:
        raise ValueError("Full upstream HF commit SHA required")
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError("HF revision resolved differently")
    root = Path(model).resolve()
    upstream = {}
    for item in info.siblings:
        if item.rfilename.endswith(".safetensors"):
            if not item.lfs or not item.lfs.sha256:
                raise ValueError("Upstream weight SHA256 unavailable")
            path = root / item.rfilename
            if not path.is_file() or sha256(path) != item.lfs.sha256:
                raise ValueError(
                    "Local weights differ from original HF base: " + item.rfilename
                )
            upstream[item.rfilename] = item.lfs.sha256
    if not upstream:
        raise ValueError("No full upstream weights")
    files = [
        p.name
        for p in root.iterdir()
        if p.is_file()
        and p.name not in ("model-manifest.json",)
        and p.suffix in (".json", ".jinja", ".txt", ".safetensors")
    ]
    if any(
        (root / name).exists()
        for name in ("trainer_state.json", "adapter_config.json", "fullft.json")
    ):
        raise ValueError("Trained checkpoint supplied as original model")
    result = export(root, files, root / "model-manifest.json")
    result["origin"] = {
        "kind": "hf_base",
        "repo_id": repo,
        "revision": revision,
        "weight_sha256": upstream,
    }
    write_json(root / "model-manifest.json", result)
    return digest(result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--hf-id", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.dry_run or not a.execute:
        print(
            json.dumps({"status": "DRY_RUN", "hf_id": a.hf_id, "revision": a.revision})
        )
        return
    print(inventory(a.model, a.hf_id, a.revision))


if __name__ == "__main__":
    main()
