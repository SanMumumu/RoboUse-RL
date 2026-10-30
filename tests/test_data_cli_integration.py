"""End-to-end production source adapter and shell migration on a synthetic ART fixture."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import shutil
import pytest
from PIL import Image


def test_real_prepare_export_rsync_import_verify_scripts(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    if not shutil.which("rsync"):
        pytest.skip("rsync unavailable")
    from agp_rl.adapters.tool_codec import TOOLS

    root = tmp_path / "source"
    public = root / "rollout_trajs/agp_official"
    art = root / "rollout_trajs/agent_trace_v1"
    for path in (
        "documents/interfaces",
        "results/seed0",
        "views",
        "episodes",
        "traces",
    ):
        (public / path).mkdir(parents=True, exist_ok=True)
    for path in ("raw", "canonical/trajectories", "training"):
        (art / path).mkdir(parents=True, exist_ok=True)
    image = public / "views/original.png"
    Image.new("RGB", (4, 4), "green").save(image)
    payload = image.read_bytes()
    h = hashlib.sha256(payload).hexdigest()
    # ART normally seeks PNG members in tar; preserve that layout in this fixture.
    import tarfile
    from io import BytesIO

    tar = public / "views/images.tar"
    with tarfile.open(tar, "w") as f:
        member = tarfile.TarInfo("original.png")
        member.size = len(payload)
        f.addfile(member, BytesIO(payload))
    with tarfile.open(tar) as f:
        offset = f.getmember("original.png").offset_data
    schemas = [{"name": name, "inputSchema": {"type": "object"}} for name in TOOLS]
    (public / "documents/interfaces/interface-76b013ef.json").write_text(
        json.dumps({"tools": schemas})
    )
    (public / "results/seed0/RELEASE_s0.csv").write_text("fixture\n")
    (public / "views/metadata.csv").write_text("fixture\n")
    target = '{"name":"observe","arguments":{}}'
    epid = "synthetic_episode"
    line = (
        json.dumps({"item": {"type": "agent_message", "text": target}}) + "\n"
    ).encode()
    trace = public / "traces/fixture.jsonl.gz"
    with gzip.open(trace, "wb") as f:
        f.write(line)
    origin = {"line": 1, "line_sha256": hashlib.sha256(line).hexdigest()}
    parquet = public / "episodes/fixture.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_id": epid,
                    "outcome": "success",
                    "seed": 0,
                    "layout_id": 1,
                    "task": "task",
                    "base_task": "task",
                    "score": 1.0,
                    "prompt_text": "actual synthetic fixture prompt",
                    "traces": [{"image_keys": ["key"]}],
                }
            ]
        ),
        parquet,
    )
    ep = {
        "episode_id": epid,
        "seed": 0,
        "layout_id": 1,
        "task": "task",
        "task_config": "task",
        "official_success": True,
        "official_score": 1.0,
        "data_quality": {"decisions": 1, "final_history_issues": []},
        "provenance": {
            "parquet": "episodes/fixture.parquet",
            "row_group": 0,
            "trace": "traces/fixture.jsonl.gz",
        },
    }
    (art / "canonical/episodes.jsonl").write_text(json.dumps(ep) + "\n")
    decision = {
        "decision_id": "fixture-d0",
        "timestep": 0,
        "visible_history": {"message_end_exclusive": 1},
        "model_output": {"text": target},
        "data_quality": {
            "sft_candidate": True,
            "input_history_complete": True,
            "exclusion_reasons": [],
        },
        "provenance": origin,
    }
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "actual synthetic fixture prompt"},
                {"type": "image", "image": "art-image://key"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": target}]},
    ]
    bundle = {
        "messages": messages,
        "message_origins": [{"line": 0}, origin],
        "decisions": [decision],
        "images": {"key": {}},
    }
    with gzip.open(art / f"canonical/trajectories/{epid}.json.gz", "wt") as f:
        json.dump(bundle, f)
    image_ref = {
        "shard": "images.tar",
        "offset": offset,
        "bytes": len(payload),
        "sha256": h,
        "width": 4,
        "height": 4,
    }
    with sqlite3.connect(art / "raw/assets.sqlite") as db:
        db.execute("create table images (key text, record text)")
        db.execute("insert into images values (?,?)", ("key", json.dumps(image_ref)))
    (art / "training/sft_samples.jsonl").write_text(
        json.dumps(
            {
                "episode_id": epid,
                "decision_id": "fixture-d0",
                "quality": {"official_success": True, "input_history_complete": True},
                "target_assistant_output": target,
                "source": {"provenance": origin},
                "prompt": {"message_end_exclusive": 1},
            }
        )
        + "\n"
    )
    (art / "dataset_manifest.json").write_text(
        json.dumps({"sft_subset_status": "VALIDATED", "counts": {"sft_samples": 1}})
    )
    names = [
        "documents/interfaces/interface-76b013ef.json",
        "results/seed0/RELEASE_s0.csv",
        "views/metadata.csv",
        "episodes/fixture.parquet",
        "traces/fixture.jsonl.gz",
    ]
    (art / "raw/source_files.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "path": n,
                    "sha256": hashlib.sha256((public / n).read_bytes()).hexdigest(),
                }
            )
            + "\n"
            for n in names
        )
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "data_root": str(root),
                "sources": [
                    {
                        "name": "seed0",
                        "seed": 0,
                        "kind": "public_art",
                        "root": "rollout_trajs/agp_official",
                        "art_root": "rollout_trajs/agent_trace_v1",
                    }
                ],
            }
        )
    )
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHON": sys.executable, "PYTHONPATH": ""}

    def script(name, *args):
        return subprocess.run(
            [
                "bash",
                str(repo / "scripts" / f"{name}.sh"),
                *map(str, args),
                "--execute",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

    prepared = tmp_path / "prepared"
    script("prepare_data", "--config", config, "--output", prepared)
    rows = [
        json.loads(s) for s in (prepared / "seed0-train.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1 and rows[0]["role"] == "train"
    assert (prepared / "seed1-val.jsonl").read_text() == "" and (
        prepared / "seed2-test.jsonl"
    ).read_text() == ""
    # Set only fixture timestamps old enough for the real export stability guard.
    stamp = tar.stat().st_mtime - 120
    os.utime(tar, (stamp, stamp))
    manifest = tmp_path / "transfer.json"
    script(
        "export_data",
        "--data-root",
        root,
        "--files",
        prepared / "required.files",
        "--manifest",
        manifest,
    )
    destination = tmp_path / "relocated"
    destination.mkdir()
    script(
        "import_data",
        "--source",
        root,
        "--data-root",
        destination,
        "--manifest",
        manifest,
    )
    script("verify_data", "--data-root", destination, "--manifest", manifest)
    from robouse.data.datasets import image_bytes

    assert image_bytes(next(iter(rows[0]["images"].values())), destination) == payload
