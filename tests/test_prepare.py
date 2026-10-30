import hashlib
import json
from pathlib import Path
from PIL import Image
from robouse.data.datasets import prepare, read_jsonl, image_bytes
from agp_rl.pipeline2b.fullft import source_adapters


def test_prepare_new_seed0_local_windows_keeps_original_rgb_and_never_reads_test(
    tmp_path, monkeypatch
):
    root = tmp_path / "read-only-source"
    root.mkdir()
    image = root / "original.png"
    Image.new("RGB", (4, 4), "blue").save(image)
    raw = image.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    candidate = {
        "episode": {
            "seed": 0,
            "task": "task",
            "layout_id": 1,
            "supervision_scope": "qualified_local_decisions",
        },
        "system_prompt": "actual tools",
        "images": {
            "original-key": {
                "path": str(image),
                "bytes": len(raw),
                "sha256": sha,
                "width": 4,
                "height": 4,
            }
        },
        "bundle": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": "art-image://original-key"}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "safe original target"}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "broken suffix"}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "rejected target"}],
                },
            ],
            "decisions": [
                {
                    "decision_id": "d0",
                    "visible_history": {"message_end_exclusive": 1},
                    "model_output": {"text": "safe original target"},
                    "data_quality": {
                        "sft_candidate": True,
                        "input_history_complete": True,
                    },
                },
                {
                    "decision_id": "d1",
                    "visible_history": {"message_end_exclusive": 3},
                    "model_output": {"text": "rejected target"},
                    "data_quality": {"sft_candidate": False},
                },
            ],
        },
    }

    def source(spec, verifier):
        assert spec["seed"] == 0
        yield {"source": "test", "reasons": []}, candidate

    monkeypatch.setattr(source_adapters, "public_candidates", source)
    cfg = {
        "data_root": str(root),
        "sources": [
            {
                "kind": "public_art",
                "name": "s0",
                "seed": 0,
                "root": ".",
                "art_root": ".",
            },
            {
                "kind": "native",
                "name": "sealed-test",
                "seed": 2,
                "root": "nonexistent-test-targets",
            },
        ],
    }
    out = tmp_path / "independent-prepared"
    rows = prepare(cfg, out)
    assert len(rows) == 1 and rows[0]["seed"] == 0
    assert rows[0]["messages"][-1]["content"][0]["text"] == "safe original target"
    assert not any("broken suffix" in str(m) for m in rows[0]["messages"])
    assert image_bytes(next(iter(rows[0]["images"].values())), root) == raw
    assert (out / "seed2-test.jsonl").read_text() == ""
    assert json.loads((out / "preparation.json").read_text())["test"] == "PENDING_TEST"
    assert (out / "required.files").read_text() == "original.png\n"
    assert "seed0-train.jsonl" in (out / "prepared.files").read_text()
