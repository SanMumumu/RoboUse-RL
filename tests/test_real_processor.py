"""Optional local processor-only contract; no model weights or CUDA allocations."""
import hashlib
import os
from pathlib import Path
import pytest
import torch


def test_real_qwen_rgb_template_and_assistant_mask(tmp_path):
    source = os.environ.get("ROBOUSE_TEST_PROCESSOR")
    if not source:
        pytest.skip("Set ROBOUSE_TEST_PROCESSOR for local processor-only acceptance")
    from transformers import AutoProcessor
    from PIL import Image
    from robouse.training.sft.train import encode

    processor = AutoProcessor.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    from robouse.models.qwen import processor_signature

    assert len(processor_signature(processor)) == 64
    path = tmp_path / "original.png"
    Image.new("RGB", (64, 64), "red").save(path)
    raw = path.read_bytes()
    key = hashlib.sha256(raw).hexdigest()
    row = {
        "window_id": "fixture",
        "images": {key: {"path": path.name, "sha256": key, "bytes": len(raw)}},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read this original RGB."},
                    {"type": "image", "image": "art-image://" + key},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": '{"name":"observe","arguments":{}}'}
                ],
            },
        ],
    }
    value = encode(row, processor, tmp_path)
    assert value["target"][-1] == processor.tokenizer.eos_token_id
    assert value["inputs"]["pixel_values"].numel() > 0
    assert value["inputs"]["input_ids"].shape[-1] > len(value["target"])
    assert (
        processor.decode(value["target"], skip_special_tokens=True).strip()
        == '{"name":"observe","arguments":{}}'
    )
