"""Policy inputs are exclusively public MCP replies and the frozen task prompt."""
from __future__ import annotations

from dataclasses import dataclass
import base64
import io
import json
from pathlib import Path

from agp_rl.io import atomic_bytes, digest, sha256

FORBIDDEN = {"reward", "reward_manager", "official_score", "official_success", "eval_result",
             "success_nums", "fail_nums", "episode_ended", "hidden_state", "privileged_state",
             "depth", "depth_image", "segmentation", "simulator_state"}


def assert_public(value):
    if isinstance(value, dict):
        bad = {str(k).lower() for k in value} & FORBIDDEN
        if bad:
            raise ValueError(f"Forbidden policy observation fields: {sorted(bad)}")
        for v in value.values():
            assert_public(v)
    elif isinstance(value, list):
        for v in value:
            assert_public(v)


@dataclass(frozen=True)
class PublicReply:
    text: str
    images: tuple[str, ...] = ()
    image_hashes: tuple[str, ...] = ()

    def message(self):
        return {"role": "user", "content": [
            {"type": "text", "text": self.text},
            *[{"type": "image", "image": path} for path in self.images]]}


def decode_mcp(result, image_dir):
    """Archive RGB bytes from MCP; never accept arbitrary resource/file URLs."""
    from PIL import Image
    image_dir = Path(image_dir)
    text, images, hashes = [], [], []
    for block in result.content:
        if block.type == "text":
            try:
                parsed = json.loads(block.text)
            except json.JSONDecodeError:
                parsed = block.text  # official MCP may label an image with a text block
            assert_public(parsed)
            text.append(block.text)
        elif block.type == "image":
            if block.mimeType not in {"image/png", "image/jpeg"}:
                raise ValueError("Only official RGB PNG/JPEG is accepted")
            raw = base64.b64decode(block.data, validate=True)
            with Image.open(io.BytesIO(raw)) as im:
                if im.mode not in {"RGB", "RGBA"}:
                    raise ValueError("Non-RGB observation refused")
                im.verify()
            import hashlib
            h = hashlib.sha256(raw).hexdigest()
            path = image_dir / (h + (".png" if block.mimeType == "image/png" else ".jpg"))
            if path.exists():
                if sha256(path) != h:
                    raise ValueError("Corrupt observation archive")
            else:
                atomic_bytes(path, raw)
            images.append(str(path.resolve()))
            hashes.append(h)
        else:
            raise ValueError(f"Unsupported MCP content type {block.type}")
    return PublicReply("\n".join(text), tuple(images), tuple(hashes))


class ObservationHistory:
    def __init__(self, system_prompt):
        self.messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]

    def append_reply(self, reply):
        if not isinstance(reply, PublicReply):
            raise TypeError("Only PublicReply may enter policy history")
        self.messages.append(reply.message())

    def append_action(self, raw_text):
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": raw_text}]})

    @property
    def hash(self):
        return digest(self.messages)
