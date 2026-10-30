"""HTTP only over loopback or SSH tunnel; credentials never enter run manifests."""
import json
import os
import urllib.request
import urllib.parse
from pathlib import Path
import base64
import hashlib
from robouse.data.datasets import write_json


class RemoteEnvironment:
    def __init__(self, url, episode_id, root, timeout=1800):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in (
            "127.0.0.1",
            "localhost",
            "::1",
        ):
            raise ValueError("Use authenticated SSH tunnel to loopback endpoint")
        self.url = url
        self.episode_id = episode_id
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.token = os.environ.get("ROBOUSE_ENV_TOKEN")
        if not self.token:
            raise ValueError("Missing ROBOUSE_ENV_TOKEN")

    def request(self, op, **values):
        request = urllib.request.Request(
            self.url,
            data=json.dumps(
                {"op": op, "episode_id": self.episode_id, **values}
            ).encode(),
            headers={
                "Authorization": "Bearer " + self.token,
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def observation(self, value):
        images = {}
        blocks = []
        for b in value.get("blocks", []):
            if b["type"] == "text":
                blocks.append(b)
                continue
            raw = base64.b64decode(b["data"], validate=True)
            key = hashlib.sha256(raw).hexdigest()
            if key != b["sha256"]:
                raise ValueError("Remote image hash mismatch")
            p = self.root / (key + ".png")
            if not p.exists():
                p.write_bytes(raw)
            from PIL import Image
            from io import BytesIO

            with Image.open(BytesIO(raw)) as im:
                width, height = im.size
            images[key] = {
                "path": p.name,
                "sha256": key,
                "bytes": len(raw),
                "width": width,
                "height": height,
            }
            blocks.append({"type": "image", "image": "art-image://" + key})
        return blocks, images

    async def reset(self, identity, group_id):
        import asyncio

        return await asyncio.to_thread(
            self.request, "reset", identity=identity, group_id=group_id
        )

    async def step(self, index, text, finish=False):
        import asyncio

        return await asyncio.to_thread(
            self.request, "step", index=index, text=text, finish=finish
        )

    async def close(self):
        import asyncio

        return await asyncio.to_thread(self.request, "close")

    def download(self, name, destination):
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".partial")
        offset = temp.stat().st_size if temp.exists() else 0
        expected = None
        with temp.open("ab") as stream:
            while True:
                value = self.request(
                    "blob", path=name, offset=offset, chunk_bytes=4 * 1024 * 1024
                )
                if value["offset"] != offset:
                    raise ValueError("Artifact range mismatch")
                if expected is not None and value["sha256"] != expected:
                    raise ValueError("Artifact changed during transfer")
                expected = value["sha256"]
                raw = base64.b64decode(value["data"], validate=True)
                stream.write(raw)
                offset += len(raw)
                if offset == value["total_bytes"]:
                    break
                if not raw:
                    raise ValueError("Incomplete artifact range")
            stream.flush()
            os.fsync(stream.fileno())
        from agp_rl.io import sha256

        if sha256(temp) != expected:
            raise ValueError("Artifact transfer SHA mismatch")
        temp.replace(path)
        return path
