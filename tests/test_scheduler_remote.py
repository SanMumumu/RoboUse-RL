import asyncio
import hashlib
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request
import pytest
import torch
from robouse.scheduler import episode
from robouse.environments.robodojo.remote import make_server
from robouse.environments.robodojo.client import RemoteEnvironment
from robouse.environments.robodojo.worker import close_state


class Engine:
    policy = "v1"

    def __init__(self):
        self.calls = 0

    async def infer(self, *args):
        self.calls += 1
        return {
            "inputs": {
                "input_ids": torch.tensor([[1, 2]]),
                "pixel_values": torch.ones(1, 3),
            },
            "tokens": torch.tensor([3]),
            "behavior_logprobs": torch.tensor([-0.7]),
            "mask": torch.tensor([True]),
            "temperature": 1.0,
            "forbidden": [],
            "text": '{"name":"observe","arguments":{}}',
            "policy_version": self.policy,
            "input_hash": "hash",
        }


class Env:
    episode_id = "episode"

    def __init__(self, root):
        self.root = root
        self.steps = 0
        self.fail = True

    async def reset(self, *args):
        return {"system": "official", "blocks": [], "initial_state_hash": "initial"}

    def observation(self, x):
        return x.get("blocks", []), {}

    async def step(self, index, text, finish=False):
        self.steps += 1
        if self.fail:
            self.fail = False
            raise ConnectionError("after persisted sampler evidence")
        return {
            "done": True,
            "blocks": [],
            "reward": 1,
            "outcome": "completed",
            "score": 1,
            "termination_reason": "native_success",
        }

    async def close(self):
        pass


def test_pending_sample_resume_never_resamples(tmp_path):
    engine = Engine()
    env = Env(tmp_path)
    with pytest.raises(ConnectionError):
        asyncio.run(
            episode(engine, env, {"task": "t", "seed": 0, "layout": 1}, "g", tmp_path)
        )
    result = asyncio.run(
        episode(engine, env, {"task": "t", "seed": 0, "layout": 1}, "g", tmp_path)
    )
    assert engine.calls == 1 and env.steps == 2
    assert result["decisions"][0]["behavior_logprobs"] == pytest.approx([-0.7])
    assert result["policy_version"] == "v1" and result["seed"] == 0


def test_http_auth_loopback_and_idempotent_wire(tmp_path, monkeypatch):
    token = "test-only-token-" + "a" * 32
    monkeypatch.setenv("ROBOUSE_ENV_TOKEN", token)

    class Adapter:
        def handle(self, r):
            return {"op": r["op"], "episode_id": r["episode_id"], "reward": None}

    server = make_server({"output": str(tmp_path), "port": 0}, Adapter())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        client = RemoteEnvironment(url, "ep", tmp_path)
        assert client.request("result")["reward"] is None
        request = urllib.request.Request(url, data=b"{}")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request)
        assert exc.value.code == 401
        assert token not in (tmp_path / "http.log").read_text()
        with pytest.raises(ValueError):
            RemoteEnvironment("http://example.com", "ep", tmp_path)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_initial_state_tolerance_and_no_alias_substitute():
    assert close_state({"physical": [1.0, 2.0]}, {"physical": [1.0 + 1e-7, 2.0]})
    assert not close_state({"physical": [1.0]}, {"physical": [1.001]})


def test_shared_context_keeps_targets_rgb_and_causal_feedback():
    from robouse.harness.context import build_context

    messages = [
        {"role": "user", "content": [{"type": "text", "text": "immutable task"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "old tool"}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Wall time: 0.5 seconds\nOutput:\nreal feedback",
                },
                {"type": "image", "image": "art-image://actual"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "unchanged target"}],
        },
    ]
    value = build_context(messages, with_target=True)
    assert value[-1] == messages[-1] and value[0] == messages[0]
    assert value[2]["content"][0]["text"] == "real feedback"
    assert value[2]["content"][1] == messages[2]["content"][1]
    assert messages[2]["content"][0]["text"].startswith("Wall time:")


def test_resumable_video_transfer_is_chunked_and_sha_verified(tmp_path, monkeypatch):
    import base64

    monkeypatch.setenv("ROBOUSE_ENV_TOKEN", "fixture-token-" + "a" * 32)
    client = RemoteEnvironment("http://127.0.0.1:8765", "episode", tmp_path)
    raw = b"synthetic archived video bytes"
    h = hashlib.sha256(raw).hexdigest()
    offsets = []

    def request(op, **values):
        assert op == "blob" and values["chunk_bytes"] <= 4 * 1024 * 1024
        offset = values["offset"]
        offsets.append(offset)
        return {
            "data": base64.b64encode(raw[offset : offset + 5]).decode(),
            "sha256": h,
            "total_bytes": len(raw),
            "offset": offset,
        }

    client.request = request
    target = tmp_path / "video.mp4"
    target.with_suffix(".mp4.partial").write_bytes(raw[:3])
    client.download("original.mp4", target)
    assert offsets[0] == 3 and target.read_bytes() == raw
