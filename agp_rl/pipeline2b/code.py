"""Native code-mode RPC with an empty filesystem and network namespace.

JavaScript's VM is a convenience, never the isolation boundary. bwrap exposes
only the trusted runner, the Node executable and its required shared libraries.
Robot calls are validated and executed by the existing parent MCP capability.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

Blocked = RuntimeError

RUNNER = Path(__file__).with_name("code_runner.js")


def sandbox_command(*args):
    node = Path(shutil.which("node") or "/missing-node").resolve(strict=True)
    result = subprocess.run(
        ["ldd", str(node)], capture_output=True, text=True, check=True, timeout=10
    )
    libraries = sorted(set(re.findall(r"(?:=>\s*)?(/[\w./+\-]+)", result.stdout)))
    command = [
        shutil.which("bwrap") or "/usr/bin/bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--tmpfs",
        "/",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/work",
        "--ro-bind",
        str(node),
        str(node),
    ]
    for path in libraries:
        if Path(path).is_file():
            command += ["--ro-bind", path, path]
    command += [
        "--ro-bind",
        str(RUNNER),
        "/runner.js",
        "--chdir",
        "/work",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        str(node),
        "--max-old-space-size=128",
        "/runner.js",
        *args,
    ]
    return command


def negative_access_test(paths):
    command = sandbox_command("--probe", json.dumps(list(map(str, paths))))
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=20,
        env={"PATH": "/usr/bin:/bin"},
    )
    if result.returncode:
        return {
            "status": "BLOCKED",
            "reason": "CODE_SANDBOX_UNAVAILABLE",
            "stderr": result.stderr[:500],
        }
    report = json.loads(result.stdout)
    return {
        "status": "PASSED" if all(report["denied"]) else "BLOCKED",
        "denied_count": sum(report["denied"]),
        "tested_count": len(paths),
        "namespace": "empty-root,pid,network,user",
        "secret_values_read": False,
    }


class CodeSession:
    def __init__(self, environment, *, timeout_seconds=120):
        self.environment = environment
        self.state = {}
        self.timeout_seconds = timeout_seconds

    async def execute(self, code):
        process = await asyncio.create_subprocess_exec(
            *sandbox_command(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            limit=64 * 1024 * 1024,
        )

        async def send(message):
            process.stdin.write((json.dumps(message) + "\n").encode())
            await process.stdin.drain()

        async def handle():
            await send(
                {
                    "kind": "execute",
                    "code": code,
                    "state": self.state,
                    "tools": list(self.environment.codec.schemas),
                    "timeout_ms": min(10000, int(self.timeout_seconds * 1000)),
                }
            )
            while True:
                line = await process.stdout.readline()
                if not line:
                    raise Blocked("CODE_SANDBOX_EXITED_WITHOUT_RESULT")
                message = json.loads(line)
                if message["kind"] == "complete":
                    self.state = message["state"]
                    return message
                if message["kind"] != "tool":
                    raise Blocked("INVALID_CODE_RPC")
                # Validation repeats the normal direct-tool contract.
                try:
                    self.environment.codec.encode(message["name"], message["arguments"])
                    reply, error = await self.environment.call(
                        {"name": message["name"], "arguments": message["arguments"]}
                    )
                    import base64

                    blocks = []
                    native_blocks = getattr(self.environment, "blocks", None)
                    if native_blocks is None:
                        raise Blocked("LOSSLESS_NATIVE_CODE_REPLY_BLOCKS_REQUIRED")
                    for block in native_blocks:
                        if block["type"] == "text":
                            blocks.append({"type": "text", "text": block["text"]})
                        else:
                            path = block["image"]
                            blocks.append(
                                {
                                    "type": "image",
                                    "mimeType": "image/png"
                                    if path.endswith(".png")
                                    else "image/jpeg",
                                    "data": base64.b64encode(
                                        Path(path).read_bytes()
                                    ).decode(),
                                }
                            )
                    await send(
                        {
                            "kind": "reply",
                            "id": message["id"],
                            "value": {"content": blocks, "isError": error},
                        }
                    )
                except (ValueError, KeyError) as exc:
                    await send(
                        {
                            "kind": "reply",
                            "id": message["id"],
                            "error": type(exc).__name__,
                        }
                    )

        try:
            return await asyncio.wait_for(handle(), timeout=self.timeout_seconds)
        finally:
            if process.returncode is None:
                process.terminate()
                await process.wait()
