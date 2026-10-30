"""RTX-local MCP transport and sandboxed JS; model stays on A800 via RPC mailbox."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from robouse.data.datasets import write_json
from agp_rl.adapters.tool_codec import ToolCodec
from agp_rl.adapters.observation_codec import decode_mcp
from agp_rl.pipeline2b.dataset import student_system_prompt
from agp_rl.pipeline2b.student_output import parse_student_output
from robouse.tools.base import Tool
from robouse.harness.tool_registry import ToolRegistry
from robouse.harness.executor import HarnessExecutor


async def run(session, job_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from agp_rl.pipeline2b.code import CodeSession
    from agp_core import features
    from robouse.environments.robodojo.native_job import source_root

    job = json.loads(job_path.read_text())
    folder = job_path.parent
    runtime = source_root() / "agp_runtime"
    env = {
        k: v
        for k, v in os.environ.items()
        if k.startswith("OURS_")
        or k in ("PATH", "HOME", "LANG", "AGP_HOME", "ROBODOJO_ROOT")
    }
    env["PYTHONPATH"] = str(runtime) + os.pathsep + str(runtime / "source/agent")
    params = StdioServerParameters(
        command=job["transport_python"],
        args=["-B", "-m", "agp_core.server", "--session", str(session)],
        env=env,
        cwd=str(runtime),
    )
    with (folder / "mcp.log").open("a") as log:
        async with stdio_client(params, errlog=log) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools = (await client.list_tools()).tools
                codec = ToolCodec(
                    {
                        t.name: t.inputSchema
                        for t in tools
                        if t.name
                        in __import__(
                            "agp_rl.adapters.tool_codec", fromlist=["TOOLS"]
                        ).TOOLS
                    }
                )
                codec.verify_live(tools)
                def dispatch(name):
                    async def invoke(arguments):
                        return await client.call_tool(name, arguments)
                    return invoke

                registry = ToolRegistry([Tool(name, schema, dispatch(name)) for name, schema in codec.schemas.items()], official_robot=True)
                executor = HarnessExecutor(registry)

                class Environment:
                    def __init__(self):
                        self.codec = codec
                        self.blocks = []
                        self.assessed = False
                        self.calls = 0
                        self.tool_seconds = 0.0

                    async def call(self, call):
                        codec.parse(codec.encode(call["name"], call["arguments"]))
                        started = time.monotonic()
                        receipt = await executor.execute(f"native-{self.calls}", codec.encode(call["name"], call["arguments"]))
                        self.calls += 1
                        self.tool_seconds += time.monotonic() - started
                        if "result" not in receipt:
                            raise RuntimeError("Native tool transport failed: " + receipt["error"])
                        result = receipt["result"]
                        reply = decode_mcp(result, folder / "images")
                        images = iter(reply.images)
                        self.blocks = [
                            {"type": "text", "text": b.text}
                            if b.type == "text"
                            else {"type": "image", "image": next(images)}
                            for b in result.content
                        ]
                        if call["name"] == "assess":
                            try:
                                self.assessed = (
                                    json.loads(result.content[0].text).get("ok") is True
                                )
                            except (ValueError, IndexError):
                                pass
                        return reply, bool(result.isError)

                native = Environment()
                code_session = CodeSession(native)
                goal, _ = await native.call({"name": "goal", "arguments": {}})
                await native.call({"name": "observe", "arguments": {}})
                flags = features.current(features.system_switches())
                prompt = features.render(
                    (runtime / "agp_core/PROMPT.md").read_text(), flags
                )
                from agp_core import task_notes

                if flags["task_notes"]:
                    prompt += task_notes.launch_section(session)[0]
                if flags["task_hints"]:
                    prompt += task_notes.launch_hint_section(session)[0]
                from robouse.environments.robodojo.native_prompt import initial_message

                blocks = initial_message(prompt, goal, None, native.blocks)["content"]
                system = student_system_prompt(codec.schemas)
                write_json(
                    folder / "observation-0.json",
                    {"blocks": blocks, "system": system, "done": False},
                )
                index = 0
                reason = "decision_limit"
                while index < job.get("max_decisions", 512):
                    action = folder / f"action-{index}.json"
                    while not action.exists():
                        if (folder / "CANCEL").exists():
                            raise asyncio.CancelledError()
                        await asyncio.sleep(0.05)
                    message = json.loads(action.read_text())
                    before_calls, before_seconds = native.calls, native.tool_seconds
                    call, error = parse_student_output(message["text"], codec)
                    result_blocks = []
                    if error:
                        result_blocks = [
                            {
                                "type": "text",
                                "text": json.dumps({"ok": False, "error": error}),
                            }
                        ]
                    elif call and call["name"] == "exec":
                        result = await code_session.execute(call["arguments"]["code"])
                        raw = result["output"]
                        if result.get("error"):
                            raw.append(
                                {
                                    "type": "text",
                                    "text": "JavaScript error: " + result["error"],
                                }
                            )
                        reply = decode_mcp(
                            SimpleNamespace(
                                content=[SimpleNamespace(**b) for b in raw]
                            ),
                            folder / "images",
                        )
                        paths = iter(reply.images)
                        result_blocks = [
                            {"type": "text", "text": b["text"]}
                            if b["type"] == "text"
                            else {"type": "image", "image": next(paths)}
                            for b in raw
                        ]
                    elif call:
                        await native.call(call)
                        result_blocks = native.blocks
                    index += 1
                    done = (
                        native.assessed
                        or index >= job.get("max_decisions", 512)
                        or message.get("finish", False)
                    )
                    write_json(
                        folder / f"observation-{index}.json",
                        {"blocks": result_blocks, "done": done, "native_tool_calls": native.calls - before_calls, "native_tool_seconds": native.tool_seconds - before_seconds},
                    )
                    if done:
                        reason = (
                            "accepted_native_assess"
                            if native.assessed
                            else "decision_limit"
                        )
                        break
                write_json(
                    session / "policy_finished.json",
                    {
                        "returncode": 0,
                        "stopped_by": reason,
                        "actor": "local_student_remote_inference",
                    },
                )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--job", type=Path, required=True)
    a = p.parse_args()
    asyncio.run(run(a.session, a.job))


if __name__ == "__main__":
    main()
