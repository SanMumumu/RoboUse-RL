"""Original native Astra/Codex episode loop with explicit model/effort admission."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from robouse.data.datasets import write_json
from robouse.environments.robodojo.native_job import source_root, verify_sources
from robouse.teacher.admission import FrontierBlocked
from robouse.teacher.provider import IsolatedCodex, isolated_environment
from robouse.teacher.loopback import direct_loopback_environment


def toml(value):
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(json.dumps(k) + " = " + toml(v) for k, v in value.items())
            + "}"
        )
    if isinstance(value, list):
        return "[" + ", ".join(toml(v) for v in value) + "]"
    return json.dumps(value)


def settings(config, session, binary, port, flags):
    values = {
        f"features.{key}": False
        for key in (
            "shell_tool",
            "multi_agent",
            "apps",
            "browser_use",
            "computer_use",
            "skill_search",
        )
    }
    values.update(
        web_search="disabled",
        project_doc_max_bytes=0,
        model_reasoning_effort=config["effort"],
        service_tier=config["service_tier"],
        default_permissions="agp_episode",
        **{
            "permissions.agp_episode.filesystem": {
                ":minimal": "read",
                str(session): "read",
                str(binary.parent.parent): "read",
            }
        },
    )
    mode = os.environ.get("OURS_TOOL_MODE", "hybrid")
    if mode not in ("hybrid", "code"):
        raise FrontierBlocked("NATIVE_TOOL_MODE_INVALID")
    servers = (
        {"robot": ["code_mode"], "robot_js": ["direct"]}
        if mode == "hybrid"
        else {"robot_js": ["direct"]}
    )
    for name, omit in servers.items():
        values.update(
            {
                f"mcp_servers.{name}.url": f"http://127.0.0.1:{port}/mcp",
                f"mcp_servers.{name}.omit_tools_from": omit,
                f"mcp_servers.{name}.tool_timeout_sec": 900,
                f"mcp_servers.{name}.required": True,
            }
        )
        for tool in (
            "goal",
            "observe",
            "measure",
            "step",
            "run",
            "action_status",
            "assess",
        ):
            if tool != "run" or flags["run"]:
                values[f"mcp_servers.{name}.tools.{tool}.approval_mode"] = "approve"
    return values


def exec_args(config, session, binary, port, flags, images, skill_setting):
    values = settings(config, session, binary, port, flags)
    args = [
        "exec",
        "--skip-git-repo-check",
        "--cd",
        str(session),
        "--json",
        "-m",
        config["model"],
    ]
    for key, value in values.items():
        args += ["-c", key + "=" + toml(value)]
    if not skill_setting.startswith("skills.config=") or "\n" in skill_setting:
        raise FrontierBlocked("INVALID_NATIVE_SKILL_DISABLE_SETTING")
    args += ["-c", skill_setting]
    for image in images:
        args += ["-i", str(image)]
    return args + ["-o", str(session / "agent_last_message.txt"), "-"]


def official_wait(runtime):
    wanted = {"signal_of", "signal_entry", "fault_signal", "wait_official"}
    nodes = [
        n
        for n in ast.parse((runtime / "agp_core/run_agent.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name in wanted
    ]
    if {n.name for n in nodes} != wanted:
        raise FrontierBlocked("NATIVE_WAIT_SOURCE_CHANGED")
    watch = {"__name__": "robouse_native_assess_watch"}
    exec(
        compile(
            (runtime / "agp_core/assess_watch.py").read_text(),
            "native-assess-watch",
            "exec",
        ),
        watch,
    )
    namespace = {
        "signal": signal,
        "subprocess": subprocess,
        "time": time,
        "accepted_assessment": watch["accepted_assessment"],
        "stop_process": watch["stop_process"],
        "ENDED_ON_ASSESS": {},
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "native-wait", "exec"),
        namespace,
    )
    return namespace["wait_official"]


def run(job_path, session):
    from agp_rl.io import sha256

    job = json.loads(job_path.read_text())
    config = job["teacher"]
    verify_sources()
    if job.get("actor_kind") != "astra_teacher" or not job.get("teacher_reservation"):
        raise FrontierBlocked("EXPLICIT_TEACHER_RESERVATION_REQUIRED")
    reservation = Path(job["teacher_reservation"]["path"])
    if sha256(reservation) != job["teacher_reservation"]["sha256"]:
        raise FrontierBlocked("TEACHER_RESERVATION_CHANGED")
    record = json.loads(reservation.read_text())
    if (
        record["episode_id"] != job["episode_id"]
        or record["identity"] != job["identity"]
    ):
        raise FrontierBlocked("RESERVATION_IDENTITY_MISMATCH")
    if time.time() > record["deadline"]:
        raise FrontierBlocked("TEACHER_RUN_BUDGET_EXPIRED")
    protocol_timeout = float(os.environ["OURS_POLICY_TIMEOUT_S"])
    if not math.isfinite(protocol_timeout) or not 0 < protocol_timeout <= 14400:
        raise FrontierBlocked("NATIVE_PROTOCOL_TIMEOUT_REQUIRED")
    runner = IsolatedCodex(
        config,
        job_path.parent / "private-teacher",
        minimum_lifetime=config["episode_wall_seconds"] + 90,
    )
    runtime = source_root() / "agp_runtime"
    sys.path[:0] = [str(runtime), str(runtime / "source/agent")]
    from agp_core import features, task_notes
    from agp_core.preload import preload

    flags = features.current()
    prompt = features.render((runtime / "agp_core/PROMPT.md").read_text(), flags)
    pins = json.loads((session / "frozen_policy.json").read_text())
    if hashlib.sha256(prompt.encode()).hexdigest() != pins["prompt_rendered_sha256"]:
        raise FrontierBlocked("NATIVE_PROMPT_PIN_MISMATCH")
    if flags["task_notes"]:
        prompt += task_notes.launch_section(session)[0]
    if flags["task_hints"]:
        prompt += task_notes.launch_hint_section(session)[0]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {
        k: v
        for k, v in os.environ.items()
        if k.startswith("OURS_")
        or k in ("PATH", "HOME", "LANG", "AGP_HOME", "ROBODOJO_ROOT")
    }
    env.update(
        PYTHONPATH=str(runtime) + os.pathsep + str(runtime / "source/agent"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    server = process = journal = timer = None
    stops = {}
    returncode = None
    started = time.monotonic()
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(number, frame):
        stops["external_signal"] = {
            "process": "teacher_launcher",
            "signal": number,
            "returncode": 128 + number,
        }
        raise SystemExit(128 + number)

    signal.signal(signal.SIGTERM, interrupted)
    parent_watch_stop = threading.Event()
    owner = json.loads((job_path.parent / "owner.json").read_text())

    def watch_parent():
        while not parent_watch_stop.wait(0.5):
            try:
                fields = (
                    Path(f"/proc/{owner['pid']}/stat")
                    .read_text()
                    .rsplit(")", 1)[1]
                    .split()
                )
                alive = fields[19] == owner["start_ticks"] and fields[0] != "Z"
            except FileNotFoundError:
                alive = False
            if not alive:
                write_json(
                    session / "teacher-infrastructure.json",
                    {"reason": "NATIVE_PARENT_EXITED"},
                )
                os.kill(os.getpid(), signal.SIGTERM)
                return

    threading.Thread(target=watch_parent, daemon=True).start()
    try:
        if config.get("local_control_journal", True):
            from robouse.teacher.journal import LocalControlJournal

            journal = LocalControlJournal(
                session, runtime_directory=config.get("private_runtime_dir")
            )
        with (session / "mcp_server.log").open("x") as log:
            server = subprocess.Popen(
                [
                    job["transport_python"],
                    "-B",
                    "-m",
                    "agp_core.server",
                    "--session",
                    str(session),
                    "--http",
                    str(port),
                ],
                env=env,
                cwd=runtime,
                stdout=log,
                stderr=log,
            )
        deadline = time.monotonic() + 120
        while True:
            if server.poll() is not None:
                raise FrontierBlocked("NATIVE_MCP_START_FAILURE")
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise FrontierBlocked("NATIVE_MCP_START_TIMEOUT")
                time.sleep(0.1)
        images = []
        if flags["preload"]:
            with direct_loopback_environment(f"http://127.0.0.1:{port}/mcp"):
                message, images = preload(f"http://127.0.0.1:{port}/mcp", session)
            prompt += message
        skill = subprocess.check_output(
            [
                job["transport_python"],
                str(runtime / "source/agent/agp/tools/codex_experiment_config.py"),
            ],
            env=env,
            text=True,
        ).strip()
        args = exec_args(config, session, runner.binary, port, flags, images, skill)
        proof = runner.admit(max_wall_seconds=config["episode_wall_seconds"])
        proof.update(budget_checked=True, reservation_sha256=sha256(reservation))
        write_json(session / "teacher-admission.json", proof)
        write_json(
            session / "teacher-request.json",
            {
                "model": config["model"],
                "effort": config["effort"],
                "service_tier": config["service_tier"],
                "args": args,
                "quota_checked": True,
                "automatic_account_switch": False,
            },
        )
        if time.time() >= record["deadline"]:
            raise FrontierBlocked(
                "TEACHER_DEADLINE_EXPIRED_AFTER_ADMISSION_NO_MODEL_START"
            )
        # Persist intent before Popen; interruption never looks like free replay.
        write_json(
            session / "model-process-start-intent.json",
            {
                "model": config["model"],
                "effort": config["effort"],
                "episode_id": job["episode_id"],
                "created_at": time.time(),
            },
        )
        with (session / "agent_events.jsonl").open("x") as out, (
            session / "agent_stderr.log"
        ).open("x") as err:
            process = subprocess.Popen(
                runner.command(
                    args,
                    session=session,
                    private_roots=[job_path.parent / "evaluator_private"],
                ),
                env=isolated_environment(runner.profile),
                stdin=subprocess.PIPE,
                stdout=out,
                stderr=err,
                text=True,
            )
            process.stdin.write(prompt)
            process.stdin.close()
            (session / "agent.pid").write_text(str(process.pid))
            cap = min(config["episode_wall_seconds"], record["deadline"] - time.time())

            def expire():
                if process.poll() is None:
                    stops["resource_watchdog"] = True
                    os.kill(os.getpid(), signal.SIGTERM)

            timer = threading.Timer(cap, expire)
            timer.daemon = True
            timer.start()
            returncode = official_wait(runtime)(
                process,
                server,
                session,
                protocol_timeout,
                flags["end_on_assess"],
                stops,
            )
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        parent_watch_stop.set()
        if timer:
            timer.cancel()
        for child in (process, server):
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        if journal:
            try:
                journal.finish()
            except Exception as exc:
                write_json(
                    session / "teacher-infrastructure.json",
                    {
                        "reason": "LOCAL_JOURNAL_ARCHIVE_FAILED",
                        "error_type": type(exc).__name__,
                    },
                )
        try:
            runner.verify_identity()
            unchanged = True
        except FrontierBlocked:
            unchanged = False
        write_json(
            session / "policy_finished.json",
            {
                "returncode": returncode,
                "stopped_by": stops.get("stopped_by"),
                "external_signal": stops.get("external_signal"),
                "resource_watchdog": stops.get("resource_watchdog"),
                "mcp_server": stops.get("mcp_server"),
                "inference_process_started": process is not None,
                "account_identity_verified": unchanged,
                "wall_seconds": time.monotonic() - started,
                "actor": "astra_teacher",
            },
        )
        signal.signal(signal.SIGTERM, previous)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", required=True, type=Path)
    p.add_argument("--session", required=True, type=Path)
    a = p.parse_args()
    try:
        run(a.job.resolve(), a.session.resolve())
    except BaseException as exc:
        write_json(
            a.session / "teacher-infrastructure.json",
            {
                "reason": str(exc)
                if isinstance(exc, FrontierBlocked)
                else type(exc).__name__,
                "teacher_calls_not_retried": True,
            },
        )
        raise


if __name__ == "__main__":
    main()
