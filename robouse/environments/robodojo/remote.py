"""Authenticated loopback server; native simulator stack stays in RTX processes."""
import hmac
import json
import os
from pathlib import Path
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_server(config, adapter=None):
    secret = os.environ.get("ROBOUSE_ENV_TOKEN")
    if not secret or len(secret) < 24:
        raise ValueError("ROBOUSE_ENV_TOKEN must have at least 24 random characters")
    if adapter is None:
        from robouse.environments.robodojo.worker import NativeWorker

        adapter = NativeWorker(config)
    logs = Path(config["output"])
    logs.mkdir(parents=True, exist_ok=True)

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            with (logs / "http-errors.log").open("a") as f:
                traceback.print_exc(file=f)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            with (logs / "http.log").open("a") as f:
                f.write(format % args + "\n")

        def do_POST(self):
            if not hmac.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + secret
            ):
                self.send_error(401)
                return
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 64 * 1024 * 1024:
                self.send_error(413)
                return
            try:
                payload = json.dumps(
                    adapter.handle(json.loads(self.rfile.read(size)))
                ).encode()
            except Exception as exc:
                if hasattr(adapter, "public_error"):
                    with (logs / "http-errors.log").open("a") as stream:
                        traceback.print_exc(file=stream)
                    status, error = adapter.public_error(exc)
                    payload = json.dumps(error).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_error(500)
                raise
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Server(("127.0.0.1", config.get("port", 8765)), Handler)


def serve(config):
    server = make_server(config)
    try:
        server.serve_forever()
    finally:
        server.server_close()
