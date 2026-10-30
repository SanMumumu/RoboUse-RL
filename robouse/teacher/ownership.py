"""Host-local PID plus start-tick receipts; a PID alone never proves ownership."""
import hashlib
import os
from pathlib import Path
import socket


def receipt(pid=None):
    pid = os.getpid() if pid is None else pid
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start_ticks": fields[19], "host_sha256": hashlib.sha256(socket.gethostname().encode()).hexdigest()}


def alive(owner):
    try:
        current = receipt(owner["pid"])
        fields = Path(f"/proc/{owner['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        return current == owner and fields[0] != "Z"
    except (OSError, KeyError, TypeError, ValueError):
        return False
