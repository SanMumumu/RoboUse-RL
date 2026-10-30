"""Keep unchanged native SQLite WAL on local RAM, then durably archive it.

Only the ControlRuntime storage location changes. RobotAdapter reservations,
native requests/results, model transcripts and RGB remain on durable storage.
No action, reset, model call, control or SQLite journal mode is modified.
"""
from __future__ import annotations

import os
from contextlib import closing
from pathlib import Path
import shutil
import sqlite3
import stat
import tempfile
import threading
import uuid

from robouse.teacher.admission import FrontierBlocked
from agp_rl.io import fsync_dir, sha256, utcnow, write_json


class LocalControlJournal:
    def __init__(self, session, *, runtime_directory=None):
        self.session = Path(session).resolve(strict=True)
        selected = runtime_directory or os.environ.get("XDG_RUNTIME_DIR")
        if not selected:
            raise FrontierBlocked("EXPLICIT_PRIVATE_RUNTIME_DIRECTORY_REQUIRED")
        runtime = Path(selected).resolve(strict=True)
        metadata = runtime.stat()
        if (
            runtime.parent != Path("/run/user")
            or not runtime.name.isdigit()
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != os.getuid()
        ):
            raise FrontierBlocked("PRIVATE_RUNTIME_DIRECTORY_PERMISSIONS_INVALID")
        mounts = []
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            mount = Path(left.split()[4])
            if runtime == mount or mount in runtime.parents:
                mounts.append((len(mount.parts), right.split()[0]))
        if not mounts or max(mounts)[1] != "tmpfs":
            raise FrontierBlocked("LOCAL_CONTROL_JOURNAL_REQUIRES_PRIVATE_TMPFS")
        self.target = self.session / "control_runtime"
        if self.target.exists() or self.target.is_symlink():
            raise FrontierBlocked("FRESH_LOCAL_CONTROL_JOURNAL_REQUIRED")
        self.local = Path(tempfile.mkdtemp(prefix="agp-control-", dir=runtime))
        self.target.symlink_to(self.local, target_is_directory=True)
        self.recovery = self.session / "control_runtime.recovery"
        self.recovery.mkdir()
        self.stopped = threading.Event()
        self.snapshot_count = 0
        self.last_snapshot_error = None
        self.thread = threading.Thread(
            target=self._snapshots, name="control-journal-backup", daemon=True
        )
        write_json(
            self.session / "local_control_journal.json",
            {
                "created_utc": utcnow(),
                "active_storage": "private tmpfs",
                "sqlite_mode_unchanged": "WAL/FULL",
                "native_source_modified": False,
                "snapshot_interval_seconds": 5,
                "hard_kill_recovery_limit": "Most recent consistent SQLite backup; raw durable robot transport remains separate.",
                "original_SIGBUS_cause": "Unknown; unsupported WAL-on-NFS exposure removed prospectively.",
            },
        )
        self.thread.start()

    def _snapshot(self):
        source = self.local / "journal.sqlite3"
        if not source.is_file():
            return
        pending = self.recovery / (".pending-" + uuid.uuid4().hex + ".sqlite3")
        try:
            with closing(
                sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=1)
            ) as reader:
                with closing(sqlite3.connect(pending, timeout=1)) as writer:
                    reader.backup(writer, pages=256, sleep=0.01)
                    writer.commit()
            with pending.open("rb") as stream:
                os.fsync(stream.fileno())
            final = self.recovery / "journal.sqlite3"
            os.replace(pending, final)
            fsync_dir(self.recovery)
            self.snapshot_count += 1
            write_json(
                self.recovery / "snapshot.json",
                {
                    "created_utc": utcnow(),
                    "snapshot": self.snapshot_count,
                    "database_sha256": sha256(final),
                    "consistent_sqlite_backup": True,
                    "contains_credentials": False,
                },
                replace=True,
            )
        finally:
            pending.unlink(missing_ok=True)

    def _snapshots(self):
        while not self.stopped.wait(5):
            try:
                self._snapshot()
            except Exception as exc:
                self.last_snapshot_error = type(exc).__name__

    def finish(self):
        """Call only after the native MCP process has stopped writing."""
        self.stopped.set()
        self.thread.join(timeout=10)
        pending = self.session / (".control-runtime-archive-" + uuid.uuid4().hex)
        shutil.copytree(self.local, pending)
        hashes = {}
        for path in sorted(pending.rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
                hashes[str(path.relative_to(pending))] = sha256(path)
        fsync_dir(pending)
        if not self.target.is_symlink() or self.target.resolve() != self.local:
            raise FrontierBlocked("LOCAL_CONTROL_JOURNAL_TARGET_CHANGED")
        self.target.unlink()
        os.replace(pending, self.target)
        fsync_dir(self.session)
        value = {
            "created_utc": utcnow(),
            "status": "ARCHIVED",
            "files": hashes,
            "snapshot_count": self.snapshot_count,
            "last_snapshot_error": self.last_snapshot_error,
            "source_storage": "private tmpfs",
            "archive_storage": "project run directory",
            "native_source_modified": False,
            "contains_credentials": False,
        }
        write_json(self.session / "local_control_journal_finished.json", value)
        shutil.rmtree(self.local)
        return value
