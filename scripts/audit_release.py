"""Scan only Git-indexed release files; never read user credential stores."""
import argparse
import ipaddress
import json
from pathlib import Path
import re
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        .decode()
        .split("\0")
    )
    patterns = {
        "private_key": r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
        "token": r"(?:gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{24,}|AKIA[A-Z0-9]{16})",
        "machine_path": r"/mnt/(?:hwdata|data)/|/home/[A-Za-z_][A-Za-z0-9_-]+/",
        "internal_host": r"ai" + r"node[0-9]+|(?:pypi\.)?hub\.ace-research\.openai\.org",
    }
    issues = []
    for name in filter(None, names):
        path = root / name
        if path.stat().st_size > 2_000_000:
            issues.append({"file": name, "kind": "large_file"})
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            issues.append({"file": name, "kind": "binary"})
            continue
        for match in re.finditer(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])", text):
            # A four-part pinned package version is not a network address.
            if text[:match.start()].rstrip().endswith("=="):
                continue
            address = match.group()
            try:
                parsed = ipaddress.ip_address(address)
            except ValueError:
                continue
            if parsed.is_private and not parsed.is_loopback and not parsed.is_unspecified:
                issues.append({"file": name, "kind": "private_ip"})
        for kind, pattern in patterns.items():
            if re.search(pattern, text):
                issues.append({"file": name, "kind": kind})
    value = {
        "tracked_files": len(list(filter(None, names))),
        "issues": issues,
        "scope": "pattern scan; no exhaustive secret-detection guarantee",
    }
    if a.output:
        Path(a.output).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value))
    raise SystemExit(bool(issues))


if __name__ == "__main__":
    main()
