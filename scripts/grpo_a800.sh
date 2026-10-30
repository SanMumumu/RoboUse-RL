#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo_root"
exec "${PYTHON:-python}" -m robouse.cli grpo_a800 "$@"
