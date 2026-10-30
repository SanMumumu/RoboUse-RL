import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
NAMES = [
    "collect_astra",
    "collect_guardian",
    "teacher_service",
    "collect_remote",
    "observe_teacher",
    "prepare_data",
    "export_data",
    "import_data",
    "verify_data",
    "sft_a800",
    "eval",
    "grpo_a800",
    "env_worker",
    "view_results",
    "status",
    "stop",
]


@pytest.mark.parametrize("name", NAMES)
def test_script_help_dry_run(name):
    env = {**os.environ, "PYTHONPATH": ""}
    for args in (["--help"], ["--dry-run"]):
        result = subprocess.run(
            ["bash", str(ROOT / "scripts" / f"{name}.sh"), *args],
            env=env,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
        if args == ["--dry-run"]:
            assert json.loads(result.stdout)["gpu_started"] is False


def test_test_requires_explicit_final_flag():
    result = subprocess.run(
        [sys.executable, "-m", "robouse.cli", "eval", "--split", "test", "--dry-run"],
        capture_output=True,
    )
    assert result.returncode != 0


def test_json_config_preserves_numeric_type():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "robouse.cli",
            "sft_a800",
            "--config",
            str(ROOT / "configs/sft4b.json"),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert isinstance(json.loads(result.stdout)["config"]["learning_rate"], float)


def test_equal_sign_cli_options_override_config_and_test_guard():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "robouse.cli",
            "eval",
            "--config",
            str(ROOT / "configs/eval.json"),
            "--split=test",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "--final-test" in result.stderr


@pytest.mark.parametrize("override", [[], ["--effort", "medium"], ["--effort=max"]])
def test_teacher_effort_defaults_and_overrides(override):
    command = [sys.executable, "-m", "robouse.cli", "collect_astra", "--dry-run"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["config"]["effort"] == "high"
    result = subprocess.run(
        command + ["--config", str(ROOT / "configs/astra_teacher.json")] + override,
        capture_output=True, text=True, check=True,
    )
    config = json.loads(result.stdout)["config"]
    expected = "high" if not override else "medium" if len(override) == 2 else "max"
    assert config["effort"] == expected
    assert config["model"] == "gpt-6-astra"
