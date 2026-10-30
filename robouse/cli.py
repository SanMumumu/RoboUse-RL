import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from robouse.data.datasets import write_json


def main():
    p = argparse.ArgumentParser(
        description="Portable RoboUse entry points. Execution is opt-in."
    )
    p.add_argument(
        "action",
        choices=[
            "collect_astra",
            "train_sft",
            "train_rl",
            "evaluate",
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
        ],
    )
    p.add_argument("--config", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--effort", help="Teacher reasoning effort; overrides config (default: high)")
    p.add_argument("--revision")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--num-gpus", type=int)
    p.add_argument("--allow-experimental", action="store_true")
    p.add_argument("--finetuning-mode")
    p.add_argument("--output", "--output-dir", dest="output")
    p.add_argument("--checkpoint", "--init-checkpoint", dest="checkpoint")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--snapshot-only", action="store_true")
    p.add_argument("--snapshot-output")
    p.add_argument("--seed", type=int)
    p.add_argument("--max-episodes", type=int)
    p.add_argument("--max-wall-seconds", type=int)
    p.add_argument("--episode-wall-seconds", type=int)
    for name in (
        "train-data",
        "val-data",
        "test-data",
        "resume",
        "manifest",
        "files",
        "source",
        "adapter",
        "policy-version",
        "identities",
        "smoke-acceptance",
    ):
        p.add_argument("--" + name)
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "."))
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--final-test", action="store_true")
    for name, default in [
        ("max-rounds", 10),
        ("groups-per-round", 8),
        ("group-size", 4),
        ("update-epochs", 1),
        ("max-env-episodes", 320),
        ("port", 8765),
    ]:
        p.add_argument("--" + name, type=int, default=default)
    a = p.parse_args()
    a.action = {"train_sft": "sft_a800", "train_rl": "grpo_a800", "evaluate": "eval"}.get(a.action, a.action)
    cfg = {}
    if a.config:
        if a.config.suffix == ".json":
            cfg = json.loads(a.config.read_text())
        else:
            import yaml

            cfg = yaml.safe_load(a.config.read_text()) or {}
    # Only explicitly supplied CLI options override configuration.
    for k, v in vars(a).items():
        option = "--" + k.replace("_", "-")
        aliases = {"output": ("--output-dir",), "checkpoint": ("--init-checkpoint",)}.get(k, ())
        if (
            k not in ("config", "action")
            and (
                any(
                    any(word == candidate or word.startswith(candidate + "=") for candidate in (option, *aliases))
                    for word in sys.argv[1:]
                )
                or k not in cfg
            )
            and v is not None
        ):
            cfg[k] = v

    def no_credentials(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in (
                    "token",
                    "password",
                    "api_key",
                    "credentials",
                    "authorization",
                ):
                    raise ValueError(
                        "Credentials belong in environment variables, never configs"
                    )
                no_credentials(item)
        elif isinstance(value, list):
            for item in value:
                no_credentials(item)

    no_credentials(cfg)
    cfg.setdefault(
        "model",
        "gpt-6-astra"
        if a.action == "collect_astra"
        else os.environ.get("MODEL_ROOT", "Qwen/Qwen3.5-4B"),
    )
    if a.action == "collect_astra":
        for key, value in {
            "effort": "high",
            "service_tier": "default",
            "max_extra_cost_usd": 0,
            "paid_credits_authorized": False,
            "automatic_retries": 0,
            "max_episodes": 0,
        }.items():
            cfg.setdefault(key, value)
    cfg.setdefault("output", os.environ.get("RUN_ROOT", "runs/default"))
    if cfg.get("num_gpus") is not None:
        if cfg["num_gpus"] != 8:
            p.error("v0.1.0 GPU training supports exactly 8 ranks; other world sizes are NOT VALIDATED")
        if len(set(cfg["gpus"].split(","))) != cfg["num_gpus"]:
            p.error("--num-gpus must match the number of distinct --gpus")
    cfg.setdefault(
        "deepspeed", str(Path(__file__).resolve().parent / "configs/zero3.json")
    )
    if cfg.get("split") == "test" and a.action == "eval" and not cfg.get("final_test"):
        p.error("Seed2 requires explicit --final-test")
    if (
        cfg.get("resume")
        and a.execute
        and not a.dry_run
        and not int(os.environ.get("WORLD_SIZE", 0))
    ):
        (Path(cfg["output"]) / "STOP").unlink(missing_ok=True)
    if a.dry_run or not a.execute:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN",
                    "action": a.action,
                    "config": cfg,
                    "gpu_started": False,
                },
                indent=2,
            )
        )
        return
    if a.action == "collect_astra":
        if cfg.get("snapshot_only"):
            if cfg.get("seed") == 2 and not a.final_test:
                p.error("Seed2 snapshot requires --final-test")
            from robouse.teacher.seal import snapshot

            print(
                json.dumps(
                    snapshot(cfg["data_root"], cfg["seed"], cfg["snapshot_output"])
                )
            )
        else:
            from robouse.teacher.collect import run

            result = run(
                cfg,
                resume=bool(cfg.get("resume")),
                final_test=a.final_test,
                check_only=bool(cfg.get("check_only")),
            )
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in result.items()
                        if k not in ("reserved", "terminal", "quota")
                    },
                    indent=2,
                )
            )
    elif a.action == "prepare_data":
        from robouse.data.datasets import prepare

        prepare(cfg, cfg["output"])
    elif a.action == "export_data":
        from robouse.data.datasets import export

        value = export(
            cfg["data_root"],
            Path(cfg["files"]).read_text().splitlines(),
            cfg["manifest"],
        )
        print(json.dumps({"files": len(value["files"]), "bytes": value["total_bytes"]}))
    elif a.action == "verify_data":
        from robouse.data.datasets import verify

        value = verify(cfg["data_root"], cfg["manifest"])
        print("VERIFIED", len(value["files"]))
    elif a.action == "import_data":
        from robouse.data.datasets import verify

        # Use a locally supplied manifest, not a mutable remote file list.
        manifest = json.loads(Path(cfg["manifest"]).read_text())
        names = [r["path"] for r in manifest["files"]]
        if any(
            Path(n).is_absolute() or ".." in Path(n).parts or "\n" in n for n in names
        ):
            raise ValueError("Unsafe manifest")
        listing = Path(str(cfg["manifest"]) + ".files")
        listing.write_text("".join(n + "\n" for n in names))
        subprocess.run(
            [
                "rsync",
                "-a",
                "--partial",
                "--append-verify",
                "--files-from=" + str(listing),
                "--",
                cfg["source"].rstrip("/") + "/",
                cfg["data_root"].rstrip("/") + "/",
            ],
            check=True,
        )
        verify(cfg["data_root"], cfg["manifest"])
    elif a.action == "sft_a800":
        if cfg.get("finetuning_mode", "full") != "full":
            p.error("PEFT is not validated in v0.1.0; only --finetuning-mode full is supported")
        if not cfg.get("revision"):
            p.error("Pin --revision to model commit (local model: inventory SHA)")
        if len(set(cfg["gpus"].split(","))) != 8:
            p.error("Eight distinct GPUs required")
        if not int(os.environ.get("WORLD_SIZE", "0")):
            if cfg.get("smoke"):
                from robouse.smoke import run

                run(cfg)
            else:
                if not cfg.get("smoke_stage"):
                    from robouse.smoke import require_acceptance

                    require_acceptance(cfg)
                from robouse.launch import distributed

                distributed(cfg)
        else:
            import torch

            torch.use_deterministic_algorithms(True)
            from robouse.training.sft.train import run

            run(cfg)
    elif a.action == "grpo_a800":
        if not cfg.get("allow_experimental"):
            p.error("Online GRPO is Experimental: real A800/Isaac logprob parity is NOT_RUN. Use --allow-experimental only for an explicitly allocated validation run; immutable behavior LP and the 0.02 parity gate remain enforced.")
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg["gpus"]
        from robouse.training.rl.grpo import run

        run(cfg)
    elif a.action == "env_worker":
        from robouse.environments.robodojo.remote import serve

        serve(cfg)
    elif a.action == "eval":
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg["gpus"]
        from robouse.evaluation.evaluator import run

        run(cfg)
    elif a.action == "view_results":
        from robouse.view import build

        build(cfg["output"])
        print(str(Path(cfg["output"]) / "offline.zip"))
    elif a.action == "status":
        root = Path(cfg["output"])
        print(
            json.dumps(
                {
                    p.name: {
                        k: v
                        for k, v in json.loads(p.read_text()).items()
                        if k not in ("consumed_groups", "reserved", "terminal")
                    }
                    for pattern in (
                        "*state.json",
                        "teacher-progress.json",
                        "progress.json",
                        "training-complete.json",
                    )
                    for p in root.glob(pattern)
                },
                indent=2,
            )
        )
    elif a.action == "stop":
        # Cooperative stop only; never signal unrelated processes.
        out = Path(cfg["output"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "STOP").touch()
        print("STOP requested; active backend must honor its request contract")


if __name__ == "__main__":
    main()
