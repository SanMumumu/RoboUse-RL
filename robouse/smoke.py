"""Independent 4-step GPU acceptance: uninterrupted vs save/interruption/resume."""
import json
from pathlib import Path
from robouse.data.datasets import write_json
from agp_rl.io import digest, sha256


def acceptance_key(config):
    from robouse.models.qwen import training_code_hash
    from robouse.data.sampling import load_sources

    return digest(
        {
            "code_sha256": training_code_hash(),
            "model": config["model"],
            "revision": config["revision"],
            "gpus": config["gpus"],
            "deepspeed_sha256": sha256(config["deepspeed"]),
            "accumulation": config.get("gradient_accumulation_steps", 4),
            "learning_rate": float(config.get("learning_rate", 1e-5)),
            "train_sources": load_sources(config, "train")[2],
            "val_sources": load_sources(config, "val")[2],
            "seed": config.get("seed", 0),
            "sampling_strategy": config.get("sampling_strategy", "coverage"),
            "samples_per_epoch": config.get("samples_per_epoch"),
        }
    )


def require_acceptance(config):
    path = config.get("smoke_acceptance")
    if not path:
        raise ValueError(
            "Formal SFT requires --smoke-acceptance from this model own GPU smoke"
        )
    record = json.loads(Path(path).read_text())
    if (
        record.get("status") != "PASS"
        or record.get("key") != acceptance_key(config)
        or record.get("world_size") != 8
    ):
        raise ValueError("GPU smoke evidence does not match model/revision/config")
    for filename, h in record["evidence"].items():
        if sha256(filename) != h:
            raise ValueError("Smoke evidence changed")


def run(config):
    from robouse.launch import distributed
    from safetensors.torch import load_file
    import torch

    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    cfg = {
        **config,
        "smoke": False,
        "smoke_stage": True,
        "max_steps": 4,
        "save_steps": 1,
        "epochs": config.get("epochs", 1),
    }
    continuous = root / "continuous"
    resumed = root / "resumed"
    distributed({**cfg, "output": str(continuous)})
    distributed({**cfg, "output": str(resumed), "stop_after_step": 2})
    distributed(
        {**cfg, "output": str(resumed), "resume": str(resumed / "checkpoint-2")}
    )
    evidence = {}
    for directory in (continuous, resumed):
        progress = directory / "progress.json"
        value = json.loads(progress.read_text())
        if value["step"] != 4 or not all(
            v > 0 for v in value["component_delta_squared"].values()
        ):
            raise ValueError("Four-step three-component update gate failed")
        if value["peak_allocated"] is None:
            raise ValueError("GPU memory evidence required")
        evidence[str(progress.resolve())] = sha256(progress)
        checkpoint = directory / "checkpoint-4"
        if (
            not list(checkpoint.glob("global_step*"))
            or len(list(checkpoint.glob("rng_state*.pth"))) < 8
        ):
            raise ValueError("Eight-rank ZeRO/RNG resume state missing")
    # Loading every safetensors shard verifies saved weights can be reloaded.
    files = sorted((continuous / "checkpoint-4").glob("*.safetensors"))
    if not files:
        raise ValueError("Missing saved full weights")
    for path in files:
        other = resumed / "checkpoint-4" / path.name
        a, b = load_file(path), load_file(other)
        if set(a) != set(b) or any(not torch.equal(a[k], b[k]) for k in a):
            raise ValueError("Exact interrupted-resume weights differ")
        evidence[str(path.resolve())] = sha256(path)
        evidence[str(other.resolve())] = sha256(other)
    result = {
        "status": "PASS",
        "world_size": 8,
        "key": acceptance_key(config),
        "evidence": evidence,
        "steps": 4,
        "resume_exact": True,
        "save_reload": True,
    }
    write_json(root / "smoke-acceptance.json", result)
    return result
