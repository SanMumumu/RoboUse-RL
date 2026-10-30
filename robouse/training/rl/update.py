"""Two-phase behavior qualification then an eight-rank HF/ZeRO group update."""
import argparse
import json
import os
from pathlib import Path
import torch
from agp_rl.io import sha256, digest
from robouse.data.datasets import write_json
from robouse.models.qwen import load, policy_version, tensor_hash, token_logprobs
from robouse.models.qwen import configure_grpo_training
from robouse.training.rl.objectives import assert_alignment
from robouse.planning import update_plan
from robouse.scheduler import atomic_torch


def qualified(config):
    torch.use_deterministic_algorithms(True)
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world != 8 and config.get("device") != "cpu":
        raise ValueError("Eight qualification ranks required")
    device = (
        f'cuda:{int(os.environ.get("LOCAL_RANK",0))}'
        if config.get("device") != "cpu"
        else "cpu"
    )
    if device != "cpu":
        torch.cuda.set_device(device)
    behavior, _ = load(config["checkpoint"], device=device)
    behavior.eval()
    if policy_version(config["checkpoint"]) != config["policy_version"]:
        raise ValueError("Behavior weight hash mismatch")
    items = []
    episodes = []
    for path in config["groups"]:
        group = json.loads(Path(path).read_text())
        for ep, adv in zip(group["episodes"], group["advantages"]):
            episodes.append((ep, adv))
    for ep_index, (ep, advantage) in enumerate(episodes):
        if ep_index % world != rank:
            continue
        total = sum(sum(bool(x) for x in d["loss_mask"]) for d in ep["decisions"])
        for turn, decision in enumerate(ep["decisions"]):
            if sha256(decision["file"]) != decision["sha256"]:
                raise ValueError("Stored sample changed")
            value = torch.load(decision["file"], map_location="cpu", weights_only=True)
            if (
                value["policy_version"] != config["policy_version"]
                or tensor_hash(value["inputs"]) != decision["input_hash"]
            ):
                raise ValueError("Input/weights identity mismatch")
            if (
                value["tokens"].tolist() != decision["generated_ids"]
                or value["behavior_logprobs"].tolist() != decision["behavior_logprobs"]
                or value["mask"].tolist() != decision["loss_mask"]
            ):
                raise ValueError(
                    "Recorded behavior tokens/probabilities/mask differ from immutable tensors"
                )
            inputs = {k: v.to(device) for k, v in value["inputs"].items()}
            tokens = value["tokens"].to(device)
            behavior.eval()
            with torch.no_grad():
                oracle = token_logprobs(
                    behavior,
                    inputs,
                    tokens,
                    temperature=value["temperature"],
                    forbidden=value["forbidden"],
                    prefix_oracle=True,
                )
                configure_grpo_training(behavior)
                training = token_logprobs(
                    behavior,
                    inputs,
                    tokens,
                    temperature=value["temperature"],
                    forbidden=value["forbidden"],
                )
            evidence_path = (
                Path(config["output"])
                / "alignment"
                / f"rank-{rank}-episode-{ep_index}-turn-{turn}.pt"
            )
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_torch(
                evidence_path,
                {
                    "sample_sha256": decision["sha256"],
                    "input_hash": value["input_hash"],
                    "policy_version": config["policy_version"],
                    "recorded_behavior": value["behavior_logprobs"],
                    "prefix_oracle": oracle.cpu(),
                    "frozen_training_backend": training.cpu(),
                    "tolerance": 0.02,
                },
            )
            error = assert_alignment(oracle.cpu(), value["behavior_logprobs"], 0.02)
            # Never overwrite value['behavior_logprobs']; both backend measures persist.
            value.update(
                trainer_old=training.cpu(),
                advantage=float(advantage),
                episode_weight=float(value["mask"].sum()) / total / len(episodes),
                alignment_error=error,
                window_id=f"{ep['episode_id']}-turn-{turn}",
            )
            items.append(value)
    del behavior
    if device != "cpu":
        torch.cuda.empty_cache()
    reference, _ = load(config["reference_checkpoint"], device=device)
    reference.eval()
    rows = []
    folder = Path(config["output"]) / "qualified"
    folder.mkdir(parents=True, exist_ok=True)
    for value in items:
        inputs = {k: v.to(device) for k, v in value["inputs"].items()}
        tokens = value["tokens"].to(device)
        with torch.no_grad():
            value["reference"] = token_logprobs(
                reference,
                inputs,
                tokens,
                temperature=value["temperature"],
                forbidden=value["forbidden"],
            ).cpu()
        path = folder / (value["window_id"] + ".pt")
        atomic_torch(path, value)
        rows.append(
            {"file": str(path), "sha256": sha256(path), "window_id": value["window_id"]}
        )
    write_json(
        folder / f"rank-{rank}.json",
        {
            "rows": rows,
            "policy": config["policy_version"],
            "reference_policy": policy_version(config["reference_checkpoint"]),
        },
    )


def train(config):
    torch.use_deterministic_algorithms(True)
    import torch.distributed as dist
    from agp_rl.pipeline2b.fullft.checkpoint import freeze_parameters
    from robouse.training.sft.trainer import trainer_class, arguments, audit_callback

    proofs = [
        json.loads(
            (Path(config["output"]) / "qualified" / f"rank-{rank}.json").read_text()
        )
        for rank in range(8)
    ]
    rows = [r for proof in proofs for r in proof["rows"]]
    order, accumulation = update_plan(len(rows), config["update_epochs"])
    prior = 0
    resume = config.get("optimizer_checkpoint")
    if resume:
        prior = json.loads((Path(resume) / "trainer_state.json").read_text())[
            "global_step"
        ]
    output = Path(config["output"]) / "update"
    output.mkdir(exist_ok=True)
    ownership = Path(config["reference_checkpoint"]) / "role-ownership.json"
    if ownership.exists():
        import shutil

        shutil.copy2(ownership, output / "role-ownership.json")
    latest = sorted(
        (
            p
            for p in output.glob("checkpoint-*")
            if (p / "CHECKPOINT_COMMITTED.json").exists()
        ),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if latest:
        resume = str(latest[-1])
    if resume:
        from robouse.checkpoints import verify

        verify(resume)
    cfg = {
        **config,
        "output": str(output),
        "gradient_accumulation_steps": accumulation,
        "save_steps": 1,
        "lr_scheduler_type": "constant",
        "max_steps": prior + config["update_epochs"],
    }
    args = arguments(
        cfg, total_steps=prior + config["update_epochs"], accumulation=accumulation
    )
    model, processor = load(config["checkpoint"])
    freeze_parameters(model, {"vision_freeze": False, "aligner_freeze": False})
    configure_grpo_training(model)

    class Episodes:
        def __len__(self):
            return len(order)

        def __getitem__(self, i):
            index = order[i]
            dummy = index is None
            row = rows[0 if dummy else index]
            if sha256(row["file"]) != row["sha256"]:
                raise ValueError("Qualified tensor changed")
            value = torch.load(row["file"], map_location="cpu", weights_only=True)
            return {
                **value,
                "target": value["tokens"],
                "dummy": dummy,
                "target_count": 0 if dummy else len(value["tokens"]),
                "window_id": "__masked_collective_tail__"
                if dummy
                else row["window_id"],
            }

    cls = trainer_class(cfg, output, grpo=True)
    trainer = cls(
        model=model,
        args=args,
        train_dataset=Episodes(),
        data_collator=lambda x: x[0],
        processing_class=processor,
        callbacks=[audit_callback(cfg, model, processor)],
    )
    trainer.model_accepts_loss_kwargs = True
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model()
    if trainer.is_world_process_zero():
        processor.save_pretrained(output)
        cp = output / f"checkpoint-{trainer.state.global_step}"
        if trainer.state.global_step != prior + config["update_epochs"]:
            raise ValueError("Update interrupted before round commit")
        write_json(
            Path(config["output"]) / "update-response.json",
            {
                "policy_version": policy_version(cp),
                "checkpoint": str(cp),
                "optimizer_rng_checkpoint": str(cp),
                "reference_policy": policy_version(config["reference_checkpoint"]),
            },
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    p.add_argument("--phase", choices=["qualify", "train"], required=True)
    a = p.parse_args()
    config = json.loads(Path(a.request).read_text())
    if a.phase == "qualify":
        qualified(config)
    else:
        train(config)


if __name__ == "__main__":
    main()
