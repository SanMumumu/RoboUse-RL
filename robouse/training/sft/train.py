"""Full-parameter SFT, seed-pinned windows and complete HF/ZeRO checkpoints."""
import json
import os
from pathlib import Path
import torch
from agp_rl.io import digest, sha256
from robouse.data.datasets import read_jsonl, check_roles, write_json
from robouse.models.qwen import messages_with_pixels, resolve_model, load, training_code_hash
from robouse.planning import window_plan
from robouse.data.sampling import load_sources, sample_indices
from robouse.training.sft.trainer import trainer_class, arguments, audit_callback
from agp_rl.policy.termination import CHAT_TEMPLATE_OPTIONS, native_termination_contract


def encode(row, processor, root):
    from robouse.harness.context import build_context

    messages = messages_with_pixels(
        build_context(row["messages"], with_target=True), row["images"], root
    )
    full = dict(
        processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=False,
            **CHAT_TEMPLATE_OPTIONS,
        )
    )
    prefix = processor.apply_chat_template(
        messages[:-1],
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        **CHAT_TEMPLATE_OPTIONS,
    )
    n = prefix["input_ids"].shape[-1]
    if not torch.equal(full["input_ids"][:, :n], prefix["input_ids"]):
        raise ValueError("Noncausal template target boundary")
    target = full["input_ids"][0, n:]
    if target.numel() < 1:
        raise ValueError("Empty target")
    end = processor.tokenizer.eos_token_id
    stops = torch.nonzero(target == end).flatten()
    if len(stops) != 1:
        raise ValueError("Assistant must contain exactly one native EOS")
    length = int(stops[0]) + 1
    target = target[:length]
    for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
        if key in full:
            full[key] = full[key][:, : n + length]
    return {
        "inputs": full,
        "target": target,
        "target_count": len(target),
        "window_id": row["window_id"],
        "dummy": False,
    }


def run(config):
    import torch.distributed as dist
    from agp_rl.pipeline2b.fullft.checkpoint import freeze_parameters

    if config.get("device") != "cpu" and int(os.environ.get("WORLD_SIZE", 0)) != 8:
        raise ValueError("Eight real ranks required")
    if config.get("finetuning_mode", "full") != "full":
        raise ValueError("Only full fine-tuning is released; PEFT is not validated in v0.1.0")
    rows, weights, train_manifests = load_sources(config, "train")
    val, _, val_manifests = load_sources(config, "val")
    manifests = train_manifests + val_manifests
    registry = []
    for folder in {Path(m["path"]).parent for m in manifests}:
        path = folder / "identity-roles.json"
        if path.exists():
            registry.extend(json.loads(path.read_text()))
    check_roles(rows + val + registry)
    if not rows or len({r["window_id"] for r in rows}) != len(rows):
        raise ValueError("Nonempty unique windows required")
    if any(r["seed"] != 0 for r in rows) or any(r["seed"] != 1 for r in val):
        raise ValueError("Seed0 train / Seed1 validation only")
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("checkpoint-*")) and not config.get("resume"):
        raise ValueError(
            "Existing checkpoints require explicit resume; use an independent output for new training"
        )
    if type(config.get("epochs", 1)) is not int or config.get("epochs", 1) < 1:
        raise ValueError("Positive integer epochs required by the frozen coverage plan")
    world = int(os.environ.get("WORLD_SIZE", 1))
    accum = config.get("gradient_accumulation_steps", 4)
    # Use public profile lengths when available, otherwise template text length
    # as a CPU bucketing estimate; never truncate RGB/history/targets.
    lengths = [
        r.get("tokens", sum(len(str(m["content"])) for m in r["messages"]))
        for r in rows
    ]
    order = []
    seed = config.get("seed", 0)
    for epoch in range(config.get("epochs", 1)):
        chosen = sample_indices(len(rows), config.get("sampling_strategy", "coverage"), weights=weights, count=config.get("samples_per_epoch"), seed=seed + epoch)
        local = window_plan([lengths[i] for i in chosen], 1, accum, seed=seed + epoch, world=world, bucket_size=config.get("bucket_size", 128))
        order.extend(None if i is None else chosen[i] for i in local)
    steps = len(order) // world // accum
    total = (
        min(steps, config["max_steps"]) if config.get("max_steps", -1) > 0 else steps
    )
    args = arguments(config, total_steps=total)
    source = resolve_model(config)
    model, processor = load(source)
    from robouse.models.qwen import processor_signature

    for file in (m["path"] for m in manifests):
        meta = Path(file).parent / "preparation.json"
        if meta.exists():
            value = json.loads(meta.read_text())
            if value.get("processor_signature") and value[
                "processor_signature"
            ] != processor_signature(processor):
                raise ValueError(
                    "Prepared windows were profiled with a different model processor/template"
                )
    freeze_parameters(model, {"vision_freeze": False, "aligner_freeze": False})
    contract = {
        "version": "seed-roles-v2",
        "train_manifests": train_manifests,
        "val_manifests": val_manifests,
        "test_data_registered_only": config.get("test_data") or config.get("datasets", {}).get("test"),
        "seed": seed,
        "sampling_strategy": config.get("sampling_strategy", "coverage"),
        "model": config["model"],
        "revision": config["revision"],
        "order": digest(order),
        "accumulation": accum,
        "world": world,
        "steps": total,
        "code_sha256": training_code_hash(),
        "deepspeed_sha256": sha256(config["deepspeed"])
        if config.get("deepspeed")
        else None,
        "training_options": {
            k: config.get(k)
            for k in (
                "learning_rate",
                "warmup_ratio",
                "lr_scheduler_type",
                "gradient_checkpointing",
                "max_context_tokens",
            )
        },
    }
    contract_path = out / "train-contract.json"
    if config.get("resume"):
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Exact-resume contract changed")
        cp = Path(config["resume"])
        from robouse.checkpoints import verify

        verify(cp)
        if not (cp / "trainer_state.json").exists():
            raise ValueError("Missing HF trainer resume state")
    if args.process_index == 0:
        from robouse.utils.manifests import write_run_manifest

        write_run_manifest(out, config, stage="sft", datasets={"train": train_manifests, "val": val_manifests})
        write_json(contract_path, contract)
        ownership = list(registry)
        for r in rows + val:
            ownership.append(
                {
                    k: r[k]
                    for k in (
                        "identity_id",
                        "seed",
                        "role",
                        "physical_aliases",
                        "layout_hash",
                        "initial_rgb_hash",
                    )
                    if k in r
                }
            )
        write_json(out / "role-ownership.json", ownership)

    class Windows:
        def __len__(self):
            return len(order)

        def __getitem__(self, i):
            index = order[i]
            dummy = index is None
            value = encode(rows[0 if dummy else index], processor, config["data_root"])
            if dummy:
                value.update(
                    dummy=True, target_count=0, window_id="__masked_collective_tail__"
                )
            if value["inputs"]["input_ids"].numel() > config.get(
                "max_context_tokens", 131072
            ):
                raise ValueError("Context limit; no truncation")
            if len(value["target"]) > config.get("max_new_tokens", 2048):
                raise ValueError(
                    "Assistant target exceeds qualified output budget; re-prepare per-model windows"
                )
            return value

    def validate(args, state):
        model.eval()
        sums = torch.zeros(2, dtype=torch.float64, device=args.device)
        with torch.no_grad():
            # All ZeRO ranks execute identical numbers of forwards. Ownership
            # counts each validation token only once in the final all-reduce.
            for i, row in enumerate(val):
                batch = encode(row, processor, config["data_root"])
                inputs = {k: v.to(args.device) for k, v in batch["inputs"].items()}
                target = batch["target"].to(args.device)
                n = inputs["input_ids"].shape[-1] - len(target)
                positions = torch.arange(n - 1, n + len(target) - 1, device=args.device)
                output = model(**inputs, use_cache=False, logits_to_keep=positions)
                if i % args.world_size == args.process_index:
                    sums[0] += torch.nn.functional.cross_entropy(
                        output.logits[0].float(), target, reduction="sum"
                    )
                    sums[1] += len(target)
        if dist.is_initialized():
            dist.all_reduce(sums)
        model.train()
        if args.process_index == 0:
            nll = float(sums[0] / sums[1]) if sums[1] else None
            write_json(
                out / f"validation-{state.global_step}.json",
                {"seed": 1, "nll": nll, "tokens": int(sums[1])},
            )
            best = (
                json.loads((out / "best-validation.json").read_text())
                if (out / "best-validation.json").exists()
                else {"nll": float("inf")}
            )
            if nll is not None and nll < best["nll"]:
                write_json(
                    out / "best-validation.json",
                    {
                        "nll": nll,
                        "checkpoint": str(out / f"checkpoint-{state.global_step}"),
                    },
                )

    cls = trainer_class(config, out)
    trainer = cls(
        model=model,
        args=args,
        train_dataset=Windows(),
        data_collator=lambda x: x[0],
        processing_class=processor,
        callbacks=[audit_callback(config, model, processor, validation=validate)],
    )
    trainer.model_accepts_loss_kwargs = True
    trainer.train(resume_from_checkpoint=config.get("resume"))
    trainer.save_model()
    if trainer.is_world_process_zero():
        processor.save_pretrained(out)
        write_json(
            out / "training-complete.json",
            {
                "status": "COMPLETED"
                if trainer.state.global_step == total
                else "INTERRUPTED",
                "step": trainer.state.global_step,
                "expected_steps": total,
                "test_targets_read": False,
            },
        )
