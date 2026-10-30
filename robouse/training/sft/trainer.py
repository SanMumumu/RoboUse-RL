"""Shared HF Trainer implementation for SFT and full-group GRPO updates."""
import json
import os
from pathlib import Path
import time
from robouse.data.datasets import write_json


def trainer_class(config, run, *, grpo=False):
    import torch
    import torch.distributed as dist
    from torch.utils.data import SequentialSampler
    from transformers import Trainer
    from robouse.training.rl.objectives import corrected_grpo, token_nll
    from robouse.models.qwen import token_logprobs

    run = Path(run)

    class FullTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(
                train_dataset if train_dataset is not None else self.train_dataset
            )

        def get_batch_samples(self, epoch_iterator, num_batches, device):
            batches = []
            for _ in range(num_batches):
                try:
                    batches.append(next(epoch_iterator))
                except StopIteration:
                    break
            count = torch.tensor(
                sum(b.get("target_count", 0) for b in batches), device=device
            )
            if dist.is_initialized():
                dist.all_reduce(count)
            return batches, count

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            dummy = inputs["dummy"]
            target = inputs["target"]
            if grpo:
                model.train()  # zero dropout configured; retain activation checkpointing
                current = token_logprobs(
                    model,
                    inputs["inputs"],
                    target,
                    temperature=inputs["temperature"],
                    forbidden=inputs["forbidden"],
                )
                loss = corrected_grpo(
                    current,
                    inputs["behavior_logprobs"],
                    inputs["trainer_old"],
                    inputs["reference"],
                    inputs["mask"],
                    inputs["advantage"],
                    beta=config.get("beta", 0.02),
                    clip=config.get("clip_epsilon", 0.2),
                )
                # HF/DeepSpeed averages ranks; custom loss disables HF's extra
                # accumulation division. Each update includes every episode once.
                loss = loss * inputs["episode_weight"] * self.args.world_size
                output = {"logits": current}
            else:
                n = inputs["inputs"]["input_ids"].shape[-1] - len(target)
                positions = torch.arange(
                    n - 1, n + len(target) - 1, device=target.device
                )
                output = model(
                    **inputs["inputs"], use_cache=False, logits_to_keep=positions
                )
                loss = token_nll(
                    output.logits[0],
                    target,
                    num_items_in_batch
                    if num_items_in_batch is not None and int(num_items_in_batch) > 0
                    else 1,
                    self.args.world_size,
                )
            if dummy:
                loss = loss * 0.0
            return (loss, output) if return_outputs else loss

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch)
            self.model.robouse_target_tokens = (
                getattr(self.model, "robouse_target_tokens", 0) + inputs["target_count"]
            )
            path = run / "consumption" / f"rank-{self.args.process_index}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "step": self.state.global_step,
                            "window_id": inputs["window_id"],
                            "dummy": inputs["dummy"],
                            "target_count": inputs["target_count"],
                        }
                    )
                    + "\n"
                )
            return loss

    return FullTrainer


def arguments(config, *, total_steps=None, accumulation=None):
    from transformers import TrainingArguments
    import os

    return TrainingArguments(
        output_dir=config["output"],
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=accumulation
        or config.get("gradient_accumulation_steps", 4),
        num_train_epochs=1,
        max_steps=total_steps or config.get("max_steps", -1),
        learning_rate=float(config.get("learning_rate", 1e-5)),
        warmup_ratio=float(config.get("warmup_ratio", 0)),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        bf16=config.get("device", "cuda") != "cpu",
        use_cpu=config.get("device") == "cpu",
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        deepspeed=config.get("deepspeed") if config.get("device") != "cpu" else None,
        save_steps=config.get("save_steps", 100),
        save_total_limit=None,
        logging_steps=1,
        report_to=["tensorboard"] if config.get("tensorboard", True) else [],
        remove_unused_columns=False,
        dataloader_drop_last=False,
        accelerator_config={"even_batches": False},
        seed=config.get("seed", 0),
        data_seed=config.get("seed", 0),
        disable_tqdm=True,
        ddp_find_unused_parameters=False,
    )


def audit_callback(config, model, processor, *, validation=None):
    import torch
    import torch.distributed as dist
    from transformers import TrainerCallback
    from agp_rl.pipeline2b.fullft.checkpoint import component

    class Audit(TrainerCallback):
        def snapshot(self):
            result = {}
            for name, p in model.named_parameters():
                group = component(name)
                if group in result:
                    continue
                if hasattr(p, "ds_id"):
                    from deepspeed.utils import safe_get_local_fp32_param

                    v = safe_get_local_fp32_param(p)
                else:
                    v = p.detach()
                if v is not None and v.numel():
                    result[group] = v.detach().flatten()[:256].float().cpu().clone()
            return result

        def on_train_begin(self, args, state, control, **kwargs):
            if args.device.type == "cuda":
                props = torch.cuda.get_device_properties(args.device)
                write_json(
                    Path(args.output_dir) / f"hardware-rank-{args.process_index}.json",
                    {
                        "device": props.name,
                        "memory_bytes": props.total_memory,
                        "world_size": args.world_size,
                        "rank": args.process_index,
                        "torch": torch.__version__,
                    },
                )
            if args.process_index == 0:
                import time as clock
                from robouse.checkpoints import MARKER

                for directory in Path(args.output_dir).glob("checkpoint-*"):
                    if directory.is_dir() and not (directory / MARKER).exists():
                        directory.rename(
                            directory.with_name(
                                "incomplete-"
                                + directory.name
                                + "-"
                                + str(clock.time_ns())
                            )
                        )
            if dist.is_initialized():
                dist.barrier()
            self.before = self.snapshot()
            self.started = time.monotonic()
            self.initial_step = state.global_step

        def on_step_end(self, args, state, control, **kwargs):
            now = self.snapshot()
            delta = torch.tensor(
                [
                    float((now[k] - self.before[k]).square().sum())
                    if k in now and k in self.before
                    else 0
                    for k in ("language", "vision", "aligner")
                ],
                device=args.device,
            )
            if dist.is_initialized():
                dist.all_reduce(delta)
            elapsed = time.monotonic() - self.started
            steps = state.global_step - self.initial_step
            tokens = torch.tensor(
                getattr(model, "robouse_target_tokens", 0), device=args.device
            )
            if dist.is_initialized():
                dist.all_reduce(tokens)
            report = {
                "tokens_per_second": float(tokens) / elapsed,
                "step": state.global_step,
                "total": state.max_steps,
                "epoch": state.epoch,
                "elapsed": elapsed,
                "eta": elapsed / steps * (state.max_steps - state.global_step)
                if steps >= 2
                else None,
                "steps_per_second": steps / elapsed,
                "component_delta_squared": dict(
                    zip(("language", "vision", "aligner"), delta.tolist())
                ),
                "peak_allocated": torch.cuda.max_memory_allocated()
                if args.device.type == "cuda"
                else None,
                "peak_reserved": torch.cuda.max_memory_reserved()
                if args.device.type == "cuda"
                else None,
            }
            if args.process_index == 0:
                write_json(Path(args.output_dir) / "progress.json", report)
            self.before = now
            if (
                Path(args.output_dir) / "STOP"
            ).exists() or state.global_step == config.get("stop_after_step"):
                control.should_save = True
                control.should_training_stop = True
            if state.global_step == state.max_steps:
                control.should_save = True

        def on_log(self, args, state, control, logs=None, **kwargs):
            if args.process_index == 0:
                path = Path(args.output_dir) / "progress.json"
                if path.exists():
                    row = json.loads(path.read_text())
                    row.update(
                        {
                            k: v
                            for k, v in (logs or {}).items()
                            if isinstance(v, (int, float))
                        }
                    )
                    write_json(path, row)

        def on_save(self, args, state, control, **kwargs):
            if args.process_index == 0:
                import shutil

                ownership = Path(args.output_dir) / "role-ownership.json"
                if ownership.exists():
                    shutil.copy2(
                        ownership,
                        Path(args.output_dir)
                        / f"checkpoint-{state.global_step}"
                        / "role-ownership.json",
                    )
                processor.save_pretrained(
                    Path(args.output_dir) / f"checkpoint-{state.global_step}"
                )
            if dist.is_initialized():
                dist.barrier()
            if validation is not None:
                validation(args, state)
            if dist.is_initialized():
                dist.barrier()
            if args.process_index == 0:
                from robouse.checkpoints import commit

                commit(
                    Path(args.output_dir) / f"checkpoint-{state.global_step}",
                    args.world_size,
                )
            if dist.is_initialized():
                dist.barrier()

    return Audit()
