"""Explicit eight-rank CPU/Gloo reference check. No CUDA APIs or model weights."""
import argparse
from pathlib import Path
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from robouse.training.rl.objectives import token_nll, corrected_grpo
from robouse.data.datasets import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if not a.execute:
        print("DRY_RUN: eight-rank CPU reference; use torchrun and --execute")
        return
    if int(os.environ.get("WORLD_SIZE", 0)) != 8:
        raise ValueError("Eight CPU ranks required")
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(8)
    base = torch.nn.Linear(2, 3, bias=False, dtype=torch.float64)
    reference = torch.nn.Linear(2, 3, bias=False, dtype=torch.float64)
    reference.load_state_dict(base.state_dict())
    model = DDP(base)

    def batch(r, k):
        x = torch.arange((r + 1) * 2, dtype=torch.float64).reshape(r + 1, 2) / 20 + k
        y = (torch.arange(r + 1) + r + k) % 3
        return x, y

    denominator = torch.tensor(
        sum(len(batch(rank, k)[1]) for k in range(2)), dtype=torch.long
    )
    dist.all_reduce(denominator)
    for k in range(2):
        x, y = batch(rank, k)
        token_nll(model(x), y, denominator, 8).backward()
    all_x = []
    all_y = []
    for r in range(8):
        for k in range(2):
            x, y = batch(r, k)
            all_x.append(x)
            all_y.append(y)
    torch.nn.functional.cross_entropy(
        reference(torch.cat(all_x)).float(), torch.cat(all_y)
    ).backward()
    sft_error = float((base.weight.grad - reference.weight.grad).abs().max())
    # GRPO: three unequal episode lengths and five explicit masked collective slots.
    model.zero_grad()
    reference.zero_grad()

    def objective(net, index):
        n = [1, 3, 7][index]
        x = torch.ones(n, 2, dtype=torch.float64) * (index + 1)
        current = net(x).log_softmax(-1)[:, 0]
        old = current.detach() - 0.05
        return (
            corrected_grpo(
                current,
                old,
                old,
                old,
                torch.ones(n, dtype=torch.bool),
                [-1.0, 0.5, 1.0][index],
                beta=0.02,
            )
            / 3
        )

    value = objective(model, rank if rank < 3 else 0) * 8
    if rank >= 3:
        value = value * 0
    value.backward()
    sum(objective(reference, i) for i in range(3)).backward()
    grpo_error = float((base.weight.grad - reference.weight.grad).abs().max())
    errors = torch.tensor([sft_error, grpo_error])
    dist.all_reduce(errors, op=dist.ReduceOp.MAX)
    if not torch.all(errors < 1e-6):
        raise ValueError(
            "Distributed normalization differs from reference: " + str(errors.tolist())
        )
    if rank == 0:
        out = Path(a.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            out,
            {
                "world_size": 8,
                "backend": "gloo",
                "device": "cpu",
                "sft_max_gradient_error": float(errors[0]),
                "grpo_max_gradient_error": float(errors[1]),
                "synthetic_reference_only": True,
            },
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
