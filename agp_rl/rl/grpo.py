"""Episode-relative GRPO; average tokens per episode, then episodes per group."""
import torch


def group_advantages(rewards, epsilon=1e-8):
    values = torch.as_tensor(rewards, dtype=torch.float64)
    if values.ndim != 1 or len(values) < 2 or not torch.isfinite(values).all():
        raise ValueError("A finite group of at least two episodes is required")
    std = values.std(unbiased=False)
    if std <= epsilon:
        return torch.zeros_like(values), False
    return (values - values.mean()) / (std + epsilon), True


def grpo_loss(current, old, reference, mask, advantage, *, clip_epsilon=0.2, beta=0.02):
    if current.shape != old.shape or current.shape != reference.shape or current.shape != mask.shape:
        raise ValueError("GRPO token shapes differ")
    if mask.dtype != torch.bool or not mask.any():
        raise ValueError("Empty/invalid action mask")
    current, old, reference = current[mask], old.detach()[mask], reference.detach()[mask]
    if not all(torch.isfinite(x).all() for x in (current, old, reference)):
        raise ValueError("Nonfinite logprob")
    ratio = torch.exp(current - old)
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage)
    delta = reference - current
    kl = delta.exp() - delta - 1
    loss = (-surrogate + beta * kl).mean()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite objective; refusing optimizer update")
    return loss, {"kl": float(kl.detach().mean()),
                  "clip_fraction": float(((ratio - 1).abs() > clip_epsilon).float().mean().detach())}
