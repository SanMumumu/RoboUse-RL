"""Token sums and sampler-to-trainer importance correction; no overwritten behavior LP."""
import torch


def assert_alignment(actual, recorded, tolerance=0.02):
    if actual.shape != recorded.shape or not actual.numel():
        raise ValueError("BEHAVIOR_LOGPROB_SHAPE_MISMATCH")
    if not torch.isfinite(actual).all() or not torch.isfinite(recorded).all():
        raise ValueError("NONFINITE_BEHAVIOR_PROBABILITY")
    error = float((actual.detach() - recorded.detach()).abs().max())
    if error > tolerance:
        raise ValueError(f"BEHAVIOR_ALIGNMENT_FAILED:{error}>{tolerance}")
    return error


def corrected_grpo(
    current, behavior, trainer_old, reference, mask, advantage, *, beta=0.02, clip=0.2
):
    """Fixed per-token TIS correction, as used for mismatched rollout/train engines.

    behavior is immutable actual sampling LP. trainer_old is a *separate* frozen
    vector measured before updates with the training forward. The correction
    converts behavior-distributed observations into the frozen trainer-policy
    measure; clipping is relative to that frozen trainer policy. No correction
    is inferred from current weights after optimization has begun.
    """
    if not all(
        x.shape == current.shape for x in (behavior, trainer_old, reference, mask)
    ):
        raise ValueError("Probability/mask shapes differ")
    if mask.dtype != torch.bool or not mask.any():
        raise ValueError("No assistant actions")
    c, b, o, r = [
        x[mask]
        for x in (current, behavior.detach(), trainer_old.detach(), reference.detach())
    ]
    if not all(torch.isfinite(x).all() for x in (c, b, o, r)):
        raise ValueError("Nonfinite log probability")
    correction = (o - b).exp().detach()
    ratio = (c - o).exp()
    surrogate = torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage
    )
    delta = r - c
    kl = delta.exp() - delta - 1
    result = (correction * (-surrogate + beta * kl)).mean()
    if not torch.isfinite(result):
        raise ValueError("Importance correction overflow")
    return result


def token_nll(logits, target, denominator, world_size=1):
    if int(denominator) < 1:
        raise ValueError("Positive global target count required")
    return (
        torch.nn.functional.cross_entropy(logits.float(), target, reduction="sum")
        * world_size
        / denominator
    )
