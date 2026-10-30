import math
import pytest
import torch
from robouse.training.rl.objectives import corrected_grpo, assert_alignment, token_nll
from robouse.planning import window_plan, update_plan
from robouse.training.rl.grpo import rank_plan


def test_framework_correction_keeps_recorded_behavior_and_gradient():
    behavior = torch.log(torch.tensor([0.2, 0.4], dtype=torch.float64))
    trainer_old = behavior + torch.tensor([0.04, -0.03], dtype=torch.float64)
    original = behavior.clone()
    current = (trainer_old + math.log(1.5)).requires_grad_()
    reference = trainer_old.clone()
    mask = torch.tensor([True, True])
    actual = corrected_grpo(
        current, behavior, trainer_old, reference, mask, 1.0, beta=0.02
    )
    expected = (
        (trainer_old - behavior).exp()
        * (-1.2 + 0.02 * ((reference - current).exp() - (reference - current) - 1))
    ).mean()
    assert torch.equal(behavior, original)
    assert actual.item() == pytest.approx(expected.item())
    g1 = torch.autograd.grad(actual, current, retain_graph=True)[0]
    g2 = torch.autograd.grad(expected, current)[0]
    assert torch.allclose(g1, g2)
    with pytest.raises(ValueError, match="BEHAVIOR_ALIGNMENT_FAILED"):
        assert_alignment(trainer_old, behavior)
    assert_alignment(behavior, behavior)


def test_variable_episode_weights_and_eight_rank_accumulation():
    theta = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    lengths = [1, 3, 7]
    losses = []
    for ep, n in enumerate(lengths):
        old = torch.full((n,), -0.7, dtype=torch.float64)
        lp = old + theta
        losses.append(
            corrected_grpo(
                lp,
                old,
                old,
                old,
                torch.ones(n, dtype=torch.bool),
                [-1.0, 0.5, 1.0][ep],
                beta=0.02,
            )
            / 3
        )
    reference = sum(losses)
    # Rank mean cancels the explicit factor eight, even with five masked tails.
    distributed = (
        sum(
            sum(losses[i] * 8 if i is not None else theta * 0 for i in p)
            for p in rank_plan(3)
        )
        / 8
    )
    assert distributed.item() == pytest.approx(reference.item())
    assert torch.autograd.grad(distributed, theta, retain_graph=True)[
        0
    ].item() == pytest.approx(torch.autograd.grad(reference, theta)[0].item())


def test_global_target_normalization_matches_concatenated_reference():
    torch.manual_seed(0)
    w = torch.randn(3, 5, requires_grad=True)
    targets = [torch.tensor([0]), torch.tensor([1, 2, 1]), torch.tensor([4, 0])]
    chunks = [w[:1], w, w[:2]]
    normalized = sum(token_nll(x, y, 6, 8) for x, y in zip(chunks, targets)) / 8
    reference = torch.nn.functional.cross_entropy(torch.cat(chunks), torch.cat(targets))
    assert torch.allclose(normalized, reference)
    assert torch.allclose(
        torch.autograd.grad(normalized, w, retain_graph=True)[0],
        torch.autograd.grad(reference, w)[0],
    )


def test_length_buckets_preserve_all_windows_without_replication():
    plan = window_plan([3, 1, 8, 2, 6, 5, 11, 7, 4], 2, 2, world=8, bucket_size=4)
    assert len(plan) % 16 == 0
    for ep in range(2):
        assert sorted(
            i for i in plan[ep * 16 : (ep + 1) * 16] if i is not None
        ) == list(range(9))
    assert plan == window_plan(
        [3, 1, 8, 2, 6, 5, 11, 7, 4], 2, 2, world=8, bucket_size=4
    )
    order, acc = update_plan(19, 3)
    assert len(order) == 72 and acc == 3
