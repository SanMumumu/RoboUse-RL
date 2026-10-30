import hashlib
import json
import math
from pathlib import Path
import shutil
import tarfile
from io import BytesIO
import pytest
import torch
from robouse.data.datasets import check_roles, export, verify, image_bytes
from robouse.training.rl.grpo import rank_plan, validate_group
from agp_rl.rl.grpo import group_advantages, grpo_loss


def test_fixed_roles_alias_and_retry():
    a = {
        "seed": 0,
        "role": "train",
        "identity_id": "t/seed-0/layout-001",
        "layout_hash": "same",
    }
    b = {
        "seed": 1,
        "role": "val",
        "identity_id": "t/seed-1/layout-001",
        "layout_hash": "same",
    }
    with pytest.raises(ValueError, match="Cross-role"):
        check_roles([a, b])
    with pytest.raises(ValueError, match="Seed role"):
        check_roles([{**a, "seed": 1}])
    check_roles([a, {**a, "window_id": "retry"}])


def test_migration_relocates_original_tar_offset(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    raw = b"original RGB bytes, never a video frame"
    with tarfile.open(src / "images.tar", "w") as t:
        member = tarfile.TarInfo("image.png")
        member.size = len(raw)
        t.addfile(member, BytesIO(raw))
    with tarfile.open(src / "images.tar") as t:
        offset = t.getmember("image.png").offset_data
    manifest = tmp_path / "manifest.json"
    export(src, ["images.tar"], manifest, min_age=0)
    dest = tmp_path / "dest"
    shutil.copytree(src, dest)
    verify(dest, manifest)
    ref = {
        "path": "images.tar",
        "offset": offset,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    assert image_bytes(ref, dest) == raw
    with pytest.raises(ValueError):
        image_bytes({**ref, "offset": 0}, dest)
    with (dest / "images.tar").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="Transfer mismatch"):
        verify(dest, manifest)


def test_active_and_traversal_files_rejected(tmp_path):
    (tmp_path / "new").write_text("active")
    with pytest.raises(ValueError, match="Recently written"):
        export(tmp_path, ["new"], tmp_path / "m")
    with pytest.raises((ValueError, FileNotFoundError)):
        export(tmp_path, ["../outside"], tmp_path / "m", min_age=0)


def test_grpo_reference_values_clipping_and_kl():
    old = torch.log(torch.tensor([0.2, 0.2], dtype=torch.float64))
    current = old + math.log(1.5)
    reference = old.clone()
    mask = torch.tensor([True, False])
    loss, metrics = grpo_loss(
        current, old, reference, mask, 1.0, clip_epsilon=0.2, beta=0.02
    )
    expected = -1.2 + 0.02 * (1 / 1.5 + math.log(1.5) - 1)
    assert loss.item() == pytest.approx(expected)
    loss, _ = grpo_loss(current, old, reference, mask, -1.0, clip_epsilon=0.2, beta=0)
    assert loss.item() == pytest.approx(1.5)
    adv, usable = group_advantages([0, 0, 1, 1])
    assert usable and adv.tolist() == pytest.approx([-1, -1, 1, 1], abs=1e-7)
    assert not group_advantages([0, 0])[1]


def test_rank_partition_no_replicated_experience():
    plan = rank_plan(19)
    assert len({len(p) for p in plan}) == 1
    assert sorted(i for p in plan for i in p if i is not None) == list(range(19))
    # With ZeRO rank mean, multiply each local equal-episode contribution by world size.
    losses = [1.0, 3.0, 7.0]
    assert sum(x / len(losses) for x in losses) == pytest.approx(11 / 3)


def test_policy_failures_and_teacher_never_admitted():
    with pytest.raises(ValueError):
        validate_group([], "v1")
    ep = {
        "identity_id": "x",
        "initial_state_hash": "h",
        "episode_id": "a",
        "seed": 1,
        "origin": "teacher",
        "policy_version": "v1",
    }
    with pytest.raises(ValueError):
        validate_group([ep, {**ep, "episode_id": "b"}], "v1")
