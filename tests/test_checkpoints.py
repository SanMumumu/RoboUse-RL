import json
from pathlib import Path
import pytest
from robouse.checkpoints import commit, verify


def test_complete_eight_rank_marker_and_corruption_rejection(tmp_path):
    (tmp_path / "trainer_state.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"synthetic-marker-fixture")
    step = tmp_path / "global_step1"
    step.mkdir()
    for rank in range(8):
        (step / f"zero_rank_{rank}_optim_states.pt").write_bytes(b"fixture")
        (tmp_path / f"rng_state_{rank}.pth").write_bytes(b"fixture")
    commit(tmp_path, 8)
    assert verify(tmp_path)["world_size"] == 8
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify(tmp_path)


def test_partial_eight_rank_checkpoint_is_not_committed(tmp_path):
    (tmp_path / "trainer_state.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="ZeRO"):
        commit(tmp_path, 8)
