import json
from pathlib import Path
import zipfile
import pytest
from robouse.view import build
from robouse.data.datasets import check_roles
from robouse.roles import normalize


def test_offline_bundle_keeps_success_and_failure_and_three_original_views(tmp_path):
    episodes = []
    for i in range(2):
        videos = {}
        for camera in ("head", "left_wrist", "right_wrist"):
            name = f"{i}-{camera}.mp4"
            (tmp_path / name).write_bytes(b"synthetic-video-fixture")
            videos[camera] = name
        episodes.append(
            {
                "episode_id": str(i),
                "policy_version": "new-fixture",
                "score": i,
                "termination_reason": "fixture",
                "timeline": [{"turn": 0, "assistant": "<script>x</script>"}],
                "videos": videos,
            }
        )
    (tmp_path / "episodes.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in episodes)
    )
    build(tmp_path)
    page = (tmp_path / "index.html").read_text()
    assert page.count("<video") == 3 and "\\u003cscript\\u003e" in page
    with zipfile.ZipFile(tmp_path / "offline.zip") as z:
        assert len(z.namelist()) == 8


def test_failed_environment_identity_and_known_physical_alias_cannot_leak():
    rows = [
        {
            "seed": 0,
            "role": "train",
            "identity_id": "a",
            "physical_aliases": ["physical-1"],
        },
        {
            "seed": 2,
            "role": "test",
            "identity_id": "b",
            "physical_aliases": ["physical-1"],
        },
    ]
    with pytest.raises(ValueError, match="Cross-role"):
        check_roles(rows)


def test_unscored_infrastructure_attempt_is_viewable_without_fake_video(tmp_path):
    (tmp_path / "episodes.jsonl").write_text(json.dumps({"episode_id": "synthetic-infra", "outcome": "infrastructure_error", "reward": None, "policy_version": "synthetic-policy", "termination_reason": "synthetic-transport-failure"}) + "\n")
    build(tmp_path)
    page = (tmp_path / "index.html").read_text()
    assert '"videos": {}' in page and '"missing_views": ["head", "left_wrist", "right_wrist"]' in page
    with zipfile.ZipFile(tmp_path / "offline.zip") as archive:
        assert archive.namelist() == ["index.html", "episodes.jsonl"]
