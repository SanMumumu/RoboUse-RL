"""A CUDA device is not proof of Isaac RGB rendering; A800 needs native evidence."""
import json
from pathlib import Path
import time
from agp_rl.io import sha256
from robouse.environments.robodojo.native_job import verify_sources
from robouse.teacher.admission import FrontierBlocked


def verify_renderer(worker, hardware, *, now=None):
    if "A800" not in hardware["name"].upper():
        return
    reference = worker.get("render_acceptance")
    if not reference or sha256(reference["path"]) != reference["sha256"]:
        raise FrontierBlocked("A800_ISAAC_RGB_NATIVE_ACCEPTANCE_REQUIRED_USE_REMOTE_RTX")
    path = Path(reference["path"]).resolve()
    proof = json.loads(path.read_text())
    now = time.time() if now is None else now
    if (proof.get("status") != "PASS" or proof.get("real_native_execution") is not True
            or proof.get("synthetic_fixture") is not False or proof.get("gpu_uuid") != hardware["uuid"]
            or proof.get("native_source_sha256") != verify_sources()
            or not 0 <= now - proof.get("observed_at", 0) <= 7 * 86400):
        raise FrontierBlocked("A800_RENDER_ACCEPTANCE_STALE_OR_MISMATCHED")
    views = proof.get("views", {})
    if set(views) != {"head", "left_wrist", "right_wrist"}:
        raise FrontierBlocked("THREE_ORIGINAL_NATIVE_RGB_VIEWS_REQUIRED")
    from PIL import Image
    for view in views.values():
        image = (path.parent / view["path"]).resolve()
        if not image.is_relative_to(path.parent) or sha256(image) != view["sha256"]:
            raise FrontierBlocked("A800_RENDER_RGB_EVIDENCE_CHANGED")
        with Image.open(image) as frame:
            frame.load()
            if min(frame.size) < 32:
                raise FrontierBlocked("A800_RENDER_RGB_EVIDENCE_INVALID")
    native = proof["native_result"]
    result = (path.parent / native["path"]).resolve()
    if not result.is_relative_to(path.parent) or sha256(result) != native["sha256"]:
        raise FrontierBlocked("A800_RENDER_NATIVE_EVIDENCE_CHANGED")
    from agp_rl.reward.official import read_official
    read_official(result, native["layout_id"])
