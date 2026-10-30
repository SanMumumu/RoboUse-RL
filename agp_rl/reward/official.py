"""Read authoritative RoboDojo _result.json details after evaluator publication.

Do not read assess, episode_metrics, or simulator private state as a reward.
Per-episode score is 0..1; official aggregate score is 0..100.
"""
from dataclasses import dataclass, asdict
import math
from pathlib import Path

from agp_rl.io import read_json, sha256


@dataclass(frozen=True)
class OfficialReward:
    layout_id: int
    success: bool
    score: float
    evidence_path: str
    evidence_sha256: str
    detail_index: str

    def training_value(self, mode="success"):
        if mode == "success":
            return float(self.success)
        if mode == "official_score":
            return self.score
        raise ValueError("Unknown reward; reward shaping is outside V1")

    def record(self):
        return asdict(self)


def read_official(path, layout_id):
    path = Path(path).resolve()
    if path.name != "_result.json":
        raise ValueError("Reward requires official _result.json")
    data = read_json(path)
    details = data["details"]
    if not isinstance(details, dict) or not details:
        raise ValueError("No completed official episodes")
    found, successes, score_sum, layouts = [], 0, 0.0, set()
    for index, detail in details.items():
        layout, success, score = detail["layout_id"], detail["success"], detail["score"]
        if type(layout) is not int or type(success) is not bool or type(score) not in (int, float):
            raise ValueError("Malformed official detail types")
        if not math.isfinite(score) or not 0 <= score <= 1 or (success and score != 1):
            raise ValueError("Invalid official per-episode score")
        if layout in layouts:
            raise ValueError("Duplicate layout in official results")
        layouts.add(layout)
        successes += success
        score_sum += score
        if layout == layout_id:
            found.append(OfficialReward(layout, success, float(score), str(path), sha256(path), str(index)))
    n = len(details)
    if (type(data.get("eval_time")) is not int or data["eval_time"] != n
            or not math.isclose(data["success_rate"], successes / n, abs_tol=1e-9)
            or not math.isclose(data["score"], score_sum / n * 100, abs_tol=1e-7)):
        raise ValueError("Official aggregate/details mismatch")
    if len(found) != 1:
        raise ValueError("Requested layout has no unique completed official result")
    return found[0]
