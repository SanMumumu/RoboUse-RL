"""Native success and observable cost accounting. Unscored attempts stay unscored."""
from collections import Counter
from statistics import mean
from agp_rl.reward.official import read_official


def _mean(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def evaluate_metrics(episodes, *, requested_count=None):
    scored, records = [], []
    for episode in episodes:
        native = episode.get("native_reward")
        if native:
            verified = read_official(native["evidence_path"], native["layout_id"]).record()
            if verified != native or episode.get("reward") != float(verified["success"]):
                raise ValueError("Native grading evidence differs")
            scored.append(verified)
        elif episode.get("reward") is not None:
            raise ValueError("Cannot score without native grader evidence")
        timeline = episode.get("timeline", [])
        counts = [t.get("native_tool_calls") for t in timeline]
        calls = sum(counts) if counts and all(c is not None for c in counts) else None
        decisions = episode.get("decisions", [])
        records.append({
            "episode_id": episode["episode_id"], "identity": episode.get("identity"),
            "seed": episode.get("seed"), "outcome": episode.get("outcome"),
            "termination_reason": episode.get("termination_reason", "unknown"),
            "terminated": bool(native), "truncated": episode.get("outcome") == "timeout" or episode.get("truncated", False),
            "native_success": native.get("success") if native else None,
            "native_score": native.get("score") if native else None,
            "native_tool_calls": calls,
            "policy_decisions": len(timeline),
            "decision_latency_seconds": _mean([t.get("decision_seconds") for t in timeline]),
            "tool_dispatch_roundtrip_seconds": _mean([t.get("tool_seconds") for t in timeline]),
            "tool_execution_latency_seconds": _mean([t.get("native_tool_seconds") for t in timeline]),
            "episode_latency_seconds": episode.get("episode_seconds"),
            "token_usage": {"prompt": sum(len(d["input_ids"]) for d in decisions), "generated": sum(len(d["generated_ids"]) for d in decisions)} if decisions else None,
            "versions": episode.get("versions", {}), "videos": episode.get("videos", {}),
            "failure_category": None if native and native["success"] else episode.get("outcome", "unknown"),
        })
    return {
        "schema": "robouse-evaluation-v1", "requested_episodes": requested_count,
        "attempted_episodes": len(episodes), "scored_episodes": len(scored),
        "unscored_episodes": len(episodes) - len(scored),
        "native_task_success_rate": _mean([float(s["success"]) for s in scored]),
        "successes": sum(bool(s["success"]) for s in scored),
        "failure_categories": dict(Counter(r["failure_category"] for r in records if r["failure_category"])),
        "tool_calls_per_episode": _mean([r["native_tool_calls"] for r in records]),
        "decision_latency_seconds": _mean([r["decision_latency_seconds"] for r in records]),
        "tool_execution_latency_seconds": _mean([r["tool_execution_latency_seconds"] for r in records]),
        "tool_dispatch_roundtrip_seconds": _mean([r["tool_dispatch_roundtrip_seconds"] for r in records]),
        "episode_latency_seconds": _mean([r["episode_latency_seconds"] for r in records]),
        "episodes": records,
    }
