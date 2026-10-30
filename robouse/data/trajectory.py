"""Public trajectory reader and causal SFT-window projection; no target repair."""
import json
from pathlib import Path
from agp_rl.io import digest
from robouse.data.schemas import Trajectory


def load_trajectory(path):
    return Trajectory.from_dict(json.loads(Path(path).read_text()))


def decision_windows(trajectory):
    """Keep independently valid decisions even when a later history is incomplete."""
    if trajectory.origin != "synthetic_fixture" and (trajectory.origin != "teacher" or trajectory.success is not True):
        return
    identity = trajectory.identity
    scope = "qualified_local_decisions" if any(not d.input_history_complete for d in trajectory.decisions) else trajectory.scope
    for index, decision in enumerate(trajectory.decisions):
        if not decision.input_history_complete or not decision.assistant:
            continue
        if not decision.context or decision.context[-1]["role"] == "assistant":
            raise ValueError("Verified pre-decision context required; do not invent history")
        messages = decision.context + [{"role": "assistant", "content": [{"type": "text", "text": decision.assistant}]}]
        yield {
            "window_id": digest({"identity": identity.__dict__, "decision": index}),
            "identity_id": f"{identity.task}/seed-{identity.seed}/{identity.initial_state}",
            "episode_id": identity.episode_id, "attempt_id": identity.attempt_id,
            "source_id": identity.source_id, "source": identity.source_id,
            "seed": identity.seed, "role": identity.role, "scope": scope,
            "origin": trajectory.origin, "messages": messages,
            "images": decision.observation.rgb,
            "tool_call": decision.tool_call, "tool_result": decision.tool_result,
        }
