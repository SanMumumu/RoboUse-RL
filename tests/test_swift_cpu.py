"""Exercise the actual pinned ms-swift scheduler/protocol on CPU."""
import asyncio
from types import SimpleNamespace
import pytest


pytest.importorskip("pydantic", minversion="2.0")
from robouse.scheduler import swift_scheduler_class


def test_swift_explicit_ids_masks_and_recorded_logprobs():
    class Engine:
        async def robouse_episode(self, *args):
            return {
                "reward": 1,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "actual student output"}],
                    }
                ],
                "policy_version": "fixture",
                "decisions": [
                    {
                        "generated_ids": [4, 7],
                        "loss_mask": [True, True],
                        "behavior_logprobs": [-0.2, -0.4],
                        "input_ids": [1, 2, 3],
                    }
                ],
            }

    scheduler = swift_scheduler_class()(Engine(), max_turns=2)
    value = asyncio.run(scheduler.run(SimpleNamespace(data_dict={"robouse": {}}), None))
    assert value.response_token_ids == [[4, 7]]
    assert value.rollout_logprobs == [[-0.2, -0.4]]
    assert value.response_loss_mask == [[1, 1]]
