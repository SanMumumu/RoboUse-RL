"""Policy backend boundary: context and original RGB in, sampled tool text out."""
from typing import Protocol


class AgentPolicy(Protocol):
    policy_version: str

    async def infer(self, messages: list[dict], images: dict, root: str) -> dict:
        """Return actual inputs, generated IDs, behavior LP, mask and policy identity."""
        ...
