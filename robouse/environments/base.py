"""Environment feedback must preserve native grading and unscored failures."""
from typing import Protocol


class Environment(Protocol):
    episode_id: str

    async def reset(self, identity: dict, group_id: str) -> dict: ...
    async def step(self, index: int, text: str, finish: bool = False) -> dict: ...
    async def close(self) -> dict: ...
