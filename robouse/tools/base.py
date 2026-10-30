"""Tools dispatch native operations; a dispatch receipt is not physical success."""
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class Tool:
    name: str
    input_schema: dict
    execute: Callable[[dict], Awaitable[Any]]
    retry_safe: bool = False
