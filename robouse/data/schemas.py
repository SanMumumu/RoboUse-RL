"""Model-independent rollout records. Paths refer to external, hash-bound assets."""
from dataclasses import asdict, dataclass, field
from typing import Any
from robouse.data.datasets import role


@dataclass(frozen=True)
class Identity:
    episode_id: str
    attempt_id: str
    source_id: str
    task: str
    seed: int
    initial_state: str
    role: str

    def __post_init__(self):
        if self.role != role(self.seed):
            raise ValueError("Seed role mismatch")
        if not all((self.episode_id, self.attempt_id, self.source_id, self.task, self.initial_state)):
            raise ValueError("Complete source/attempt/environment identity required")


@dataclass
class Observation:
    rgb: dict[str, dict[str, Any]]
    instruction: str
    depth: dict[str, dict[str, Any]] = field(default_factory=dict)
    feedback: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.rgb or not self.instruction:
            raise ValueError("Original RGB and language instruction required")
        for ref in [*self.rgb.values(), *self.depth.values()]:
            if not ref.get("path") or len(ref.get("sha256", "")) != 64:
                raise ValueError("External asset path and SHA256 required")


@dataclass
class Decision:
    observation: Observation
    context: list[dict[str, Any]]
    assistant: str
    tool_call: dict[str, Any] | None
    tool_result: dict[str, Any] | None
    policy_version: str
    input_history_complete: bool = True
    generation: dict[str, Any] | None = None


@dataclass
class Trajectory:
    identity: Identity
    decisions: list[Decision]
    reward: float | None
    success: bool | None
    terminated: bool
    truncated: bool
    termination_reason: str
    origin: str
    scope: str = "full_trajectory"
    schema_version: str = "robouse-rollout-v1"

    def __post_init__(self):
        if self.origin not in ("teacher", "student_on_policy", "synthetic_fixture"):
            raise ValueError("Explicit teacher/student/fixture origin required")
        if self.scope not in ("full_trajectory", "qualified_local_decisions"):
            raise ValueError("Unknown trajectory scope")
        if self.success is not None and not self.terminated:
            raise ValueError("Success requires a native terminal verdict")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        record = dict(value)
        record["identity"] = Identity(**record["identity"])
        record["decisions"] = [
            Decision(**{**d, "observation": Observation(**d["observation"])})
            for d in record["decisions"]
        ]
        return cls(**record)
