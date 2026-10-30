# Architecture

RoboUse-RL trains a multimodal **tool-using agent**, rather than a continuous action predictor. The model emits a tool-call envelope or a bounded JavaScript program using registered robot tools. The harness validates and dispatches it. Only the native controller issues physical control signals, and only the native grader establishes task success.

![RoboUse-RL framework](assets/framework.svg)

| Boundary | Implementation | Input → output | Dependencies |
|---|---|---|---|
| Policy | `agents/base.py`, `agents/qwen_agent.py` | Causal messages + original RGB → actual generated IDs/text, mask, behavior log probabilities | Qwen processor and `models/qwen.py` |
| Model | `models/qwen.py` | Original model or committed checkpoint → multimodal model and processor | Hugging Face Transformers; pinned revision |
| Context | `harness/context.py` | Recorded visible messages → shared causal context | No hidden reasoning or target synthesis |
| Memory | `harness/memory.py`, `scheduler.py` | Episode history + call/decision identity → context and replay receipts | Local memory; durable JSON/tensor journal for online episodes |
| Tools | `tools/base.py`, `harness/tool_registry.py` | Exact name/arguments + frozen JSON Schema → validated call | Existing `agp_rl.adapters.tool_codec` |
| Execution | `harness/executor.py` | Validated call → dispatch/error receipt | Explicit native callback; retries require an idempotent tool |
| Environment | `environments/base.py`, `environments/robodojo/` | Reset/step/close → original RGB, visible feedback, native score evidence | External RoboDojo, XPolicyLab, Isaac, MCP, Node.js |
| Data | `data/schemas.py`, `trajectory.py`, `datasets.py`, `sampling.py` | Original trajectories and identity manifests → qualified decision windows and deterministic mixtures | Selected original source adapters; no model-specific encoding in loader |
| SFT | `training/sft/` | Seed0 windows + Seed1 validation → full HF/ZeRO checkpoint | HF Trainer + Accelerate + DeepSpeed |
| RL | `training/rl/`, `scheduler.py`, `collect.py` | Independent comparable student episodes → qualified GRPO updates | Extracted ms-swift scheduler; immutable sampler evidence |
| Evaluation | `evaluation/` | All requested environment identities + checkpoint → native metrics, evidence, videos | Same student rollout and native grader |
| Teacher | `teacher/` | Frozen identities + explicit selected CLI/account/budget → sealed native teacher trajectories | External supported Codex CLI, read-only selected authentication, RTX simulator |

The small `agp_rl/` package contains retained converters, tool codecs, the official reward reader and checkpoint contracts. It is deliberately retained where extraction preserves already tested behavior. `_native/` is a byte-pinned AGP tool/control snapshot; `_vendor/` contains attributed Apache-2.0 scheduler/protocol code. Neither is a complete simulator distribution.

## Adding a policy or environment

Implement `AgentPolicy.infer(messages, images, root)` and preserve exact input tensors, generated IDs, behavior LP, loss mask and policy identity. A backend that only returns regenerated text is insufficient for on-policy RL. `QwenAgent` is used by the production `HfEngine`; it shares SFT's processor and context contract.

Implement the asynchronous `Environment` protocol and the observation/artifact transport used by `RemoteEnvironment`. Return native terminal evidence separately from visible feedback. Implement reset comparability and request idempotency before using grouped RL. The existing RoboDojo adapter runs the official evaluator unchanged and restricts it to a selected task/seed/layout.

`ToolRegistry` accepts environment-specific tools; `official_robot=True` additionally enforces the seven exact frozen RoboDojo tools. `HarnessExecutor` is used by the native student actor. A dispatch receipt always leaves `physical_success=None`: tool completion, robot motion and task success are distinct observations. Generic executor receipts are episode-local; production action/decision replay is durable in the scheduler and remote worker mailboxes.

## Context and recovery

The shared native v4 context normalization removes only known transport headers from recorded tool feedback; it preserves image references and assistant targets. The online scheduler persists an actual sampled decision before sending an action. Recovery restores that decision and RNG rather than sampling again. Tool errors remain visible to the policy; no teacher correction, invented history or model-output repair is inserted.
