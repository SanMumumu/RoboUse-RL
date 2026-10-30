# v0.1.0 migration audit

This is the engineering migration inventory. It distinguishes source reuse from runtime validation; it is not an experiment report. Detailed source-relative paths and SHA comparisons are retained locally in `agent/evidence/migration-inventory.json` and are excluded from Git.

| Research area | Release action | Retained/refactored implementation | Verified boundary | External dependencies |
|---|---|---|---|---|
| `agent/sft_chain_20261029/code/` and existing fullft | Reuse selected converters, losses, checkpoint contracts; refactor release Trainer orchestration | `agp_rl/pipeline2b/fullft/`, `training/sft/` | CPU HF optimization/resume and processor tests; A800 NOT_RUN | Transformers, Accelerate, DeepSpeed |
| Teacher native collection | Reuse admission, journal, preload, native wait and grading; isolate configuration and queue | `teacher/`, `environments/robodojo/` | Fake RPC/worker and synthetic seal/resume only; real calls NOT_RUN | Selected external Codex CLI, bwrap, explicit account/quota/allocation, Isaac |
| Original trajectory processing | Reuse ART/native adapters and raw image/tar verification; expose trajectory schema and weighted dataset interface | `data/`, `agp_rl/data/` | Synthetic ART/tar/native relocation and causal-window checks | PyArrow for ART/parquet, original RGB assets |
| Qwen3.5 | Reuse processor/template/termination and exact generated-token sampler | `agents/qwen_agent.py`, `models/qwen.py` | Actual local 2B processor; 2B/4B GPU training NOT_RUN | Original HF checkpoint/revision |
| GRPO | Keep multi-round scheduler, immutable behavior LP, prefix oracle, frozen backend correction, reference KL, durable policy updates | `training/rl/`, `scheduler.py`, `collect.py` | CPU numeric references; original research raw LP reconstruction failures acknowledged; release A800 parity NOT_RUN | Eight GPU ranks + native student episodes |
| Tool schemas/execution | Keep strict official seven-tool codec and sandboxed JS; expose reusable registry/executor | `tools/`, `harness/`, selected `agp_rl/adapters/` | CPU schema/error/idempotency tests | MCP and Node.js for native tool/JS execution |
| Context/memory | Keep full-causal native v4 normalization and durable sampled-decision replay | `harness/context.py`, `scheduler.py` | CPU causal and pending-sample/RNG tests | Original visible context; no hidden reasoning |
| RoboDojo bridge/controller | Keep 64 byte-pinned official AGP files; move maintained launch/transport code into adapter package | `environments/robodojo/`, `_native/` | Inventory and CPU ownership/transport tests only | External RoboDojo/XPolicyLab/Isaac/assets |
| Evaluation/grading/videos | Reuse official result reader and original videos; unify metrics/provenance/output names | `evaluation/`, `view.py` | Synthetic native-result validation and offline contracts; actual new-policy video NOT_RUN | Native grader and supported cameras |
| Checkpoints | Reuse full optimizer/RNG commit/checksum contracts; force final checkpoint publication | `checkpoints.py`, `training/sft/trainer.py` | Real tiny-model CPU save/load/exact resume | ZeRO GPU resume NOT_RUN |
| Reference schedulers | Extract attributed ms-swift 4.2.1 scheduler/protocol; consult Guava, AgenticRoboRL and ResearchBoost-Agent | `_vendor/`, provenance.json | CPU lifecycle tests | Apache-2.0 retained; no entire ThirdGIT |
| Legacy experiment scripts | Do not migrate host/account-specific launchers, live observers, experiment results or custom optimizers | Public scripts/configs replace launch configuration only | No source/live job changed | Private historical evidence remains outside release |
| Historical LoRA | Not released as a supported mode | Explicit unsupported-mode diagnostic | Historical research validation is not release validation | PEFT integration Planned |

## starVLA comparison

Audited the current `starVLA_dev` checkout at `985c2238627481f25476fbee478083812dd82a21` through the actual repository tree, training YAML, RoboDojo recipe/launcher, packaging and benchmark documentation. References: [source tree](https://github.com/starVLA/starVLA/tree/985c2238627481f25476fbee478083812dd82a21), [training config](https://github.com/starVLA/starVLA/blob/985c2238627481f25476fbee478083812dd82a21/starVLA/config/training/starvla_cotrain_libero.yaml), [RoboDojo guide](https://github.com/starVLA/starVLA/blob/985c2238627481f25476fbee478083812dd82a21/examples/simBenchmarks/RoboDojo/README.md).

Adopted explicit module responsibilities, model-independent loader outputs, config-driven launchers, separate benchmark environments and documented training/evaluation recipes. RoboUse-RL uses native tool-call decisions and causal Agent trajectories; no VLA action head, normalization/action chunk policy or unrelated StarVLA source is copied. Its benchmark claims and released-model scores are not RoboUse-RL results.

## Exclusions

No active rollout tree, mixed Seed0/1 cache, Seed1-trained initialization, model weights, raw datasets, source-project `.git`, private logs/account state, GPU leases, entire ThirdGIT checkout or old videos are copied into the public commit. Existing source and experiments remain recoverable. Public synthetic fixtures are marked synthetic and carry no native success claim.

## Collection follow-up audit

Read-only review additionally covered the research `runtime_v14/rpc_policy.py`, bounded Admission wrapper, failure recorder and independent v20 Guardian. The credential-free classifier and subprocess fixture were reused directly; host-bound paths, account ordering, proxy functions, state paths, leases and live progress were not copied. The public wrapper preserves prior/identity/write checks, the three-attempt bound and no episode replay. Guardian, failure spool, native/processor observer and A800↔RTX service are portable additions around existing collection/HTTP/transfer primitives. The running source collector was not imported, patched, restarted or sampled by this work. Source-reported rollout progress is not a release experiment result.
