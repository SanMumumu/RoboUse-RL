## Release RoboUse-RL v0.1.0

RoboUse-RL extracts the working tool-using Agent learning path from a private research workspace into an installable modular package. Policy/model, causal context/memory, schema-bound tools, native environment/controller integration, data preparation, SFT, experimental GRPO and native evaluation have explicit responsibilities. The repository contains selected reusable code and a public synthetic RGB fixture, rather than a research-directory dump.

This release retains the HF Trainer/DeepSpeed full-parameter SFT path and original processor/assistant-mask/checkpoint contracts. Data roles are fixed to Seed0 train, Seed1 validation and explicitly gated Seed2 final test. The public CLI supports training configuration, multiple weighted sources, deterministic sampling, original HF/local model identity and resume. Native evaluation records original score evidence, observable timing/token/identity/version metadata and three-view videos where available.

Online GRPO is Experimental. CPU reference tests cover group-relative advantages, clipped objective/reference KL, variable episode weighting, global normalization and immutable sampling evidence. Actual A800 sampler/oracle parity and RoboDojo closed-loop operation have not been run for this release. Execution requires explicit experimental opt-in and preserves the 0.02 behavior alignment gate; no mock data or teacher episode is substituted for on-policy student data.

### Validation

- Independent editable installation plus declared CPU test extras completed; **108 CPU tests passed**, including original local 2B processor/RGB/mask validation.
- Actual tiny random-model HF CPU optimizer/checkpoint/RNG resume is distinct from GPU acceptance.
- Eight actual CPU/Gloo ranks passed independent references: SFT gradient error 2.78e-17; GRPO 6.94e-18. Wheel/native-source verification and indexed release scans passed.
- GitHub Actions CPU workflow is provided; hosted status is separate from local results.
- A800 2B/4B smoke, full GPU SFT, actual Online GRPO, Isaac/native validation/test, real teacher calls and new-policy videos: **NOT_RUN**.

### Known limitations

Only full fine-tuning is released; PEFT, additional policy/environment backends and ROS are Planned. The reference sampler has no KV cache and is not throughput-optimized. Native operation depends on external RoboDojo/XPolicyLab/Isaac/assets and explicit resource allocations. Optional teacher collection requires an external supported CLI and selected-account admission. No benchmark or SOTA claims are made.

Original research history, active jobs and data remain untouched. The release branch preserves the target main commit, with prior export history backed up outside the public repository. Source/data licenses and third-party attribution are retained; private data, credentials, weights, logs and server paths are excluded.

### Collection follow-up

Adds the portable pre-model Admission policy from the research runtime: at most three typed metadata checks in one unfinished attempt, no blind retry for 401/403/429/permission/quota/IO, same-identity positive prior for unknown internal RPC recovery, and immutable sealed failure exclusions. A lane-local bootstrap fault is isolated while other admitted lanes continue.

Adds an independent bounded Guardian and credential-free failure spool/Markdown recorder. Adds the A800 control interface over authenticated loopback/SSH to an RTX teacher service, pinned server plans, idempotent campaign ownership/resume and verified original-data transfer after writers terminate. A800-native Isaac is gated on separate real GPU/native/RGB evidence.

The observer separates native terminal publication, raw artifact verification, causal-window qualification and opt-in local per-model processor acceptance. Runtime first-terminal/processor completion is never inferred from process startup or synthetic tests. Real teacher/A800/Isaac validation remains NOT_RUN; no active research collector or existing account/GPU lease was modified.
