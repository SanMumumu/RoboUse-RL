# Data and split protocol

`robouse-rollout-v1` is a model-independent public trajectory record. See the complete, runnable [synthetic fixture](../examples/trajectory_sft/fixture/trajectory-seed0.json) and `robouse.data.schemas`.

A trajectory contains:

- `identity`: original episode, attempt, source, task, seed, initial state and role.
- `decisions`: RGB references, optional depth references, instruction, pre-decision visible context, original assistant output, structured tool call/result and environment feedback. Policy version and optional generation evidence are explicit.
- `reward`, `success`: native results or null for unscored attempts.
- `terminated`, `truncated`, `termination_reason`: terminal and resource-bound outcomes.
- `origin`: `teacher`, `student_on_policy` or `synthetic_fixture`.
- `scope`: `full_trajectory` or `qualified_local_decisions`.

The complete trajectory and each local window have separate quality decisions. `decision_windows()` keeps decisions with verifiable complete input history and an original target. A missing later segment does not erase an earlier valid window. Conversion never fabricates missing history or hidden reasoning. Qualified teacher windows must come from native-success episodes; synthetic public fixtures are explicitly unscored and are not experiment evidence.

Prepared JSONL rows use `window_id`, `identity_id`, `seed`, `role`, `messages`, `images`, source/attempt provenance and `scope`. The last message is the **only** assistant target. Prior assistant turns, user observations and tool feedback are context, not loss targets. Original-data adapters also retain teacher model/effort and known physical aliases. Unknown legacy attempt metadata remains null rather than being invented.

Images are referenced by SHA256 keys (`art-image://<sha>`). Each asset records a root-relative `path`, original byte count and SHA256. Tar-backed assets additionally specify the **original tar** byte offset. Never repack a tar and reuse its offsets. `DATA_ROOT` or `--data-root` relocates assets without changing the archived byte identity. The public fixture uses tiny ASCII PPM RGB files so it needs no binary Git payload or private data.

## Fixed roles

| Seed | Role | Allowed use |
|---|---|---|
| 0 | train | SFT gradients; student RL identities |
| 1 | val | Sealed teacher NLL; native identity-complete validation and checkpoint selection |
| 2 | test | Explicit final native test only |

The identity registry includes failed episodes and known physical aliases, not just teacher successes. Cross-role identities, retries, derived trajectory/content hashes and known physical aliases are rejected. Freeze role manifests before preparing a new dataset version; old mixed-seed caches are not admissible. Start SFT from original HF weights, not a model trained with Seed1 gradients.

Seed2 targets are not opened by SFT preparation or training. `--test-data` registers a protected path only. Missing Seed2 remains `PENDING_TEST`; another seed is never substituted. Native validation/test enumerate the requested environment identities, including teacher-failure scenarios. Offline teacher val NLL is not native success evaluation.

## Multiple sources and sampling

A single `--train-data` / `--val-data` path is supported. Alternatively use:

```yaml
datasets:
  train:
    - {name: teacher_a, path: /path/to/seed0-a.jsonl, weight: 3.0}
    - {name: teacher_b, path: /path/to/seed0-b.jsonl, weight: 1.0}
  val:
    - {name: heldout, path: /path/to/seed1-val.jsonl}
  test:
    - {path: /path/to/sealed-seed2.jsonl}  # registered only; unopened by SFT
sampling_strategy: weighted_replacement
samples_per_epoch: 10000
seed: 0
```

`coverage` (default) visits every training window once per epoch and applies deterministic length bucketing. Source weights affect `weighted_replacement`, which requires an explicit epoch sample count: a source's total sampling mass equals its weight. Repeated draws are intentional training samples, not new episodes. Duplicate window IDs across sources are rejected. The frozen order, source file hashes, weights, split manifest and seed are saved in the training contract.

## Raw preparation and migration

Edit [configs/data.json](../configs/data.json) with root-relative source paths, identity manifests, frozen native tool schemas and sealed snapshot hashes. Use a **new output directory**:

```bash
bash scripts/prepare_data.sh --config /path/to/data.json \
  --data-root /path/to/raw --output /path/to/raw/prepared-v1 --execute
bash scripts/export_data.sh --data-root /path/to/raw \
  --files /path/to/raw/prepared-v1/required.files \
  --manifest /path/to/raw/transfer.json --execute
# Destination: copy transfer.json first, then resume rsync without --delete.
bash scripts/import_data.sh --source user@source:/path/to/raw \
  --data-root /path/to/new-root --manifest /path/to/transfer.json --execute
bash scripts/verify_data.sh --data-root /path/to/new-root \
  --manifest /path/to/transfer.json --execute
```

`required.files`, `required-images.files` and `prepared.files` separate dependencies and prepared metadata. The transfer manifest contains relative paths, per-file bytes/SHA and total bytes. Export refuses recently written/changing files; choose immutable snapshots. `rsync --partial --append-verify` supports resumption; no deletion synchronization is used. Migrate model weights separately and never through Git. Raw model-input RGB is migrated directly; video frames are not substitutes.

Processor profiles are model-specific. A 2B profile/cache is not a 4B acceptance result; preparation records the processor/template signature and training checks it against the actual processor.
