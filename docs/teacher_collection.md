# Optional Astra teacher collection

The isolated teacher collector reuses the original native MCP/CLI episode loop and official grader. It does not connect to an existing research collector or change its code, login, data or GPU allocation. It is an optional integration requiring an externally installed supported Codex CLI; base installation and SFT do not depend on it.

Default settings are `model: gpt-6-astra`, `effort: high`, `service_tier: default`. Both model and effort are configurable in [astra_teacher.json](../configs/astra_teacher.json), or with `--model` / `--effort`; the selected CLI's live catalog must advertise the exact requested values. There is no fixed effort enumeration or silent fallback. The actual raw native transcript must confirm the requested model/effort before being attributed as valid teacher evidence. Catalog metadata does not establish inference entitlement. See the [official app-server description](https://developers.openai.com/siwc/token-sharing-open-source/codex-app-server).

Configure pinned CLI and bwrap binaries, their SHA256/version, frozen identity JSONL, a **new** dataset version/root and explicit RTX worker allocations. Each identity carries task/seed/layout plus original `layout_path` and `layout_sha256`. Optional base-task/physical aliases participate in cross-role checks. Resource allocation evidence contains `allocated`, `owner`, `gpu`, and an actual Unix `expires_at`; its file SHA, owner and lock path are explicit configuration. The collector refuses a GPU with an existing compute process and does not allocate/reset/preempt GPUs. Default concurrency is one; more requires explicit independent RTX allocations and shared worker/account locks.

Select one existing authentication profile through environment variables only:

```bash
export ASTRA_AUTH_HOME=/path/to/selected-existing-codex-home
export ASTRA_ACCOUNT_SHA256=EXPECTED_SELECTED_ACCOUNT_FINGERPRINT
# RTX: non-generative metadata checks only; no thread or model turn.
bash scripts/collect_astra.sh --config /path/to/astra-teacher.json \
  --check-only --output /path/to/runs/astra-check --execute
# RTX: only within an explicitly authorized included-usage budget/allocation.
bash scripts/collect_astra.sh --config /path/to/astra-teacher.json --effort high \
  --max-episodes 8 --max-wall-seconds 86400 --episode-wall-seconds 14400 \
  --output /path/to/runs/astra-teacher-v1 --execute
```

Authentication is read-only mounted, not copied or refreshed. All known windows and weekly quota must have reliable remaining usage strictly greater than 20%; unknown, expired, mismatched or closed quota fails admission. `max_extra_cost_usd: 0` disallows extra-credit fallback, not subscription usage. No account rotation, paid-credit purchase or alternative-model bypass is implemented. `--check-only` performs account/rate-limit/model metadata RPC only, never a model question. Real capability/entitlement and collection are **NOT_RUN** in this release.

Persisted reservations freeze model/effort, identities, CLI/code hashes, episode count and wall deadline before launches. Resume with the same command plus `--resume /path/to/runs/astra-teacher-v1`. A failed or already-started attempt is not automatically replayed and budgets do not reset. `scripts/status.sh` and `scripts/stop.sh` report/request cooperative stop. Process termination checks owned PID/start ticks; unrelated jobs are never killed.

Each seed outputs raw original public RGB, native transcript/events, videos, request/admission proof, official results, terminal reason and hashes. Infrastructure failure and human interruption remain unscored. Only stable complete artifacts are published in `trajectory-index.json`, `terminal.json`, `artifacts.jsonl` and `indexes/episodes.jsonl`; original tar offsets are not altered. Live local control journals use private tmpfs SQLite WAL/FULL and preserve archive bytes.

Create a sealed cutoff without a model or simulator:

```bash
bash scripts/collect_astra.sh --snapshot-only --seed 1 \
  --data-root /path/to/new-teacher-data \
  --snapshot-output /path/to/new-teacher-data/seed-1/snapshots/frozen-v1.jsonl --execute
```

Add this as a `kind: native` source in the preparation config with `sealed_manifest` and `sealed_manifest_sha256`. Only Seed0 enters SFT/RL. Seed1 teacher samples are val NLL only; Seed2 collection/snapshot requires `--final-test` and never feeds training or correction. Old teacher results are never relabeled as a new run or student results.

For a persistent independent Guardian, bounded same-attempt Admission checks, automatic failure journal, observer and A800 control/verified data pull, see [A800 collection deployment](collection_a800.md). These paths do not attach to an existing research collector.
