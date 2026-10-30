# Teacher collection from an A800 node

An A800 host can coordinate collection and receive verified original data while an RTX host runs Isaac, native tools and the optional cloud teacher CLI. This is the default supported placement interface. The transport/queue/guard contracts are CPU-tested; actual A800↔RTX teacher collection and real new-runtime terminal/processor acceptance remain **NOT_RUN** in this release.

No command here attaches to, updates or restarts an existing research collector. Prepare a new checkout, run directory and data version. Keep an existing collection campaign's account, processes and resource leases unchanged.

## RTX: private collection configuration and service

Copy `configs/astra_teacher.json`, or `configs/astra_teacher_8workers.json` for eight explicitly allocated renderers. Fill original identity/layout hashes, pinned CLI/bwrap paths and hashes, selected model/effort, interpreter/benchmark paths, data root, worker resource leases and shared lock paths. Effort defaults to **high** and remains configurable. Do not infer GPU allocation from device visibility.

For a 2,100-attempt campaign, explicitly set a sufficient `max_episodes`, `max_wall_seconds` and optional `target_valid_scores` in the private server-side teacher config. `max_episodes: 0` in the example intentionally prevents unbudgeted execution. `target_valid_scores` counts native-scored episodes, including valid native policy failures, rather than successes. Exhausted identities below target are reported honestly; no failed episode is replenished automatically.

The service config contains `teacher_config` pointing to that private JSON and its own fresh `output` directory. Model/account/identity/renderer selection stays on the server; the A800 API cannot upload a command or credentials.

```bash
# RTX, within an existing explicit allocation and teacher-usage authorization.
export ASTRA_AUTH_HOME=/path/to/selected-existing-profile
export ASTRA_ACCOUNT_SHA256=EXPECTED_SELECTED_ACCOUNT_FINGERPRINT
export ROBOUSE_ENV_TOKEN='YOUR_SHARED_OUT_OF_BAND_RANDOM_SECRET_24_PLUS_CHARACTERS'
bash scripts/teacher_service.sh --config /path/to/private-teacher-service.json --execute
```

The server binds `127.0.0.1` using the existing authenticated environment HTTP server. Starting the service itself does not call the teacher or initialize Isaac. `teacher_start` launches the independent Guardian and canonical local collector only after the pinned plan and bounded budget match.

## A800: inspect, pin and launch a campaign

```bash
# A800: keep this SSH tunnel running in a separate terminal.
ssh -N -L 8766:127.0.0.1:8766 user@rtx-node
export ROBOUSE_ENV_TOKEN='THE_SAME_OUT_OF_BAND_RANDOM_SECRET'

# A800: inspect configuration only; this is not live quota/entitlement proof.
bash scripts/collect_remote.sh --config configs/astra_a800_remote.json \
  --check-only --output /path/to/a800-runs/plan-review --execute
```

Review the returned model/effort, placement and maximum concurrency. Pin `plan_sha256` in a private copy of `configs/astra_a800_remote.json`; it binds server configuration, identity file, selected account fingerprint and collector code. A changed plan is rejected. The default example identifies a new Seed0 campaign; use a separately frozen Seed1 identity list for validation collection, never mix the roles.

```bash
# A800: actual collection, only after filling the private files and allocation.
bash scripts/collect_remote.sh --config /path/to/a800-collection.json \
  --plan-sha256 REVIEWED_REMOTE_PLAN_SHA256 \
  --max-episodes 2100 --max-wall-seconds 604800 \
  --output /path/to/a800-runs/astra-high-v1 \
  --data-root /path/to/a800-data/astra-high-v1 --execute

# Resume that exact campaign and budget; no duplicate start or reset deadline.
bash scripts/collect_remote.sh --config /path/to/a800-collection.json \
  --plan-sha256 REVIEWED_REMOTE_PLAN_SHA256 \
  --max-episodes 2100 --max-wall-seconds 604800 \
  --output /path/to/a800-runs/astra-high-v1 \
  --data-root /path/to/a800-data/astra-high-v1 \
  --resume /path/to/a800-runs/astra-high-v1 --execute

bash scripts/status.sh --output /path/to/a800-runs/astra-high-v1 --execute
# An active control client forwards this cooperative stop to the remote campaign.
bash scripts/stop.sh --output /path/to/a800-runs/astra-high-v1 --execute
# This also works after the control client disconnects, using its saved ownership.
bash scripts/collect_remote.sh --config /path/to/a800-collection.json \
  --output /path/to/a800-runs/astra-high-v1 --stop-remote --execute
```

A client disconnect does not kill Guardian or existing episodes. An uncertain start cannot automatically spawn a second campaign. A frozen plan can claim one campaign ID; changing the ID does not permit replay of its sealed failures. Resume preserves the original wall deadline, reservations, terminal identities and held lanes. An explicit operator resume may continue an unexpired stopped queue, but cannot clear safety/permission/quota/integrity holds or reopen failed attempts.

Before moving unfinished work from another collector, create a frozen **identity-only** exclusion manifest from a sealed cutoff containing every already-scored or sealed-failed identity:

```json
{"task":"TASK","seed":1,"layout":123,"outcome":"infrastructure_error","physical_aliases":["KNOWN_ALIAS"]}
```

Register its path and SHA256 in the private teacher config's `excluded_identity_files`. Exclusions are immutable on resume, never read teacher targets and retain role/physical-alias ownership checks. Supply a disjoint frozen remaining-identity list; do not point a new A800 campaign at a live writer's queue or claim previously counted episodes as new release results.

## Independent Guardian and observer on the execution host

The service launches Guardian automatically. For a direct RTX-local deployment without the service:

```bash
bash scripts/collect_guardian.sh --config /path/to/private-teacher.json \
  --max-episodes 2100 --max-wall-seconds 604800 \
  --output /path/to/new-run --execute
```

Only the canonical collector dispatches. Guardian holds its own singleton lock, observes local PID/start-tick receipts and permits at most four durable controller starts. It does not kill a model for low instantaneous utilization, reset budgets, rotate accounts, recharge or start training. Transient per-lane bootstrap/resource failures are isolated; healthy lanes still undergo fresh admission and continue. Real global account/permission/quota holds close new admission.

Start the observer separately, with the **actual execution-host campaign output/data paths** (the service places campaigns below its output and teacher data roots):

```bash
# RTX: published artifacts and native terminal evidence only; no model calls.
bash scripts/observe_teacher.sh --output /path/to/teacher-service/campaigns/CAMPAIGN_ID \
  --data-root /path/to/teacher-data/CAMPAIGN_ID \
  --watch --max-wall-seconds 604800 --execute

# Optional per-student processor acceptance; no model weights or training.
bash scripts/observe_teacher.sh --output /path/to/teacher-service/campaigns/CAMPAIGN_ID \
  --data-root /path/to/teacher-data/CAMPAIGN_ID \
  --processor-model /path/to/original-Qwen3.5-2B \
  --processor-revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --tool-schema /path/to/frozen-native-tool-schemas.json \
  --watch --max-wall-seconds 604800 --execute
```

The schema file contains `{"schemas": ...}` for the frozen seven official tools. The processor check profiles a qualifying Seed0/Seed1 success decision using the existing causal converter and exact assistant mask; Seed2 targets are never tokenized. It records the actual processor signature. A 2B PASS does not establish 4B acceptance. It can remain pending while native collection continues.

`teacher-acceptance.json` separates `first_complete_terminal`, `raw_artifact_integrity`, `causal_window_validation` and `processor`. Incremental artifact verification checks newly published index/asset hashes; `--full-verify` rehashes all published artifacts. A processor configuration failure does not falsely label raw native artifacts as corrupt. The run-local [failure journal](SEED1_ROLLOUT_FAILURES.md) retains controlled incidents and Unknown causes.

## Original data transfer

After Guardian/controller/native writers are terminal, the A800 client pulls only the sealed data tree. It transfers original RGB/transcripts/videos/native results with file-relative paths and SHA/byte counts, verifies the mirror, and preserves any original tar bytes/offsets. Active files, login/profile directories, private GT and live logs are not exposed. The original selected authentication remains on the RTX side. Existing export/import/verify scripts remain available for large resumable rsync transfers without `--delete`.

## A800-local Isaac rendering

Local rendering remains unverified. The collector checks the actual GPU name; an A800 requires `workers[].render_acceptance` with a pinned proof path/SHA before any Isaac worker starts. The independent proof must report real native execution, the same GPU UUID and native-source hash, a recent timestamp, the original head/left-wrist/right-wrist images and authoritative native result with SHA references. Synthetic fixture declarations, missing views, stale evidence and UUID/source mismatches are rejected.

This is a fail-closed acceptance boundary, not an installation recipe or a claim that A800 renders Isaac successfully. Use the remote RTX path unless that node has passed its own real native RGB validation. No A800 renderer, real teacher episode or current research-runtime acceptance was executed by the release tests.
