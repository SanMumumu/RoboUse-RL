# Collection failure knowledge base

This portable guide is derived from the research Seed1 collector's observed failure classes and bounded recovery policy. It contains no private run paths, account assignments, credentials or claimed release benchmark results. The title preserves the Seed1 operational origin; the same controls apply to Seed0 collection. Seed1 remains validation-only.

## Admission boundaries

- Metadata recovery is **within the same unfinished pre-model attempt**, at most three total checks with 2/5-second backoff. It is not a model, tool, action or episode retry.
- A `-32603` internal error is not a diagnosis of proxy/VPN failure. Retrying that unknown error requires a fresh verified prior for the same selected account, credential fingerprint, model/effort and quota, plus current identity/write checks.
- HTTP 401/403/429, permission, quota, login/identity and local storage failures do not receive blind retries. Unknown quota remains closed for model launches. No account rotation, credits or model fallback is performed.
- Once a model-start intent/PID/events exists, the prestart recovery path cannot repeat it. Sealed failed identities remain terminal. Frozen exclusion manifests protect failures imported from an earlier data version.
- Guardian supervises only the canonical collector. It never calls a model, changes native grading, resets budgets or relaunches sealed episodes. Four controller starts is the maximum persistent budget; each metadata check separately has a hard maximum of three attempts.

## Failure classes

| Class | Interpretation and minimum action | Automatic behavior |
|---|---|---|
| `account/read` or `model/list` RPC `-32603` | Root cause Unknown. Preserve typed RPC evidence; verify the same-account prior and local writes | At most three prestart metadata checks; no inference probe |
| Metadata timeout / known transport fault | A metadata transport fault, not proof of permission or quota | Bounded read-only checks while identity/storage remain valid; no proxy reconfiguration |
| 401 / 403 / 429 / permission / quota | Actual provider/account hold | Stop new model admission immediately; existing in-flight episodes are not killed merely because quota closes |
| Read-only filesystem / lost writable mount | Host write success does not prove the simulator namespace remains writable | Hold the affected path/lane; preserve evidence. No remount, chmod, global restart or proxy workaround |
| Stale witness / stale observation | An old heartbeat alone does not prove process/GPU/storage failure | Check owned PID+start ticks, terminal state and positive current evidence. No idle-utilization-based kill |
| One lane's bootstrap/resource failure | Other lanes can still provide fresh admission/resource proof | Persist the failed lane hold and failed identity; healthy lanes continue. All lanes held is an explicit stop, not fake progress |
| Raw artifact / native-score mismatch | Data integrity is unproven | `RAW_INTEGRITY_HOLD.json` closes new dispatch. Do not rewrite a score or erase the original record |
| Processor acceptance pending/failed | Distinct from native terminal and artifact integrity | Record it separately; never relabel a processor configuration error as corrupted native data |
| A800 CUDA available but renderer unproven | CUDA availability does not establish Isaac RGB/Vulkan support | Refuse local rendering without hash-bound real native/camera evidence; use remote RTX |

The public implementation does not copy host-specific proxy functions, credentials, GPU leases or account ordering from the research collector. Recovery of a real namespace/mount or persistent service fault requires independent positive evidence; a short CPU test is not a production acceptance result.

## Automatic recorder

`failure-spool/*.json` contains controlled classifications and typed method/code/status/attempt fields. Raw provider bodies and argv/auth responses are never retained by this recorder. Guardian idempotently folds the spool into the **run-local** `SEED1_ROLLOUT_FAILURES.md` under a file lock and fsync/atomic publication. The checked-in guide is not modified by a run. A failed document write leaves the original spool intact.

Each entry includes UTC, classification, stage, attempt, decision, root cause `Unknown` and source SHA256. Recovery records state exactly which metadata proof was re-established. They do not claim that the underlying infrastructure root cause was fixed.

## Acceptance remains evidence-based

Native terminal publication, original artifact integrity, causal SFT-window qualification and processor/template acceptance are different checks. `observe_teacher.sh` reads only published records and verifies the original native grader evidence. Processor checks are opt-in, local, per-model and never read Seed2 targets or load model weights. Missing results remain `PENDING` / `NOT_RUN`. See [A800 collection](collection_a800.md).
