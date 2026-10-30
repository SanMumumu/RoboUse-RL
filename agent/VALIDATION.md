# v0.1.0 validation record

This record separates implementation checks from GPU/native experiments. Detailed local logs, internal source SHA inventories and history backups are under ignored `agent/evidence/`; they are not benchmark results.

| Check | Status | Actual evidence / limit |
|---|---|---|
| `pip install -e .` | PASS | Installed `robouse-rl==0.1.0` in an independent venv; no research environment modified |
| `pip install -e '.[test,cpu-test]'` | PASS | Independent CPU torch 2.7.0+cpu / torchvision 0.22.0+cpu, Transformers 5.8.1, PyArrow and other declared extras installed; no old project PYTHONPATH |
| Installed console and public imports | PASS | `robouse --help`, data/harness/training/evaluation imports from outside the source directory |
| Final complete CPU suite | PASS | **108 passed** in 49.29s in that independent environment; optional local original 2B processor provided read-only |
| Public RGB trajectory fixture | PASS | Qualified early decision survives incomplete later history; schema/byte hashes verified; synthetic and unscored |
| Data migration | PASS on synthetic original-format data | Actual prepare/export/rsync import/verify and relocated tar offsets; not a full real dataset migration |
| Tool schema / Harness | PASS | Strict schema, errors, bounded explicit retries, idempotency and physical-success boundary; no robot execution |
| SFT mask / normalization | PASS | Actual original Qwen3.5-2B processor/RGB/assistant-mask test, independent global target-token gradient reference |
| Checkpoint save/load/resume | PASS | Actual tiny randomly initialized CPU model with HF optimizer/scheduler/RNG; exact interrupted-resume weights |
| GRPO numerical references / rejection tests | PASS | Group advantage, clipping, fixed reference KL, variable-length weights and immutable behavior LP assertions |
| Eight actual CPU/Gloo ranks | PASS | SFT maximum gradient error 2.7755575615628914e-17; GRPO 6.938893903907228e-18; synthetic references only |
| Native grading / failure accounting / viewer | PASS on CPU fixtures | Score evidence rehashed; infrastructure/interruption remains unscored even with a published native JSON; unavailable video views shown honestly |
| Target training/evaluation CLI / setup | PASS | Help and dry-run; real training/evaluation functions connected, unsupported RL opt-in and final-test guards reject before execution |
| Wheel / native-source inventory | PASS | Wheel build; all 64 byte-pinned native source files and public core modules present; final artifact hashes stored locally |
| Indexed release scan / whitespace | PASS | No configured token/private-key, private IP, internal-host, machine-path, binary or >2 MB file findings; pattern scan is not exhaustive secret detection |
| GitHub Actions | IN_PROGRESS at publication | [Hosted CPU contracts](https://github.com/SanMumumu/RoboUse-RL/actions); hosted results remain separate from local tests |
| A800 2B and 4B four-step smoke | NOT_RUN | No verified A800 allocation; no GPU/simulator job launched |
| A800 full SFT / ZeRO resume | NOT_RUN | Requires separate matching model-specific GPU smoke on allocated hardware |
| Actual Qwen3.5-4B processor | NOT_RUN | No 2B result is substituted |
| Actual online GRPO / A800 LP parity | NOT_RUN / Experimental | Original raw LP reconstruction failures acknowledged; immutable behavior LP and 0.02 prefix-oracle gate retained |
| Native validation/test / new policy videos | NOT_RUN | External compatible RoboDojo/XPolicyLab/Isaac/assets and allocations required |
| Real Astra metadata/inference/collection | NOT_RUN | Admission/budget/resume/seal tested using fake RPC/workers and synthetic native traces only |
| Complete real dataset/model transfer | NOT_RUN | Real datasets/weights excluded from Git; scripts and dependency manifests provided |
| Seed2 preparation/evaluation | PENDING_TEST | No substitute seed or SFT test-target access |
| Fresh-clone installation / CPU suite | PASS | Independent clone without the research project PYTHONPATH: imports, editable installation, release scan and all 108 tests passed in 48.48s |
| GitHub push / PR | PASS | Authenticated account verified as SanMumumu; normal push to release/v0.1.0 and [PR #1](https://github.com/SanMumumu/RoboUse-RL/pull/1) targeting main; remote head verified |

A first training import in the new base-only venv failed because optional torch/test dependencies were absent. After independent CPU-extra installation and the collection follow-up, the final imports and all 108 tests passed. Initial engineering failures are retained in local logs; they are not hidden or reported as robot experiments.

Release development preserves the existing target main commit and prior export history backup. No original project file, active rollout source, environment, account/login, GPU lease, existing job, model weight or historical dataset was changed. Fresh-clone delivery checks and actual remote publication status are reported separately when executed.

## Admission / Guardian / A800 collection follow-up

- PASS: typed metadata failure classification and actual subprocess JSON-RPC fixtures recover the same pre-model attempt within three checks; 401/403/429, permissions, quota and IO stop without blind retries. Unknown internal faults require the same fresh identity/credential/model/quota prior and current write proof. No real teacher endpoint was contacted.
- PASS: a global admission hold prevents another prepared lane's model start; hold publication and start intent/Popen share a lock. Lane-local transient bootstrap faults preserve failures while healthy lanes continue.
- PASS: independent Guardian singleton/start budget, exact stopped-run resume, frozen failure exclusions and credential-free/idempotent failure journal. No sealed failure, budget or deadline is reset.
- PASS: actual localhost authenticated HTTP transport with a synthetic remote campaign tests pinned plans, idempotent ownership, no uncertain-start replay, controlled errors, original byte/SHA transfer and exact resume. Renderer, collector and model endpoints in that test are fixtures, not Isaac/Astra.
- PASS: artifact/native result validation and per-model local processor acceptance are separate. The real local 2B processor is checked against a synthetic sealed trace without loading weights. Synthetic/absent A800 render proof is rejected.
- PASS: new scripts support help/config/dry-run. Fresh-clone delivery checks passed and PR #1 was created after authenticated account and repository permission verification.
- NOT_RUN: actual A800↔RTX teacher collection, A800-local Isaac RGB rendering, cloud teacher inference, real new-runtime first complete terminal and processor acceptance. This PR does not attach its observer to the live research collector or mark that collector accepted. Source-reported progress remains separate from release test evidence.
