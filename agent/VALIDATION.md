# v0.1.0 validation record

This record separates implementation checks from GPU/native experiments. Detailed local logs, internal source SHA inventories and history backups are under ignored `agent/evidence/`; they are not benchmark results.

| Check | Status | Actual evidence / limit |
|---|---|---|
| `pip install -e .` | PASS | Installed `robouse-rl==0.1.0` in an independent venv; no research environment modified |
| `pip install -e '.[test,cpu-test]'` | PASS | Independent CPU torch 2.7.0+cpu / torchvision 0.22.0+cpu, Transformers 5.8.1, PyArrow and other declared extras installed; no old project PYTHONPATH |
| Installed console and public imports | PASS | `robouse --help`, data/harness/training/evaluation imports from outside the source directory |
| Final complete CPU suite | PASS | **76 passed** in 36.66s in that independent environment; optional local original 2B processor provided read-only |
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
| GitHub Actions | NOT_RUN | CPU workflow supplied; no hosted pass claimed before actual push |
| A800 2B and 4B four-step smoke | NOT_RUN | No verified A800 allocation; no GPU/simulator job launched |
| A800 full SFT / ZeRO resume | NOT_RUN | Requires separate matching model-specific GPU smoke on allocated hardware |
| Actual Qwen3.5-4B processor | NOT_RUN | No 2B result is substituted |
| Actual online GRPO / A800 LP parity | NOT_RUN / Experimental | Original raw LP reconstruction failures acknowledged; immutable behavior LP and 0.02 prefix-oracle gate retained |
| Native validation/test / new policy videos | NOT_RUN | External compatible RoboDojo/XPolicyLab/Isaac/assets and allocations required |
| Real Astra metadata/inference/collection | NOT_RUN | Admission/budget/resume/seal tested using fake RPC/workers and synthetic native traces only |
| Complete real dataset/model transfer | NOT_RUN | Real datasets/weights excluded from Git; scripts and dependency manifests provided |
| Seed2 preparation/evaluation | PENDING_TEST | No substitute seed or SFT test-target access |
| GitHub push / PR | BLOCKED_AUTH at initial check | Target main observed; no configured HTTPS credential, GH_TOKEN/GITHUB_TOKEN or gh CLI; no publication claimed |

A first training import in the new base-only venv failed because optional torch/test dependencies were absent. After independent CPU-extra installation the final imports and all 76 tests passed. Initial engineering failures are retained in local logs; they are not hidden or reported as robot experiments.

Release development preserves the existing target main commit and prior export history backup. No original project file, active rollout source, environment, account/login, GPU lease, existing job, model weight or historical dataset was changed. Fresh-clone delivery checks and actual remote publication status are reported separately when executed.
