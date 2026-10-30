# Reproducibility and validation boundaries

The release freezes package versions, upstream base-model revisions, role manifests, per-file asset hashes, context version, tool schemas, checkpoint state and actual policy identity. Training saves resolved configuration, Git SHA, random seed, dataset source hashes/weights and deterministic order. SFT test paths are registration-only. Full and local trajectory quality are recorded separately.

## Installation and CPU checks

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
# CPU tests: install a CPU torch build before test extras.
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test,cpu-test]'
python -m pytest -q
# Public fixture: no internal data, credentials, weights, simulator or teacher.
python examples/trajectory_sft/read_fixture.py --output outputs/fixture-prepared
```

The optional `ROBOUSE_TEST_PROCESSOR` points to an already downloaded original Qwen3.5 processor. Its CPU test checks real RGB encoding, template signature and assistant mask without loading model weights. No implicit model download or simulator launch occurs in tests. Tests using synthetic/fake transports are explicitly labeled; they do not establish online reward or physical success.

```bash
# Eight actual CPU processes, synthetic independent global-gradient references.
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 \
  -m robouse.cpu_distributed --output outputs/cpu-ddp.json --execute
python scripts/audit_release.py
```

GitHub Actions runs installation/import, CPU tests and an eight-process Gloo numerical reference. Hosted CI status must be read from the actual GitHub workflow; a local pass is not a hosted Actions pass.

## Evidence matrix

See [release validation](../agent/VALIDATION.md) for actual executed checks and limitations. The retained research history includes limited train-instance overfitting diagnostics; these are not released benchmark claims or new student validation/test results. No benchmark success rate, SOTA assertion or real online RL result is published by this release.

| Check | Evidence type | Hardware/native conclusion |
|---|---|---|
| Import/install/public fixture/tools/harness | CPU synthetic contracts | No robot execution |
| HF optimizer/save/load/resume | Actual tiny randomly initialized CPU model | No Qwen A800 acceptance |
| Original Qwen processor | Local processor + synthetic RGB only | No model weights or training |
| SFT mask/global normalization | Independent CPU numerical references | A800 kernels NOT_RUN |
| GRPO objective/LP assertions | CPU reference and rejection tests | A800 sampler/oracle parity NOT_RUN |
| Eight-rank gradient comparison | Eight real CPU/Gloo ranks, synthetic data | No eight-GPU training |
| Native AGP inventory | Byte hashes of 64 preserved files | Does not prove simulator compatibility |
| RoboDojo reset/score/video contracts | CPU adapters and synthetic artifacts | Real Isaac episodes NOT_RUN |
| Teacher admission/seal/resume | Fake RPC/workers + synthetic native traces | Real teacher calls NOT_RUN |

The final release checks used independently installed CPU dependencies and no old project PYTHONPATH. Only the optional original 2B processor files were read from a local model directory; no weights were loaded for that check. A clean install of the complete pinned A800 CUDA stack remains a separate NOT_RUN acceptance item until performed on an allocated node.

## Preservation and release history

Original research files, environments, live jobs, model weights and data remain untouched. The prior independent export history is backed up in a Git bundle outside the public repository. The release branch is based on the target repository's existing `main`, preserving its original commit; it adds a reviewed source snapshot rather than importing private research history. No orphan rewrite or force push is necessary.

Dataset/model/video manifests and detailed internal migration inventories live outside Git. Public source excludes accounts, tokens, server-specific paths, internal addresses, raw datasets, weights and logs. Pattern scans are a useful gate, not an exhaustive secret-detection guarantee. The native and vendored component licenses/provenance are retained in LICENSE/NOTICE and `_vendor/provenance.json`.
