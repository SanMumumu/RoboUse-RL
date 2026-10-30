# Training

SFT is a real HF Trainer path with full multimodal fine-tuning. A800 2B/4B execution remains **NOT_RUN** for this release. The CPU tiny-model test performs real optimizer updates and exact checkpoint/RNG resume, but does not establish Qwen GPU performance.

## Installation and original models

Use Python 3.11 and an independent environment. Install base interfaces with `pip install -e .`. For training, install the pinned CUDA stack from `requirements-train.lock` using the index in the lock, or install `.[train]` after selecting the appropriate PyTorch build. The complete A800 locked environment installation is **NOT_RUN**. RTX/Isaac belongs in a separate simulator environment; use `.[transport]` for the MCP transport side.

The 2B and 4B YAML recipes pin their HF commit revisions. To use a local directory, verify original weights against the same upstream revision:

```bash
python -m robouse.model_inventory --model /path/to/Qwen3.5-2B \
  --hf-id Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc --execute
# The printed inventory digest is the local --revision below.
```

Do not use previously Seed1-trained weights as the base. Processor, tokenizer, template and termination tokens are loaded from the selected model; no 2B encoding cache is implicitly reused for 4B.

## Full-parameter SFT on the A800 node

```bash
# Preview: no GPU is touched without --execute.
bash scripts/train_sft.sh --config configs/sft/qwen35_2b.yaml \
  --train-data /path/to/prepared/seed0-train.jsonl \
  --val-data /path/to/prepared/seed1-val.jsonl \
  --test-data /path/to/prepared/seed2-test.jsonl \
  --data-root /path/to/data --output-dir outputs/sft_exp01 --num-gpus 8 --dry-run

# Independent model-specific 4-step acceptance. Requires allocated eight GPUs.
bash scripts/train_sft.sh --config configs/sft/qwen35_2b.yaml \
  --train-data /path/to/prepared/seed0-train.jsonl \
  --val-data /path/to/prepared/seed1-val.jsonl --data-root /path/to/data \
  --output-dir outputs/smoke2b --num-gpus 8 --smoke --execute

# Formal run after that exact model/data/config passes smoke.
bash scripts/train_sft.sh --config configs/sft/qwen35_2b.yaml \
  --train-data /path/to/prepared/seed0-train.jsonl \
  --val-data /path/to/prepared/seed1-val.jsonl \
  --test-data /path/to/prepared/seed2-test.jsonl --data-root /path/to/data \
  --output-dir outputs/sft_exp01 --num-gpus 8 \
  --smoke-acceptance outputs/smoke2b/smoke-acceptance.json --execute
```

The acceptance run compares uninterrupted four-step training with stop/save/reload/resume, checks language/vision/connector changes on all ranks, records peak GPU memory and verifies complete checkpoint state. Configure enough epochs/data for four genuine optimizer steps. 2B acceptance cannot substitute for 4B acceptance. No CPU offload is assumed: the default ZeRO-3 config leaves it disabled; enable it only after an allocated-node memory measurement justifies it, then rerun matching acceptance.

For the original A800 interface and a local 4B model:

```bash
bash scripts/sft_a800.sh --model /path/to/Qwen3.5-4B --revision LOCAL_MODEL_INVENTORY_SHA \
  --train-data /path/to/prepared/seed0-train.jsonl \
  --val-data /path/to/prepared/seed1-val.jsonl \
  --test-data /path/to/prepared/seed2-test.jsonl --data-root /path/to/data \
  --gpus 0,1,2,3,4,5,6,7 --output /path/to/runs/sft4b \
  --smoke-acceptance /path/to/smoke4b/smoke-acceptance.json --execute
```

Use the 4B recipe for 4B training; its own processor and smoke are required. `--test-data` only registers the protected role, without opening test targets.

## Resume and outputs

Repeat the identical command with `--resume outputs/sft_exp01/checkpoint-N`. Preserve the same data, seed, world size, accumulation, order, model revision, code and loss options. The contract rejects incompatible resumes. Checkpoint commit markers bind full weights, optimizer/scheduler, per-rank RNG, processor and role ownership. Incomplete checkpoints are preserved under a separate name. The final step always saves a full checkpoint and runs Seed1 validation; best-checkpoint selection uses Seed1 NLL only.

Training uses eight distinct ranks and a deterministic global window plan. Accelerate shards each real slot once; masked dummy tails only balance collective counts and contribute zero loss. Accumulated SFT loss is normalized by the **global assistant target-token count**, including variable lengths. Only the final assistant target (including native EOS) is supervised; image/user/tool feedback and prior assistant history are excluded. No target truncation or JSON repair is allowed.

Console output is a single summary/tqdm; `full.log`, TensorBoard, `progress.json`, consumption records, `run_manifest.json`, data hashes, seed and Git SHA remain in the output directory. Loss, LR, throughput, peak GPU memory and measured ETA are recorded; ETA is unknown during warmup. SFT needs no simulator.

Only `finetuning_mode: full` is supported in v0.1.0. Historical research LoRA experiments are not migrated as a validated release feature; selecting a PEFT mode fails explicitly.

## Experimental online GRPO

This optimizes **tool-using student trajectories**, not continuous robot actions. Run the Isaac environment worker on an explicitly allocated RTX node; sampling and updates run on A800. See [evaluation](evaluation.md) for the authenticated loopback/SSH arrangement.

```bash
# Preview the real path, without CUDA or mock substitution.
bash scripts/train_rl.sh --config configs/rl/grpo_qwen35_2b.yaml \
  --init-checkpoint /path/to/sft_checkpoint --output-dir outputs/rl_exp01 \
  --num-gpus 8 --dry-run

# Bounded validation only; edit the config with frozen Seed0 identities/env endpoints.
bash scripts/train_rl.sh --config configs/rl/grpo_qwen35_2b.yaml \
  --init-checkpoint /path/to/sft_checkpoint --output-dir outputs/rl_smoke \
  --num-gpus 8 --max-rounds 1 --groups-per-round 2 --group-size 2 \
  --update-epochs 1 --max-env-episodes 4 --allow-experimental --execute
```

Execution without `--allow-experimental` fails before touching CUDA. This is a maturity gate, not a claim of numerical or simulator failure in the released CPU tests. The original research LP reconstruction did fail on its then-current backend; this extraction retains actual sampling LP, an exact prefix oracle and a separate frozen training-backend LP. Actual A800 sampler/oracle qualification must pass the unchanged **0.02** assertion; it has not been run here. A failed qualification stops the run and preserves evidence. No threshold weakening, overwritten behavior LP or teacher-as-student replay is used.

Each group consists of independent complete student episodes from one comparable native initial state. The policy is frozen for collection. Stored evidence includes original images, exact tensors, raw generated IDs, behavior LP, mask, native reward and weight hash. Relative advantages, clipped objective, fixed SFT-reference KL and per-episode length weighting have independent CPU references. Training-backend importance correction uses a separately frozen LP vector and never replaces the immutable behavior LP.

Groups, rounds, update epochs, episode budget, RNG, policy and optimizer checkpoints are durable. After an update the next sampler verifies the new weight hash. All-zero-variance rounds honestly record zero updates with no extra sampling. Infrastructure failures remain unscored. Resume uses the same configuration plus `--resume outputs/rl_exp01`; no unbounded replenishment is performed. Multi-round production behavior and real A800/Isaac parity remain **Experimental / NOT_RUN**.
