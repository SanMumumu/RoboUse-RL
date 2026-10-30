# RoboDojo evaluation

Evaluation uses all explicitly requested environment identities, not only successful teacher scenarios. Only the unchanged RoboDojo `_result.json` grader is authoritative. Agent claims, an accepted `assess` call, tool dispatch and visible feedback are not success labels.

## Machine responsibilities

- **A800 node:** student inference, SFT and GRPO updates.
- **RTX node:** Isaac rendering, native controller, tools and grader. RoboDojo and XPolicyLab must already be installed with required assets and compatible simulator/transport environments.
- **Laptop or local workstation:** downloaded offline results, or an SSH tunnel to the loopback viewer.

The public repository includes the selected AGP tool/control snapshot, not Isaac assets or an entire RoboDojo checkout. Configure `benchmark`, `simulator_python`, `transport_python`, `renderer_gpu`, `sensor_version` and `controller_version` in a private copy of [env_worker.json](../configs/env_worker.json). Unknown sensor/controller versions remain null. Use an independent checkout/environment; never modify a running evaluator's imports.

On the allocated **RTX node**, supply a shared random authentication secret via the environment (never a config or Git file):

```bash
export ROBOUSE_ENV_TOKEN='YOUR_OUT_OF_BAND_RANDOM_SECRET_AT_LEAST_24_CHARACTERS'
bash scripts/env_worker.sh --config /path/to/rtx-worker.json \
  --output /path/to/runs/rtx-worker --port 8765 --execute
```

On the **A800 node**, open the SSH tunnel and set the same secret:

```bash
ssh -N -L 8765:127.0.0.1:8765 user@rtx-node
# Separate terminal:
export ROBOUSE_ENV_TOKEN='THE_SAME_OUT_OF_BAND_SECRET'
```

Keep `env_urls: [http://127.0.0.1:8765]` in the evaluation/RL config. The worker binds loopback; the client rejects public HTTP endpoints. Authentication and SSH protect the transport. A800 does not render Isaac RGB.

## Seed1 validation and explicit Seed2 final test

Create frozen identity JSONL with `task`, `seed`, `layout` and known `physical_aliases`. The worker resets the same native task/seed/layout and groups check native physical/static/seed state at tolerance 1e-6 without exposing private GT to the policy. Include every required identity even if its teacher failed.

```bash
# A800: Seed1 validation; configure identities as the complete Seed1 list.
bash scripts/evaluate.sh --config configs/evaluation/robodojo.yaml \
  --checkpoint /path/to/checkpoint --split val \
  --identities /path/to/seed1-identities.jsonl --output-dir outputs/val_exp01 --execute

# A800: explicit final test only; sealed Seed2 identities, never training data.
bash scripts/evaluate.sh --config configs/evaluation/robodojo.yaml \
  --checkpoint /path/to/checkpoint --split test --final-test \
  --identities /path/to/seed2-identities.jsonl --output-dir outputs/eval_exp01 --execute
```

The requested target test command without `--final-test` intentionally fails. Missing Seed2 is `PENDING_TEST`; do not substitute Seed0/Seed1. Real native validation/test for the release is **NOT_RUN**.

## Output contract

`metrics.json`, `episodes.jsonl`, `episode_metrics.json`, `run_manifest.json` and `videos/<episode-id>/` are written below the output directory. Raw decision tensors, original RGB, policy hashes, visible tool timeline and native score evidence remain under each episode directory.

Metrics distinguish requested, attempted, scored and unscored episodes. `native_task_success_rate` is successes divided by **native-scored episodes**; coverage and unscored counts are separate. It is null if nothing was scored. Native evidence is rehashed and checked against the recorded reward. Infrastructure and operator interruptions never become reward zero. Partial successful episodes remain visible if a later attempt fails; the error is still raised.

Per-episode records include native outcome/score, task/seed/initial state, actual tool calls, policy decisions, decision latency, native MCP tool duration, A800↔RTX dispatch roundtrip, end-to-end episode latency, observable token counts, failure category and component versions. Unknown values are null. Native tool timing measures the registered MCP operation (including controller feedback); it is not a fabricated GPU inference estimate. JS programs may issue multiple native calls; counts come from the native dispatch boundary.

The worker preserves original three-view MP4 when supported. Public output names are `head`, `left_wrist` and `right_wrist`; native `top` is mapped to `head`. Both successes and failures remain visible. Missing/unfinished video is not replaced with an old policy's recording.

```bash
# A800 or laptop after copying the run: build a self-contained HTML/MP4 ZIP.
bash scripts/view_results.sh --output-dir outputs/eval_exp01 --execute
# Optional local web viewer. Bind loopback only.
python -m http.server 8000 --bind 127.0.0.1 --directory outputs/eval_exp01
```

Open `http://127.0.0.1:8000/` or extract `offline.zip` and open its HTML offline. The viewer never starts sampling, changes denominators or rewrites scores.
