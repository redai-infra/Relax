# P3O A100×4 recipes

These launchers compare P3O and GRPO under matched on-policy and controlled
rollout-mismatch scenarios. They target four colocated GPUs and submit the
training driver through Ray Jobs.

中文版请参阅 [README_zh.md](README_zh.md)。

## Required environment

Set these paths before a non-dry run:

```bash
export P3O_MODEL_DIR=/path/to/model
export P3O_TRAIN_DATA=/path/to/train.jsonl
export P3O_EVAL_DATA=/path/to/eval.jsonl
export P3O_OUTPUT_ROOT=/path/to/output
export P3O_MEGATRON_DIR=/path/to/Megatron-LM
export P3O_RAY_DASHBOARD=http://ray-dashboard-host:8265
```

`P3O_EVAL_DATA` is required in `formal` mode and is optional in `smoke` mode.
The model, training data, and Megatron paths must exist before the Ray job is
submitted. Each run records its resolved arguments, command, Git identity,
logs, Ray status, exit code, and per-step rollout JSONL beneath
`P3O_OUTPUT_ROOT`. Set `P3O_ROLLOUT_RESULT_DIR` only when an external evidence
layout requires a different raw-rollout destination; the resolved path is
recorded in `run_identity.env`.

The Ray job runtime explicitly disables inherited HTTP proxies. SGLang checks
engine health and registers workers through node-local IP addresses; allowing
host proxy variables into Ray workers can leave healthy engines stuck behind
the proxy instead of completing the startup barrier.

Formal mode defaults to DeepScaleR's `problem`/`answer` fields. Smoke mode
defaults to the commonly used `question`/`answer` schema. Set `P3O_INPUT_KEY`
and `P3O_LABEL_KEY` explicitly when the selected asset uses another schema;
both resolved keys are recorded in `run_identity.env`.

Formal mode also defaults to the `deepscaler` rule-based verifier, which reads
Qwen-Thinking's `</think>` suffix and a final `\\boxed{...}` answer. Smoke mode
retains the `mopd` default for legacy GSM8K-style assets. Set `P3O_RM_TYPE`
explicitly when a smoke uses DeepScaleR or another reward contract; the
resolved reward type is recorded in `run_identity.env`.

Formal evaluation defaults to the `deepscaler` dataset name, 16 samples per
prompt, a 4096-token response cap, temperature 1.0, and top-p 0.95. Bounded
resource studies may set `P3O_EVAL_NAME`, `P3O_EVAL_N_SAMPLES`,
`P3O_EVAL_MAX_RESPONSE_LEN`, `P3O_EVAL_TEMPERATURE`, and `P3O_EVAL_TOP_P`.
These values affect evaluation only and are recorded in `run_identity.env`;
paired algorithms must use identical values.

The default `P3O_ROLLOUT_SHUFFLE=1` retains ordinary training behavior. Set it
to `0` only with a pre-materialized fixed prompt schedule for paired evidence;
the setting is recorded so a shuffled run cannot be mistaken for the fixed
comparison.

Set `P3O_DETERMINISTIC_INFERENCE=1` for paired experiments that require common
per-sample sampling seeds across P3O and GRPO. The resolved flag is recorded in
run identity. This controls sampling randomness only; after the first update,
different policy weights can and should produce different responses for the
same seed.

Formal mode sources `scripts/models/qwen3-4B.sh` and targets
Qwen3-4B-Thinking-2507. Smoke mode sources `scripts/models/qwen3-0.6B.sh`.
Set `P3O_MODEL_CONFIG` only when deliberately validating another compatible
model configuration; the resolved path is recorded in `run_identity.env`.
The formal launcher overrides the generic 4B script's RoPE base to `5000000`,
matching this checkpoint's `config.json`; smoke remains at `1000000`. A
deliberate compatible override can use `P3O_MODEL_ROTARY_BASE`, and its value is
also recorded in run identity.

## Active P3O contract

The formal P3O path uses `--p3o-ess-scope micro-batch`,
`--p3o-kl-mode proxy_safe`, and monitoring margins
`--clip-low/--clip-high 0.2`. `proxy_safe` has the same forward value as the
FeynRL-compatible sampled-token proxy and corrects only the extreme negative
log-ratio gradient. `exact` is available only through the pure full-vocabulary
verification helper, not as a CLI mode, because rollout data stores
selected-token log-probabilities rather than behavior logits.

P3O owns a dedicated policy-loss dispatch and is mutually exclusive with
`--use-opd`; an OPD teacher loss, OPD advantage replacement, or OPD-only
reward would define an unvalidated hybrid objective. The reward/verifier name
`P3O_RM_TYPE=mopd` is unrelated to the `--use-opd` training feature and remains
valid for compatible datasets.

Formal defaults are G=16, global batch 64, micro-batch 1, rollout batch 4,
response length 4096, and 30 optimizer steps (`--num-rollout 30`). The planned
paired seeds are 42, 123, and 2026. Smoke remains G=4, global batch 16,
response length 128, and one optimizer step.

The following environment variables expose the aligned settings without
changing scenario scripts:

```bash
export P3O_ESS_SCOPE=micro-batch  # or step for capability/replay validation
export P3O_KL_MODE=proxy_safe     # proxy for golden parity
export P3O_CLIP_LOW=0.2
export P3O_CLIP_HIGH=0.2
export P3O_SEED=42
export P3O_RM_TYPE=deepscaler # required when smoke mode is paired with DeepScaleR
```

Ray workers inherit normal proxy settings by default. On clusters where an
injected outbound proxy intercepts SGLang's node-local readiness probes, set
`P3O_CLEAR_RUNTIME_PROXIES=1` to clear proxy variables inside the job runtime.
This setting is opt-in and recorded in `run_identity.env` because it also
disables proxy access for every worker in the job.

If A100-40GB capacity prevents a 4B pilot, reduce pilot response length first
while keeping micro-batch size 1 and record the deviation. Do not treat reduced
smoke runs as formal evidence or silently reduce the three-seed comparison.
For a response-preserving resource fallback, set `P3O_ACTIVATION_RECOMPUTE=1`
to add whole-layer uniform activation recomputation and set
`P3O_LOG_PROBS_CHUNK_SIZE` to a positive token count for chunked log-probability
and entropy reductions. Both settings apply identically to P3O and GRPO and are
recorded in run identity; the default `0`/`-1` leaves the original path intact.

## Scenarios

| Scenario                   | Update interval | Temperature override | Meaning                                                           |
| -------------------------- | --------------: | -------------------: | ----------------------------------------------------------------- |
| `on_policy`                |               1 |                  off | Synchronize every rollout with the normal sampling configuration. |
| `periodic_sync_interval_3` |               3 |                  off | Introduce only periodic rollout-policy staleness.                 |
| `temperature_0p6`          |               1 |                  0.6 | Change only the behavior-policy temperature.                      |
| `temperature_1p2`          |               1 |                  1.2 | Change only the behavior-policy temperature.                      |

P3O and GRPO launchers for the same scenario share all non-algorithm
configuration. Temperature scenarios preserve `top_p`, `top_k`, response
limits, and evaluation sampling settings.

## Running

```bash
bash examples/algorithms/p3o/run_p3o_on_policy_a100x4.sh
bash examples/algorithms/p3o/run_grpo_on_policy_a100x4.sh
bash examples/algorithms/p3o/run_p3o_periodic_sync_interval_3_a100x4.sh
bash examples/algorithms/p3o/run_p3o_temperature_0p6_a100x4.sh
```

For a one-rollout check, select any scenario through the smoke wrapper:

```bash
bash examples/algorithms/p3o/run_p3o_smoke.sh p3o_temperature_1p2
```

Use `P3O_DRY_RUN=1` to print the resolved training arguments without checking
assets or submitting a Ray job.

## Policy-age metric

`train/p3o/rollout_policy_age_rollouts` measures the difference between the
current rollout ID and the rollout-policy snapshot ID that generated the batch.
Its unit is rollouts, not optimizer steps. A periodic refresh affects the next
rollout; metrics for the batch at the refresh boundary still describe the
snapshot that generated that batch.
