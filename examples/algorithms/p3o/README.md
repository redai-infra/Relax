# P3O A100×4 recipes

These launchers compare P3O and GRPO under matched on-policy and controlled
rollout-mismatch scenarios. They target four colocated GPUs and submit the
training driver through Ray Jobs.

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
logs, Ray status, and exit code beneath `P3O_OUTPUT_ROOT`.

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
