# Task 22: Hybrid-async text performance

This benchmark adapts the repository's Hybrid-async text recipe to Qwen3-0.6B and exactly two GPUs. It runs three paired trials; every trial executes the baseline and optimized recipe with the same seed and effective workload.

Both variants retain the same reference forward, zero-coefficient KL path, weight-update frequency, and effective workload. The only A/B variable is `--update-weight-buffer-size`: the baseline uses the framework default of 512 MiB and the optimized recipe uses 1 GiB. The larger buffer reduces the number of weight-publication chunks for this 1.40 GiB checkpoint while retaining two chunks so conversion and transfer can still overlap. Selecting `TASK22_VARIANT=baseline` is the rollback.

## Fixed configuration

| Item | Value |
| --- | --- |
| Model | `~/model/Qwen3-0.6B` |
| GPUs | 2, split as actor 1 + rollout 1 |
| Data | 16 local arithmetic records |
| Repetitions | 3 paired trials |
| Steps | 10 per component run, hard-capped at 20 |
| Batch | 8 prompts x 4 samples = 32 |
| Response cap | 512 tokens |
| Hybrid staleness | 2 |
| Baseline buffer | 512 MiB |
| Optimized buffer | 1 GiB |

## Run

The Ray head must expose only the two experiment GPUs. Activate the workspace environment and run:

```bash
cd /home/zhengbaowei/relax_ft/Relax
source ../.venv/bin/activate
CUDA_VISIBLE_DEVICES=2,3 TOTAL_TRIALS=3 \
  bash benchmarks/task22_hybrid_async_text/run_paired_trials.sh
```

Raw logs, manifests, submitted commands, and one-second GPU samples are written to ignored `benchmark_artifacts/`. The analyzer writes the reviewable report, CSV tables, and SVG curve to `benchmarks/results/task22-hybrid-async-text/`.
