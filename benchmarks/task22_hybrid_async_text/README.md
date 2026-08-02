# Task 22: Hybrid-async text performance

This benchmark adapts the repository's Hybrid-async text recipe to Qwen3-0.6B and exactly two GPUs. It runs three paired trials; every trial executes three variants with the same seed and effective workload.

1. `baseline` preserves the original zero-KL reference forward and uses an 8192-token dynamic-batch budget.
2. `zero_kl` removes `--use-kl-loss` and `--ref-load` while the KL coefficient is exactly zero. This is an objective-equivalent configuration fix, measured separately from the main optimization.
3. `optimized` builds on `zero_kl` and publishes rollout weights every two actor updates with `--update-weights-interval 2`.

All variants keep the 512 MiB weight-update buffer, 8192/8192 train/log-prob token budgets, model, data, generated workload, global batch, optimizer updates, and max staleness fixed. The main acceptance target is at least 5% higher response-token throughput for `optimized` versus `zero_kl`, with no runtime errors, unexpected NaN/Inf, missing samples, or TIS instability. Interval two intentionally allows one additional actor update of policy freshness; `TASK22_VARIANT=zero_kl UPDATE_WEIGHTS_INTERVAL=1` rolls it back, while `TASK22_VARIANT=baseline` rolls back both changes.

## Fixed configuration

| Item                        | Value                                                               |
| --------------------------- | ------------------------------------------------------------------- |
| Model                       | `~/model/Qwen3-0.6B`                                                |
| GPUs                        | 2, split as actor 1 + rollout 1                                     |
| Data                        | 16-row ModelScope GSM8K slice (`AI-ModelScope/gsm8k`, `main/train`) |
| Repetitions                 | 3 paired trials, with all 3 variants in every trial                 |
| Steps                       | 20 per component run                                                |
| Performance window          | Logged steps 5-15 inclusive                                         |
| Batch                       | 8 prompts x 4 samples = 32                                          |
| Response cap                | 512 tokens                                                          |
| Hybrid staleness            | 2                                                                   |
| Weight-update buffer        | 512 MiB for every variant                                           |
| Train/log-prob token budget | 8192/8192 for every variant                                         |
| Weight publication interval | 1 (`baseline`, `zero_kl`); 2 (`optimized`)                          |

## Run

The Ray head must expose only the two experiment GPUs. Activate the workspace environment and run:

```bash
cd /home/zhengbaowei/relax_ft/Relax
source ../.venv/bin/activate
CUDA_VISIBLE_DEVICES=2,3 TOTAL_TRIALS=3 \
  bash benchmarks/task22_hybrid_async_text/run_paired_trials.sh
```

Raw logs, manifests, submitted commands, and one-second GPU samples are written to ignored `benchmark_artifacts/task22-hybrid-async-text-v3/`. The analyzer writes the reviewable report, CSV tables, and SVG curve to `benchmarks/results/task22-hybrid-async-text/`.

The dataset helper downloads `AI-ModelScope/gsm8k` with `ms download --repo-type dataset`, then slices the first 16 rows from `main/train` into `benchmarks/data/task22_gsm8k_main16.jsonl` for training.
