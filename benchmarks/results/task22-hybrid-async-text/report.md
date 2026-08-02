# Task 22 Hybrid-async text performance report

## Result

Three paired trials on commit `056f4b47c2022910c780f497d42283a803dc2ea7` show that pruning the zero-coefficient KL reference path changes response throughput by +6.05% and E2E latency by -6.20%. Publishing rollout weights every two actor updates then changes throughput by another +13.48% and latency by -10.74%. The combined change is +20.34% throughput and -16.28% latency versus baseline.
Acceptance: **PASS** against the frozen target of at least +5% response throughput for `optimized` versus `zero_kl`.

| variant   | update interval | framework/E2E step (s) | response tok/s | samples/s | weight publish/step (s) |      TIS | actor peak MiB | rollout peak MiB |
| --------- | --------------: | ---------------------: | -------------: | --------: | ----------------------: | -------: | -------------: | ---------------: |
| baseline  |               1 |            3.893/3.909 |         4206.2 |     8.220 |                   0.664 | 0.999900 |          43978 |            80621 |
| zero_kl   |               1 |            3.670/3.667 |         4460.5 |     8.721 |                   0.652 | 1.000026 |          43974 |            80621 |
| optimized |               2 |            3.234/3.273 |         5061.8 |     9.895 |                   0.311 | 0.999972 |          51168 |            80623 |

## Per-run evidence

| variant   | run | steps | E2E p50/p95 (s) | response tok/s |  reward |         loss |      TIS | samples | errors |
| --------- | --: | ----: | --------------: | -------------: | ------: | -----------: | -------: | ------: | -----: |
| baseline  |   1 |    20 |     4.000/4.000 |         4234.5 | -1.0000 |            0 | 0.999837 |     352 |      0 |
| baseline  |   2 |    20 |     4.000/4.000 |         4215.7 | -1.0000 |            0 | 0.999934 |     352 |      0 |
| baseline  |   3 |    20 |     4.000/4.000 |         4168.3 | -1.0000 |            0 | 0.999931 |     352 |      0 |
| zero_kl   |   1 |    20 |     4.000/4.000 |         4542.0 | -1.0000 |            0 | 1.000155 |     352 |      0 |
| zero_kl   |   2 |    20 |     4.000/4.000 |         4403.7 | -1.0000 |            0 | 0.999925 |     352 |      0 |
| zero_kl   |   3 |    20 |     4.000/5.000 |         4435.8 | -0.9943 |  5.66312e-06 | 0.999998 |     352 |      0 |
| optimized |   1 |    20 |     3.000/4.000 |         5090.9 | -1.0000 |            0 | 0.999968 |     352 |      0 |
| optimized |   2 |    20 |     3.000/4.000 |         5049.3 | -1.0000 |            0 | 1.000041 |     352 |      0 |
| optimized |   3 |    20 |     3.000/4.000 |         5045.3 | -0.9943 | -1.50305e-05 | 0.999906 |     352 |      0 |

## Paired changes

| run | zero-KL throughput | interval-two throughput | total throughput | total latency |     seed |
| --: | -----------------: | ----------------------: | ---------------: | ------------: | -------: |
|   1 |             +7.26% |                 +12.08% |          +20.22% |       -16.28% | 20260802 |
|   2 |             +4.46% |                 +14.66% |          +19.77% |       -16.28% | 20260803 |
|   3 |             +6.42% |                 +13.74% |          +21.04% |       -16.28% | 20260804 |

## Fixed workload and method

- Hardware: physical GPUs 2 and 3, both NVIDIA RTX PRO 6000 Blackwell; actor 1 GPU, rollout 1 GPU.
- Host: 128 logical CPUs (INTEL(R) XEON(R) GOLD 6530), 503.5 GiB RAM.
- Runtime: Python 3.12.3; Torch 2.11.0+cu130 (CUDA 13.0); driver 580.173.02; Ray 2.56.1; SGLang 0.5.12.post1; Transformers 5.6.0.
- Model: `/home/zhengbaowei/model/Qwen3-0.6B`, model SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- Data: ModelScope `AI-ModelScope/gsm8k` main subset of 16 prompts, SHA256 `c992e09c748ae19e82ad6d4fda099eae01ce987414372a47834c901403e2c7e4`; no hand-written large dataset.
- Each component run: 20 steps, 8 prompts/step, 4 samples/prompt, effective batch 32, response cap 512.
- Async policy: Hybrid, max staleness 2 and TIS enabled. Baseline/zero-KL publish every step; optimized publishes every two steps.
- Dynamic-batch budgets: 8192 training tokens and 8192 log-prob tokens per GPU for every variant.
- Performance window: logged steps 5-15 inclusive (11 observations). Primary throughput uses high-resolution framework `perf/step_time`; the E2E interval from actor completion N-1 to N is secondary because those completion logs have only one-second timestamp resolution.
- GPU window: actor completion 4 through completion 15. Utilization and peak memory use only this window; utilization includes idle-zero samples.
- Trial order uses a three-way rotation to reduce warm-cache and order bias.

## Optimization rationale

`baseline` deliberately preserves the original misconfiguration: `--use-kl-loss --kl-loss-coef 0.00` plus `--ref-load`. `zero_kl` removes the unused reference path while retaining the 8192-token dynamic-batch budget. Because the KL term is multiplied by exactly zero, this removes compute but does not change the scalar objective.

`optimized` builds on `zero_kl` and sets `--update-weights-interval 2`. Hybrid skips the rollout pause, weight transfer, and resume endpoint on odd completed steps, while still publishing at interval boundaries and the final step. A configured evaluation also forces publication before evaluation.

This method intentionally trades one additional actor update of rollout-policy freshness for lower publication overhead. `--max-staleness 2` bounds the existing asynchronous pipeline, while TIS, loss, reward, and clipping metrics are correctness guardrails rather than evidence of long-horizon convergence equivalence.

The logs record 21 weight publications per zero-KL job and 11 per optimized job, including the common initialization publication.

All variants keep the weight-update buffer fixed at 512 MiB; the previously tested 1 GiB buffer is intentionally excluded from this experiment.

## Rejected directions

- Increasing the weight-update buffer from 512 MiB to 1 GiB improved throughput by only +1.12% across three trials.
- Increasing train/log-prob budgets to 12288/24576 reduced throughput by -1.52% versus `zero_kl`: log-prob forward became faster, but actor training regressed enough to outweigh it.

## Correctness guardrails

All 9 component jobs completed 20 steps. Unexpected non-finite metrics: 0; runtime errors: 0.
Every job produced 640 samples. Per-run mean TIS ranges from 0.999837 to 1.000155; maximum mean TIS clip fraction is 0.000000.
The existing unknown-device `perf/device_peak_tflops=inf` sentinel is counted separately and is not treated as a training anomaly.
This short experiment establishes runtime equivalence guards, not long-horizon convergence equivalence.

## Reproduction and rollback

```bash
cd /home/zhengbaowei/relax_ft/Relax
CUDA_VISIBLE_DEVICES=2,3 TOTAL_TRIALS=3 bash benchmarks/task22_hybrid_async_text/run_paired_trials.sh
```

Run `TASK22_VARIANT=zero_kl UPDATE_WEIGHTS_INTERVAL=1` to disable only interval-two publication, or `TASK22_VARIANT=baseline` to roll back both changes. Raw logs, manifests, submitted commands, and one-second GPU samples are under `benchmark_artifacts/task22-hybrid-async-text-v3/`.

Generated files: `summary.csv`, `step_metrics.csv`, and `throughput_curves.svg`.
