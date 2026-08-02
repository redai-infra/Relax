# Task 22 Hybrid-async text performance report

## Result

Three paired trials on commit `92b39cc0c270a553b4918caa0c9ffed4448b58ff` show that pruning the zero-coefficient KL reference path changes response throughput by +7.10% and E2E latency by -6.25%. Applying the 12288/24576 train/log-prob token budgets then changes throughput by another -1.52% and latency by +1.67%. The combined change is +5.46% throughput and -4.69% latency versus baseline.
Acceptance: **NOT MET** against the frozen target of at least +5% response throughput for `optimized` versus `zero_kl`.

| variant   | train/log-prob budget | microbatches | framework/E2E step (s) | response tok/s | samples/s | GPU util | actor peak MiB | rollout peak MiB |
| --------- | --------------------: | -----------: | ---------------------: | -------------: | --------: | -------: | -------------: | ---------------: |
| baseline  |             8192/8192 |         3.00 |            3.912/3.879 |         4185.5 |     8.179 |    55.6% |          43978 |            80621 |
| zero_kl   |             8192/8192 |         3.00 |            3.652/3.636 |         4482.5 |     8.762 |    57.3% |          43974 |            80621 |
| optimized |           12288/24576 |         1.67 |            3.706/3.697 |         4414.2 |     8.634 |    59.6% |          56414 |            80623 |

## Per-run evidence

| variant   | run | steps | E2E p50/p95 (s) | response tok/s |  reward |         loss |      TIS | samples | errors |
| --------- | --: | ----: | --------------: | -------------: | ------: | -----------: | -------: | ------: | -----: |
| baseline  |   1 |    20 |     4.000/4.000 |         4166.8 | -0.9943 | -1.40967e-05 | 1.000232 |     352 |      0 |
| baseline  |   2 |    20 |     4.000/4.000 |         4185.2 | -1.0000 |            0 | 1.000040 |     352 |      0 |
| baseline  |   3 |    20 |     4.000/4.000 |         4204.6 | -1.0000 |            0 | 0.999919 |     352 |      0 |
| zero_kl   |   1 |    20 |     4.000/4.000 |         4547.2 | -1.0000 |            0 | 0.999962 |     352 |      0 |
| zero_kl   |   2 |    20 |     4.000/4.000 |         4446.1 | -1.0000 |            0 | 0.999839 |     352 |      0 |
| zero_kl   |   3 |    20 |     4.000/4.000 |         4454.4 | -1.0000 |            0 | 1.000071 |     352 |      0 |
| optimized |   1 |    20 |     4.000/4.000 |         4432.8 | -0.9943 | -1.17127e-06 | 1.000015 |     352 |      0 |
| optimized |   2 |    20 |     4.000/4.000 |         4412.5 | -1.0000 |            0 | 0.999983 |     352 |      0 |
| optimized |   3 |    20 |     4.000/4.000 |         4397.3 | -0.9943 | -7.40283e-06 | 0.999894 |     352 |      0 |

## Paired changes

| run | zero-KL throughput | token-budget throughput | total throughput | total latency |     seed |
| --: | -----------------: | ----------------------: | ---------------: | ------------: | -------: |
|   1 |             +9.13% |                  -2.51% |           +6.38% |        -6.98% | 20260802 |
|   2 |             +6.23% |                  -0.76% |           +5.43% |        -4.65% | 20260803 |
|   3 |             +5.94% |                  -1.28% |           +4.58% |        -2.38% | 20260804 |

## Fixed workload and method

- Hardware: physical GPUs 2 and 3, both NVIDIA RTX PRO 6000 Blackwell; actor 1 GPU, rollout 1 GPU.
- Host: 128 logical CPUs (INTEL(R) XEON(R) GOLD 6530), 503.5 GiB RAM.
- Runtime: Python 3.12.3; Torch 2.11.0+cu130 (CUDA 13.0); driver 580.173.02; Ray 2.56.1; SGLang 0.5.12.post1; Transformers 5.6.0.
- Model: `/home/zhengbaowei/model/Qwen3-0.6B`, model SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- Data: ModelScope `AI-ModelScope/gsm8k` main subset of 16 prompts, SHA256 `c992e09c748ae19e82ad6d4fda099eae01ce987414372a47834c901403e2c7e4`; no hand-written large dataset.
- Each component run: 20 steps, 8 prompts/step, 4 samples/prompt, effective batch 32, response cap 512.
- Async policy: Hybrid, max staleness 2, TIS enabled, weight update interval 1.
- Performance window: logged steps 5-15 inclusive (11 observations). Primary throughput uses high-resolution framework `perf/step_time`; the E2E interval from actor completion N-1 to N is secondary because those completion logs have only one-second timestamp resolution.
- GPU window: actor completion 4 through completion 15. Utilization and peak memory use only this window; utilization includes idle-zero samples.
- Trial order uses a three-way rotation to reduce warm-cache and order bias.

## Optimization rationale

`baseline` deliberately preserves the original misconfiguration: `--use-kl-loss --kl-loss-coef 0.00` plus `--ref-load`. `zero_kl` removes the unused reference path while retaining the 8192-token dynamic-batch budget. Because the KL term is multiplied by exactly zero, this removes compute but does not change the scalar objective.

`optimized` builds on `zero_kl` and uses a 12288-token training budget plus a 24576-token forward-only log-prob budget. The role-specific budgets target two backward microbatches and one forward microbatch instead of three each, reducing scheduling and kernel-launch overhead without changing samples, generated-token caps, global batch, optimizer updates, staleness, or weight-publication frequency.

All variants keep the weight-update buffer fixed at 512 MiB; the previously tested 1 GiB buffer is intentionally excluded from this experiment.

## Correctness guardrails

All 9 component jobs completed 20 steps. Unexpected non-finite metrics: 0; runtime errors: 0.
The existing unknown-device `perf/device_peak_tflops=inf` sentinel is counted separately and is not treated as a training anomaly.
This short experiment establishes runtime equivalence guards, not long-horizon convergence equivalence.

## Reproduction and rollback

```bash
cd /home/zhengbaowei/relax_ft/Relax
CUDA_VISIBLE_DEVICES=2,3 TOTAL_TRIALS=3 bash benchmarks/task22_hybrid_async_text/run_paired_trials.sh
```

Run `TASK22_VARIANT=zero_kl MAX_TOKENS_PER_GPU=8192` to disable only the token-budget optimization, or `TASK22_VARIANT=baseline` to roll back both changes. Raw logs, manifests, submitted commands, and one-second GPU samples are under `benchmark_artifacts/task22-hybrid-async-text-v2/`.

Generated files: `summary.csv`, `step_metrics.csv`, and `throughput_curves.svg`.
