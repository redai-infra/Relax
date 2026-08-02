# Task 22 Hybrid-async text performance report

## Result

Three paired trials on commit `2b1c7955e2ff84ea6ef6240b9575e6329f260ab5` show a response-throughput change of +1.12% and an end-to-end step-latency change of -1.05% after increasing the weight-update buffer from 512 MiB to 1 GiB.

| variant   | E2E step (s) | response tok/s | samples/s | weight update (s) | GPU util | actor GPU | rollout GPU | peak MiB |
| --------- | -----------: | -------------: | --------: | ----------------: | -------: | --------: | ----------: | -------: |
| baseline  |        3.958 |         4135.7 |     8.086 |             0.692 |    49.7% |     73.5% |       25.8% |    80639 |
| optimized |        3.917 |         4182.1 |     8.172 |             0.684 |    50.4% |     73.5% |       27.3% |    80747 |

## Per-run evidence

| variant   | run | steps | E2E p50/p95 (s) | response tok/s |  reward |         loss |      TIS | samples | errors |
| --------- | --: | ----: | --------------: | -------------: | ------: | -----------: | -------: | ------: | -----: |
| baseline  |   1 |    10 |     4.000/5.000 |         4091.6 | -1.0000 |            0 | 1.000127 |     320 |      0 |
| baseline  |   2 |    10 |     4.000/4.000 |         4090.3 | -1.0000 |            0 | 1.000058 |     320 |      0 |
| baseline  |   3 |    10 |     4.000/4.000 |         4225.2 | -1.0000 |            0 | 0.999926 |     320 |      0 |
| optimized |   1 |    10 |     4.000/4.000 |         4225.9 | -0.9938 | -6.49178e-06 | 1.000003 |     320 |      0 |
| optimized |   2 |    10 |     4.000/5.000 |         4093.0 | -1.0000 |            0 | 0.999912 |     320 |      0 |
| optimized |   3 |    10 |     4.000/4.000 |         4227.5 | -1.0000 |            0 | 0.999960 |     320 |      0 |

## Paired changes

| run | response throughput | E2E step latency |     seed |
| --: | ------------------: | ---------------: | -------: |
|   1 |              +3.28% |           -3.12% | 20260802 |
|   2 |              +0.07% |           +0.00% | 20260803 |
|   3 |              +0.05% |           +0.00% | 20260804 |

## Fixed workload and method

- Hardware: physical GPUs 2 and 3, both NVIDIA RTX PRO 6000 Blackwell; actor 1 GPU, rollout 1 GPU.
- Host: 128 logical CPUs (INTEL(R) XEON(R) GOLD 6530), 503.5 GiB RAM.
- Runtime: Python 3.12.3; Torch 2.11.0+cu130 (CUDA 13.0); driver 580.173.02; Ray 2.56.1; SGLang 0.5.12.post1; Transformers 5.6.0.
- Model: `/home/zhengbaowei/model/Qwen3-0.6B`, model SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- Data: ModelScope `AI-ModelScope/gsm8k` main subset of 16 prompts, SHA256 `c992e09c748ae19e82ad6d4fda099eae01ce987414372a47834c901403e2c7e4`; no hand-written large dataset.
- Each component run: 10 steps, 8 prompts/step, 4 samples/prompt, effective batch 32, response cap 512.
- Async policy: Hybrid, max staleness 2, TIS enabled, weight update interval 1.
- Stable observations: completed steps 2-9, where step N latency is the interval from actor completion N-1 to N. This includes post-train coordination and weight publication omitted by framework `perf/step_time`.
- GPU window: actor completion 1 through completion 9. Utilization is the arithmetic mean of all one-second samples, including idle zeros.
- Trial order is baseline/optimized, optimized/baseline, baseline/optimized to reduce order bias.

## Optimization rationale

Both variants load the same reference model and execute the same reference forward, actor training, rollout generation, and per-step weight publication. The baseline uses the framework's 512 MiB weight-update buffer; the optimized variant uses 1 GiB. For the 1.40 GiB checkpoint this reduces publication chunking while preserving two-stage conversion/transfer overlap. No sample, token, batch, step, objective, reward, staleness, or synchronization-frequency setting changes.

`perf/update_weights_time` is a secondary diagnostic. The framework resets metrics before synchronization, so the value logged in actor perf record N measures the publication preceding that record's training interval; it is not used as the primary latency denominator.

## Correctness guardrails

All 6 component jobs completed their configured steps. Unexpected non-finite metrics: 0; runtime errors: 0.
The existing unknown-device `perf/device_peak_tflops=inf` sentinel is counted separately and is not treated as a training anomaly.
This short experiment establishes runtime equivalence guards, not long-horizon convergence equivalence.

## Reproduction and rollback

```bash
cd /home/zhengbaowei/relax_ft/Relax
CUDA_VISIBLE_DEVICES=2,3 TOTAL_TRIALS=3 bash benchmarks/task22_hybrid_async_text/run_paired_trials.sh
```

Run `TASK22_VARIANT=baseline` (512 MiB) to roll back the buffer change. Raw logs, manifests, submitted commands, and one-second GPU samples are under `benchmark_artifacts/task22-hybrid-async-text/`.

Generated files: `summary.csv`, `step_metrics.csv`, and `throughput_curves.svg`.
