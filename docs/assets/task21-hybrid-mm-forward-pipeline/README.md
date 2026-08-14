# Task 21: Hybrid-async multimodal pipeline - fresh-process first-step GPU results

## Protocol

- Performance-code commit: `780c742793fea54acdb28cb40ffb591fb909ba51`.
- Base commit: `9a5674afde12f608698ab4f60cdb9849a0eb6cb3`.
- Hardware: physical RTX A6000 GPUs `1,2,3,4,6`; four actor GPUs (`TP=2`, `CP=2`) plus one rollout GPU.
- Workload: Qwen3-VL-8B-Instruct, OpenR1-Multimodal, 256 generated samples per run, response cap 1024, one real optimizer step.
- Design: paired seeds `20260811`–`20260814`, ABBAABBA order; baseline and experiment differ only in `--hybrid-pipeline-overlap`.
- Statistics: four fresh-process paired runs, each containing one optimizer step. The analyzer validated every run and all preregistered first-step targets.

These results measure first-step latency and throughput under the fixed Qwen3-VL,
TP2 x CP2 x DP1 profile. They do not establish steady-state training
throughput. A steady-state claim requires multi-step runs with warmup excluded
and later measurement windows declared before analysis.

## Result summary

| Metric | Baseline | Experiment | Result |
|---|---:|---:|---:|
| step token throughput, arithmetic mean | 575.68 token/s | 644.69 token/s | paired geometric-mean speedup **12.01%** |
| hybrid phase-1 time, arithmetic mean | 165.38 s | 130.38 s | paired geometric-mean reduction **21.21%** |
| end-to-end step time, arithmetic mean | 342.24 s | 305.61 s | **10.70% lower** |
| producer/actor overlap | 0/4 runs | 4/4 runs | **100%** experiment overlap |
| first-step-window GPU utilization, arithmetic mean | 40.01% | 44.37% | +4.36 percentage points |
| sampled peak VRAM, maximum | 47,380 MiB | 47,140 MiB | no regression |

Every paired run exceeded the preregistered 5% throughput target: `10.22%`, `12.21%`, `12.80%`, and `12.82%`. Every pair also reduced phase-1 time by `19.11%`–`23.53%`.

The throughput plot shows both total-token and response-token throughput for each matched seed. The experiment point is above its baseline point in every pair.

![Step throughput](task21_step_throughput.png)

The phase-1 plot shows the producer/actor overlap interval, phase-1 latency, and producer lead. The experiment overlaps producer work with actor forward in all four runs, while the baseline deliberately disables that overlap.

![Phase-1 overlap](task21_phase1_overlap.png)

The GPU plot combines the 500 ms NVML time series with first-step-window utilization, idle ratio, and sampled peak VRAM. It describes these fresh-process runs only.

![GPU utilization and VRAM](task21_gpu_util_vram.png)

The quality plot pairs reward, response length, truncation, loss, gradient norm, and PPO KL by seed. Workload, reward, response length, and truncation are exact within each pair; loss differs by at most `6.21e-5`, PPO KL remains within `2.37e-11`, and every value is finite. Seed `20260814` shows a disclosed gradient-norm difference (`1.0033` vs `0.4775`) caused by different asynchronous microbatch grouping; the deterministic replay parity run controls grouping and bounds the patched-path gradient-norm difference to `0.207%`.

![Correctness and quality](task21_correctness_quality.png)

The window summary shows each paired throughput speedup and the geometric mean against the preregistered 5% threshold.

![Window and paired summary](task21_window_summary.png)

## Correctness evidence

The final replay parity dataset contains 1,024 samples across four actor ranks. Baseline/control/experiment agree exactly for tokens, response lengths, loss masks, rewards, raw rewards, advantages, returns, truncation flags, multimodal tensors, and dynamic-microbatch schedules. Loss is exactly `0.03446025028824806`; experiment log-probability maximum absolute difference is `2.3842e-7`; gradient-norm symmetric relative difference is `0.207%` (below the BF16 0.5% guardrail).

Full raw logs, manifests, TensorBoard events, timeline JSONL, NVML CSV, summaries, and generated figures are stored under `/data01/LWX/relax-task21/`. Compact paired data and checksums are included in this directory.
