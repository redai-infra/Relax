# Task 21: Hybrid-async multimodal pipeline benchmark results

## Method

The reference Hybrid path receives rollout samples through the TransferQueue,
runs any required old-policy/reference forwards on actor GPUs, computes
advantages over the merged rollout batch, and then performs the optimizer
forward/backward pass. This PR isolates three changes to that path:

- **ProcessorPool (P):** multimodal Hugging Face preprocessing runs in spawned
  worker processes and returns the same prompt IDs and training tensors through
  an IPC-safe payload. This removes Python GIL contention from the rollout event
  loop without changing model execution or the sample contract.
- **Train-forward reuse (R):** in the guarded true-on-policy case, the dedicated
  actor old-policy forward is omitted. The optimizer forward supplies both the
  differentiable policy log-probability and its detached old-policy value, so
  their PPO ratio is exactly one while gradients still flow through the current
  policy. When rollout and train weights may differ, TIS continues to compare
  the detached train-side value with the rollout-engine log-probability.
- **Schedule-matched chunk forwarding (S):** the actor restores its weights once,
  then fetches and forwards each fixed-size TransferQueue chunk as it becomes
  available. Chunks are merged by global sample index, and training replays the
  recorded dynamic-microbatch grouping so scheduling does not introduce an
  artificial PPO ratio.

All controls are default-off. Reuse fails closed unless forwards are
deterministic, exactly one optimizer mini maps to each rollout partition, and
the rollout log-probabilities have the expected context-parallel layout. Chunk
forwarding has its own topology and feature compatibility checks and cannot be
combined with reuse in the reference launcher.

## 60-step steady-state result

The superseding steady-state campaign ran B/P/P+R/P+S on benchmark commit
`e2be8cd158609cc2dfae72b7ba92df72cacb3091` with paired seeds `20260820` and
`20260821`. Every formal run used a fresh process, 60 optimizer updates, physical
RTX A6000 GPUs `0,1,2,4,7`, actor TP2 x CP2 x DP1 on four GPUs, and one rollout
engine on one GPU. Steps `0-9` are warmup; the declared steady window is `10-59`.

| Condition | Steady token throughput mean (range) | Mean step time | Paired geometric-mean result |
| --- | ---: | ---: | ---: |
| B | 1,103.25 (964.68-1,241.83) token/s | 179.62 s | reference |
| P | 1,157.92 (1,131.16-1,184.69) token/s | 170.52 s | +5.77% vs B |
| P+R | 1,522.04 (1,475.17-1,568.91) token/s | 129.21 s | **+39.00% vs B**, **+31.42% vs P** |
| P+S | 1,229.04 (1,136.58-1,321.50) token/s | 162.82 s | +11.97% vs B, +5.87% vs P |

P+R is the only path that improves token throughput against both B and P in
both paired seeds. Its per-seed gains are `+52.92%` and `+26.34%` vs B, and
`+24.52%` and `+38.70%` vs P. The paired geometric-mean step-time reduction is
`27.43%` vs B and `24.19%` vs P.

ProcessorPool alone has a positive paired geometric mean but is not consistent
per seed (`+22.81%`, `-8.91%` vs B). P+S is positive against B in both seeds
(`+17.82%`, `+6.42%`), but its incremental result over P changes sign
(`-4.06%`, `+16.83%`). Those ablations remain reported, but the sustained
throughput claim is limited to P+R.

All eight included runs completed 60/60 optimizer steps with
`training_exit_status=0`, `validation_exit_status=0`, `exit_status=0`, finite
loss/reward/gradient series, and `validation=passed`. One S60 attempt interrupted
by an external GPU lease was excluded; the automatic retry completed and is the
formal seed-`20260820` S60 run. Curves below show all steps directly from
TensorBoard scalar exports without smoothing or interpolation; the band is the
two-seed range and the shaded `0-9` interval is warmup.

![60-step gradient norm](sixty-step/task21_60step_grad_norm.png)

![60-step training loss](sixty-step/task21_60step_loss.png)

![60-step raw reward](sixty-step/task21_60step_reward.png)

The per-step data, compact summary, full campaign report, and paired CSVs are
available in
[`sixty_step_training_curves.csv`](sixty-step/sixty_step_training_curves.csv)
and [`sixty_step_summary.json`](sixty-step/sixty_step_summary.json), with the
complete analyzer output in
[`sixty_step_campaign_report.json`](sixty-step/sixty_step_campaign_report.json).

## Historical first-step protocol

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

## Historical first-step result summary

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
