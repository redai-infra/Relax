# Hybrid-Async Multimodal Perf Comparison — H20 (Qwen3-VL-4B)

Saved 2026-07-29 before shutting down the H20 remote server (region-42.seetacloud.com).

## Environment

- 2x NVIDIA H20 (96GB each), Hopper/sm_90
- Model: Qwen/Qwen3-VL-4B-Instruct
- Dataset: lmms-lab/multimodal-open-r1-8k-verified (converted via scripts/tools/process_openr1.py)
- Script: scripts/training/multimodal/run-qwen3-vl-4B-2xgpu-openr1mm-hybrid-async.sh
- Transformer Engine rebuilt from source for sm_90 (environment image was originally built targeting a different GPU arch)

## Config used for all 4 runs

- `--attention-backend auto`, `--recompute-granularity selective`
- prompt_len=2048, response_len=4096, context_len=6144, max-tokens-per-gpu=6144
- rollout-batch-size=16, global-batch-size=64, n-samples-per-prompt=4
- num-rollout=30, num-iters-per-train-update=1 (syncs weights every iteration)
- Checkpoint saving disabled (--save removed) — not needed for perf benchmarking, also avoids disk exhaustion
- --use-clearml enabled

## Code states

- `actor_baseline.py` — relax/backends/megatron/actor.py at HEAD (no hybrid-async weight-sync overlap optimization)
- `actor_optimized.py` — same file with the hybrid-async optimization (commit 51afba8: "perf(hybrid): overlap actor->rollout weight sync with next step")

## Runs

| Log                         | Code      | Result               |
| --------------------------- | --------- | -------------------- |
| logs/baseline_h20_run1.log  | baseline  | Job succeeded, 0 OOM |
| logs/baseline_h20_run2.log  | baseline  | Job succeeded, 0 OOM |
| logs/optimized_h20_run1.log | optimized | Job succeeded, 0 OOM |
| logs/optimized_h20_run2.log | optimized | Job succeeded, 0 OOM |

## Results (avg over 30 steps, excluding step 0 which includes warmup/first-load overhead)

| Metric                  | baseline avg | optimized avg | delta                         |
| ----------------------- | ------------ | ------------- | ----------------------------- |
| `perf/step_time`        | 49.90s       | 53.26s        | **+6.75% (optimized slower)** |
| `perf/actor_train_time` | 40.37s       | 43.38s        | **+7.46% (optimized slower)** |

Raw extracted values: `perf_extract.txt` (step_time), `actor_train_extract.txt` (actor_train_time).

## Finding

On H20 (high NVLink bandwidth, weight broadcast itself is cheap) with weight sync forced every
iteration (`--num-iters-per-train-update 1`), the hybrid-async overlap optimization measured a
**net regression** (~7%), not an improvement. Likely explanation: the optimization's fixed
per-sync overhead (CPU-side double-buffer snapshot copy via `weights_backuper.copy()`, plus
ThreadPoolExecutor submit/join bookkeeping) is paid every iteration regardless of overlap, while
the benefit (hiding weight-push latency behind next-iteration compute) shrinks when the
underlying broadcast is already fast and frequent. This is the opposite of the earlier RTX 5090
2B-model test (which showed a small ~3.8% improvement, but was measured on a memory-constrained
setup requiring unfused attention + full recompute workarounds that inflated per-step compute
time and diluted the effect in the other direction).

Suggests the optimization's benefit is conditional: helpful when weight-sync latency is large
relative to compute (slower interconnect, bigger models, less frequent syncs), and can be a net
negative when sync is cheap and frequent. Worth stating this boundary explicitly in the
GitHub issue rather than a blanket "faster" claim.

## Prior RTX 5090 (2B model) results, for reference

(Original logs are gone — that server was already stopped before this H20 run. Numbers were
computed from log timestamps during that session and reported inline in conversation.)

- baseline avg step time: ~33.3s/step
- optimized avg step time: ~32.0s/step (~3.8% faster)
- Required workarounds due to 32GB VRAM limit: `--attention-backend unfused`,
  `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`,
  `--max-tokens-per-gpu 3072` — all of which inflate per-step compute time and were flagged as
  likely suppressing the optimization's true relative benefit.

______________________________________________________________________

## 2026-07-30 — DCS-based weight sync (replaces the ThreadPoolExecutor overlap above)

The `51afba8` ThreadPoolExecutor overlap above was reverted. This round tests a different
approach: reuse the same DCS (Distributed Checkpoint Service) NCCL/GLOO device-direct broadcast
that pure fully-async mode already uses for `update_weights_fully_async`, instead of hybrid's old
CUDA-IPC `UpdateWeightFromTensor` push. Rationale and design: `d4b741b` ("perf(hybrid): push
actor->rollout weights via DCS"). Final working commit after fixing 5 real-hardware-only bugs:
`2c27d43` (chain: `d4b741b` → `9172abb` → `1438e84` → `5b85208` → `d3c4861` → `2c27d43`).

### Code states

- baseline: commit `02e2408` (pre-DCS hybrid, synchronous `UpdateWeightFromTensor` CUDA-IPC push)
- optimized: commit `2c27d43` (DCS-based synchronous push, same branch
  `feat/task21-hybrid-async-multimodal-perf`)

### Bugs found and fixed during real-GPU validation

All five were invisible to static/CPU-only review in the prior session and only surfaced once
actually run against real Megatron-Bridge multimodal weights on GPU:

1. **`9172abb`** — `set_rollout_manager()` still gated the init-time `update_weights()` call on
   `(not fully_async or hybrid)`, but hybrid no longer has a `weight_updater` (that got replaced
   by the DCS `checkpoint_engine_client`) → `AttributeError: 'MegatronTrainRayActor' object has no attribute 'weight_updater'`. Fixed by aligning hybrid's condition with fully_async's (both skip
   the init push; first sync happens lazily via DCS on the first real train step).
2. **`1438e84`** — `DeviceDirectBackend._named_params_and_buffers()`'s `weights_getter` branch fed
   the NCCL/GLOO broadcast CPU-pinned tensors straight from `TensorBackuper`, but the
   broadcast/`all_gather_param` requires device tensors → `RuntimeError: No backend type associated with device type cpu`. Fixed by adding an H2D copy (`tensor.to(self.device)`) when
   the source is a CPU tensor, mirroring `UpdateWeightFromTensor`'s equivalent copy for its own
   CUDA-IPC push.
3. **`5b85208` / `d3c4861`** (two failed attempts before the real fix) — `all_gather_param`
   unconditionally asserts `hasattr(param, "tensor_model_parallel")` for TP-gather, but Megatron
   dynamically attaches `tensor_model_parallel`/`partition_dim`/`partition_stride`/`parallel_mode`
   as plain Python attributes on live `nn.Parameter` objects — not part of tensor storage, so
   `torch.empty_like()` (used by `TensorBackuper.backup()` to build its CPU snapshot) silently
   drops them → `AssertionError`. First attempt copied the attrs at snapshot-creation time; still
   failed because the live param itself hadn't been backfilled with defaults yet at that point for
   this model's bridge-mode word_embeddings. Second attempt called Megatron's own
   `set_defaults_if_not_set_tensor_model_parallel_attributes` on the snapshot; still failed with
   the *same* assertion.
4. **`2c27d43`** — the actual root cause of #3: the H2D-copy fix from bug #2
   (`tensor.to(self.device)`) itself allocates a **brand-new** Tensor object, which drops
   whatever attributes bugs #3's fixes had just attached to the CPU snapshot — the attribute loss
   was happening one step further downstream than either #3 fix attempt was looking. Fixed by
   re-copying the TP attributes onto the device-moved tensor via Megatron's own
   `copy_tensor_model_parallel_attributes(moved, tensor)` plus a manual `parallel_mode` copy.
   Confirmed working live: training passed step 0's weight sync and proceeded cleanly through all
   30 steps on both runs.

### Runs

| Log                              | Code      | Result                               |
| -------------------------------- | --------- | ------------------------------------ |
| logs/baseline_run1_20260730.log  | `02e2408` | Job succeeded, 30/30 steps, 0 errors |
| logs/baseline_run2_20260730.log  | `02e2408` | Job succeeded, 30/30 steps, 0 errors |
| logs/optimized_run1_20260730.log | `2c27d43` | Job succeeded, 30/30 steps, 0 errors |
| logs/optimized_run2_20260730.log | `2c27d43` | Job succeeded, 30/30 steps, 0 errors |

### Results (avg over 30 steps, excluding step 0)

| Metric                     | baseline avg | optimized avg              | delta                         |
| -------------------------- | ------------ | -------------------------- | ----------------------------- |
| `perf/step_time`           | 51.32s       | 52.34s                     | **+1.99% (optimized slower)** |
| `perf/actor_train_time`    | 41.58s       | 41.53s                     | -0.13% (no change)            |
| `perf/log_probs_time`      | 7.62s        | 7.93s                      | **+4.11% (optimized slower)** |
| `perf/train_wait_time`     | 9.47s        | 10.54s                     | **+1.07s (optimized slower)** |
| `perf/update_weights_time` | 1.09s        | *(not logged — see below)* | n/a                           |

### Finding: the DCS swap did **not** deliver the expected win — real regression, not noise

Both optimized runs land consistently around 52.2–52.5s step_time, vs. both baseline runs at
50.0–52.6s (avg 51.3s) — this is a small but **reproducible** ~2% regression, not run-to-run
variance. `log_probs_time`/`log_probs_tflops` are also consistently ~4% worse in both optimized
runs. This directly contradicts the premise of the optimization (see `d4b741b` and the original
plan): swapping the CUDA-IPC push for DCS's NCCL/GLOO broadcast was expected to be at least
neutral, ideally cheaper, since fully-async mode already uses the identical path.

**Root cause, found by reading the code, not just the numbers:** `train_hybrid`
(`relax/backends/megatron/actor.py`, near line 1348) calls
`run(self.checkpoint_engine_client.update_weights_for_rollout(rollout_only=True))` **synchronously**
in the main line of the function, *outside* the `timer("train")` / `inverse_timer("train_wait")`
block — this was a deliberate "hard swap, no ThreadPoolExecutor" design decision (see the plan:
"DCS's own broadcast/stream handling is what's supposed to provide any overlap, not a
`ThreadPoolExecutor` wrapper"). In practice, DCS's broadcast does **not** provide free overlap
either — it's still a blocking call from the caller's perspective. Because the push happens after
the current step's `timer("train")` closes, its cost doesn't show up as its own metric (there is
no `perf/update_weights_time` field anywhere in the optimized logs — confirmed by diffing the full
set of `perf/*` keys between baseline and optimized logs, they're otherwise identical). Instead, it
shows up as inflated `train_wait_time` on the **next** step: optimized's `train_wait_time` is
~1.07s higher than baseline's (10.54s vs 9.47s) — almost exactly matching baseline's own
`update_weights_time` (~1.09s). The weight-push cost didn't go away; it just moved into an
uninstrumented gap and is (if anything) slightly more expensive as measured by total step_time,
plus something in the log_probs phase got ~4% slower (not yet root-caused — possibly PCIe/NVLink
contention from the CPU→GPU H2D copies added in bug #2/#4, or memory-bandwidth pressure from the
extra `tensor.to(device)` allocation happening once per parameter every sync).

**Conclusion:** on this 2xH20 setup, hard-swapping hybrid's weight sync from CUDA-IPC
`UpdateWeightFromTensor` to a synchronous DCS push is a **net regression** (~2% slower step time),
not an improvement. The original ThreadPoolExecutor approach (tested 2026-07-29, also a regression
at this scale) and this DCS approach share the same underlying issue: neither actually removes or
overlaps the weight-sync cost off the critical path — one paid CPU-thread overhead for no real
concurrency benefit, the other trades one synchronous push mechanism for another with no
measured net gain. A real overlap win would require either (a) making the DCS push genuinely
asynchronous (start the broadcast, return immediately, join before the *following* step's weight
read — not before the *next* iteration's train step) or (b) confirming DCS's broadcast has lower
fixed latency than CUDA-IPC at larger scale (more GPUs / bigger models) where the fixed per-call
overhead seen here would be a proportionally smaller cost. Recommend reporting this null result on
GitHub issue #21 rather than merging this as a performance win.

______________________________________________________________________

## 2026-07-30 (later same day) — split `train_hybrid`'s training step into `num_iters_per_train_update` iterations

### What we set out to do

`docs/en/guide/hybrid-training.md`'s "Next Steps" section documented a known gap: hybrid mode's
`num_iters_per_train_update` flag was supposed to control how many pieces the *training* step gets
split into (so optimizer updates could be pipelined with TransferQueue consumption and peak
training-side memory would drop), but the merged batch was actually still trained in a single
`train(...)` call regardless of that flag's value.

### What we found before writing any code

Reading `relax/backends/megatron/actor.py`'s `train_hybrid` and `relax/backends/megatron/data.py`'s
`get_data_iterator` showed the docs were stale, not just incomplete:

- `train_hybrid`'s **forward** phase (phase 1) sub-batch count already comes from
  `build_rollout_minibatch_plan`, which is driven by `--num-steps-per-rollout` (falling back to
  `rollout_batch_size * n_samples_per_prompt // global_batch_size` if unset) — **not** by
  `--num-iters-per-train-update` at all.
- `get_data_iterator` already supports splitting a merged batch into multiple separate optimizer
  steps: if `rollout_data` carries a `ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY` list of length N, it
  builds an N-length `num_microbatches` schedule, and `train()`'s loop (`relax/backends/megatron/ model.py`) already runs one full `train_one_step` (one real `optimizer.step()`) per entry. This
  machinery was added by `679022e` ("feat(megatron): split rollout mini batches"), which landed
  *after* the hybrid-training guide was written — the guide was simply never updated.
- `train_hybrid` already sets `ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY` from phase 1's rollout-mini
  counts, so whenever `--num-steps-per-rollout` > 1, the merged training step **already** ran as
  multiple optimizer steps. `--num-iters-per-train-update` genuinely had zero effect on hybrid mode
  — it's only read by pure fully-async's separate `actor_fwd`/`reference` services
  (`compute_ref_log_prob`/`compute_actor_log_prob`) and the standalone `advantages` component, none
  of which are deployed under `--hybrid`.
- Two of the six existing hybrid launch scripts (`run-qwen35-27B-8xgpu-openr1mm-hybrid-async.sh`,
  `run-qwen3-vl-2b-2xgpu-openr1mm-hybrid-async.sh`) pass `--num-iters-per-train-update 2` expecting
  it to matter — it silently didn't.

### What we implemented (kept)

`relax/backends/megatron/actor.py`, `train_hybrid`, phase 3: after phase 1/2 finish (forward +
merge + advantages), the merged batch is now **re-chunked** into `num_iters_per_train_update`
equal-size pieces (raising `RuntimeError` if it doesn't divide evenly) and that list is what gets
written to `ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY` for `get_data_iterator`/`train()` to consume —
independent of however many sub-batches phase 1 used. This makes `--num-iters-per-train-update`
control the training-step split exactly as documented, while `--num-steps-per-rollout` continues to
control the forward-phase split, unchanged. Default `num_iters_per_train_update=1` reproduces the
old single-optimizer-step behavior exactly (verified: all 6 existing hybrid scripts pass the new
divisibility check).

### What we attempted and reverted: making the DCS push asynchronous too

Since a genuinely async, double-buffered weight push (`hybrid_send_0`/`hybrid_send_1`, ported from
the reverted `51afba8` ThreadPoolExecutor design) was flagged as the natural next step to *also*
overlap weight sync with the now-pipelined training iterations, we implemented it: a background
`ThreadPoolExecutor(max_workers=1)` fires `checkpoint_engine_client.update_weights_for_rollout(...)`
against an isolated CPU snapshot tag, joined at the top of the *next* iteration's push instead of
blocking immediately. It worked correctly (verified on real GPU: no deadlocks, no race conditions,
correct weight updates every rollout, tested with `--num-iters-per-train-update 2`) but delivered no
speed benefit, and was **reverted** — see results below. The code currently in this repo pushes
weights **synchronously**, same as `2c27d43`.

### Runs

| Log                                      | Config                                                      | Result                     |
| ---------------------------------------- | ----------------------------------------------------------- | -------------------------- |
| `logs/syncdcs_split_run1_20260730.log`   | sync DCS push, `--num-iters-per-train-update 2`             | Job succeeded, 30/30 steps |
| `logs/syncdcs_split_run2_20260730.log`   | sync DCS push, `--num-iters-per-train-update 2`             | Job succeeded, 30/30 steps |
| `logs/optimized_async_run1_20260730.log` | async DCS push (reverted), `--num-iters-per-train-update 2` | Job succeeded, 30/30 steps |
| `logs/optimized_async_run2_20260730.log` | async DCS push (reverted), `--num-iters-per-train-update 2` | Job succeeded, 30/30 steps |

### Results (avg over 30 steps, excluding step 0)

| Metric                  | baseline (`02e2408`, CUDA-IPC) | DCS sync, no split (`2c27d43`) | **DCS sync + split×2 (kept)**      | DCS async + split×2 (reverted) |
| ----------------------- | ------------------------------ | ------------------------------ | ---------------------------------- | ------------------------------ |
| `perf/step_time`        | 51.32s                         | 52.34s                         | **55.24s (+5.5% vs DCS-no-split)** | 56.03s (+7.0% vs DCS-no-split) |
| `perf/actor_train_time` | 41.58s                         | 41.53s                         | **44.34s (+6.8%)**                 | 44.86s (+8.0%)                 |
| `perf/log_probs_time`   | 7.62s                          | 7.93s                          | **8.09s (+2.0%)**                  | 8.24s (+3.9%)                  |

### Finding

Splitting the training step (`--num-iters-per-train-update 2`) is a **net regression** here, and it
is a *reproducible* one — both sync-push runs land at 56.42s/54.07s, both async-push runs land at
56.36s/55.69s. Critically, **sync vs. async push barely matters** (55.24s vs 56.03s, ~0.8s apart —
within run-to-run noise): the regression is dominated by the training split itself, not by whether
the weight push blocks. This makes sense given how the split actually pays for itself: each extra
optimizer step re-pays the DP gradient all-reduce, and — since this script enables
`--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d` — an extra round of CPU-offloaded
optimizer-state H2D/D2H copies per rollout, while each step's batch is half the size (worse compute
of overhead ratio at this small 2-GPU scale). The theoretical benefit ("pipeline optimizer updates
with TransferQueue consumption, lower peak training memory") does not apply to `train_hybrid`
specifically: by the time phase 3 begins, phase 1 has already fully drained TransferQueue for this
rollout, so there is no data-fetch to overlap with — the split's only possible benefit here is
reduced peak memory, which this 96GB-per-GPU / 4B-model config never needed.

**Conclusion:** the code fix (making `--num-iters-per-train-update` actually control `train_hybrid`'s
training split, instead of silently doing nothing) is a correctness/documentation-accuracy fix worth
keeping regardless of perf — two existing launch scripts were already passing this flag believing it
did something. But turning it on (setting it above 1) is a **measured regression** at this 2xH20
scale and should not be recommended as a default; `--num-iters-per-train-update 1` (the default)
preserves the original behavior exactly. The async DCS push was implemented, validated correct on
real hardware, and then reverted — it neither helped nor hurt beyond the split's own cost, so there
is no reason to carry its added complexity (double-buffering, background thread, extra guard
conditions for `--disable-weights-backuper`/`--keep-old-actor`) in the codebase.

This is the third confirmed null result on this issue (after the `51afba8` ThreadPoolExecutor
overlap and the `d4b741b`→`2c27d43` DCS weight-sync swap): every "overlap"/"split" idea tried so far
has been a net regression at 2-GPU scale, because the fixed fetch/sync/optimizer-step overhead being
paid more often outweighs any concurrency benefit when compute-per-step is already small and GPU
memory is not the bottleneck.
