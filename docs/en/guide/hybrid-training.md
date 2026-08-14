# Hybrid Training Mode

## Overview

**Hybrid mode** is a third execution mode in Relax that sits between [Colocate (Sync)](./architecture.md) and [Fully Async](./fully-async-training.md). It combines:

- the **streaming data pipeline** of Fully Async (TransferQueue + `max-staleness` for off-policy tolerance), with
- the **in-process weight sharing** of Colocate (TensorBackuper + `_switch_model`, so ref / actor_fwd / advantages all run on the actor's own GPUs).

Concretely, Actor and Rollout still run on **separate GPU placement groups** (like Fully Async), but the actor no longer ships weights to standalone ActorFwd / Reference / Advantages services. Instead it cycles a single set of weights between `actor`, `ref`, `old_actor`, and `teacher` tags via a CPU/GPU `TensorBackuper`, computing every forward pass locally and pushing weights to rollout through the sync `UpdateWeightFromTensor` path.

### Mode Comparison

| Dimension           | Colocate (Sync)                          | Fully Async                                                  | Hybrid                                                                              |
| ------------------- | ---------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| **GPU layout**      | Actor and Rollout time-share same GPUs   | Actor / Rollout / ActorFwd / Reference each have own GPUs    | Actor and Rollout on separate GPUs; ref / actor_fwd / adv share actor's GPUs        |
| **Data pipeline**   | TransferQueue, batch-synchronous         | TransferQueue + StreamingDataLoader, fully streaming         | TransferQueue optimizer minis; optional producer-chunk actor-forward pipeline       |
| **Weight sync**     | In-process tensor copy                   | NCCL broadcast via DCS (Checkpoint Engine)                   | Sync `UpdateWeightFromTensor` to rollout; TensorBackuper for ref/actor_fwd          |
| **Staleness**       | `max_staleness = 0` (strict on-policy)   | Configurable `max_staleness`                                 | Configurable `max_staleness`                                                        |
| **Roles deployed**  | `actor`, `critic`, `rollout`             | `actor`, `critic`, `rollout`, `advantages`, `reference`, `actor_fwd` | `actor`, `critic`, `rollout` (same as Colocate; ref/actor_fwd live inside actor)    |
| **`--balance-data`**| Supported                                | Not supported                                                | **Supported** (one of hybrid's reasons to exist)                                    |

### When to Use Hybrid

Pick **Hybrid** when:

- You want the throughput benefits of dedicated rollout GPUs and pipelined data flow, but
- Your model is large enough that running independent ref / actor_fwd services would waste GPUs, or
- You need `--balance-data` (load-balanced micro-batching across DP ranks), which pure Fully Async cannot provide.

Pick **Fully Async** when you have spare GPUs for separate ref / actor_fwd / advantages services and want true cross-step pipelining.

Pick **Colocate** when GPU count is tight and you can tolerate serial rollout → train cycles.

______________________________________________________________________

## Architecture

### Role Layout

Hybrid uses the same role set as Colocate — only `actor`, `critic` (optional), and `rollout` are deployed as Ray Serve services. The decision lives in `relax/core/registry.py`:

```python
def process_role(config):
    if config.hybrid:
        # hybrid mode: actor handles ref/actor_fwd internally
        # via _switch_model, only need actor + rollout services
        return ROLES_COLOCATE
    if config.fully_async:
        ...
```

But unlike Colocate, the actor and rollout placement groups are **disjoint**, matching Fully Async semantics. From `relax/core/controller.py`:

```python
if colocate and not self.config.hybrid:
    # Sync colocate: actor and rollout share GPUs via time-sharing (offload/onload)
    actor_rollout_pgs = create_placement_group(num_gpus=num_gpus)
else:
    # fully_async (pure or hybrid): actor and rollout use separate GPUs
    actor_rollout_pgs = None
```

### Flag Resolution

`--hybrid` is the only public switch. `relax/utils/arguments.py` resolves it into the two underlying flags downstream machinery already understands:

```python
if args.hybrid:
    args.fully_async = True
    args.colocate = True
```

Passing `--fully-async --colocate` directly is rejected; use `--hybrid` instead. This single-switch design keeps `args.hybrid` as the canonical hybrid-only branch in the registry, controller dispatch, and `train_hybrid` call site.

### Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        Controller (Orchestrator)                           │
│                     relax/core/controller.py                               │
│                                                                            │
│       ┌───────────────────────────────────┐    ┌────────────────────┐      │
│       │            Actor Service          │    │  Rollout Service   │      │
│       │  (own placement group, N GPUs)    │    │ (own PG, M GPUs)   │      │
│       │                                   │    │   SGLang engines   │      │
│       │  ┌────────────────────────────┐   │    └─────────┬──────────┘      │
│       │  │ TensorBackuper             │   │              ▲                 │
│       │  │  tags: actor / ref /       │   │              │                 │
│       │  │        old_actor / teacher │   │              │                 │
│       │  │  _switch_model(tag) swaps  │   │              │                 │
│       │  │  weights on the same GPUs  │   │              │                 │
│       │  └────────────────────────────┘   │              │                 │
│       │  train_hybrid():                  │              │                 │
│       │    ├─ ref forward   (switch:ref)  │              │                 │
│       │    ├─ actor forward (switch:actor)│                                │
│       │    ├─ advantages    (in-process)  │              │                 │
│       │    └─ train         (switch:actor)│                                │
│       └──────────────┬────────────────────┘              │                 │
│                      │ UpdateWeightFromTensor (sync) ───┘                  │
└──────────────────────┼─────────────────────────────────────────────────────┘
                       ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      TransferQueue (Data Plane)                           │
│  Rollout writes train_N partition incrementally ──► Actor consumes        │
│  in sub-batches via get_meta(batch_size, batch_index) with max-staleness  │
└───────────────────────────────────────────────────────────────────────────┘
```

______________________________________________________________________

## The `train_hybrid` Loop

`relax/backends/megatron/actor.py:708` implements the hybrid training step in three phases:

1. **Collect optimizer minis and compute forward log-probs**

   The actor requests each optimizer mini derived from
   `rollout_batch_size * n_samples_per_prompt / global_batch_size`. For each mini it:

   - pulls data from TransferQueue (`_get_data_from_transfer_queue("train", rollout_id, fields, batch_size, batch_index)`)
   - runs `_switch_model("ref")` (if ref weights are backed up) and computes ref log-probs
   - runs `_switch_model("teacher")` (if OPD teacher weights are backed up) and computes teacher log-probs
   - runs `_switch_model("old_actor" or "actor")` and computes current actor log-probs
   - appends the enriched mini to an in-memory list

   With the optional incremental actor-forward pipeline enabled, one optimizer
   mini is consumed in fixed sample windows aligned to the rollout producer's
   nominal transfer threshold. Physical producer put grouping may vary; this
   actor split does not change the advantage or optimizer boundary.

2. **Merge sub-batches and compute advantages globally**

   All sub-batch dicts are concatenated into one `rollout_data`, then `compute_advantages_and_returns(self.args, rollout_data)` runs once over the merged batch. This is the **key correctness reason** for the two-phase design — advantage normalization must see the full DP-group batch, not per-sub-batch slices.

3. **Train on the merged batch and push weights**

   A single `train(...)` call runs the optimizer step on the merged batch. Afterwards the actor backs up the new weights to the `actor` tag and (on the ref-update interval) refreshes the `ref` tag, then calls `self.update_weights()` to push the updated weights to rollout via `UpdateWeightFromTensor`.

The forward phase remains bounded to one fetched unit at a time, while the
merged training step preserves Colocate-style global statistics.

______________________________________________________________________

## Configuration

### Required Flags

| Flag                            | Purpose                                                                              |
| ------------------------------- | ------------------------------------------------------------------------------------ |
| `--hybrid`                      | Enable hybrid mode (resolves to `fully_async=True, colocate=True` internally)        |
| `--resource '{...}'`            | Declare `actor` and `rollout` placement groups separately, e.g. `{"actor":[1,4],"rollout":[1,4]}` |
| `--num-iters-per-train-update`  | Producer transfer threshold and the fixed actor fetch/forward chunk count |
| `--max-staleness`               | Off-policy budget (0 = strict on-policy, >0 allows staleness)                        |

### Optional but Common

| Flag                          | Notes                                                                                                    |
| ----------------------------- | -------------------------------------------------------------------------------------------------------- |
| `--balance-data`              | Supported in hybrid (rejected in pure fully-async). Enable for DP load balancing.                        |
| `--num-data-storage-units`    | Number of TransferQueue storage actors.                                                                  |
| `--use-streaming-dataset`     | Stream prompts from disk instead of loading into memory.                                                 |
| `--ref-update-interval`       | Periodically refresh the cached ref weights from the latest actor weights.                               |

### Default Overrides

When `--hybrid` is set, `relax/utils/arguments.py` defaults the following (unless the user passes them explicitly):

- `offload_train = False` and `offload_rollout = False` — actor and rollout are on separate GPUs, so no offload needed
- `compute_advantages_and_returns = True` — actor must compute advantages internally
- `fully_async = True`, `colocate = True` — derived from `--hybrid`

::: warning
`--balance-data` requires `--hybrid` if you also want a streaming pipeline. The combination `--fully-async --balance-data` (without `--hybrid`) is rejected at argument parse time.
:::

### Incremental actor-forward pipeline

Hybrid can optionally request fixed sample-count actor chunks from TransferQueue
instead of waiting for a complete optimizer mini:

| Option | Default | Purpose |
| --- | --- | --- |
| `--hybrid-pipeline-forward` | off | Fetch and forward each fixed actor chunk as soon as enough samples are ready |
| `--hybrid-pipeline-overlap` / `--no-hybrid-pipeline-overlap` | on | Forward immediately, or fetch all identical chunks first for a schedule-matched performance control |
| `--hybrid-pipeline-trace-dir PATH` | unset | Write content-free producer, fetch, restore, forward, advantage, and optimizer events |
| `--hybrid-pipeline-fetch-timeout-s SECONDS` | `600` | Fail an incomplete chunk wait with rollout/mini/chunk context |

The switch is deliberately off by default. With the reference Qwen3.5-9B
recipe (`global_batch_size=256`, `num_iters_per_train_update=2`, DP=1), the
producer targets 128 samples per transfer, while the actor always requests two
complete 128-sample chunks. Producer `async_put` grouping is intentionally not
part of the contract: `FIRST_COMPLETED` coalescing, tail flush, or backfill can
legitimately produce one, two, or more puts as long as all 256 samples and their
global-index fingerprint are conserved. The optional path:

1. restores the actor exactly once for the optimizer mini, before waiting for
   the first chunk;
2. fetches and forwards chunk 0 while rollout can continue producing chunk 1;
3. fetches and forwards chunk 1;
4. orders every per-sample field by `BatchMeta.global_indexes`;
5. records each chunk's dynamically balanced microbatch schedule, translates
   it to the canonical merged-batch indexes, and replays the same sample
   grouping and order during training;
6. computes advantages once over all 256 samples and performs one optimizer
   step.

It does not change producer transfer policy, multimodal preprocessing, pixel
tensor values, GRPO group boundaries, reward normalization, or optimizer
semantics. The additional actor fetch is intended to expose rollout/actor
overlap, not to reduce work.

The schedule replay is a correctness requirement, not a performance tuning
heuristic. Packed multimodal kernels can produce batch-shape-dependent BF16
rounding differences. Comparing chunked old-policy log-probs with a differently
packed full-batch training forward would therefore create a non-zero PPO ratio
before any weight update. Replaying the exact chunk microbatches keeps the
old-policy and training forward shapes aligned while retaining one optimizer
update over the complete mini. For this reason, the switch requires
`--use-dynamic-batch-size`; an incompatible batch mode fails during actor
startup.

`--no-hybrid-pipeline-overlap` changes only the ordering of the same chunk
operations: the actor fetches every chunk before starting any chunk forward.
It preserves the chunk-local dynamic schedules and their single merged
optimizer update. This is the registered baseline for a causal performance
comparison. Omitting `--hybrid-pipeline-forward` remains the compatibility
rollback, but its full-batch packing is not a schedule-matched performance
control.

The first implementation is intentionally limited and fails fast instead of
silently falling back:

| Dimension | Supported with the switch enabled |
| --- | --- |
| Mode | Hybrid |
| Workload | Multimodal, dynamic-batch GRPO |
| Forward roles | Actor only; no ref/KL, teacher/OPD, old actor, critic, or routing replay |
| Parallel topology | TP=2, DP=1, PP=1, VPP=1, CP=2, EP=1, ETP=1 |
| Offload | `offload_train=False`, `offload_rollout=False` |
| Dropout | Attention and hidden dropout both zero |
| Batch policy | Dynamic microbatching; exactly one fixed optimizer mini per rollout (`rollout_batch_size * n_samples_per_prompt == global_batch_size`); no partial or dynamic-global batch |
| Log-prob source | Actor-computed log-prob; no true-on-policy or rollout-log-prob shortcut |
| TensorBackuper | Normal enabled backuper with only the `actor` tag |

Chunk sizes must reconstruct the optimizer mini exactly and remain a multiple
of `n_samples_per_prompt`. Duplicate, missing, underfilled, or overfilled
`BatchMeta.global_indexes` terminate the step with an actionable error.
Startup also requires TransferQueue `>=0.1.10.dev0` with
`BatchMeta.global_indexes` and the `async_put(custom_meta=..., is_last=...)`
contract; an incompatible installation fails before Ray workers or rollout
producers can write data.

The reference launcher exposes the options as environment variables:

```bash
HYBRID_PIPELINE_FORWARD=1 \
HYBRID_PIPELINE_TRACE_DIR=/data01/LWX/relax-task21/runs/smoke/timeline \
HYBRID_PIPELINE_FETCH_TIMEOUT_S=600 \
NUM_ITERS_PER_TRAIN_UPDATE=4 \
bash scripts/training/multimodal/run-qwen35-9B-8xgpu-openr1mm-hybrid-async.sh \
  hybrid-async
```

The same launcher provides default-preserving experiment overrides so a smoke
run or paired benchmark does not require editing the script:

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_CONFIG_FILE` | `qwen35-9B.sh` | Select the model-parallel configuration script |
| `MODEL_NAME` / `MODEL_RUN_NAME` | `Qwen3.5-9B` / `qwen35-9b` | Select the checkpoint subdirectory and log prefix |
| `MODEL_CHECKPOINT_DIR` / `REFERENCE_CHECKPOINT_DIR` | `${MODEL_DIR}/${MODEL_NAME}` | Pin actor and reference inputs explicitly |
| `CHECKPOINT_SAVE` | `1` | Set to `0` to omit all `--save*` arguments and use separate rollout/TensorBoard outputs |
| `ROLLOUT_RESULT_DIR` / `TENSORBOARD_DIR` | No override while saving; `${EXP_DIR}/rollout_result` and `${EXP_DIR}/tensorboard_log` without saving | Preserve raw results and scalars in no-save runs; `TENSORBOARD_DIR` is exported for MetricsService |
| `NUM_ITERS_PER_TRAIN_UPDATE` | `2` | Set the prompt-group-aligned producer and actor chunk count for each optimizer mini |
| `HYBRID_PIPELINE_OVERLAP` | `1` | Set to `0` only with `HYBRID_PIPELINE_FORWARD=1` to run the schedule-matched no-overlap control |
| `ROLLOUT_MAX_RESPONSE_LEN` / `ROLLOUT_MAX_PROMPT_LEN` / `ROLLOUT_MAX_CONTEXT_LEN` | `10240` / `2048` / `12288` | Pin generation and context limits |
| `ACTOR_MAX_TOKENS_PER_GPU` | `12288` | Bound dynamic-batch actor microbatch tokens |
| `HYBRID_ACTOR_GPUS` / `HYBRID_ROLLOUT_GPUS` | `4` / `4` | Build the Hybrid placement resource |
| `ROLLOUT_NUM_GPUS_PER_ENGINE` | `2` | Set each SGLang engine's tensor-parallel width |
| `SGLANG_MEM_FRACTION_STATIC` | `0.8` | Set SGLang's static memory fraction |

Before submitting a Ray Job, the launcher validates positive integers, context
capacity, rollout-GPU divisibility, and that actor GPU counts are multiples of
the current TP=2, CP=2 topology. Invalid combinations fail before Ray workers
start instead of silently degrading.

With the reference `global_batch_size=256` and
`n_samples_per_prompt=8`, setting `NUM_ITERS_PER_TRAIN_UPDATE=4` creates four
64-sample stages. Each stage therefore contains eight complete prompt groups.
Baseline and experiment runs in a strict pair both enable chunk forwarding and
use the same value. The baseline sets `HYBRID_PIPELINE_OVERLAP=0`, so it fetches
all four stages before their forwards; the experiment sets it to `1`, so each
ready stage is forwarded while later stages are still produced.

For example, this is a no-save Qwen3-VL-8B configuration for a constrained
machine. It may be used for a resource-qualified smoke or paired performance
benchmark, but its results must be labeled separately from the default 8-GPU
Qwen3.5 recipe:

```bash
MODEL_CONFIG_FILE="${MODEL_CONFIG_DIR}/qwen3-vl-8B.sh" \
MODEL_NAME=Qwen3-VL-8B-Instruct \
MODEL_RUN_NAME=qwen3-vl-8b \
CHECKPOINT_SAVE=0 \
ROLLOUT_MAX_RESPONSE_LEN=512 \
ROLLOUT_MAX_PROMPT_LEN=2048 \
ROLLOUT_MAX_CONTEXT_LEN=2560 \
ACTOR_MAX_TOKENS_PER_GPU=6144 \
NUM_ITERS_PER_TRAIN_UPDATE=4 \
HYBRID_ACTOR_GPUS=4 \
HYBRID_ROLLOUT_GPUS=1 \
ROLLOUT_NUM_GPUS_PER_ENGINE=1 \
HYBRID_PIPELINE_FORWARD=1 \
bash scripts/training/multimodal/run-qwen35-9B-8xgpu-openr1mm-hybrid-async.sh \
  hybrid-async
```

After changing the model, resource topology, or length limits, compare only
against a baseline with the identical configuration. Do not pool such runs
with the default 8-GPU recipe.

Trace files are separated by hostname, role, PID, and global rank. They contain
timestamps, counts, token totals, multimodal tensor byte counts, CUDA peaks,
and an irreversible global-index fingerprint; prompts, responses, images, and
sample tensors are never serialized.

Validate one run:

```bash
python scripts/tools/analyze_hybrid_pipeline_benchmark.py \
  --run-dir /data01/LWX/relax-task21/runs/smoke \
  --validate-only
```

Compare two baseline and two experiment runs and generate the registered CSV,
JSON, and curve artifacts:

```bash
python scripts/tools/analyze_hybrid_pipeline_benchmark.py \
  --run-dir /data01/LWX/relax-task21/runs/A1-baseline-seed20260728 \
  --run-dir /data01/LWX/relax-task21/runs/A2-baseline-seed20260729 \
  --run-dir /data01/LWX/relax-task21/runs/B1-experiment-seed20260728 \
  --run-dir /data01/LWX/relax-task21/runs/B2-experiment-seed20260729 \
  --output-dir /data01/LWX/relax-task21/comparison \
  --enforce-targets
```

For every paired seed, launch the baseline with
`HYBRID_PIPELINE_FORWARD=1 HYBRID_PIPELINE_OVERLAP=0` and the experiment with
`HYBRID_PIPELINE_FORWARD=1 HYBRID_PIPELINE_OVERLAP=1`. All other manifest
fields, including candidate commit, model, data, limits, topology, seed, image,
and hardware fingerprint, must match.

The analyzer validates event closure, one restore and optimizer step, sample
conservation, same-host monotonic timing, finite metrics, producer/fetch
fingerprints, steady-step coverage, and the registered performance thresholds.
Producer put count is diagnostic only; actor fetch/forward count remains
strictly fixed. Strict producer overlap means the first actor forward starts
before the producer starts its final put. The final put completion is retained
only as a transfer-stage diagnostic because its trace write can lose a
scheduling race with the consumer. With `--enforce-targets`, the analyzer also
checks that the baseline fetched all chunks before its first forward, that the
experiment forwarded before its final fetch completed, per-run strict producer
overlap, step-time p95,
the configured expected-GPU NVML coverage, peak VRAM,
token/multimodal-byte workload, and the
raw-reward, truncation-rate, staleness, same-weight PPO KL, and policy
clip-fraction guardrails. Same-weight `abs(train/ppo_kl)` and
`abs(train/pg_clipfrac)` must each remain at or below `1e-7`. It uses `sum(step_tokens) /
sum(step_time)` for aggregate throughput rather than averaging per-step rates.
The truncation guardrail reads the rollout-side
`rollout/truncated_ratio`; the training-side `rollout/truncated` scalar is not
used because chunk aggregation can sum that value more than once per step.
GPU utilization and the below-10% idle ratio use only 500 ms NVML samples
whose wall time falls inside the registered steady step intervals reconstructed
from TensorBoard `perf/step_time`; sampled peak VRAM remains a full-run safety
metric.
The strict comparison additionally requires a clean run manifest, identical
candidate/image/TransferQueue identities, verified static-input hashes,
dependency freeze, wheel hash, launcher log, and zero training/validation/final
exit-status artifacts for every run.
The manifest also records and cross-checks `max_staleness`, global/rollout
batch sizes, samples per prompt, and actor chunk count, so CLI expectations
cannot silently disagree with the measured workload.
The registered four-stage protocol requires exactly four actor fetch/forward
chunks of 64 samples. Producer puts may be regrouped by asynchronous completion;
their events must close and their aggregate sample count and fingerprint must
match the actor-consumed workload exactly.
Before any paired statistics are calculated, the analyzer also requires
identical model/config/data paths, prompt/response/context limits, actor token
budget, actor/rollout resource topology, SGLang determinism and memory
settings, physical-to-container GPU mapping, checkpoint mode, and debug
capture/replay settings. Missing workload fields or any mismatch fail closed.
The hostname and a SHA-256 fingerprint over GPU UUID, model, PCI address, and
driver version must also match across the comparison.
Paired global-index fingerprints must match exactly. When deterministic SGLang
inference is enabled, total, response, and multimodal-byte workloads must also
match exactly rather than within a tolerance. The comparison JSON reports the
mean, median, range, population standard deviation, and coefficient of
variation across paired repeats.
The staleness curve is the trace-derived producer lead at the first actor
forward: the largest completed producer rollout ID minus the actor rollout ID
at that timestamp. The current rollout is considered ready once its actor fetch
completes even if the producer trace write is scheduled slightly later. The
lead must remain at most the manifest's configured `max_staleness` value
(`2` in the reference recipe), and its paired steady mean may increase by at
most `0.25`.

The higher-confidence Task 21 mechanisms should be benchmarked as separate
cumulative ablations, following the accepted PR #201 protocol. In particular,
compare ProcessorPool and true-on-policy train-forward log-prob reuse separately
from chunk overlap. The labels below intentionally do not reuse PR #201's
`A1/A2/A3` names because this branch does not bundle its owner/ref group payload
dedup, whose paired mean regressed samples/s in that PR:

```text
B   : MM_PROCESSOR_POOL_SIZE=0 HYBRID_REUSE_TRAIN_LOGPROBS=0 HYBRID_PIPELINE_FORWARD=0
P   : MM_PROCESSOR_POOL_SIZE=8 HYBRID_REUSE_TRAIN_LOGPROBS=0 HYBRID_PIPELINE_FORWARD=0
P+R : MM_PROCESSOR_POOL_SIZE=8 HYBRID_REUSE_TRAIN_LOGPROBS=1 HYBRID_PIPELINE_FORWARD=0
P+S : MM_PROCESSOR_POOL_SIZE=8 HYBRID_REUSE_TRAIN_LOGPROBS=0 HYBRID_PIPELINE_FORWARD=1
```

Run at least 40 optimizer steps per fresh process, exclude warmup, use at least
two independent runs per condition, and report the range as well as the mean.
`HYBRID_REUSE_TRAIN_LOGPROBS=1` requires deterministic forwards, one optimizer
mini per rollout partition, and TIS when `max_staleness > 0`. It intentionally
cannot be combined with `HYBRID_PIPELINE_FORWARD=1` in the reference launcher,
so the two effects remain attributable.

To roll back, omit `--hybrid-pipeline-forward` (or set
`HYBRID_PIPELINE_FORWARD=0`). No checkpoint or dataset conversion is needed.
Before widening the support matrix, add collective-order and restore-count
tests for the new DP/PP/VPP or role graph, then rerun frozen-input parity,
multimodal smoke, and paired performance measurements.

`--steady-windows 0-0` deliberately measures a fresh-process first optimizer
step. It is useful when only one training step fits the fixed resource window,
but it is not a steady-state throughput claim. Label it as a paired first-step
benchmark, balance launch order across at least two seeds, and use later
multi-step windows whenever resources permit. The analyzer records this as
`measurement_scope=fresh_process_first_step`; later windows with at least three
selected steps are recorded as `measurement_scope=steady_state`.

______________________________________________________________________

## Quick Start

A reference launch script for an 8-GPU multimodal hybrid run lives at
`scripts/training/multimodal/run-qwen35-9B-8xgpu-openr1mm-hybrid-async.sh`.

The hybrid invocation it builds:

```bash
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 4], "rollout": [1, 4]}' \
    --max-staleness 2 \
    --num-data-storage-units 1 \
    --num-iters-per-train-update 2 \
    --balance-data \
    --hybrid \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"
```

Key points in this configuration:

- 8 total GPUs split 4 + 4 between actor and rollout
- `max-staleness 2` — actor may consume rollout output up to 2 steps behind the freshest weights
- `num-iters-per-train-update 2` — rollout targets half-batch transfers and
  the optional incremental path performs two fixed 128-sample actor fetches;
  the physical producer put count may vary
- `balance-data` — DP load balancing enabled
- GRPO algorithm with `--use-tis`; KL/ref forward is disabled in this recipe

______________________________________________________________________

## Troubleshooting

### `train_hybrid(rollout_id=N) batch_index=K stalled for ... seconds`

This warning fires in `relax/backends/megatron/actor.py` when the actor's TransferQueue poll for the next sub-batch keeps returning empty while the partition is not marked `all_consumed`. Typical causes:

- Rollout under-filled this partition (dropped samples without refilling).
- Rollout is paused on a health-check failure or restart.
- Staleness budget exhausted: rollout cannot produce new data because it is waiting for fresh weights.

Check rollout-side logs and partition status before assuming a code bug.

### `--balance-data is not supported in pure fully-async mode`

You passed `--fully-async --balance-data` without `--hybrid`. Either drop `--balance-data` or switch to `--hybrid`, which supports DP-balanced data.

### Rollout sees stale weights for a long time

Hybrid uses the sync `UpdateWeightFromTensor` path at the end of each `train_hybrid` call. If you see large weight-update gaps, check:

- `update_weights()` timing in actor logs
- Whether rollout health-checks are paging the actor (`_check_services_health()` is called before weight sync)

______________________________________________________________________

## Next Steps

Planned follow-ups for hybrid mode:

- **Integrate DCS for weight sync** — replace the current synchronous `UpdateWeightFromTensor` path with the Distributed Checkpoint Service so weight broadcast to rollout can overlap with the next training iteration, closing the remaining sync gap at the end of every `train_hybrid` call.
- **Split `train_actor` into `num_iters_per_train_update` iterations** — today `num_iters_per_train_update` only chunks the forward phase; the merged training step still runs once on the full global batch. Extend the actor train step to also iterate `num_iters_per_train_update` times so optimizer updates can be pipelined with TransferQueue consumption and peak training-side memory drops further.

Related docs:

- [Fully Async Training Pipeline](./fully-async-training.md) — the streaming-data engine hybrid borrows
- [Architecture](./architecture.md) — overview of Relax's service layering
- [Update Weights Pipeline](./update-weights-pipeline.md) — how `UpdateWeightFromTensor` and DCS differ
