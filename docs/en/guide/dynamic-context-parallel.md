# Dynamic Context Parallelism

Dynamic Context Parallelism lets Megatron training use a different context parallel size for each micro-batch.

## Overview

Static Context Parallelism (CP) chooses one `context_parallel_size` for the whole training job. If `--rollout-max-context-len = a` and `--max-tokens-per-gpu = b`, long-context training often needs `cp_size = ceil(a / b)` so the longest sequence can fit. That fixed CP size is then used by every micro-batch, including short packed micro-batches.

Dynamic Context Parallelism removes that global binding. For each micro-batch, Relax estimates the smallest CP size needed by the longest sequence in that micro-batch, rounds it up to a power of two, and uses the corresponding Megatron dynamic CP group. Short micro-batches can therefore run with a smaller CP size and more effective data-parallel sub-groups, reducing CP communication latency.

This is most useful when the rollout data has uneven sequence lengths, especially for multimodal workloads where packing can place several short samples in the same micro-batch while `--rollout-max-context-len` is still large enough to require CP for rare long samples.

::: warning
Current validation is primarily for Megatron actor training in colocate mode. Validate separately before relying on Dynamic CP for R3, Fully Async, Hybrid, or SFT production runs.
:::

## Architecture

```
Static CP group size = max_cp

Micro-batch 1: short sequences
┌──────────────────────────────────────────────┐
│ dynamic_cp_size = 1                          │
│ ranks become max_cp independent DP subgroups │
└──────────────────────────────────────────────┘

Micro-batch 2: medium sequences
┌──────────────────────────────────────────────┐
│ dynamic_cp_size = 2                          │
│ static CP group is split into CP subgroups   │
└──────────────────────────────────────────────┘

Micro-batch 3: longest sequences
┌──────────────────────────────────────────────┐
│ dynamic_cp_size = max_cp                     │
│ same behavior as static CP                   │
└──────────────────────────────────────────────┘
```

Dynamic CP is implemented in the Megatron backend:

| Area | Implementation |
|---|---|
| Argument validation | `relax/backends/megatron/arguments.py` validates `--dynamic-context-parallel`, derives the maximum static CP size, and requires dynamic batching. |
| Micro-batch CP decision | `relax/backends/megatron/cp_utils.py` computes `dynamic_cp_size = next_power_of_2(ceil(max_seq_len / max_tokens_per_gpu))`. |
| Data split and merge | `dynamic_cp_split_data` balances samples by token length across CP sub-groups, and `dynamic_cp_merge_output` reconstructs full per-sample outputs before write-back. |
| Training batch preparation | `relax/backends/megatron/data.py` attaches `dynamic_cp_size`, `dynamic_cp_rank`, and CP-aware packed sequence metadata to each micro-batch. |
| VLM compatibility | `relax/backends/megatron/model.py` swaps the bridge CP group for the current micro-batch and patches GatedDeltaNet to read the micro-batch CP size. |

Relax differs from the verl implementation in two important ways. First, it supports multimodal batches by passing unsplit VLM inputs and CP-aware packed sequence parameters to Megatron Bridge. Second, when a micro-batch is subdivided, Relax uses sequence-length balancing instead of splitting only by sample count, and it increases the dynamic CP size when needed so every sub-group receives data.

## Features

1. **Per-micro-batch CP size**: Short packed micro-batches can use `cp_size = 1` or `2` while long micro-batches still use the maximum CP size.
2. **Token-balanced sub-groups**: Relax balances samples by token length when a static CP group is split into smaller dynamic CP groups.
3. **No empty sub-groups**: If the micro-batch has fewer samples than the number of possible sub-groups, Relax grows the dynamic CP size to keep collective calls symmetric.
4. **Multimodal path support**: VLM batches keep unsplit inputs for Megatron Bridge, while loss and log-prob helpers use the dynamic CP metadata for slicing and gathering.
5. **Forward-output reconstruction**: Forward-only outputs are gathered from the dynamic CP group and then across sub-groups, restoring the original micro-batch order.

## Quick Start

Enable Dynamic CP together with dynamic batching:

```bash
python3 relax/entrypoints/train.py \
  --dynamic-context-parallel \
  --use-dynamic-batch-size \
  --rollout-max-context-len 32768 \
  --max-tokens-per-gpu 8192 \
  --calculate-per-token-loss \
  # ... other training args
```

For this example, Relax derives the maximum static CP size as:

```text
ceil(32768 / 8192) = 4
next_power_of_2(4) = 4
```

At runtime, a micro-batch whose longest sequence is only `6000` tokens can use `dynamic_cp_size = 1`, while a micro-batch near `32768` tokens uses `dynamic_cp_size = 4`.

::: tip
If a base script already sets `--dynamic-context-parallel`, the required Relax-side packing switch is `--use-dynamic-batch-size` with `--max-tokens-per-gpu`.
:::

## Configuration

| Flag | Required | Notes |
|---|---|---|
| `--dynamic-context-parallel` | Yes | Enables Megatron dynamic CP group creation and Relax's per-micro-batch CP path. |
| `--use-dynamic-batch-size` | Yes | Required by validation. Dynamic CP relies on token-budget micro-batches. |
| `--rollout-max-context-len` | Yes | Used as the maximum possible context length. |
| `--max-tokens-per-gpu` | Yes | Per-GPU token budget. Dynamic CP uses it as `max_seqlen_per_dp_cp_rank`. |
| `--calculate-per-token-loss` | Yes | Required when CP or Dynamic CP is enabled by Megatron Bridge. |
| `--log-probs-max-tokens-per-gpu` | Optional | Forward-only log-prob computation can use a different token budget; when unset, it follows `--max-tokens-per-gpu`. |

Choose `--max-tokens-per-gpu` lower than `--rollout-max-context-len` when you actually need CP. If the token budget already covers the maximum context length, Dynamic CP usually collapses to `cp_size = 1` and brings little benefit.

Relax derives and overwrites the maximum static `context_parallel_size` during Megatron argument validation:

```text
max_cp = next_power_of_2(ceil(rollout_max_context_len / max_tokens_per_gpu))
```

The global `world_size` must be divisible by:

```text
tensor_model_parallel_size * pipeline_model_parallel_size * max_cp
```

## Validation and Performance

Internal experiments on a 5k multimodal dataset with uneven sequence lengths, an average of 4.5 images per sample, and average image resolution around `673 x 841` showed:

| Workload | Training throughput | End-to-end throughput |
|---|---:|---:|
| Qwen3-VL-4B | +12% | +9% |
| Qwen3-35B-A3B | +12% | +2% |

The end-to-end gain depends on how much of the rollout is dominated by actor training. If rollout or data loading is the bottleneck, Dynamic CP can still improve the training section without moving the full pipeline by the same amount.

## Best Practices

1. **Use it for high variance lengths**: Dynamic CP pays off when many micro-batches are much shorter than `--rollout-max-context-len`.
2. **Keep token budgets realistic**: Start from the largest per-GPU token budget that is stable without OOM, then reduce only when long samples still exceed memory.
3. **Watch the logs**: Relax logs messages such as `[dynamic_cp] micro-step dynamic_cp_size=...`; a healthy mixed-length job should show more than one CP size.
4. **Validate new modes separately**: Treat R3, Fully Async, Hybrid, and SFT as separate validation targets before production use.
5. **Compare train and end-to-end metrics**: A train-only improvement may be diluted by rollout, reward, or weight-update time.

## Troubleshooting

### `--dynamic-context-parallel requires --use-dynamic-batch-size`

Add both flags. Dynamic CP depends on token-budget micro-batch packing:

```bash
--dynamic-context-parallel \
--use-dynamic-batch-size \
--max-tokens-per-gpu 8192
```

### `world_size must be a multiple of tp*pp*cp`

Relax computes `cp` from `--rollout-max-context-len` and `--max-tokens-per-gpu`. Increase the total world size, lower TP/PP, or increase `--max-tokens-per-gpu` so the derived maximum CP size is smaller.

### Dynamic CP logs always show the same CP size

Check whether the dataset actually has short packed micro-batches. If every micro-batch has a longest sequence near `--rollout-max-context-len`, Dynamic CP behaves like static CP.

### VLM runs fail inside Megatron Bridge

Confirm that the run uses a Megatron version whose `initialize_model_parallel` accepts `dynamic_context_parallel`. Relax checks this during validation, but mismatched Bridge or Megatron pins can still surface as model-side errors.

## Next Steps

- [Performance Tuning](./performance-tuning.md) - Tune dynamic batching and parallelism for throughput.
- [Configuration](./configuration.md) - Review the full training argument reference.
- [OOM Troubleshooting](./oom-troubleshooting.md) - Pick safe token budgets for long-context jobs.
