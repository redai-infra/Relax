# Task 32 v1 — GDN chunkwise context parallel: MCore backport manifest

This file is the file-level and hunk-level record required by the Task 32 RFC
review: what was taken from upstream, what was deliberately left behind, and why.
It describes the delta between `20260506-85bced0ae.patch` (previous) and
`20260805-85bced0ae.patch` (current). Both patches apply to the same source tree,
so the whole difference below *is* the Task 32 change.

**v1 depends on NVIDIA/Megatron-LM#3282 only.** NVIDIA/Megatron-LM#5664 is
explicitly out of scope per the RFC review and contributes nothing to this patch.

## Base

| Thing                                      | Value                                                                                          |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| Megatron-Bridge                            | `2faedbf6fe3c422835a44b2b360cadcb2a116a54`                                                     |
| Megatron-LM (`.dev.commit` of that Bridge) | `85bced0ae6ab46f61a0fd774074a3273daf6ae02`                                                     |
| Tree assembly                              | `cp -r Bridge/src/megatron` then `rsync` MCore's `megatron/` over it (see `docker/Dockerfile`) |

## Upstream sources

| Ref                                                                                        | State           | Used for                                                                                      |
| ------------------------------------------------------------------------------------------ | --------------- | --------------------------------------------------------------------------------------------- |
| [Megatron-LM #3282](https://github.com/NVIDIA/Megatron-LM/pull/3282), merged as `5139086e` | merged to `dev` | The chunkwise CP feature. The only MCore source used by v1.                                   |
| flash-linear-attention / fla-core `0.4.2`                                                  | released        | `fla.ops.cp.build_cp_context`; `cp_context=` on `causal_conv1d` and `chunk_gated_delta_rule`. |

`5139086e` is ~597 commits ahead of `85bced0a`, and it edits the same regions of
`gated_delta_net.py` / `transformer_config.py` that Relax's own patch edits, so a
cherry-pick is not possible. Everything below is a selective port onto the pinned
tree.

## Design constraint from the RFC review

The GDN CP mode is **static for the whole process**. `linear_cp_mode` is read by
both the construction-time head check and by `GatedDeltaNet.forward`; there is no
per-call override and nothing in a forward writes to `self` or to the shared
config. Dynamic context parallelism varies only `cp_group` / `local_cp_size`,
never the algorithm.

## Files added / changed

### `megatron/core/context_parallel_layout.py` — new, +324

**Byte-identical to `5139086e` below the module docstring.** Verified with
`diff` against the upstream file; the only change is a Relax provenance note in
the docstring. Contains:

| Symbol                                                       | Purpose                                                                                                     |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| `get_thd_context_parallel_rank_indices`                      | Reference description of both partitions. Tests assert Relax's `slice_with_cp` / `gdn_cp_slice` against it. |
| `zigzag_to_contiguous_chunks`, `contiguous_to_zigzag_chunks` | Public entry points; dispatch on whether `cu_seqlens` is given.                                             |
| `_zigzag_contiguous_thd_swap`                                | Packed THD: one packed-token all-to-all. Builds its routing from `cu_seqlens` on each call.                 |
| `_zigzag_contiguous_chunk_swap`                              | SBHD: one chunk-level all-to-all.                                                                           |

### `megatron/core/packed_seq_params.py` — +17

- `resolve_cp_group()` from `5139086e`. The single place where "dynamic
  per-micro-batch CP group, else the construction-time one" is decided, so every
  consumer derives size/rank from the same group object.
- `local_cp_size` / `cp_group` already existed on `85bced0a`; not re-backported.
- The dataclass field list is **untouched**.

### `megatron/core/transformer/transformer_config.py` — +50 / −16

- `linear_cp_mode` field and the head-divisibility rule from `5139086e`: headwise
  needs `heads % (tp * cp) == 0`, chunkwise needs `heads % tp == 0`.
- **Relax adaptation:** the default is `"headwise"`, not upstream's `"chunkwise"`.
  Upgrading the image must not silently change the algorithm an existing recipe
  runs.
- **Relax adaptation:** `"all_gather"` is accepted as a third declared value,
  using the same TP-only head rule as chunkwise. That is what makes a
  non-divisible geometry constructible for Relax's all-gather fallback, and it
  means the declared config equals the resolved `--gdn-cp-mode` instead of
  declaring one mode while running another.
- Unknown values — including an unresolved `"auto"` — assert at construction.
- **Excluded:** `gdn_conv_pad_alignment`, `gdn_pre_gated_delta_rule_fusion` and
  their interaction asserts. Neither field exists on this base.

### `megatron/core/ssm/gated_delta_net.py` — +290 / −42

- `_resolve_cp_routing()`: resolve the CP group once via `resolve_cp_group`, then
  give the whole group to exactly one of headwise / chunkwise and `None` to the
  other (`None` is treated as size 1 everywhere downstream). No process group is
  ever created in a forward.
- Validates `PackedSeqParams.local_cp_size == cp_group.size()`; a mismatch means
  a collective would run on the wrong group.
- `cp_size == 1` short-circuits *before* the mode is read, so a CP=1 micro-batch
  is legal under any declared mode.
- `linear_cp_mode="all_gather"` raises if MCore's own forward is reached with
  `cp_size > 1`: that mode is implemented by the Relax wrapper, so arriving here
  means the wrapper was not installed.
- zigzag ↔ contiguous conversion around the conv + scan; `cp_context` built once
  per forward and passed to both FLA kernels.
- `_resolve_cu_seqlens` gains the `cp_size` divisibility check from `5139086e`.
- **Relax adaptation — backwards compatibility:**
  - `_prepare_qkv_for_gated_delta_rule` takes `cp_size_headwise` as an *optional*
    argument defaulting to `self.cp_size`, so Relax's existing all-gather
    fallback (`relax/backends/megatron/model.py`) keeps calling it unchanged.
  - `get_parameter_local_cp` accepts a `None` group as size 1.
  - `cp_context=` is only passed when chunkwise is active, so with chunkwise off
    the FLA call is byte-identical to before the backport — and still works
    against FLA 0.4.1.
- **Preserved Relax fixes** (both from the previous patch, unchanged):
  - `torch._dynamo.config.patch(disable=True)` around
    `_prepare_qkv_for_gated_delta_rule` (Qwen3.6 `torch.compile` failure);
  - `param[tuple(slices)]` in `get_parameter_local_cp` (multi-dim basic indexing).
- **Excluded from `5139086e`:** the `_forward_compute` split and `recompute_gdn`
  selective-recompute wrapper; `gdn_pre_gated_delta_rule_fusion` /
  `_fused_streamed_pre_gated_delta_rule`; `gdn_conv_pad_alignment` conv padding;
  the `_a2a_cp_to_hp` / `_a2a_hp_to_cp` refactor of the headwise path; the rename
  of `get_parameter_local_cp` to `get_parameter_local_cp_headwise`. All are
  independent changes from the 597-commit gap and would alter existing paths'
  structure or numerics for no Task 32 benefit.

### Explicitly not in this patch

| Thing                                            | Why                                                                                                                                                                                                                        |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Anything from NVIDIA/Megatron-LM#5664            | Out of scope for v1 per the RFC review. No `cp_partition_mode`, no route tensors, no `prebuild_thd_cp_partition_routes`, no `pad_between_seqs`. The THD swap rebuilds its routing per call, which is `5139086e` behaviour. |
| `megatron/core/extensions/transformer_engine.py` | The THD output-length fix in the RFC was conditional on adopting `pad_between_seqs`. That representation is not backported and no test on this base reproduces the mismatch.                                               |
| `pyproject.toml` / `uv.lock`                     | Relax installs FLA from `docker/Dockerfile`, not from upstream package metadata.                                                                                                                                           |

## Verifying this patch is exactly what it claims

```bash
# byte-identity of the new module against upstream, below the docstring
curl -s https://raw.githubusercontent.com/NVIDIA/Megatron-LM/5139086e/megatron/core/context_parallel_layout.py \
  > /tmp/up.py
diff <(sed -n '/^from typing import/,$p' <patched-tree>/megatron/core/context_parallel_layout.py) \
     <(sed -n '/^from typing import/,$p' /tmp/up.py)   # must be empty

# no #5664 content anywhere
grep -rn 'cp_partition_route\|prebuild_thd_cp\|pad_between_seqs' <patched-tree>/megatron/  # must be empty
```
