# Task40 P3O CP forward diagnostics

This directory contains the Batch 3 harness for locating the first BF16 THD
forward divergence between CP1 and CP2. It is a diagnostic package only: it
does not change Relax production code and Batch 3 does not run the cluster
commands.

## Provenance and deliberate differences

The harness is an extension of two tools that were already exercised before
this batch; it is not a new independent implementation.

1. The 2026-08-14 Step-0 oracle custom-init hook is
   `scripts/local_p3o_task40/p0_oracle_hook.py` (SHA-256
   `d53fdded5f601e0611231520166850dd49f05aad754e36f8adcef0333d31bd8c`).
   `p3o_forward_dump.py` retains its custom-init entry point, rank/DP/CP/TP/PP
   runtime metadata, `dump_details`-relative output, detached CPU tensor
   serialization, and SHA-256 key over the full int64 token sequence. The
   oracle's P3O S1/S2/N wrapping and FP32 native-attention replacement are not
   copied: this batch observes forward activations only and is BF16-only.
2. The module discovery and forward-hook technique comes from
   `relax/backends/megatron/model_provider.py::_install_cp_probe` (the
   `CP-PROBE` block). The new hook calls that existing probe first, then extends
   its `named_modules()`/`with_kwargs=True` approach from one attention-shape
   sample to every decoder layer and every captured log-prob micro-batch. It
   writes tensors rather than log lines and adds global token ownership for the
   CP zig-zag layout. Production `model_provider.py` is not modified.

The two frozen 8/14 BF16 command sources used for parameter comparison are:

- DP4CP1 command SHA-256:
  `7e7fe652a059b9a88b0122a5fde7fa0c13a2d073b9cf667a7fc2eeea86895bae`
- DP2CP2 command SHA-256:
  `fb5882f8b92d2b241c7910f842b32851a8379b2592feb1693b8cd74683c237e3`

`run_forward_dump.sh` keeps their model, fixture, seed, rollout shape, global
batch, MBS1, optimizer, P3O step-scope settings, FlashAttention, no-recompute,
and deterministic rollout arguments. Its only training-argv additions are the
three diagnostic hook paths plus `--dump-details`; topology changes are limited
to `--resource` and `--context-parallel-size`, as shown below. The non-semantic
TensorBoard experiment name and output paths use the new topology/run names.
DP2CP1 uses the same 8/14 DP2 resource template (`actor=[1,2]`,
`rollout=[1,4]`) with the frozen Step-0 fixture. THD and BF16 are the same
resolved defaults used by the 8/14 BF16 commands and are enforced again by
`configure()`.

| Template | Actor resource | Rollout resource |  DP |  CP | Precision | QKV | MBS |
| -------- | -------------- | ---------------- | --: | --: | --------- | --- | --: |
| `dp4cp1` | `[1,4]`        | `[1,4]`          |   4 |   1 | BF16      | THD |   1 |
| `dp2cp2` | `[1,4]`        | `[1,4]`          |   2 |   2 | BF16      | THD |   1 |
| `dp2cp1` | `[1,2]`        | `[1,4]`          |   2 |   1 | BF16      | THD |   1 |

There is intentionally no precision argument or FP32 branch in the launcher.

## Capture scope

The frozen commands pass `--use-rollout-logprobs`, so the actor intentionally
skips a separate current-policy log-prob forward. The hooks therefore select
the first production P3O stats forward through `before_train_step`; for a
configuration that does execute actor log-prob forward, `before_log_prob`
selects that earlier equivalent path. Capture stops after exactly
`global_batch_size / (DP * MBS)` completed rank-local micro-batches, before the
gradient forward can be duplicated on disk. Activation recomputation is
disabled in every template. A run therefore has exactly one dump for each
non-dummy rank-local micro-batch.

The hook observes both module-local aliases of `get_batch`: the ordinary
forward path in `model.py` and the P3O sufficient-statistics path in
`p3o_step.py`. This is required because each module imports the callable by
value; replacing only one alias does not observe the other path.

For each decoder layer `NNN`, the required stage set is:

- `layer_NNN.block.input` and `.output`;
- `layer_NNN.self_attention.input` and `.output`;
- `layer_NNN.qkv_projection.input` and `.output`;
- `layer_NNN.attention_query`, `.attention_key`, and `.attention_value` at the
  core-attention call boundary;
- `layer_NNN.attention_output` from core attention;
- `logits` from the model output.

Hooks serialize tensors immediately after each stage to bound host memory.
The diagnostic sync and I/O overhead is intentional and must not be used for
performance measurements.

## Artifact contract

For a run directory `<run>`, `p3o_forward_dump.py finalize-manifest` produces
this layout:

```text
<run>/
├── command.sh
├── resolved_args.txt
├── stdout_stderr.log
├── exit_code.txt
├── manifest.json
└── dump/
    ├── runtime_rank<R>.json
    ├── manifest_rank<R>.json
    └── rank<R>/micro<M>/
        ├── metadata.json
        ├── token_metadata.pt
        ├── layer_000.block.input.pt
        ├── ...
        └── logits.pt
```

`R` is the actor global rank, not DP or CP rank; `M` is that rank's zero-based
micro-batch index within the captured log-prob pass. Every stage tensor is
stored losslessly on CPU with its original dtype. `metadata.json` maps each
stage to its file, shape, dtype, token axis, and finite/non-finite result.

### Global sample and token key

The topology-independent sample key is
`sha256(full_sample_token_ids.to(int64).contiguous().bytes)`, matching the 8/14
oracle audit. The global tensor-row key is the pair
`(sample_sha256, global_zero_based_token_index)`.

`token_metadata.pt` supplies the exact row mapping:

- `sample_keys`: full-sequence SHA-256 values for this micro-batch;
- `full_token_ids`: the corresponding pre-CP token tensors;
- `local_sample_indices[i]`: index into `sample_keys`, or `-1` for trailing
  pack padding;
- `local_token_indices[i]`: global position inside that sample, or `-1` for
  trailing pack padding;
- `local_chunk_indices[i]`: global CP chunk number. For CP size `C`, rank `r`
  owns chunks `r` and `2C-1-r`; CP1 uses chunk `0`;
- `local_real_mask[i]`: false for within-sample CP padding and trailing pack
  padding. The comparator never treats padding as a real global token;
- `derived_position_ids`: the monotonic global positions that Relax/Megatron
  synthesizes because the model call passes `position_ids=None`;
- `position_ids_argument`: the direct model argument (`None` for this frozen
  command contract);
- `cu_seqlens_q` and `cu_seqlens_kv`: direct CPU copies from
  `PackedSeqParams`.

An identical full token sequence appearing twice would make a content-only
sample key ambiguous. Both the dumper contract and comparator treat that as a
hard harness failure rather than silently merging samples.

### Per-capture metadata

`metadata.json` is written only after the root model forward completes. It
contains:

- `format_version`, `complete`, `phase`, and `phase_micro_batch_index`;
- global rank/world size plus DP/CP/TP/PP rank and world size;
- `qkv_format`, `micro_batch_size`, sample keys, total/response/max sequence
  lengths, `max_seqlen_q`, and `max_seqlen_kv`;
- whether the direct position argument was `None`, the token-key contract, and
  the `token_metadata.pt` path;
- all stage paths, shapes, dtypes, token axes, and finite flags.

`runtime_rank<R>.json` additionally records resolved precision, fixture path and
SHA-256, capture phase, and the two migration sources. `manifest_rank<R>.json`
lists completed micro-batches for one rank. Top-level `manifest.json` requires
all expected ranks and capture metadata, checks every tensor file and finite
flag, and records fixture hashes and file counts.

## Failure contract

A cluster cell is failed if any of the following holds, even when Ray exits 0:

- resolved precision is not BF16, layout is not THD, or MBS is not 1;
- fixture SHA-256 differs across ranks or from
  `48538d165386dc94006613d857c022a7ba2e979bdc31bc617374eee2dc3c35b8`;
- a rank, non-dummy log-prob micro-batch, required stage, tensor file,
  `token_metadata.pt`, or completed `metadata.json` is missing;
- a tensor contains NaN/Inf, lacks an unambiguous token axis, or its token-axis
  length disagrees with the local ownership metadata;
- a sample hash is ambiguous, a global sample+token key is duplicated, CP ranks
  do not reconstruct the same real-token key set as CP1, or token IDs/derived
  positions disagree;
- `manifest.json.complete` is false;
- the comparator reports any contract error or a tensor exceeds its supplied
  `atol`/`rtol`. Defaults are both `1e-6`; Batch 5 must report max-abs,
  relative-L2, first divergent stage/token, and CP chunk-boundary statistics.

The comparator exits 0 only for `FORWARD_MATCH`; `FORWARD_MISMATCH` exits 1 and
preserves `ROOT_CAUSE.json` and `ROOT_CAUSE.md`. Classification follows the
evidence location: input/key coverage failure → `INPUT_PACKING_BUG`, first
internal layer divergence → `ATTENTION_KERNEL_ORDER`, and logits-only
divergence after equal internal stages → `LOGPROB_EXTRACTION_BUG`.

## Commands (Batch 4 only)

Do not run these during Batch 3. On the designated 4×A100 cluster, invoke one
fresh run ID per cell; the common launcher refuses to overwrite a run directory.

```bash
bash scripts/local_p3o_task40/run_forward_dump.sh dp4cp1 <run-id>
bash scripts/local_p3o_task40/run_forward_dump.sh dp2cp2 <run-id>
bash scripts/local_p3o_task40/run_forward_dump.sh dp2cp1 <run-id>
```

The resolved run path is
`.../runs/step0_dump/<topology>/seed_42/<run-id>/`. Compare completed runs with:

```bash
python scripts/local_p3o_task40/compare_forward_dumps.py \
  --reference <dp4cp1-run> \
  --candidate <dp2cp2-run> \
  --output-dir <evidence-root>/analysis
```

Batch 3 validation is limited to local imports, compilation, shell syntax, and
static command inspection. It does not submit Ray jobs or start Batch 4.
