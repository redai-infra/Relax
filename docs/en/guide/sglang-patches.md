# SGLang Runtime Patches

Relax injects runtime enhancements into SGLang at engine start-up via monkey-patching, rather than editing SGLang source or `cp`-overwriting system files. Each patch is gated by its own env flag, applied in `relax/backends/sglang/sglang_engine.py::_launch_server_with_patches`, and any combination may be enabled independently.

## Overview

| Patch | env flag | Injection point | Purpose |
|-------|----------|-----------------|---------|
| Routing Replay | `RELAX_OPTIMIZE_ROUTING_REPLAY=1` | scheduler subprocess (`_patched_run_scheduler_process`) | async D→H copy of routed-experts buffers, removing default-stream sync |
| OPD Pre-expanded | `RELAX_OPD_PREEXPANDED_PATCH=1` | main process | accept `opd_preexpanded_raw` pre-expanded `input_ids`+raw images, skip decode→retokenize |
| DeepEyes Qwen-VL | `RELAX_DEEPEYES_QWEN_VL_PATCH=1` | main process | support DeepEyes multi-turn pre-tokenized `input_ids`: collapse consecutive `<|image_pad|>` and restore the original token sequence + mrope |

## How the DeepEyes Qwen-VL patch reaches the SGLang subprocess

The primary switch for this patch is the CLI argument `--deepeyes-qwen-vl-patch` (default `False`, i.e. stock SGLang Qwen-VL behavior). The propagation chain:

1. The training script passes `--deepeyes-qwen-vl-patch` (e.g. `examples/deepeyes/run_deepeyes_r3.sh`, gated by the `DEEPEYES_PATCH` env var, default `0` = off / stock SGLang behavior; DeepEyes multi-turn training requires `DEEPEYES_PATCH=1` to enable it).
2. `relax/backends/sglang/sglang_engine.py::_init_normal` sees `args.deepeyes_qwen_vl_patch` truthy and sets `os.environ["RELAX_DEEPEYES_QWEN_VL_PATCH"] = "1"` (same pattern as `--optimize-routing-replay`).
3. `_init_normal` spawns the SGLang subprocess via `multiprocessing.Process(target=_launch_server_with_patches, start_method='spawn')`, which inherits the parent's `os.environ`.
4. `_launch_server_with_patches` checks the env flag and, if set, calls `apply_deepeyes_qwen_vl_patch`.

> You may also skip the CLI arg and `export RELAX_DEEPEYES_QWEN_VL_PATCH=1` directly, forwarding it via `RELAX_PROPAGATE_ENV_VARS` (the equivalent programmatic path, matching the OPD patch). Both hit the same env flag; either works.

## DeepEyes Qwen-VL Patch

**Module**: `relax/backends/sglang/patches/qwen_vl_patch.py`

**Why**: DeepEyes multi-turn VLM rollout (`examples/deepeyes/rollout.py::_run_inference_step`) sends **pre-tokenized** `input_ids: list[int]` to SGLang instead of text. The stock `QwenVLImageProcessor.process_mm_data_async` decodes→retokenizes, causing:

- **A. Token drift**: decode→retokenize is not lossless; token count may change, misaligning mrope positions.
- **B. N×M explosion**: the N accumulated `<|image_pad|>` tokens are re-expanded by `load_mm_data` into N×M.

**The two real changes injected** (all other upstream logic untouched):

1. `_strip_image_token`: collapse consecutive `<|image_pad|>` into a single placeholder before `load_mm_data`.
2. `original_input_ids` save/restore: save the caller's original `input_ids`, let the mm pipeline build `pixel_values`/grids as usual, then restore the token sequence and **recompute mrope from the restored `input_ids`**.

> **Why re-implement the method body instead of a plain wrapper**: mrope must be recomputed from the restored original `input_ids`, while the original method computes mrope internally from the retokenized sequence. A plain wrapper that only rewrites `input_ids` in the return value would misalign `input_ids` and mrope — exactly the bug this patch fixes. Recomputing mrope requires the method-internal artifact `ret` (carrying `image_grid_thw`/`video_grid_thw`/`second_per_grid_ts`), unavailable outside. The patch therefore rewrites `process_mm_data_async`, **delegating to upstream primitives** `self.load_mm_data` / `self.process_and_combine_mm_data` / `MRotaryEmbedding.get_rope_index` / the upstream module-level `preprocess_video`, inlining the two changes. Unrelated helpers (`smart_resize`, `preprocess_video`, `get_mm_data`, `__init__`, …) are **not** copied — they stay upstream.

**Idempotent**: `_PATCH_FLAG = "_relax_deepeyes_patched"` marks a patched class; a second call is a no-op.

**No effect on normal requests**: when `input_text` is a `str`, `_strip_image_token` passes through, `original_input_ids` stays `None`, and the method follows the original path.

## SGLang upgrade checklist

After upgrading SGLang, verify each patch's upstream dependencies still exist:

### DeepEyes Qwen-VL Patch
- `sglang.srt.multimodal.processors.qwen_vl.QwenVLImageProcessor` exists and has:
  - `process_mm_data_async` (the method being replaced)
  - `load_mm_data` (instance method, signature `prompt, image_data, video_data, audio_data, multimodal_tokens`)
  - `process_and_combine_mm_data` (instance method, returns `(mm_items, input_ids, ret)`)
- `sglang.srt.multimodal.processors.qwen_vl.preprocess_video` (module-level async function)
- `sglang.srt.layers.rotary_embedding.MRotaryEmbedding.get_rope_index` (keyword args: `spatial_merge_size, image_token_id, video_token_id, vision_start_token_id, model_type, tokens_per_second, input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts, use_audio_in_video, audio_seqlens, audio_token_id, audio_start_token_id, position_id_per_seconds`)
- `sglang.srt.managers.schedule_batch.MultimodalProcessorOutput.from_dict` (sglang ≥0.5.12; if absent, the patch falls back to returning a dict)

**Fail-fast**: `apply_qwen_vl_patches` calls `_verify_upstream_api` before binding; any missing symbol raises `RuntimeError` stating "Relax requires sglang 0.5.12.post1". `apply_deepeyes_qwen_vl_patch` additionally wraps this in try/except so a patch failure never blocks engine start-up (warning only).

### OPD Pre-expanded Patch
- Also replaces `QwenVLImageProcessor.process_mm_data_async`; depends on `self._processor.image_processor`, `self.get_mm_items_offset`, `MRotaryEmbedding.get_rope_index`, `MultimodalProcessorOutput`. See `relax/utils/opd/opd_sglang_patch.py`.

### Routing Replay Patch
- Replaces methods on `_RoutedExpertsCapturerReal`. See `relax/backends/sglang/routing_replay_patch.py`.

## Verification

- Unit tests: `pytest tests/backends/sglang/test_qwen_vl_patch.py -v` (stubs sglang, no GPU needed).
- End-to-end: run `examples/deepeyes/run_deepeyes_r3.sh` in an environment with the target SGLang version + GPU; output must be token-for-token identical to the old `cp` overlay at `temperature=0`.

## Deprecated

`examples/deepeyes/qwen_vl.py` (the old whole-file `cp` overlay onto `/sgl-workspace/...`) has been deleted and replaced by this patch. Clean up any remaining `/sgl-workspace` references in your environment.
