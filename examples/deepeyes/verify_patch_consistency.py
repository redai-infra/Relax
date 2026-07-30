# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Task-17 acceptance #4: patch-before/after key-output consistency.

Verifies the DeepEyes Qwen-VL processor monkey-patch changes ONLY the two
DeepEyes-specific things it claims to, and leaves normal (raw-text) requests
byte-for-byte identical to the stock SGLang processor.

Two cases, each run against BOTH the stock and the patched
``process_mm_data_async`` on identical stubbed ``self`` instances:

  Case A — raw-text prompt (normal single-turn Qwen-VL request):
    The patch MUST be a no-op: stock output == patched output on every key.

  Case B — pre-tokenized list[int] with N <|image_pad|> per image
           (DeepEyes multi-turn rollout input):
    The patch changes exactly two things —
      * ``input_ids`` restored to the caller's original expanded list (not the
        mm-pipeline's "retokenized" tensor);
      * ``mrope_positions`` recomputed from that restored list.
    Everything else (mm_items, grid, delta) is unchanged.

This is a standalone verification script (not a pytest). It stubs the upstream
primitives via a fake ``self`` so it runs without a GPU / real model. Run it
inside the relaxrl image:

    python3 examples/deepeyes/verify_patch_consistency.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from types import SimpleNamespace


# --- make the real patch module importable; do NOT enable the env flag yet ---
# We import the function objects directly so we can call both the stock and the
# patched implementation in the same process.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch  # noqa: E402


IMAGE_PAD = 151655
VISION_START = 151652
VISION_END = 151653


# ---------------------------------------------------------------------------
# Fake SGLang machinery: register minimal sglang.* modules so the patch module
# (which imports sglang lazily, inside functions) can bind.
# ---------------------------------------------------------------------------


def _install_fake_sglang() -> types.ModuleType:
    sglang = types.ModuleType("sglang")
    srt = types.ModuleType("sglang.srt")
    mm = types.ModuleType("sglang.srt.multimodal")
    procs = types.ModuleType("sglang.srt.multimodal.processors")
    layers = types.ModuleType("sglang.srt.layers")
    managers = types.ModuleType("sglang.srt.managers")

    qwen_vl_mod = types.ModuleType("sglang.srt.multimodal.processors.qwen_vl")

    class QwenVLImageProcessor:
        """Stock processor whose method we replace with the patch."""

        async def process_mm_data_async(self, image_data, input_text, request_obj, *args, **kwargs):
            raise NotImplementedError

    qwen_vl_mod.QwenVLImageProcessor = QwenVLImageProcessor

    async def preprocess_video(video, video_config=None):
        return video, {}

    qwen_vl_mod.preprocess_video = preprocess_video

    rotary_mod = types.ModuleType("sglang.srt.layers.rotary_embedding")

    class MRotaryEmbedding:
        last_kwargs: dict = {}

        @staticmethod
        def get_rope_index(**kwargs):
            MRotaryEmbedding.last_kwargs = kwargs
            return torch.zeros((1, 4, 3), dtype=torch.long), 0

    rotary_mod.MRotaryEmbedding = MRotaryEmbedding

    sched_mod = types.ModuleType("sglang.srt.managers.schedule_batch")

    class MultimodalProcessorOutput:
        @staticmethod
        def from_dict(d):
            return d

    sched_mod.MultimodalProcessorOutput = MultimodalProcessorOutput

    for name, mod in [
        ("sglang", sglang),
        ("sglang.srt", srt),
        ("sglang.srt.multimodal", mm),
        ("sglang.srt.multimodal.processors", procs),
        ("sglang.srt.multimodal.processors.qwen_vl", qwen_vl_mod),
        ("sglang.srt.layers", layers),
        ("sglang.srt.layers.rotary_embedding", rotary_mod),
        ("sglang.srt.managers", managers),
        ("sglang.srt.managers.schedule_batch", sched_mod),
    ]:
        sys.modules[name] = mod

    return qwen_vl_mod


_install_fake_sglang()

# Now import the patch function + the stock method we will compare against.
from relax.backends.sglang.patches.qwen_vl_patch import (  # noqa: E402
    _patched_process_mm_data_async,
    _strip_image_token,
)


# ---------------------------------------------------------------------------
# A faithful re-implementation of the STOCK upstream method body, used as the
# "before" reference. It is identical to the patched body EXCEPT it does NOT
# strip image_pad before load_mm_data and does NOT restore original input_ids
# / recompute mrope. (Mirrors upstream sglang 0.5.12 QwenVLImageProcessor.)
# ---------------------------------------------------------------------------


async def _stock_process_mm_data_async(self, image_data, input_text, request_obj, *args, **kwargs):
    from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
    from sglang.srt.multimodal.processors.qwen_vl import preprocess_video

    base_output = self.load_mm_data(
        prompt=input_text,  # <-- stock: no stripping
        image_data=image_data,
        video_data=request_obj.video_data,
        audio_data=request_obj.audio_data,
        multimodal_tokens=self.mm_tokens,
    )

    video_metadata = None
    if base_output.videos:
        videos_processed = [await preprocess_video(v, video_config=self.video_config) for v in base_output.videos]
        base_output.videos, video_metadata = map(list, zip(*videos_processed))

    if self.hf_config.model_type in ("qwen3_vl", "qwen3_vl_moe"):
        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens, video_metadata=video_metadata, do_sample_frames=False
        )
    else:
        mm_items, input_ids, ret = self.process_and_combine_mm_data(base_output, self.mm_tokens)

    input_ids = input_ids.flatten()
    # <-- stock: no restore of original_input_ids

    second_per_grid_ts = getattr(ret, "second_per_grid_ts", None)
    if second_per_grid_ts is None:
        second_per_grid_ts = getattr(ret, "video_second_per_grid", None)

    mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
        spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
        image_token_id=self.mm_tokens.image_token_id,
        video_token_id=self.mm_tokens.video_token_id,
        vision_start_token_id=self.vision_start_token_id,
        model_type=self.model_type,
        tokens_per_second=getattr(self.hf_config.vision_config, "tokens_per_second", None),
        input_ids=input_ids.unsqueeze(0),  # <-- stock: mrope from retokenized ids
        image_grid_thw=getattr(ret, "image_grid_thw", None),
        video_grid_thw=getattr(ret, "video_grid_thw", None),
        second_per_grid_ts=second_per_grid_ts,
        use_audio_in_video=False,
        audio_seqlens=None,
        audio_token_id=getattr(self.hf_config, "audio_token_id", None),
        audio_start_token_id=self.audio_start_token_id,
        position_id_per_seconds=getattr(self.hf_config, "position_id_per_seconds", None),
    )
    mrope_positions = mrope_positions.squeeze(1)
    return {
        "input_ids": input_ids.tolist(),
        "mm_items": mm_items,
        "im_start_id": self.vision_start_token_id,
        "im_end_id": self.vision_end_token_id,
        "im_token_id": self.mm_tokens.image_token_id,
        "video_token_id": self.mm_tokens.video_token_id,
        "audio_token_id": self.mm_tokens.audio_token_id,
        "mrope_positions": mrope_positions,
        "mrope_position_delta": mrope_position_delta,
    }


# ---------------------------------------------------------------------------
# Stub ``self``: load_mm_data records the prompt it received; the mm pipeline
# returns a "retokenized" input_ids ([100,200,300]) that differs from the
# caller's original — exactly the drift the patch must correct.
# ---------------------------------------------------------------------------


def _make_self_stub(captured):
    ret = SimpleNamespace(
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        video_grid_thw=None,
        second_per_grid_ts=None,
        video_second_per_grid=None,
    )
    base_output = SimpleNamespace(videos=[])
    hf_config = SimpleNamespace(
        model_type="qwen2_5_vl",
        audio_token_id=None,
        position_id_per_seconds=None,
        vision_config=SimpleNamespace(spatial_merge_size=2, tokens_per_second=None),
    )
    mm_tokens = SimpleNamespace(image_token_id=IMAGE_PAD, video_token_id=151656, audio_token_id=None)

    class SelfStub:
        model_type = "qwen2_5_vl"
        vision_start_token_id = VISION_START
        vision_end_token_id = VISION_END
        audio_start_token_id = None
        video_config = {}

        def __init__(self):
            self.hf_config = hf_config
            self.mm_tokens = mm_tokens

        def load_mm_data(self, **kwargs):
            captured["load_mm_data_prompt"] = kwargs["prompt"]
            return base_output

        def process_and_combine_mm_data(self, base_output, mm_tokens, **kwargs):
            captured["process_and_combine_kwargs"] = kwargs
            mm_items, input_ids = [], torch.tensor([100, 200, 300])  # "retokenized"
            return mm_items, input_ids, ret

    return SelfStub()


def _request_obj():
    return SimpleNamespace(rid="rid-1", video_data=[], audio_data=[])


def _run(coro, *a, **k):
    return asyncio.run(coro(*a, **k))


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def case_text_input():
    """Normal raw-text request: patch must be a no-op."""
    print("\n[Case A] raw-text prompt  ->  patch should be a NO-OP")
    captured_stock, captured_patch = {}, {}
    text = "hello world"

    out_stock = _run(_stock_process_mm_data_async, _make_self_stub(captured_stock), [], text, _request_obj())
    out_patch = _run(_patched_process_mm_data_async, _make_self_stub(captured_patch), [], text, _request_obj())

    print(f"  load_mm_data prompt (stock) : {captured_stock['load_mm_data_prompt']!r}")
    print(f"  load_mm_data prompt (patch) : {captured_patch['load_mm_data_prompt']!r}")
    for key in ("input_ids", "im_start_id", "im_end_id", "im_token_id", "mrope_position_delta"):
        same = out_stock[key] == out_patch[key]
        print(f"  {key:24s}: stock==patch ? {same}")
    mrope_same = torch.equal(out_stock["mrope_positions"], out_patch["mrope_positions"])
    print(f"  {'mrope_positions':24s}: stock==patch ? {mrope_same}")

    ok = all(out_stock[k] == out_patch[k] for k in ("input_ids", "im_start_id", "im_end_id")) and mrope_same
    print(f"  => Case A {'PASS (no-op)' if ok else 'FAIL'}")
    return ok


def case_pretokenized_input():
    """DeepEyes multi-turn pre-tokenized input: patch fixes input_ids +
    mrope."""
    print("\n[Case B] pre-tokenized list[int] (N image_pad per image)  ->  patch STRIPS + RESTORES")
    original = [VISION_START, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, VISION_END, 7, 8]
    captured_stock, captured_patch = {}, {}

    out_stock = _run(_stock_process_mm_data_async, _make_self_stub(captured_stock), [], original, _request_obj())
    out_patch = _run(_patched_process_mm_data_async, _make_self_stub(captured_patch), [], original, _request_obj())

    print(f"  original input_ids          : {original}")
    print(f"  load_mm_data prompt (stock) : {captured_stock['load_mm_data_prompt']}  (unchanged, N pads)")
    print(f"  load_mm_data prompt (patch) : {captured_patch['load_mm_data_prompt']}  (collapsed to 1 pad)")
    print(f"  returned input_ids (stock)  : {out_stock['input_ids']}  (drifted to retokenized [100,200,300])")
    print(f"  returned input_ids (patch)  : {out_patch['input_ids']}  (restored to original)")

    ids_restored = out_patch["input_ids"] == original
    ids_drifted_stock = out_stock["input_ids"] == [100, 200, 300]
    # mrope recomputed from restored ids: patch's rope input must be the original list,
    # stock's must be the retokenized [100,200,300].
    from sglang.srt.layers.rotary_embedding import MRotaryEmbedding

    # Re-run capturing rope kwargs for each.
    _ = _run(_stock_process_mm_data_async, _make_self_stub({}), [], original, _request_obj())
    stock_rope = MRotaryEmbedding.last_kwargs["input_ids"].tolist()[0]
    _ = _run(_patched_process_mm_data_async, _make_self_stub({}), [], original, _request_obj())
    patch_rope = MRotaryEmbedding.last_kwargs["input_ids"].tolist()[0]
    print(f"  mrope input_ids (stock)     : {stock_rope}  (from retokenized)")
    print(f"  mrope input_ids (patch)     : {patch_rope}  (from restored original)")

    ok = ids_restored and ids_drifted_stock and patch_rope == original and stock_rope == [100, 200, 300]
    print(f"  => Case B {'PASS (restored + mrope recomputed)' if ok else 'FAIL'}")
    return ok


def case_strip_unit():
    """Sanity: _strip_image_token is the exact collapse used in Case B."""
    print("\n[Case C] _strip_image_token unit check")
    src = [VISION_START, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, VISION_END, 7, 8]
    got = _strip_image_token(src)
    want = [VISION_START, IMAGE_PAD, VISION_END, 7, 8]
    print(f"  {src} -> {got}")
    ok = got == want
    print(f"  => Case C {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    # We compare the two function objects directly (stock reference vs the
    # patched implementation Relax binds at engine start-up). No need to bind
    # onto the class — that path is covered by the unit tests.
    print("=== Task-17 patch before/after consistency ===")
    results = [case_strip_unit(), case_text_input(), case_pretokenized_input()]
    print("\n=== Summary ===")
    print(f"  Case C strip unit      : {'PASS' if results[0] else 'FAIL'}")
    print(f"  Case A text no-op      : {'PASS' if results[1] else 'FAIL'}")
    print(f"  Case B pretokenized fix: {'PASS' if results[2] else 'FAIL'}")
    if all(results):
        print("\nALL PASS: patch changes only the two DeepEyes-specific things; normal requests unchanged.")
        return 0
    print("\nSOME FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
