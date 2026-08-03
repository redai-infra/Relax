# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the DeepEyes Qwen-VL processor monkey-patch.

SGLang is not installed in the test environment, so the upstream modules the
patch binds to (``sglang.srt.multimodal.processors.qwen_vl``,
``sglang.srt.layers.rotary_embedding``, ``sglang.srt.managers.schedule_batch``)
are stubbed in ``sys.modules``. The patch module itself imports only
``numpy``/``torch`` at top level (all sglang imports are lazy, inside
functions), so importing it is always light.
"""

from __future__ import annotations

import sys
import types

import pytest


torch = pytest.importorskip("torch")

from relax.backends.sglang.patches.qwen_vl_patch import (  # noqa: E402
    _PATCH_FLAG,
    _patched_process_mm_data_async,
    _strip_image_token,
    _verify_upstream_api,
    apply_qwen_vl_patches,
)


IMAGE_PAD = 151655
VISION_START = 151652
VISION_END = 151653


# ---------------------------------------------------------------------------
# _strip_image_token — pure list logic, no sglang needed.
# ---------------------------------------------------------------------------


def test_strip_image_token_passes_through_non_list():
    assert _strip_image_token("a string prompt") == "a string prompt"
    assert _strip_image_token(None) is None
    assert _strip_image_token(42) == 42


def test_strip_image_token_empty_or_single():
    assert _strip_image_token([]) == []
    assert _strip_image_token([IMAGE_PAD]) == [IMAGE_PAD]
    assert _strip_image_token([1]) == [1]


def test_strip_image_token_collapses_run_to_single():
    src = [VISION_START, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, VISION_END, 1, 2]
    assert _strip_image_token(src) == [VISION_START, IMAGE_PAD, VISION_END, 1, 2]


def test_strip_image_token_keeps_surrounding_markers_and_multiple_runs():
    # Two images in one sequence, each must collapse independently while the
    # surrounding <|vision_start|>/<|vision_end|> markers stay intact.
    src = [
        VISION_START,
        IMAGE_PAD,
        IMAGE_PAD,
        VISION_END,
        10,
        11,
        VISION_START,
        IMAGE_PAD,
        VISION_END,
    ]
    assert _strip_image_token(src) == [
        VISION_START,
        IMAGE_PAD,
        VISION_END,
        10,
        11,
        VISION_START,
        IMAGE_PAD,
        VISION_END,
    ]


def test_strip_image_token_respects_custom_id():
    # Non-default image-token id is honoured.
    src = [9, 9, 9, 1]
    assert _strip_image_token(src, image_token_id=9) == [9, 1]


def test_strip_image_token_does_not_touch_isolated_pads():
    # Non-adjacent image_pad tokens are all kept (each is its own run of 1).
    assert _strip_image_token([IMAGE_PAD, 1, IMAGE_PAD]) == [IMAGE_PAD, 1, IMAGE_PAD]


# ---------------------------------------------------------------------------
# Fake SGLang module machinery for apply / verify / method tests.
# ---------------------------------------------------------------------------


def _make_fake_processor_cls():
    class QwenVLImageProcessor:
        # Real upstream methods the patch binds onto / calls via self.
        def load_mm_data(self, *args, **kwargs):  # pragma: no cover - replaced by stubs
            raise NotImplementedError

        def process_and_combine_mm_data(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

        async def process_mm_data_async(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    return QwenVLImageProcessor


def _install_fake_sglang(monkeypatch, processor_cls=None, *, with_preprocess_video=True):
    """Register minimal fake ``sglang.*`` modules in ``sys.modules``.

    Returns the ``qwen_vl`` module object so tests can read back the processor
    class / preprocess_video that the patch bound to.
    """
    processor_cls = processor_cls or _make_fake_processor_cls()

    sglang = types.ModuleType("sglang")
    srt = types.ModuleType("sglang.srt")
    mm = types.ModuleType("sglang.srt.multimodal")
    procs = types.ModuleType("sglang.srt.multimodal.processors")
    layers = types.ModuleType("sglang.srt.layers")
    managers = types.ModuleType("sglang.srt.managers")

    qwen_vl_mod = types.ModuleType("sglang.srt.multimodal.processors.qwen_vl")
    qwen_vl_mod.QwenVLImageProcessor = processor_cls

    if with_preprocess_video:

        async def preprocess_video(video, video_config=None):
            return video, {}

        qwen_vl_mod.preprocess_video = preprocess_video

    rotary_mod = types.ModuleType("sglang.srt.layers.rotary_embedding")

    class MRotaryEmbedding:
        @staticmethod
        def get_rope_index(**kwargs):
            # Return a 2D-like tensor so .squeeze(1) works; capture kwargs via
            # a side channel for assertions.
            MRotaryEmbedding.last_kwargs = kwargs
            return torch.zeros((1, 4, 3), dtype=torch.long), 0

    rotary_mod.MRotaryEmbedding = MRotaryEmbedding

    sched_mod = types.ModuleType("sglang.srt.managers.schedule_batch")

    class MultimodalProcessorOutput:
        @staticmethod
        def from_dict(d):
            return d

    sched_mod.MultimodalProcessorOutput = MultimodalProcessorOutput

    # Register the full parent chain so ``from sglang.srt... import ...`` works.
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
        monkeypatch.setitem(sys.modules, name, mod)

    return qwen_vl_mod, rotary_mod


@pytest.fixture
def fake_sglang(monkeypatch):
    """Install fake sglang modules with a *fresh* processor class per test so
    the ``_PATCH_FLAG`` marker never leaks between tests."""
    qwen_vl_mod, rotary_mod = _install_fake_sglang(monkeypatch)
    return qwen_vl_mod, rotary_mod


# ---------------------------------------------------------------------------
# apply_qwen_vl_patches — idempotency + binding.
# ---------------------------------------------------------------------------


def test_apply_binds_patched_method_and_marks_flag(fake_sglang):
    qwen_vl_mod, _ = fake_sglang
    proc = qwen_vl_mod.QwenVLImageProcessor

    assert not getattr(proc, _PATCH_FLAG, False)
    assert proc.process_mm_data_async is not _patched_process_mm_data_async

    assert apply_qwen_vl_patches() is True

    assert getattr(proc, _PATCH_FLAG, False) is True
    assert proc.process_mm_data_async is _patched_process_mm_data_async


def test_apply_is_idempotent(fake_sglang):
    qwen_vl_mod, _ = fake_sglang
    proc = qwen_vl_mod.QwenVLImageProcessor

    apply_qwen_vl_patches()
    bound = proc.process_mm_data_async

    # Second call must be a no-op: returns False, same method reference.
    assert apply_qwen_vl_patches() is False
    assert proc.process_mm_data_async is bound


# ---------------------------------------------------------------------------
# _verify_upstream_api — fail-fast on missing symbols.
# ---------------------------------------------------------------------------


def test_verify_passes_when_api_present(fake_sglang):
    # Should not raise.
    _verify_upstream_api()


def test_verify_fails_fast_when_process_mm_data_async_missing(monkeypatch):
    class Bare:
        load_mm_data = staticmethod(lambda *a, **k: None)
        process_and_combine_mm_data = staticmethod(lambda *a, **k: None)

    _install_fake_sglang(monkeypatch, processor_cls=Bare)
    with pytest.raises(RuntimeError, match="process_mm_data_async"):
        _verify_upstream_api()


def test_verify_fails_fast_when_preprocess_video_missing(monkeypatch):
    _install_fake_sglang(monkeypatch, with_preprocess_video=False)
    with pytest.raises(RuntimeError, match="preprocess_video"):
        _verify_upstream_api()


def test_apply_does_not_raise_on_verify_failure(monkeypatch):
    """``apply_deepeyes_qwen_vl_patch`` must swallow failures so it never
    blocks server start-up — but the direct ``apply_qwen_vl_patches`` still
    surfaces them."""
    _install_fake_sglang(monkeypatch, with_preprocess_video=False)
    from relax.backends.sglang.patches.qwen_vl_patch import apply_deepeyes_qwen_vl_patch

    # Must not raise.
    apply_deepeyes_qwen_vl_patch()


# ---------------------------------------------------------------------------
# _patched_process_mm_data_async — the two Relax changes in isolation.
# ---------------------------------------------------------------------------


def _make_self_stub(captured):
    """Build a fake ``self`` whose ``load_mm_data`` /
    ``process_and_combine_mm_data`` record their inputs into ``captured`` and
    return tensors the method body can traverse (image-only, non-qwen3
    path)."""
    from types import SimpleNamespace

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

    mm_tokens = SimpleNamespace(
        image_token_id=IMAGE_PAD,
        video_token_id=151656,
        audio_token_id=None,
    )

    class SelfStub:
        def __init__(self):
            self.model_type = "qwen2_5_vl"
            self.vision_start_token_id = VISION_START
            self.vision_end_token_id = VISION_END
            self.audio_start_token_id = None
            self.video_config = {}
            self.hf_config = hf_config
            self.mm_tokens = mm_tokens

        def load_mm_data(self, **kwargs):
            captured["load_mm_data"] = kwargs
            return base_output

        def process_and_combine_mm_data(self, base_output, mm_tokens, **kwargs):
            captured["process_and_combine"] = kwargs
            # "Re-tokenized" ids that differ from the caller's original — the
            # patch must restore the original over this.
            mm_items, input_ids = [], torch.tensor([100, 200, 300])
            return mm_items, input_ids, ret

    return SelfStub()


def _make_request_obj():
    from types import SimpleNamespace

    return SimpleNamespace(rid="rid-1", video_data=[], audio_data=[])


def test_patched_method_text_input_is_passthrough(fake_sglang):
    """Raw-text (str) prompts must NOT trigger strip/restore: load_mm_data
    receives the text unchanged and the returned input_ids come from the mm
    pipeline (the 'retokenized' tensor), not a restored list."""
    qwen_vl_mod, rotary_mod = fake_sglang
    apply_qwen_vl_patches()

    captured: dict = {}
    self_obj = _make_self_stub(captured)

    result = _run(
        _patched_process_mm_data_async,
        self_obj,
        image_data=[],
        input_text="hello world",
        request_obj=_make_request_obj(),
    )

    # No stripping happened: prompt is the original str.
    assert captured["load_mm_data"]["prompt"] == "hello world"
    # No restore: returned input_ids are the mm pipeline's tensor.
    assert result["input_ids"] == [100, 200, 300]


def test_patched_method_list_input_strips_and_restores(fake_sglang):
    """Pre-tokenized list[int] with N image_pads per image: load_mm_data must
    receive the *collapsed* prompt (one image_pad), while the returned
    input_ids and the mrope input must be the *original* expanded list."""
    qwen_vl_mod, rotary_mod = fake_sglang
    apply_qwen_vl_patches()

    original = [VISION_START, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, VISION_END, 7, 8]
    captured: dict = {}
    self_obj = _make_self_stub(captured)

    result = _run(
        _patched_process_mm_data_async,
        self_obj,
        image_data=[],
        input_text=original,
        request_obj=_make_request_obj(),
    )

    # Change #1: prompt passed to load_mm_data is collapsed (3 -> 1 image_pad).
    assert captured["load_mm_data"]["prompt"] == [VISION_START, IMAGE_PAD, VISION_END, 7, 8]

    # Change #2: returned input_ids are the ORIGINAL expanded list, not [100,200,300].
    assert result["input_ids"] == original

    # mrope recompute used the restored (original) input_ids, not the
    # 'retokenized' [100,200,300].
    rope_kwargs = rotary_mod.MRotaryEmbedding.last_kwargs
    rope_input_ids = rope_kwargs["input_ids"]
    assert rope_input_ids.tolist()[0] == original


def test_patched_method_returns_multimodal_processor_output_when_available(fake_sglang):
    """When sglang exposes ``MultimodalProcessorOutput.from_dict`` the result
    is routed through it (version-agnostic return wrapping, mirroring OPD)."""
    apply_qwen_vl_patches()
    captured: dict = {}
    self_obj = _make_self_stub(captured)

    result = _run(
        _patched_process_mm_data_async, self_obj, image_data=[], input_text="x", request_obj=_make_request_obj()
    )
    # The fake from_dict returns the dict as-is; the key assertion is that no
    # exception was raised and the shape is the upstream return contract.
    assert "mrope_positions" in result
    assert "mm_items" in result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro_func, *args, **kwargs):
    import asyncio

    return asyncio.run(coro_func(*args, **kwargs))
