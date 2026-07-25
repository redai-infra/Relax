# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for relax.utils.data.processing_utils.

Imports are deferred to fixtures because processing_utils pulls in the heavy
imageio / soundfile / transformers / torch stack at module level, which trips a
numpy ABI mismatch in this CI image during pytest collection.
"""

import threading

import pytest


@pytest.fixture(scope="module")
def adapt_processor_kwargs():
    from relax.utils.data.processing_utils import adapt_processor_kwargs as fn

    return fn


@pytest.fixture()
def processing_utils():
    from relax.utils.data import processing_utils as module

    module.shutdown_encode_executor()
    yield module
    module.shutdown_encode_executor()


@pytest.mark.parametrize(
    ("cpu_count", "expected"),
    [(4, 4), (12, 12), (32, 32), (72, 32), (128, 32), (None, 8)],
)
def test_encode_executor_uses_cpu_aware_default(monkeypatch, processing_utils, cpu_count, expected):
    monkeypatch.setattr(processing_utils.os, "cpu_count", lambda: cpu_count)

    executor = processing_utils.configure_encode_executor()

    assert executor._max_workers == expected


@pytest.mark.parametrize("workers", [1, 8, 32, 64])
def test_encode_executor_uses_explicit_worker_count(processing_utils, workers):
    executor = processing_utils.configure_encode_executor(workers)

    assert executor._max_workers == workers


@pytest.mark.parametrize("workers", [0, -1, -32])
def test_encode_executor_rejects_non_positive_values(processing_utils, workers):
    with pytest.raises(ValueError, match="positive integer"):
        processing_utils.configure_encode_executor(workers)


@pytest.mark.parametrize("workers", [True, 1.5, "8"])
def test_encode_executor_rejects_non_integer_values(processing_utils, workers):
    with pytest.raises(TypeError, match="integer"):
        processing_utils.configure_encode_executor(workers)


def test_encode_executor_same_configuration_is_idempotent(processing_utils):
    first = processing_utils.configure_encode_executor(8)
    second = processing_utils.configure_encode_executor(8)

    assert first is second
    assert processing_utils.get_encode_executor() is first


def test_encode_executor_rejects_reconfiguration(processing_utils):
    original = processing_utils.configure_encode_executor(8)

    with pytest.raises(RuntimeError, match="already initialized with 8 workers"):
        processing_utils.configure_encode_executor(16)

    assert processing_utils.get_encode_executor() is original


def test_encode_executor_shutdown_allows_reinitialization(processing_utils):
    first = processing_utils.configure_encode_executor(4)

    processing_utils.shutdown_encode_executor()

    with pytest.raises(RuntimeError, match="cannot schedule new futures after shutdown"):
        first.submit(lambda: None)

    second = processing_utils.configure_encode_executor(4)
    assert second is not first


@pytest.mark.asyncio
async def test_async_encode_uses_configured_executor(monkeypatch, processing_utils):
    processing_utils.configure_encode_executor(1)
    monkeypatch.setattr(
        processing_utils,
        "encode_image_for_rollout_engine",
        lambda image: (image, threading.current_thread().name),
    )

    image, thread_name = await processing_utils.async_encode_image_for_rollout_engine("image")

    assert image == "image"
    assert thread_name.startswith("relax-encode")


class _FakeQwenVLProcessor:
    """Stand-in for a standard HF VLM processor (Qwen-VL / Qwen-Omni shape)."""

    def __call__(self, text=None, images=None, videos=None, audio=None, **kwargs):
        raise AssertionError("not invoked in tests")


class KimiK25Processor:
    """Class-name match for the K2.x adapter branch — body is irrelevant."""

    def __call__(self, messages=None, medias=None, text=None, return_tensors="pt", **kwargs):
        raise AssertionError("not invoked in tests")


class KimiK26Processor(KimiK25Processor):
    """Future K2.x variants must keep getting the K2 adapter via class-name
    prefix."""


def test_adapt_processor_kwargs_default_passthrough(adapt_processor_kwargs):
    proc = _FakeQwenVLProcessor()
    mm = {"images": ["pil_img_1"], "videos": [], "audio": []}
    extra = {"return_tensors": None, "images_kwargs": {"return_tensors": "pt"}}

    out = adapt_processor_kwargs(proc, mm, extra)

    assert out == {**mm, **extra}, "Non-K2 processors must see the original shape unchanged"


def test_adapt_processor_kwargs_default_handles_none_inputs(adapt_processor_kwargs):
    proc = _FakeQwenVLProcessor()
    assert adapt_processor_kwargs(proc, None, None) == {}
    assert adapt_processor_kwargs(proc, None, {"foo": 1}) == {"foo": 1}
    assert adapt_processor_kwargs(proc, {"images": ["x"]}, None) == {"images": ["x"]}


def test_adapt_processor_kwargs_kimi_k25_translates_images_to_medias(adapt_processor_kwargs):
    proc = KimiK25Processor()
    mm = {"images": ["pil_a", "pil_b"], "videos": [], "audio": []}

    out = adapt_processor_kwargs(proc, mm, extra_kwargs={"return_tensors": None})

    assert out == {
        "medias": [
            {"type": "image", "image": "pil_a"},
            {"type": "image", "image": "pil_b"},
        ],
        # Forced "pt" wins over the None coming from build_processor_kwargs;
        # the K2.x processor's tokenizer call needs real tensors.
        "return_tensors": "pt",
    }
    assert "images" not in out and "videos" not in out and "audio" not in out


def test_adapt_processor_kwargs_kimi_future_variant_uses_same_branch(adapt_processor_kwargs):
    proc = KimiK26Processor()
    out = adapt_processor_kwargs(proc, {"images": ["x"]}, None)
    assert out["medias"] == [{"type": "image", "image": "x"}]


def test_adapt_processor_kwargs_kimi_no_images_returns_only_return_tensors(adapt_processor_kwargs):
    proc = KimiK25Processor()
    # No medias → caller relies on text-only K2 branch (still legal as long as text is provided).
    out = adapt_processor_kwargs(proc, {"images": [], "videos": [], "audio": []}, None)
    assert out == {"return_tensors": "pt"}


def test_adapt_processor_kwargs_kimi_warns_on_unsupported_modalities(caplog, adapt_processor_kwargs):
    proc = KimiK25Processor()
    mm = {"images": ["x"], "videos": ["v"], "audio": ["a"]}
    with caplog.at_level("WARNING"):
        out = adapt_processor_kwargs(proc, mm, None)

    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "video" in msgs.lower()
    assert "audio" in msgs.lower()
    assert "videos" not in out
    assert "audio" not in out
    assert out["medias"] == [{"type": "image", "image": "x"}]


def test_adapt_processor_kwargs_kimi_drops_conflicting_extra_kwargs(adapt_processor_kwargs):
    """build_processor_kwargs adds images_kwargs / videos_kwargs /
    audio_kwargs.

    + return_tensors=None.

    K2.x ignores the per-modality dicts via **kwargs but would crash on
    duplicate return_tensors; the adapter must own the kwarg space.
    """
    proc = KimiK25Processor()
    extra = {
        "return_tensors": None,
        "images_kwargs": {"return_tensors": "pt"},
        "videos_kwargs": {"return_tensors": "pt"},
        "audio_kwargs": {"return_tensors": "pt"},
    }
    out = adapt_processor_kwargs(proc, {"images": ["x"]}, extra)
    assert out == {
        "medias": [{"type": "image", "image": "x"}],
        "return_tensors": "pt",
    }


@pytest.mark.parametrize(
    "cls_name, expected_kimi",
    [
        ("KimiK25Processor", True),
        ("KimiK26Processor", True),
        ("KimiK2Processor", True),
        ("Qwen2VLProcessor", False),
        ("Qwen3OmniProcessor", False),
        ("KimiAudio", False),  # doesn't end with Processor
        ("SomeOtherKimiK2Tokenizer", False),  # doesn't end with Processor
    ],
)
def test_adapt_processor_kwargs_class_name_match(cls_name, expected_kimi, adapt_processor_kwargs):
    fake_cls = type(cls_name, (), {"__call__": lambda self, **kw: None})
    proc = fake_cls()
    out = adapt_processor_kwargs(proc, {"images": ["x"]}, None)
    if expected_kimi:
        assert "medias" in out
        assert "images" not in out
    else:
        assert "images" in out
        assert "medias" not in out
