# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the prompt-data based SFT streaming dataset."""

import asyncio
import json
from pathlib import Path

import pytest
import torch

from relax.engine.sft.dataset.streaming import (
    ProcessedSample,
    SFTMultimodalContractError,
    SFTStreamingDataset,
    _canonicalize_messages,
    _expand_loss_mask_via_alignment,
    pack_samples_for_tq,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class _FakeTokenizer:
    """Minimal tokenizer with generation-mask chat-template support."""

    chat_template = "{% generation %}assistant{% endgeneration %}"

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,  # noqa: ARG002
        tokenize=True,  # noqa: ARG002
        return_tensors=None,  # noqa: ARG002
        return_dict=False,
        return_assistant_tokens_mask=False,
        **kwargs,  # noqa: ARG002
    ):
        ids: list[int] = []
        masks: list[int] = []
        next_id = 1
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                text = "".join(part.get("text", "") for part in content if part.get("type") == "text")
            else:
                text = content
            n = max(1, len(text))
            ids.extend(range(next_id, next_id + n))
            masks.extend([1 if message["role"] in ("assistant", "function_call") else 0] * n)
            next_id += n
        input_ids = torch.tensor([ids], dtype=torch.long)
        if return_assistant_tokens_mask:
            return {"input_ids": input_ids, "assistant_masks": [masks]}
        return input_ids


def test_streaming_dataset_reads_prompt_data_messages(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        ],
    )

    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys=None,
        seed=42,
        prefetch_max_cached=0,
    )

    ds.shuffle(0)
    samples, crossed = ds.get_batch(1)

    assert crossed is False
    assert len(samples) == 1
    assert samples[0].loss_mask.sum().item() == len("Answer")
    assert samples[0].source_idx == 0
    ds.stop()


def test_pack_samples_for_tq_marks_samples_as_sft(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "A"},
                ]
            }
        ],
    )

    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys=None,
        seed=42,
        prefetch_max_cached=0,
    )

    ds.shuffle(0)
    samples, _ = ds.get_batch(1)
    batch = pack_samples_for_tq(samples)

    assert batch is not None
    assert batch["response_lengths"] == batch["total_lengths"]
    assert batch["response_lengths"][0] == len(batch["tokens"][0])
    assert sum(batch["loss_masks"][0]) == len("A")
    ds.stop()


def _make_text_only_sample() -> ProcessedSample:
    """A ProcessedSample with no multimodal inputs (text-only)."""
    return ProcessedSample(
        tokens=torch.tensor([1, 2, 3], dtype=torch.long),
        loss_mask=torch.tensor([0, 1, 1], dtype=torch.long),
        total_length=3,
        multimodal_train_inputs=None,
        source_idx=0,
    )


def test_pack_samples_for_tq_omits_multimodal_field_for_text_only_batch():
    # Default behaviour: an all-text batch carries no multimodal key.
    batch = pack_samples_for_tq([_make_text_only_sample()])

    assert batch is not None
    assert "multimodal_train_inputs" not in batch


def test_pack_samples_for_tq_forces_multimodal_field_for_text_only_batch():
    # A VL run (multimodal_keys configured) must always emit the field so the
    # consumer's fixed TQ field list stays satisfied even for text-only batches.
    batch = pack_samples_for_tq([_make_text_only_sample()], force_multimodal_field=True)

    assert batch is not None
    assert "multimodal_train_inputs" in batch
    assert batch["multimodal_train_inputs"] == [None]


def test_streaming_dataset_builds_messages_from_prompt_and_label_keys(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"prompt": "What is 2+2?", "answer": "4"}])

    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="prompt",
        label_key="answer",
        multimodal_keys=None,
        seed=42,
        prefetch_max_cached=0,
    )

    ds.shuffle(0)
    samples, _ = ds.get_batch(1)

    assert len(samples) == 1
    assert samples[0].loss_mask.sum().item() == len("4")
    ds.stop()


def test_streaming_dataset_rejects_messages_when_label_key_is_set(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A"},
                ],
                "answer": "extra",
            }
        ],
    )

    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key="answer",
        multimodal_keys=None,
        seed=42,
        prefetch_max_cached=0,
    )

    with pytest.raises(TypeError, match="--label-key is set"):
        ds.get_canonical_sample(0)
    ds.stop()


def test_streaming_dataset_rejects_prompt_string_without_label_key(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"prompt": "No label"}])

    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="prompt",
        label_key=None,
        multimodal_keys=None,
        seed=42,
        prefetch_max_cached=0,
    )

    with pytest.raises(TypeError, match="--label-key is not set"):
        ds.get_canonical_sample(0)
    ds.stop()


def test_streaming_dataset_collects_inline_structured_image_url(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "_sample_id": "inline-image-url",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the image."},
                            {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
                        ],
                    },
                    {"role": "assistant", "content": "Answer"},
                ],
                "images": [],
            }
        ],
    )

    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        seed=42,
        prefetch_max_cached=0,
    )

    sample = ds.get_canonical_sample(0)

    assert sample.images == ["https://example.test/image.png"]
    assert sample.messages[0].content[1] == {
        "type": "image",
        "image": "https://example.test/image.png",
    }
    ds.stop()


@pytest.mark.parametrize(
    ("messages", "expected_images"),
    [
        (
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.test/inline.png"}},
                    ],
                },
                {"role": "user", "content": "<image>\nDescribe both images."},
                {"role": "assistant", "content": "Answer"},
            ],
            ["https://example.test/inline.png", "/data/top-level.png"],
        ),
        (
            [
                {"role": "user", "content": "<image>\nFirst image."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.test/inline.png"}},
                    ],
                },
                {"role": "assistant", "content": "Answer"},
            ],
            ["/data/top-level.png", "https://example.test/inline.png"],
        ),
    ],
)
def test_streaming_dataset_preserves_mixed_inline_and_top_level_image_order(
    tmp_path: Path,
    messages: list[dict],
    expected_images: list[str],
):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"messages": messages, "images": ["/data/top-level.png"]}])
    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        seed=42,
        prefetch_max_cached=0,
    )

    sample = ds.get_canonical_sample(0)

    assert sample.images == expected_images
    ds.stop()


def test_streaming_dataset_preserves_repeated_inline_image_url(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    image_url = "https://example.test/repeated.png"
    _write_jsonl(
        path,
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                    {"role": "assistant", "content": "Answer"},
                ],
                "images": [],
            }
        ],
    )
    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        seed=42,
        prefetch_max_cached=0,
    )

    sample = ds.get_canonical_sample(0)

    assert sample.images == [image_url, image_url]
    ds.stop()


@pytest.mark.parametrize("image_url", [{"url": 123}, {"url": ""}, ""])
def test_streaming_dataset_rejects_malformed_inline_image_url(tmp_path: Path, image_url):
    path = tmp_path / "train.jsonl"
    rows = _invalid_image_url_rows()
    rows[0]["messages"][0]["content"][1]["image_url"] = image_url
    _write_jsonl(path, rows[:1])
    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        seed=42,
        prefetch_max_cached=0,
    )

    with pytest.raises(SFTMultimodalContractError, match="top_level_required_count"):
        ds.get_canonical_sample(0)
    ds.stop()


def _invalid_image_url_rows() -> list[dict]:
    return [
        {
            "_sample_id": "invalid-image-url",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the image."},
                        {"type": "image_url", "image_url": {}},
                    ],
                },
                {"role": "assistant", "content": "Bad sample"},
            ],
            "images": [],
        },
        {
            "_sample_id": "text-only",
            "messages": [
                {"role": "user", "content": "Text question"},
                {"role": "assistant", "content": "Good sample"},
            ],
            "images": [],
        },
    ]


def test_streaming_dataset_invalid_multimodal_defaults_to_error(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, _invalid_image_url_rows()[:1])
    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        seed=0,
        prefetch_max_cached=0,
    )
    ds.shuffle(0)

    try:
        with pytest.raises(SFTMultimodalContractError, match="sample idx=0, sample_id='invalid-image-url'"):
            ds.get_batch(1)
    finally:
        ds.stop()


@pytest.mark.parametrize("batch_mode", ["inline", "async", "prefetch"])
def test_streaming_dataset_invalid_multimodal_skip_refills_batch(tmp_path: Path, caplog, batch_mode: str):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, _invalid_image_url_rows())
    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        seed=0,
        prefetch_max_cached=4 if batch_mode == "prefetch" else 0,
        prefetch_chunk_size=1,
        prefetch_num_workers=1,
        invalid_multimodal_strategy="skip",
    )
    ds.shuffle(0)

    try:
        if batch_mode == "async":
            samples, _ = asyncio.run(ds.get_batch_async(1))
        else:
            samples, _ = ds.get_batch(1)
        assert [sample.source_idx for sample in samples] == [1]
        assert "SFTStreamingDataset[invalid-multimodal=skip]" in caplog.text
        assert "sample idx=0, sample_id='invalid-image-url'" in caplog.text
    finally:
        ds.stop()


def test_streaming_dataset_rejects_unknown_invalid_multimodal_strategy(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, _invalid_image_url_rows())

    with pytest.raises(ValueError, match="invalid_multimodal_strategy must be one of"):
        SFTStreamingDataset(
            path=str(path),
            tokenizer=_FakeTokenizer(),
            processor_pool=None,
            capacity=None,
            prompt_key="messages",
            label_key=None,
            multimodal_keys={"image": "images"},
            prefetch_max_cached=0,
            invalid_multimodal_strategy="ignore",
        )


def test_streaming_dataset_rejects_duplicate_inline_and_top_level_image_sources(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the image."},
                            {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
                        ],
                    },
                    {"role": "assistant", "content": "Answer"},
                ],
                "images": ["/data/image.png"],
            }
        ],
    )

    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        seed=42,
        prefetch_max_cached=0,
    )

    with pytest.raises(
        SFTMultimodalContractError,
        match=r"'inline_count': 1, 'top_level_required_count': 0.*'top_level_count': 1",
    ):
        ds.get_canonical_sample(0)
    ds.stop()


def test_streaming_dataset_accepts_literal_image_placeholder_with_matching_top_level_image(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "messages": [
                    {"role": "user", "content": "<image>\nDescribe the image."},
                    {"role": "assistant", "content": "Answer"},
                ],
                "images": ["/data/image.png"],
            }
        ],
    )

    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        seed=42,
        prefetch_max_cached=0,
    )

    sample = ds.get_canonical_sample(0)

    assert sample.images == ["/data/image.png"]
    assert sample.messages[0].content[0] == {"type": "image", "image": "/data/image.png"}
    ds.stop()


def test_streaming_dataset_prefetch_and_inline_yield_same_samples(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {"messages": [{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": f"A{i}"}]}
            for i in range(8)
        ],
    )

    def _collect(prefetch: int) -> list[int]:
        ds = SFTStreamingDataset(
            path=str(path),
            tokenizer=_FakeTokenizer(),
            processor_pool=None,
            capacity=None,
            prompt_key="messages",
            label_key=None,
            multimodal_keys=None,
            seed=42,
            prefetch_max_cached=prefetch,
        )
        ds.shuffle(0)
        seen: list[int] = []
        for _ in range(2):
            batch, _ = ds.get_batch(4)
            seen.extend(s.source_idx for s in batch)
        ds.stop()
        return seen

    inline_order = _collect(0)
    prefetch_order = _collect(64)
    assert inline_order == prefetch_order
    assert sorted(inline_order) == list(range(8))


# ----------------------------------------------------------------------
# _expand_loss_mask_via_alignment
# ----------------------------------------------------------------------


def test_expand_loss_mask_alignment_text_only_is_identity():
    short = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    mask = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    out = _expand_loss_mask_via_alignment(
        short_ids=short, short_mask=mask, expanded_ids=short, pad_token_ids=frozenset({99})
    )
    assert out.tolist() == [0, 1, 1, 0]


def test_expand_loss_mask_alignment_single_image_pad_run():
    pad = 99
    short = torch.tensor([1, pad, 5], dtype=torch.long)
    short_mask = torch.tensor([0, 0, 1], dtype=torch.long)
    expanded = torch.tensor([1, pad, pad, pad, pad, pad, 5], dtype=torch.long)
    out = _expand_loss_mask_via_alignment(
        short_ids=short, short_mask=short_mask, expanded_ids=expanded, pad_token_ids=frozenset({pad})
    )
    assert out.tolist() == [0, 0, 0, 0, 0, 0, 1]


def test_expand_loss_mask_alignment_multiple_pad_kinds():
    img, aud = 99, 88
    short = torch.tensor([1, img, 2, aud, 3], dtype=torch.long)
    short_mask = torch.tensor([0, 0, 0, 0, 1], dtype=torch.long)
    expanded = torch.tensor([1, img, img, img, 2, aud, aud, 3], dtype=torch.long)
    out = _expand_loss_mask_via_alignment(
        short_ids=short, short_mask=short_mask, expanded_ids=expanded, pad_token_ids=frozenset({img, aud})
    )
    assert out.tolist() == [0, 0, 0, 0, 0, 0, 0, 1]


def test_expand_loss_mask_alignment_disagreement_raises():
    short = torch.tensor([1, 2, 3], dtype=torch.long)
    expanded = torch.tensor([1, 999, 3], dtype=torch.long)
    with pytest.raises(ValueError, match="alignment failed"):
        _expand_loss_mask_via_alignment(
            short_ids=short,
            short_mask=torch.zeros_like(short),
            expanded_ids=expanded,
            pad_token_ids=frozenset(),
        )


def test_expand_loss_mask_alignment_trailing_mismatch_raises():
    short = torch.tensor([1, 2], dtype=torch.long)
    expanded = torch.tensor([1, 2, 3], dtype=torch.long)
    with pytest.raises(ValueError, match="alignment ended early"):
        _expand_loss_mask_via_alignment(
            short_ids=short,
            short_mask=torch.zeros_like(short),
            expanded_ids=expanded,
            pad_token_ids=frozenset(),
        )


def test_canonicalize_messages_extracts_tool_calls():
    """Raw OpenAI-style assistant messages may carry a ``tool_calls`` field;
    ``_canonicalize_messages`` must propagate it onto ``CanonicalMessage`` so
    the chat template can render the tool calls.

    Messages without the field stay None.
    """
    tool_call = {"type": "function", "function": {"name": "f", "arguments": {"x": 1}}}
    raw = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [tool_call]},
        {"role": "tool", "content": "ok"},
    ]
    msgs = _canonicalize_messages(raw, require_response=True)
    assert msgs[0].tool_calls is None
    assert msgs[1].tool_calls == [tool_call]
    assert msgs[2].tool_calls is None
