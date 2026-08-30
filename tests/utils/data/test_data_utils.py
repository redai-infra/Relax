# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for shared data message normalization."""

import pytest

from relax.utils.data.data_utils import build_messages, collect_message_multimodal_data, process_raw_sample


@pytest.mark.parametrize(
    "image_url",
    [
        "https://example.test/image.png",
        {"url": "https://example.test/image.png"},
    ],
)
def test_build_messages_normalizes_inline_image_url(image_url):
    row = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image."},
                    {"type": "image_url", "image_url": image_url},
                ],
            }
        ]
    }

    messages = build_messages(
        row,
        prompt_key="messages",
        system_prompt=None,
        as_conversation=True,
        multimodal_keys=None,
    )

    assert messages[0]["content"][1] == {
        "type": "image",
        "image": "https://example.test/image.png",
    }
    assert collect_message_multimodal_data(messages)["image"] == ["https://example.test/image.png"]


def test_build_messages_uses_top_level_media_without_mutating_input():
    row = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {}},
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ],
        "images": ["/data/image.png"],
    }

    first_messages = build_messages(
        row,
        prompt_key="messages",
        system_prompt=None,
        as_conversation=True,
        multimodal_keys={"image": "images"},
    )
    second_messages = build_messages(
        row,
        prompt_key="messages",
        system_prompt=None,
        as_conversation=True,
        multimodal_keys={"image": "images"},
    )

    assert row["images"] == ["/data/image.png"]
    assert first_messages == second_messages
    assert first_messages[0]["content"][0] == {"type": "image", "image": "/data/image.png"}
    assert collect_message_multimodal_data(first_messages)["image"] == ["/data/image.png"]


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return f"rendered:{messages[-1]['content']}"


def test_process_raw_sample_surfaces_teacher_prompt_as_metadata():
    sample = process_raw_sample(
        {"text": "question", "teacher_text": "privileged question", "label": "answer"},
        _Tokenizer(),
        processor=None,
        prompt_key="text",
        label_key="label",
        teacher_prompt_key="teacher_text",
        apply_chat_template=True,
    )

    assert sample.teacher_prompt is None
    assert sample.metadata["opd_teacher_prompt"] == "rendered:privileged question"


def test_process_raw_sample_without_teacher_key_leaves_no_privilege_metadata():
    sample = process_raw_sample(
        {"text": "question", "label": "answer"},
        _Tokenizer(),
        processor=None,
        prompt_key="text",
        label_key="label",
        teacher_prompt_key=None,
        apply_chat_template=True,
    )

    assert sample.teacher_prompt is None
    assert "opd_teacher_prompt" not in sample.metadata
