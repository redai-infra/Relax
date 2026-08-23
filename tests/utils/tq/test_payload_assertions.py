# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only tests for TransferQueue payload byte-identity assertions."""

from __future__ import annotations

import struct
import warnings
from typing import Any

import numpy as np
import pytest
import torch

from tests.utils.tq._payload_assertions import diff_digests, leaf_digests


class NonTensorData:
    """Minimal tensordict-compatible wrapper for an optional-dependency-free
    test."""

    def __init__(self, data: Any) -> None:
        self.data = data


class NonTensorStack:
    """Minimal tensordict-compatible wrapper for an optional-dependency-free
    test."""

    def __init__(self, data: Any) -> None:
        self._data = data

    def tolist(self) -> Any:
        return self._data


def test_leaf_digests_distinguishes_raw_bytes_from_value_equality():
    positive_zero = leaf_digests(torch.tensor([0.0], dtype=torch.float32))
    negative_zero = leaf_digests(torch.tensor([-0.0], dtype=torch.float32))

    assert positive_zero["payload"][:2] == negative_zero["payload"][:2]
    assert positive_zero["payload"][2] != negative_zero["payload"][2]
    assert diff_digests(positive_zero, negative_zero) == [
        f"payload: sha256 mismatch (expected {positive_zero['payload'][2]}, got {negative_zero['payload'][2]})"
    ]


def test_scalar_nan_payload_bits_are_not_collapsed_by_repr():
    first = struct.unpack("!d", bytes.fromhex("7ff8000000000001"))[0]
    second = struct.unpack("!d", bytes.fromhex("7ff8000000000002"))[0]

    assert repr(first) == repr(second) == "nan"
    assert leaf_digests(first) != leaf_digests(second)


def test_dict_paths_do_not_collapse_dotted_keys_into_nested_keys():
    digests = leaf_digests({"a.b": 1, "a": {"b": 2}})

    assert len(digests) == 2
    assert set(digests) == {"payload['a.b']", "payload.a.b"}


def test_non_string_dict_keys_fail_loudly():
    with pytest.raises(TypeError, match="Unsupported payload dict key at payload: int"):
        leaf_digests({1: "value"})


def test_leaf_digests_preserves_dtype_shape_and_nested_tensor_rows():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        nested = torch.nested.nested_tensor(
            [torch.tensor([1, 2], dtype=torch.int16), torch.tensor([3], dtype=torch.int16)]
        )
    payload = {
        "array": np.array([[1, 2]], dtype=np.uint16),
        "nested": nested,
        "tensor": torch.tensor([[1, 2]], dtype=torch.int32),
    }

    digests = leaf_digests(payload)

    assert digests["payload.array"][:2] == ("np.uint16", "1x2")
    assert digests["payload.nested[0]"][:2] == ("torch.int16", "2")
    assert digests["payload.nested[1]"][:2] == ("torch.int16", "1")
    assert digests["payload.tensor"][:2] == ("torch.int32", "1x2")


def test_leaf_digests_unwraps_non_tensor_containers():
    wrapped = NonTensorStack([NonTensorData({"text": "hello"}), NonTensorData(None)])

    assert leaf_digests(wrapped) == leaf_digests([{"text": "hello"}, None])


def test_leaf_digests_rejects_unknown_leaf_type():
    with pytest.raises(TypeError, match=r"Unsupported payload leaf at payload\.bad: object"):
        leaf_digests({"bad": object()})


def test_diff_digests_reports_missing_and_extra_leaves():
    expected = leaf_digests({"expected": 1})
    actual = leaf_digests({"actual": 1})

    problems = diff_digests(expected, actual)

    assert len(problems) == 2
    assert problems[0].startswith("payload.actual: unexpected extra leaf")
    assert problems[1].startswith("payload.expected: missing")
