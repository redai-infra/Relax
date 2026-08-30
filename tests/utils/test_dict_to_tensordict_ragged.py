# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch

from relax.utils.utils import dict_to_tensordict


def test_dict_to_tensordict_preserves_mixed_empty_and_nonempty_rows() -> None:
    result = dict_to_tensordict({"opd_topk_teacher_log_probs": [[], [-1.0, -2.0]]}, batch_size=2)

    rows = result["opd_topk_teacher_log_probs"]
    assert rows.layout == torch.jagged
    assert rows[0].numel() == 0
    assert torch.equal(rows[1], torch.tensor([-1.0, -2.0]))


def test_dict_to_tensordict_preserves_nonempty_row_before_empty_row() -> None:
    result = dict_to_tensordict({"opd_topk_teacher_log_probs": [[-1.0, -2.0], []]}, batch_size=2)

    rows = result["opd_topk_teacher_log_probs"]
    assert rows[0].tolist() == [-1.0, -2.0]
    assert rows[1].numel() == 0
    assert rows[0].dtype == rows[1].dtype


def test_dict_to_tensordict_preserves_all_empty_ragged_rows() -> None:
    result = dict_to_tensordict({"opd_topk_token_ids": [[], []]}, batch_size=2)

    rows = result["opd_topk_token_ids"]
    assert rows.layout == torch.jagged
    assert rows[0].numel() == 0
    assert rows[1].numel() == 0


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("opd_topk_token_ids", [11, 12]),
        ("opd_topk_ksz", [2]),
    ],
)
@pytest.mark.parametrize("empty_first", [True, False])
def test_dict_to_tensordict_preserves_mixed_integer_rows(field, values, empty_first) -> None:
    rows = [[], values] if empty_first else [values, []]

    result = dict_to_tensordict({field: rows}, batch_size=2)
    converted = result[field]
    nonempty_index = 1 if empty_first else 0
    empty_index = 0 if empty_first else 1

    assert converted.layout == torch.jagged
    assert converted[empty_index].numel() == 0
    assert converted[nonempty_index].tolist() == values
    assert converted[empty_index].dtype == converted[nonempty_index].dtype
