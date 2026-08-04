# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from examples.mem_agent.compare_results import compare_pair


def test_compare_pair_uses_absolute_percentage_points():
    passed = compare_pair("50", {"sub_em_pct": 41.0}, {"sub_em_pct": 43.9})
    failed = compare_pair("50", {"sub_em_pct": 41.0}, {"sub_em_pct": 44.1})

    assert passed["absolute_gap_pp"] == pytest.approx(2.9)
    assert passed["passed"] is True
    assert failed["passed"] is False
