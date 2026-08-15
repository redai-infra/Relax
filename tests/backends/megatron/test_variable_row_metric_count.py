# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
VALID_ROWS_KEY = "agentic_variable_row_valid_rows"


def _load_metric_count_helper():
    path = REPO_ROOT / "relax" / "backends" / "megatron" / "loss.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definition = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_metric_num_samples"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            definition,
        ],
        type_ignores=[],
    )

    def get_rollout_valid_row_flags(batch):
        return batch.get(VALID_ROWS_KEY)

    namespace = {"RolloutBatch": dict, "get_rollout_valid_row_flags": get_rollout_valid_row_flags}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["_metric_num_samples"]


METRIC_NUM_SAMPLES = _load_metric_count_helper()


def test_train_metrics_exclude_synthetic_rows_and_legacy_count_is_unchanged():
    variable_rows = {
        "response_lengths": [3, 4, 1, 1],
        VALID_ROWS_KEY: [True, True, False, False],
    }
    assert METRIC_NUM_SAMPLES(variable_rows) == 2
    assert METRIC_NUM_SAMPLES({"response_lengths": [3, 4, 1, 1]}) == 4
