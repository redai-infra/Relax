# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import ast
import hashlib
import os
import time
from argparse import Namespace
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
PADDING_KEY = "agentic_variable_row_padding"
IDENTITY_KEY = "agentic_row_identity"
IDENTITY_TAG_KEY = "agentic_row_identity_tag"
IDENTITY_TAGS_FIELD = "agentic_row_identity_tags"
MAX_ROWS_ENV = "RELAX_AGENTIC_MAX_EXPORTED_ROWS_PER_SAMPLE"


def _load_helpers():
    path = REPO_ROOT / "relax" / "backends" / "megatron" / "actor.py"
    names = {
        "_use_agentic_variable_row_mode",
        "_agentic_variable_row_max_padded_rows",
        "_extract_agentic_variable_row_padding_flags",
        "_agentic_row_identity_tag",
        "_agentic_identity_tag_value",
        "_extract_agentic_variable_row_identities",
        "_validate_agentic_variable_row_partition_identities",
        "_validate_agentic_variable_row_partition",
        "_validate_agentic_variable_row_window",
        "_agentic_variable_row_drain_action",
        "_agentic_variable_row_stream_drained_consensus",
    }
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == names
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *definitions,
        ],
        type_ignores=[],
    )
    namespace = {
        "AGENTIC_MAX_EXPORTED_ROWS_PER_SAMPLE_ENV": MAX_ROWS_ENV,
        "AGENTIC_VARIABLE_ROW_PADDING_KEY": PADDING_KEY,
        "AGENTIC_ROW_IDENTITY_KEY": IDENTITY_KEY,
        "AGENTIC_ROW_IDENTITY_TAG_KEY": IDENTITY_TAG_KEY,
        "AGENTIC_ROW_IDENTITY_TAGS_FIELD": IDENTITY_TAGS_FIELD,
        "AGENTIC_ROW_IDENTITY_SCHEMA_VERSION": 1,
        "Any": Any,
        "hashlib": hashlib,
        "Mapping": Mapping,
        "Namespace": Namespace,
        "dist": None,
        "os": os,
        "torch": None,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


HELPERS = _load_helpers()


def _load_variable_row_collector():
    path = REPO_ROOT / "relax" / "backends" / "megatron" / "actor.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    actor_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor"
    )
    collector = next(
        node
        for node in actor_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_collect_agentic_variable_row_batches"
    )
    host_class = ast.ClassDef(
        name="_CollectorHost",
        bases=[],
        keywords=[],
        body=[collector],
        decorator_list=[],
    )

    class _Tensor:
        def __init__(self, values):
            self.value = values[0]

        def clone(self):
            return _Tensor([self.value])

        def item(self):
            return self.value

    class _Torch:
        int64 = object()

        @staticmethod
        def tensor(values, *, dtype, device):
            assert dtype is _Torch.int64
            assert device == "cuda:0"
            return _Tensor(values)

    class _Dist:
        class ReduceOp:
            MIN = object()
            MAX = object()
            SUM = object()

        @staticmethod
        def all_reduce(_tensor, *, op, group):
            assert op in (_Dist.ReduceOp.MIN, _Dist.ReduceOp.MAX, _Dist.ReduceOp.SUM)
            assert group == "dp-group"

    stream_drained_consensus = HELPERS._agentic_variable_row_stream_drained_consensus
    stream_drained_consensus.__globals__["torch"] = _Torch
    stream_drained_consensus.__globals__["dist"] = _Dist

    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            host_class,
        ],
        type_ignores=[],
    )
    namespace = {
        "_agentic_variable_row_drain_action": HELPERS._agentic_variable_row_drain_action,
        "_agentic_variable_row_max_padded_rows": HELPERS._agentic_variable_row_max_padded_rows,
        "_agentic_variable_row_stream_drained_consensus": stream_drained_consensus,
        "_extract_agentic_variable_row_padding_flags": (HELPERS._extract_agentic_variable_row_padding_flags),
        "_extract_agentic_variable_row_identities": (HELPERS._extract_agentic_variable_row_identities),
        "_validate_agentic_variable_row_partition": HELPERS._validate_agentic_variable_row_partition,
        "_validate_agentic_variable_row_partition_identities": (
            HELPERS._validate_agentic_variable_row_partition_identities
        ),
        "_validate_agentic_variable_row_window": HELPERS._validate_agentic_variable_row_window,
        "AGENTIC_ROW_IDENTITY_TAGS_FIELD": IDENTITY_TAGS_FIELD,
        "device_utils": SimpleNamespace(make_current_torch_device=lambda: "cuda:0"),
        "dist": _Dist,
        "logger": SimpleNamespace(info=lambda *args, **kwargs: None),
        "mpu": SimpleNamespace(
            get_context_parallel_world_size=lambda: 1,
            get_data_parallel_group=lambda **_kwargs: "dp-group",
            get_data_parallel_world_size=lambda **_kwargs: 1,
        ),
        "os": os,
        "time": time,
        "timer": lambda _name: nullcontext(),
        "torch": _Torch,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["_CollectorHost"]


VARIABLE_ROW_COLLECTOR = _load_variable_row_collector()


def _mode_args(**overrides):
    values = {
        "group_rm": True,
        "agentic_custom_advantage_path": "graphgpo.advantage",
        "use_dynamic_batch_size": True,
        "fully_async": False,
        "hybrid": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _custom_meta_batch(custom_meta):
    class _NonListSampleView:
        pass

    class _BatchMeta:
        samples = _NonListSampleView()

        def get_all_custom_meta(self):
            return custom_meta

    return _BatchMeta()


def _batch_meta(flags):
    return _custom_meta_batch([{PADDING_KEY: flag, "total_lengths": 7} for flag in flags])


def _identity_tag(row_id: str) -> int:
    digest = hashlib.sha256(row_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") & ((1 << 63) - 1)


def _real_identity(*, group_id: str, turn_index: int, turn_count: int) -> dict[str, Any]:
    row_ids = [f"row-{group_id}-{idx}" for idx in range(turn_count)]
    return {
        "schema_version": 1,
        "padding": False,
        "row_id": row_ids[turn_index],
        "rollout_group_id": group_id,
        "policy_version": "7",
        "task_id": f"task-{group_id}",
        "trajectory_id": f"trajectory-{group_id}",
        "turn_id": f"turn_{turn_index:03d}",
        "turn_index": turn_index,
        "sample_index": int(group_id),
        "terminal": turn_index == turn_count - 1,
        "truncated": False,
        "total_length": 7,
        "response_length": 1,
        "action_token_count": 1,
        "group_row_count": turn_count,
        "group_trajectory_count": 1,
        "group_row_ids_sha256": hashlib.sha256("\n".join(sorted(row_ids)).encode()).hexdigest(),
        "partition_expected_group_count": 2,
    }


def _padding_identity(row_index: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "padding": True,
        "row_id": f"padding-{row_index}",
        "rollout_group_id": None,
        "policy_version": "7",
        "task_id": None,
        "trajectory_id": None,
        "turn_id": None,
        "turn_index": None,
        "sample_index": None,
        "terminal": False,
        "truncated": False,
        "total_length": 7,
        "response_length": 1,
        "action_token_count": 0,
        "group_row_count": 0,
        "group_trajectory_count": 0,
        "group_row_ids_sha256": None,
        "partition_expected_group_count": 2,
    }


def _identity_batch(identities: list[dict[str, Any]]):
    custom_meta = [
        {
            PADDING_KEY: identity["padding"],
            "total_lengths": 7,
            IDENTITY_KEY: identity,
            IDENTITY_TAG_KEY: _identity_tag(identity["row_id"]),
        }
        for identity in identities
    ]
    rollout_data = {
        "total_lengths": [7] * len(identities),
        "response_lengths": [1] * len(identities),
        IDENTITY_TAGS_FIELD: [_identity_tag(identity["row_id"]) for identity in identities],
    }
    return rollout_data, _custom_meta_batch(custom_meta)


def test_variable_row_mode_is_default_off_and_rejects_async_or_hybrid(monkeypatch):
    monkeypatch.delenv(MAX_ROWS_ENV, raising=False)
    assert HELPERS._use_agentic_variable_row_mode(_mode_args()) is False

    monkeypatch.setenv(MAX_ROWS_ENV, "50")
    assert HELPERS._use_agentic_variable_row_mode(_mode_args()) is True
    assert HELPERS._use_agentic_variable_row_mode(_mode_args(group_rm=False)) is False
    assert HELPERS._use_agentic_variable_row_mode(_mode_args(agentic_custom_advantage_path=None)) is False
    assert HELPERS._use_agentic_variable_row_mode(_mode_args(use_dynamic_batch_size=False)) is False
    assert HELPERS._use_agentic_variable_row_mode(Namespace()) is False

    with pytest.raises(ValueError, match="synchronous mode only"):
        HELPERS._use_agentic_variable_row_mode(_mode_args(fully_async=True))
    with pytest.raises(ValueError, match="synchronous mode only"):
        HELPERS._use_agentic_variable_row_mode(_mode_args(fully_async=True, hybrid=True))
    with pytest.raises(ValueError, match="synchronous mode only"):
        HELPERS._use_agentic_variable_row_mode(_mode_args(hybrid=True))

    monkeypatch.setenv(MAX_ROWS_ENV, "not-a-number")
    with pytest.raises(ValueError, match="positive integer"):
        HELPERS._use_agentic_variable_row_mode(_mode_args())


def test_max_padded_rows_uses_exported_row_bound(monkeypatch):
    monkeypatch.setenv(MAX_ROWS_ENV, "3")
    args = Namespace(global_batch_size=8, rollout_batch_size=2, n_samples_per_prompt=2)
    assert HELPERS._agentic_variable_row_max_padded_rows(args) == 16

    monkeypatch.delenv(MAX_ROWS_ENV)
    with pytest.raises(ValueError, match=MAX_ROWS_ENV):
        HELPERS._agentic_variable_row_max_padded_rows(args)
    monkeypatch.setenv(MAX_ROWS_ENV, "0")
    with pytest.raises(ValueError, match="positive integer"):
        HELPERS._agentic_variable_row_max_padded_rows(args)


def test_padding_meta_parser_accepts_only_explicit_booleans():
    assert HELPERS._extract_agentic_variable_row_padding_flags(_batch_meta([False, True, False]), 3) == [
        False,
        True,
        False,
    ]

    with pytest.raises(RuntimeError, match="metadata size mismatch"):
        HELPERS._extract_agentic_variable_row_padding_flags(_batch_meta([False]), 2)
    with pytest.raises(RuntimeError, match="missing"):
        HELPERS._extract_agentic_variable_row_padding_flags(_custom_meta_batch([{}]), 1)
    with pytest.raises(RuntimeError, match="must be bool"):
        HELPERS._extract_agentic_variable_row_padding_flags(_batch_meta([1]), 1)
    with pytest.raises(RuntimeError, match="must be a mapping"):
        HELPERS._extract_agentic_variable_row_padding_flags(_custom_meta_batch([None]), 1)
    with pytest.raises(RuntimeError, match="get_all_custom_meta"):
        HELPERS._extract_agentic_variable_row_padding_flags(SimpleNamespace(samples=[]), 1)


def test_identity_parser_binds_custom_metadata_to_reordered_tensor_rows():
    identities = [
        _real_identity(group_id="0", turn_index=0, turn_count=1),
        _padding_identity(0),
    ]
    rollout_data, batch_meta = _identity_batch(identities)

    assert HELPERS._extract_agentic_variable_row_identities(batch_meta, rollout_data) == identities

    rollout_data[IDENTITY_TAGS_FIELD] = list(reversed(rollout_data[IDENTITY_TAGS_FIELD]))
    with pytest.raises(RuntimeError, match="identity mismatch"):
        HELPERS._extract_agentic_variable_row_identities(batch_meta, rollout_data)


def test_partition_identity_validation_is_order_independent_and_rejects_restart_residue():
    identities = [
        *[_real_identity(group_id="0", turn_index=idx, turn_count=2) for idx in range(2)],
        *[_real_identity(group_id="1", turn_index=idx, turn_count=3) for idx in range(3)],
        _padding_identity(0),
    ]
    identities = [identities[idx] for idx in (3, 0, 5, 4, 2, 1)]
    summary = HELPERS._validate_agentic_variable_row_partition_identities(
        identities,
        expected_group_count=2,
        expected_trajectories_per_group=1,
    )
    assert summary == {
        "policy_version": "7",
        "group_count": 2,
        "real_row_count": 5,
        "padding_row_count": 1,
    }

    duplicate_after_restart = [*identities, identities[0]]
    with pytest.raises(RuntimeError, match="duplicate row_id"):
        HELPERS._validate_agentic_variable_row_partition_identities(
            duplicate_after_restart,
            expected_group_count=2,
            expected_trajectories_per_group=1,
        )

    residual_group = [identity for identity in identities if identity.get("rollout_group_id") != "1"]
    with pytest.raises(RuntimeError, match="residual or missing"):
        HELPERS._validate_agentic_variable_row_partition_identities(
            residual_group,
            expected_group_count=2,
            expected_trajectories_per_group=1,
        )

    incomplete_turn = [identity for identity in identities if identity.get("row_id") != "row-1-1"]
    with pytest.raises(RuntimeError, match="missing or has duplicate physical rows"):
        HELPERS._validate_agentic_variable_row_partition_identities(
            incomplete_turn,
            expected_group_count=2,
            expected_trajectories_per_group=1,
        )


def test_window_progress_accepts_full_and_partial_then_rejects_overflow():
    consumed = HELPERS._validate_agentic_variable_row_window(
        actual_global_rows=8,
        global_batch_size=8,
        total_padded_rows_before=0,
        max_padded_rows=16,
    )
    assert consumed == 8
    consumed = HELPERS._validate_agentic_variable_row_window(
        actual_global_rows=5,
        global_batch_size=8,
        total_padded_rows_before=consumed,
        max_padded_rows=16,
    )
    assert consumed == 16

    with pytest.raises(RuntimeError, match="between 1 and global_batch_size"):
        HELPERS._validate_agentic_variable_row_window(
            actual_global_rows=0,
            global_batch_size=8,
            total_padded_rows_before=0,
            max_padded_rows=16,
        )
    with pytest.raises(RuntimeError, match="exceeded"):
        HELPERS._validate_agentic_variable_row_window(
            actual_global_rows=1,
            global_batch_size=8,
            total_padded_rows_before=16,
            max_padded_rows=16,
        )


def test_partition_padding_is_order_independent_but_still_bounded():
    assert (
        HELPERS._validate_agentic_variable_row_partition(
            total_physical_rows=16,
            total_actual_rows=13,
            global_batch_size=8,
        )
        == 3
    )

    with pytest.raises(RuntimeError, match="at least one global batch of padding"):
        HELPERS._validate_agentic_variable_row_partition(
            total_physical_rows=16,
            total_actual_rows=8,
            global_batch_size=8,
        )


def test_collector_accepts_real_rows_after_a_window_containing_padding(monkeypatch):
    monkeypatch.setenv(MAX_ROWS_ENV, "50")
    batches = [
        _identity_batch(
            [
                *[_real_identity(group_id="0", turn_index=idx, turn_count=5) for idx in range(5)],
                *[_padding_identity(idx) for idx in range(3)],
            ]
        ),
        _identity_batch([_real_identity(group_id="1", turn_index=idx, turn_count=8) for idx in range(8)]),
    ]

    collector = VARIABLE_ROW_COLLECTOR()
    collector.role = "actor"
    collector.args = Namespace(
        advantage_estimator="grpo",
        distributed_timeout_minutes=1,
        global_batch_size=8,
        multimodal_keys=None,
        n_samples_per_prompt=1,
        rollout_batch_size=2,
        use_critic=False,
        use_opd=False,
    )

    def _get_data(*_args, **_kwargs):
        if batches:
            return batches.pop(0)
        return None, None

    collector._get_data_from_transfer_queue = _get_data
    collector.all_consumed = lambda *_args, **_kwargs: not batches

    result = collector._collect_agentic_variable_row_batches(
        rollout_id=0,
        task_name="actor_train",
        data_fields=["total_lengths"],
    )

    rollout_batches, _metas, local_counts, global_counts, valid_flags = result
    assert len(rollout_batches) == 2
    assert local_counts == [8, 8]
    assert global_counts == [5, 8]
    assert sum(valid_flags) == 13
    assert valid_flags == [True, True, True, True, True, False, False, False, *([True] * 8)]


def test_drain_state_machine_aligns_dp_readiness_and_requires_a_window():
    action = HELPERS._agentic_variable_row_drain_action
    assert action(ready_min=1, ready_max=1, stream_drained=False, accepted_windows=0) == "accept"
    assert action(ready_min=0, ready_max=1, stream_drained=False, accepted_windows=0) == "wait"
    assert action(ready_min=0, ready_max=0, stream_drained=False, accepted_windows=0) == "retry"
    assert action(ready_min=0, ready_max=0, stream_drained=True, accepted_windows=1) == "done"
    with pytest.raises(RuntimeError, match="before yielding"):
        action(ready_min=0, ready_max=0, stream_drained=True, accepted_windows=0)
    with pytest.raises(RuntimeError, match="pending after"):
        action(ready_min=1, ready_max=1, stream_drained=True, accepted_windows=1)


def test_stream_drained_requires_data_parallel_consensus():
    class _Tensor:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class _Torch:
        int64 = object()

        @staticmethod
        def tensor(values, *, dtype, device):
            assert dtype is _Torch.int64
            assert device == "cuda:0"
            return _Tensor(values[0])

    class _Dist:
        class ReduceOp:
            MIN = object()

        observed_group = None

        @classmethod
        def all_reduce(cls, tensor, *, op, group):
            assert op is cls.ReduceOp.MIN
            cls.observed_group = group
            # Simulate another DP rank that has not observed drained yet.
            tensor.value = 0

    consensus = HELPERS._agentic_variable_row_stream_drained_consensus
    consensus.__globals__["torch"] = _Torch
    consensus.__globals__["dist"] = _Dist

    assert (
        consensus(
            stream_drained=True,
            device="cuda:0",
            dp_group="dp-group",
        )
        is False
    )
    assert _Dist.observed_group == "dp-group"
