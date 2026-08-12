# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


pytest.importorskip("ray", reason="Ray is an optional test dependency")
pytest.importorskip("megatron", reason="Megatron is an optional test dependency")

from relax.distributed.ray import placement_group
from relax.distributed.ray import rollout as rollout_module


def test_create_rollout_manager_non_global_dataset_skips_epoch_query(monkeypatch) -> None:
    get_num_rollout_per_epoch = Mock(side_effect=AssertionError("must not query epoch length"))
    manager = SimpleNamespace(
        get_num_rollout_per_epoch=SimpleNamespace(remote=get_num_rollout_per_epoch),
    )
    remote_constructor = Mock(return_value=manager)
    rollout_manager_cls = SimpleNamespace(
        options=Mock(return_value=SimpleNamespace(remote=remote_constructor)),
    )
    monkeypatch.setattr(rollout_module, "RolloutManager", rollout_manager_cls)
    monkeypatch.setattr(placement_group, "_get_head_node_id", lambda: "01" * 28)

    args = Namespace(
        loss_type="grpo",
        rollout_global_dataset=False,
        num_rollout_per_epoch=None,
        num_epoch=None,
        num_rollout=20,
        check_weight_update_equal=False,
        offload_rollout=False,
    )

    returned_manager, num_rollout_per_epoch = placement_group.create_rollout_manager(args, pg=None)

    assert returned_manager is manager
    assert num_rollout_per_epoch is None
    get_num_rollout_per_epoch.assert_not_called()


def test_create_rollout_manager_preserves_pre_resolved_missing_epoch_boundary(monkeypatch) -> None:
    get_num_rollout_per_epoch = Mock(side_effect=AssertionError("must not recompute pre-resolved sizing"))
    manager = SimpleNamespace(
        get_num_rollout_per_epoch=SimpleNamespace(remote=get_num_rollout_per_epoch),
    )
    rollout_manager_cls = SimpleNamespace(
        options=Mock(return_value=SimpleNamespace(remote=Mock(return_value=manager))),
    )
    monkeypatch.setattr(rollout_module, "RolloutManager", rollout_manager_cls)
    monkeypatch.setattr(placement_group, "_get_head_node_id", lambda: "01" * 28)

    args = Namespace(
        loss_type="grpo",
        rollout_global_dataset=True,
        num_rollout_per_epoch=None,
        num_epoch=None,
        num_rollout=20,
        check_weight_update_equal=False,
        offload_rollout=False,
    )

    _, num_rollout_per_epoch = placement_group.create_rollout_manager(args, pg=None)

    assert num_rollout_per_epoch is None
    assert args.num_rollout_per_epoch is None
    get_num_rollout_per_epoch.assert_not_called()


def test_create_rollout_manager_normalizes_fallback_zero_epoch_size(monkeypatch) -> None:
    get_num_rollout_per_epoch = Mock(return_value="epoch-size-ref")
    manager = SimpleNamespace(
        get_num_rollout_per_epoch=SimpleNamespace(remote=get_num_rollout_per_epoch),
    )
    rollout_manager_cls = SimpleNamespace(
        options=Mock(return_value=SimpleNamespace(remote=Mock(return_value=manager))),
    )
    monkeypatch.setattr(rollout_module, "RolloutManager", rollout_manager_cls)
    monkeypatch.setattr(placement_group, "_get_head_node_id", lambda: "01" * 28)
    monkeypatch.setattr(placement_group.ray, "get", lambda ref: 0)

    args = Namespace(
        loss_type="grpo",
        rollout_global_dataset=True,
        num_epoch=None,
        num_rollout=20,
        check_weight_update_equal=False,
        offload_rollout=False,
    )

    _, num_rollout_per_epoch = placement_group.create_rollout_manager(args, pg=None)

    assert num_rollout_per_epoch is None
    assert args.num_rollout_per_epoch is None
    get_num_rollout_per_epoch.assert_called_once_with()


def test_create_rollout_manager_rejects_non_divisible_fallback_epoch_dataset(monkeypatch) -> None:
    get_num_rollout_per_epoch = Mock(side_effect=AssertionError("must reject before resolving epoch steps"))
    manager = SimpleNamespace(
        get_num_rollout_per_epoch=SimpleNamespace(remote=get_num_rollout_per_epoch),
    )
    rollout_manager_cls = SimpleNamespace(
        options=Mock(return_value=SimpleNamespace(remote=Mock(return_value=manager))),
    )
    data_source = SimpleNamespace(lengths=SimpleNamespace(remote=Mock(return_value="length-ref")))
    monkeypatch.setattr(rollout_module, "RolloutManager", rollout_manager_cls)
    monkeypatch.setattr(placement_group, "_get_head_node_id", lambda: "01" * 28)
    monkeypatch.setattr(placement_group.ray, "get", lambda ref: 10)

    args = Namespace(
        loss_type="grpo",
        rollout_global_dataset=True,
        rollout_batch_size=6,
        num_epoch=2,
        num_rollout=None,
        check_weight_update_equal=False,
        offload_rollout=False,
    )

    with pytest.raises(ValueError, match="must be divisible"):
        placement_group.create_rollout_manager(args, pg=None, data_source=data_source)

    get_num_rollout_per_epoch.assert_not_called()
