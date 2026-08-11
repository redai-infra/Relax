# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from relax.engine.rollout import bootstrap


def test_resolve_rl_num_rollout_populates_actor_config_before_service_creation(monkeypatch) -> None:
    config = Namespace(
        loss_type="grpo",
        rollout_global_dataset=True,
        rollout_batch_size=16,
        num_epoch=3,
        num_rollout=20,
    )
    lengths_remote = Mock(return_value="length-ref")
    data_source = SimpleNamespace(lengths=SimpleNamespace(remote=lengths_remote))
    monkeypatch.setattr(bootstrap.ray, "get", lambda ref: 64)

    bootstrap.resolve_rl_num_rollout(config, data_source)

    lengths_remote.assert_called_once_with()
    assert config.num_rollout_per_epoch == 4
    assert config.num_rollout == 12


def test_resolve_rl_num_rollout_handles_non_global_dataset_without_epoch_query(monkeypatch) -> None:
    config = Namespace(
        loss_type="grpo",
        rollout_global_dataset=False,
        num_epoch=None,
        num_rollout=20,
    )
    data_source = SimpleNamespace(lengths=SimpleNamespace(remote=Mock()))
    ray_get = Mock()
    monkeypatch.setattr(bootstrap.ray, "get", ray_get)

    bootstrap.resolve_rl_num_rollout(config, data_source)

    ray_get.assert_not_called()
    assert config.num_rollout_per_epoch is None
    assert config.num_rollout == 20


def test_resolve_rl_num_rollout_requires_num_rollout_for_non_global_dataset() -> None:
    config = Namespace(
        loss_type="grpo",
        rollout_global_dataset=False,
        num_epoch=None,
        num_rollout=None,
    )

    with pytest.raises(ValueError, match="num_rollout must be positive"):
        bootstrap.resolve_rl_num_rollout(config, SimpleNamespace())


def test_explicit_num_rollout_can_cross_small_global_dataset_without_epoch_boundary(monkeypatch) -> None:
    config = Namespace(
        loss_type="grpo",
        rollout_global_dataset=True,
        rollout_batch_size=16,
        num_epoch=None,
        num_rollout=20,
    )
    data_source = SimpleNamespace(lengths=SimpleNamespace(remote=Mock(return_value="length-ref")))
    monkeypatch.setattr(bootstrap.ray, "get", lambda ref: 8)

    bootstrap.resolve_rl_num_rollout(config, data_source)

    assert config.num_rollout_per_epoch is None
    assert config.num_rollout == 20


def test_num_epoch_rejects_small_global_dataset(monkeypatch) -> None:
    config = Namespace(
        loss_type="grpo",
        rollout_global_dataset=True,
        rollout_batch_size=16,
        num_epoch=2,
        num_rollout=None,
    )
    data_source = SimpleNamespace(lengths=SimpleNamespace(remote=Mock(return_value="length-ref")))
    monkeypatch.setattr(bootstrap.ray, "get", lambda ref: 8)

    with pytest.raises(ValueError, match="Cannot derive num_rollout from num_epoch"):
        bootstrap.resolve_rl_num_rollout(config, data_source)
