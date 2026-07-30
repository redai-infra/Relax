# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for synchronous RLOO registry entries."""

from types import SimpleNamespace

import pytest


pytest.importorskip("megatron.core")

from relax.core.registry import ALGOS, ROLES_COLOCATE, process_role  # noqa: E402


def test_rloo_registry_matches_grpo_role_topology():
    assert "rloo" in ALGOS
    assert ALGOS["rloo"] == ALGOS["grpo"]


def test_rloo_sync_process_role_uses_colocate_path():
    config = SimpleNamespace(
        debug_rollout_only=False,
        debug_train_only=False,
        fully_async=False,
        hybrid=False,
        loss_type="policy_loss",
        advantage_estimator="rloo",
    )

    assert process_role(config) is ROLES_COLOCATE


def test_default_grpo_registry_entry_remains_available():
    assert "grpo" in ALGOS
