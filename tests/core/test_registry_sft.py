# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for SFT registry entries."""

from types import SimpleNamespace

import pytest


# `relax.core.registry` eagerly imports `relax.components.advantages`, which
# imports `megatron.core` at module level. Skip the whole module when megatron
# is unavailable (e.g. GitHub CI without GPU stack).
pytest.importorskip("megatron.core")

from relax.core.registry import ALGOS, ROLES_SFT_ONLY, process_role  # noqa: E402


def _cfg(**kwargs):
    defaults = dict(
        debug_rollout_only=False,
        debug_train_only=False,
        fully_async=False,
        hybrid=False,
        loss_type="policy_loss",
        advantage_estimator="grpo",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_algos_has_sft_entry():
    assert "sft" in ALGOS
    assert "actor" in ALGOS["sft"]


def test_process_role_returns_sft_only_when_loss_type_sft():
    roles = process_role(_cfg(loss_type="sft"))
    assert roles is ROLES_SFT_ONLY
    assert {r.value for r in roles} == {"actor", "sft"}


def test_process_role_keeps_rl_path_unchanged():
    from relax.core.registry import ROLES_COLOCATE

    roles = process_role(_cfg(loss_type="policy_loss"))
    assert roles is ROLES_COLOCATE


def test_process_role_debug_flags_take_precedence_over_sft():
    """debug_rollout_only / debug_train_only 仍优先于 sft（保留 RL 调试惯例）。"""
    from relax.core.registry import ROLES_ROLLOUT_ONLY, ROLES_TRAIN_ONLY

    assert process_role(_cfg(debug_rollout_only=True, loss_type="sft")) is ROLES_ROLLOUT_ONLY
    assert process_role(_cfg(debug_train_only=True, loss_type="sft")) is ROLES_TRAIN_ONLY


def test_roles_main_enum_includes_sft():
    from relax.core.registry import ROLES

    assert ROLES.sft.value == "sft"


def test_roles_sft_only_includes_actor_and_sft():
    assert {r.value for r in ROLES_SFT_ONLY} == {"actor", "sft"}


def test_algos_sft_has_sft_class_wired():
    from relax.components.sft import SFT
    from relax.core.registry import ALGOS, ROLES

    assert ALGOS["sft"][ROLES.sft] is SFT


def test_algos_sft_actor_class_unchanged():
    from relax.components.actor import Actor
    from relax.core.registry import ALGOS, ROLES

    assert ALGOS["sft"][ROLES.actor] is Actor
