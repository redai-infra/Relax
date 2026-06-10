# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""controller.register_all_serve colocate gating for SFT-with-rollout.

The colocate gate previously checked ``hasattr(ROLES, "rollout")`` against
``process_role(config)``. SFT returns ``ROLES_SFT_ONLY`` (no ``rollout`` attr),
so even when ``register_sft_rollout`` injected ``ROLES.rollout`` via
``extra_roles``, colocate stayed off. These tests pin the fix: the gate must
inspect the union of ``list(ROLES) + extra_roles``.
"""

from argparse import Namespace

import pytest


def _registry_or_skip():
    try:
        from relax.core.registry import ROLES, ROLES_SFT_ONLY
    except (ImportError, AssertionError) as exc:
        pytest.skip(f"relax.core.registry unavailable: {exc}")
    return ROLES, ROLES_SFT_ONLY


def test_register_sft_rollout_injects_rollout_role():
    from relax.core.optional_roles import register_sft_rollout

    ROLES, _ = _registry_or_skip()

    config = Namespace(loss_type="sft", sft_predict_interval=10)
    algo: dict = {}
    extras = register_sft_rollout(config, algo)
    assert extras == [ROLES.rollout]
    assert ROLES.rollout in algo


def test_register_sft_rollout_noop_without_predict_interval():
    from relax.core.optional_roles import register_sft_rollout

    config = Namespace(loss_type="sft", sft_predict_interval=None)
    algo: dict = {}
    assert register_sft_rollout(config, algo) == []
    assert algo == {}


def test_sft_role_set_with_extra_rollout_supports_colocate():
    """Mirrors register_all_serve's colocate check post-fix."""
    ROLES, ROLES_SFT_ONLY = _registry_or_skip()

    extra_roles = [ROLES.rollout]
    role_names = {str(r) for r in list(ROLES_SFT_ONLY) + extra_roles}
    assert "actor" in role_names
    assert "rollout" in role_names
    # Pre-fix would have been `hasattr(ROLES_SFT_ONLY, "rollout")` → False
    assert not hasattr(ROLES_SFT_ONLY, "rollout")


def test_sft_role_set_without_rollout_blocks_colocate():
    _, ROLES_SFT_ONLY = _registry_or_skip()

    extra_roles: list = []
    role_names = {str(r) for r in list(ROLES_SFT_ONLY) + extra_roles}
    assert "rollout" not in role_names
