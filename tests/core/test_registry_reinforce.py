# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Registry tests for the REINFORCE++ / REINFORCE++-baseline algorithm entries
(task #29).

The two variants must be registered in ``ALGOS`` (otherwise the Controller
rejects the key in ``register_all_serve``) and must reuse the GRPO service
topology (no critic) — they differ only in advantage/loss computation.
"""

from types import SimpleNamespace

import pytest


# `relax.core.registry` eagerly imports `relax.components.advantages`, which
# imports `megatron.core` at module level. Skip when megatron is unavailable.
pytest.importorskip("megatron.core")

from relax.components.actor import Actor  # noqa: E402
from relax.components.actor_fwd import ActorFwd  # noqa: E402
from relax.components.advantages import Advantages  # noqa: E402
from relax.components.rollout import Rollout  # noqa: E402
from relax.core.registry import ALGOS, ROLES, ROLES_COLOCATE, process_role  # noqa: E402


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


class TestReinforceRegistry:
    """The two REINFORCE++ variants are registered and reuse the GRPO topology."""

    @pytest.mark.parametrize("key", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
    def test_variant_is_registered(self, key):
        assert key in ALGOS

    @pytest.mark.parametrize("key", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
    def test_variant_shares_grpo_topology(self, key):
        expected = {
            ROLES.rollout: Rollout,
            ROLES.actor: Actor,
            ROLES.advantages: Advantages,
            ROLES.reference: ActorFwd,
            ROLES.actor_fwd: ActorFwd,
        }
        assert ALGOS[key] == expected
        # REINFORCE++ is actor-only (no value head), unlike PPO.
        assert ROLES.critic not in ALGOS[key]

    @pytest.mark.parametrize("key", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
    def test_process_role_returns_colocate_for_sync(self, key):
        assert process_role(_cfg(advantage_estimator=key)) is ROLES_COLOCATE
