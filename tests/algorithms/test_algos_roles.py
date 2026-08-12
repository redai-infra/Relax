# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""ALGOS must cover every registered algorithm, with the right roles.

`ALGOS` was a hand-written dict, one literal block per algorithm. That is the
kind of table an estimator can be accepted by argparse yet be missing from --
`reinforce_plus_plus` and `reinforce_plus_plus_baseline` were exactly that for a
while, crashing `controller.register_all_serve` with `ValueError: Algorithm key
'...' not registered in ALGOS` until they were added by hand. Deriving the table
from the registry removes the class of bug rather than one instance of it.

The roles are not uniform: PPO needs a critic and the policy-gradient
estimators do not, so the derivation reads `AlgorithmSpec.needs_critic` instead
of handing every algorithm the same set. These tests pin both halves.

`relax.core.registry` imports the component classes, which import megatron, so
the behavioural checks are gated on that. The source-level checks are not: they
are the ones that run on the CPU-only CI runner, and they are what catches a
regression back to hand-written entries.
"""

import pathlib

import pytest


REGISTRY_PATH = pathlib.Path(__file__).resolve().parents[2] / "relax" / "core" / "registry.py"

try:
    from relax.core.registry import ALGOS, ROLES

    HAS_MEGATRON = True
except ImportError:  # pragma: no cover - depends on the runner
    ALGOS = ROLES = None
    HAS_MEGATRON = False

requires_megatron = pytest.mark.skipif(not HAS_MEGATRON, reason="relax.core.registry requires megatron")


# ---------------- runs everywhere, including CPU-only CI ----------------


def test_algos_is_derived_from_the_registry_not_hand_written():
    src = REGISTRY_PATH.read_text(encoding="utf-8")

    assert "ALGORITHM_SPECS.items()" in src, "ALGOS no longer derives its entries from the registry"
    for name in ("grpo", "gspo", "sapo", "cispo", "ppo"):
        assert f'"{name}": {{' not in src, f"ALGOS hand-writes a role dict for {name} again"


def test_role_topology_is_driven_by_needs_critic():
    """Which component classes an algorithm binds must come from the spec.

    Scoped to the ALGOS derivation on purpose. `process_role` still branches on
    the literal "ppo" to pick the role *iteration order*; that is the
    controller's orchestration surface, it is untouched by this change, and
    folding it in would be a separate proposal.
    """
    src = REGISTRY_PATH.read_text(encoding="utf-8")

    assert "needs_critic=spec.needs_critic" in src, "ALGOS no longer derives the critic role from AlgorithmSpec"


def test_sft_stays_a_separate_literal_entry():
    """SFT is selected by loss_type, so it must not come from the estimator
    registry."""
    src = REGISTRY_PATH.read_text(encoding="utf-8")
    assert 'ALGOS["sft"]' in src


def test_sft_is_not_an_estimator():
    from relax.algorithms import list_algorithm_names

    assert "sft" not in list_algorithm_names()


# ---------------- needs the real component classes ----------------


@requires_megatron
def test_every_registered_algorithm_has_a_role_mapping():
    """The regression that used to crash the controller at startup."""
    from relax.algorithms import list_algorithm_names

    missing = [name for name in list_algorithm_names() if name not in ALGOS]
    assert not missing, f"{missing} would raise ValueError in controller.register_all_serve"


@requires_megatron
@pytest.mark.parametrize("name", ["reinforce_plus_plus", "reinforce_plus_plus_baseline", "gdpo"])
def test_previously_missing_algorithms_are_covered(name):
    """These three are why the derivation exists."""
    assert name in ALGOS


@requires_megatron
def test_role_set_differs_only_by_the_critic():
    from relax.algorithms import ALGORITHM_SPECS

    base = {ROLES.rollout, ROLES.actor, ROLES.advantages, ROLES.reference, ROLES.actor_fwd}
    for name, spec in ALGORITHM_SPECS.items():
        expected = base | {ROLES.critic} if spec.needs_critic else base
        assert set(ALGOS[name]) == expected, f"{name} has unexpected roles"


@requires_megatron
def test_ppo_keeps_its_critic():
    """PPO is the one value-based estimator; dropping Critic here would leave
    `controller.register_all_serve` skipping the role and the run training
    without a value function."""
    from relax.components.critic import Critic

    assert ALGOS["ppo"][ROLES.critic] is Critic


@requires_megatron
def test_policy_gradient_algorithms_have_no_critic():
    for name in ("grpo", "gspo", "sapo", "cispo", "gdpo", "reinforce_plus_plus"):
        assert ROLES.critic not in ALGOS[name], f"{name} would start a critic service it never uses"


@requires_megatron
def test_sft_roles_are_unchanged():
    assert set(ALGOS["sft"]) == {ROLES.sft, ROLES.actor}


@requires_megatron
def test_each_algorithm_gets_an_independent_role_dict():
    """controller.py copies and then mutates these; sharing one dict object
    would leak an optional role from one algorithm into all the others."""
    from relax.algorithms import list_algorithm_names

    names = list_algorithm_names()
    for left, right in zip(names, names[1:], strict=False):
        assert ALGOS[left] is not ALGOS[right], f"{left} and {right} share one dict object"
