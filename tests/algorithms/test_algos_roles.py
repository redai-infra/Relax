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
@pytest.mark.parametrize("name", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_previously_missing_algorithms_are_covered(name):
    """These two are why the derivation exists."""
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
    for name in ("grpo", "gspo", "sapo", "cispo", "reinforce_plus_plus"):
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


# ---------------- a second critic algorithm must need no new name checks ----------------


def _register_second_critic_algorithm(monkeypatch):
    """Add a value-based estimator to the registry for the duration of a test.

    Returns its name. Uses PPO's own spec so the only thing under test is
    whether the pipeline keys off `needs_critic` or off the literal `"ppo"`.
    """
    import dataclasses

    from relax.algorithms import ALGORITHM_SPECS
    from relax.algorithms import spec as spec_module

    name = "ppo_second"
    clone = dataclasses.replace(ALGORITHM_SPECS["ppo"], name=name)
    monkeypatch.setitem(spec_module.ALGORITHM_SPECS, name, clone)
    assert ALGORITHM_SPECS[name].needs_critic is True
    return name


def _args_for(name, **overrides):
    from argparse import Namespace

    base = dict(
        advantage_estimator=name,
        multimodal_keys=None,
        kl_coef=0.0,
        fully_async=False,
        hybrid=False,
        use_opd=False,
        use_rollout_logprobs=False,
        true_on_policy_mode=False,
        debug_rollout_only=False,
        debug_train_only=False,
        loss_type=None,
    )
    base.update(overrides)
    return Namespace(**base)


def test_a_second_critic_algorithm_gets_the_critic_rollout_fields(monkeypatch):
    """`values` must reach the advantages consumer, unprompted by any name.

    This is the failure the registry was supposed to make impossible: argparse
    and `ALGOS` accept a second value-based estimator, and then the value
    plumbing silently does not switch on because it compares against `"ppo"`.
    """
    from relax.utils.training.data_fields import build_data_fields

    name = _register_second_critic_algorithm(monkeypatch)

    ppo_fields = build_data_fields(_args_for("ppo"), consumer="advantages")
    new_fields = build_data_fields(_args_for(name), consumer="advantages")
    assert new_fields == ppo_fields, "a second critic algorithm sees different fields than PPO"
    assert "values" in new_fields

    # and the critic consumer's own set, which is the base set rather than the actor's
    assert build_data_fields(_args_for(name), consumer="critic") == build_data_fields(
        _args_for("ppo"), consumer="critic"
    )


def test_a_second_critic_algorithm_is_told_it_needs_a_critic_resource(monkeypatch):
    """Startup validation is keyed on the capability, not on the name."""
    import pytest as _pytest

    from relax.utils.training.ppo_utils import validate_ppo_config

    name = _register_second_critic_algorithm(monkeypatch)

    with _pytest.raises(ValueError, match="requires a 'critic' entry"):
        validate_ppo_config(_args_for(name, resource={"actor": "a"}))

    # a non-critic estimator is still waved through
    validate_ppo_config(_args_for("grpo", resource={"actor": "a"}))


@requires_megatron
def test_a_second_critic_algorithm_walks_the_critic_role_topology(monkeypatch):
    """`process_role` decides which roles the controller walks at all."""
    from relax.core.registry import process_role

    name = _register_second_critic_algorithm(monkeypatch)

    # identity, not membership: every role set carries a `critic` member and
    # `ALGOS` is what filters it out, so comparing member names would pass even
    # if the second algorithm fell through to the non-critic topology.
    assert process_role(_args_for(name)) is process_role(_args_for("ppo"))
    assert process_role(_args_for(name)) is not process_role(_args_for("grpo"))

    # and the fully-async split follows too, rather than only the colocate one
    assert process_role(_args_for(name, fully_async=True)) is process_role(_args_for("ppo", fully_async=True))
