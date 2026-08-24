# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the algorithm registry."""

import inspect

import pytest

from relax.algorithms import get_algorithm, list_algorithm_names
from relax.algorithms.spec import ALGORITHM_SPECS, AlgorithmSpec


EXPECTED_NAMES = [
    "grpo",
    "gspo",
    "sapo",
    "cispo",
    "ppo",
    "reinforce_plus_plus",
    "reinforce_plus_plus_baseline",
]


def test_all_expected_algorithms_registered():
    for name in EXPECTED_NAMES:
        assert name in ALGORITHM_SPECS, f"{name} missing from ALGORITHM_SPECS"


def test_spec_name_matches_dict_key():
    for key, spec in ALGORITHM_SPECS.items():
        assert spec.name == key


def test_spec_is_frozen():
    spec = get_algorithm("grpo")
    with pytest.raises(Exception):
        spec.name = "mutated"


def test_get_algorithm_unknown_name_raises_with_available_names():
    with pytest.raises(KeyError) as exc:
        get_algorithm("does_not_exist")
    assert "grpo" in str(exc.value)


def test_list_algorithm_names_matches_registry_keys():
    assert list_algorithm_names() == list(ALGORITHM_SPECS.keys())


def test_grpo_family_shares_one_advantage_fn():
    """grpo/gspo/sapo/cispo are identical at the advantage layer."""
    ids = {get_algorithm(n).advantage_fn for n in ("grpo", "gspo", "sapo", "cispo")}
    assert ids == {"grpo_broadcast"}


def test_reward_normalizer_ids_match_current_behavior():
    for name in ("grpo", "gspo", "sapo", "cispo"):
        assert get_algorithm(name).reward_normalizer == "group_mean_std"
    assert get_algorithm("reinforce_plus_plus_baseline").reward_normalizer == "group_mean"
    for name in ("ppo", "reinforce_plus_plus"):
        assert get_algorithm(name).reward_normalizer == "none"


def test_is_group_normalized_matches_the_legacy_whitelist():
    """The pre-registry whitelist, exactly.

    group.
    """
    legacy = {"grpo", "gspo", "sapo", "cispo", "reinforce_plus_plus_baseline"}
    actual = {n for n in EXPECTED_NAMES if get_algorithm(n).is_group_normalized}
    assert actual == legacy


def test_gspo_is_the_only_sequence_level_kl():
    seq = {n for n in EXPECTED_NAMES if get_algorithm(n).kl_level == "sequence"}
    assert seq == {"gspo"}


def test_gspo_is_the_only_one_needing_full_log_probs():
    need = {n for n in EXPECTED_NAMES if get_algorithm(n).needs_full_log_probs}
    assert need == {"gspo"}


def test_ppo_is_the_only_algorithm_needing_a_critic():
    """`needs_critic` is what `relax/core/registry.py` reads to decide whether
    ALGOS binds the Critic component, so this set is load-bearing."""
    critic = {n for n in EXPECTED_NAMES if get_algorithm(n).needs_critic}
    assert critic == {"ppo"}


def test_reinforce_family_requires_normalize_advantages():
    for name in ("reinforce_plus_plus", "reinforce_plus_plus_baseline"):
        assert get_algorithm(name).requires_normalize_advantages is True
    assert get_algorithm("grpo").requires_normalize_advantages is False


def test_policy_loss_ids_match_current_behavior():
    assert get_algorithm("sapo").policy_loss_fn == "sapo"
    assert get_algorithm("cispo").policy_loss_fn == "cispo"
    for name in ("grpo", "gspo", "ppo", "reinforce_plus_plus", "reinforce_plus_plus_baseline"):
        assert get_algorithm(name).policy_loss_fn == "ppo_clip"


def test_defaults_are_permissive():
    spec = AlgorithmSpec(name="x", reward_normalizer="none", advantage_fn="a", policy_loss_fn="ppo_clip")
    assert spec.kl_level == "token"
    assert spec.needs_full_log_probs is False
    assert spec.needs_critic is False
    assert spec.requires_normalize_advantages is False


def test_spec_module_has_no_heavy_imports():
    """The registry must import on a CPU-only runner with just torch
    available."""
    import relax.algorithms.spec as spec_mod

    src = inspect.getsource(spec_mod)
    for banned in (
        "import megatron",
        "from megatron",
        "import ray",
        "from ray",
        "import transfer_queue",
        "import tensordict",
        "from relax.components",
        "from relax.backends",
    ):
        assert banned not in src, f"spec.py must not import {banned}"


def test_every_spec_identifier_resolves_to_a_registered_implementation():
    """A typo in the registry must not wait until the first batch to
    surface."""
    from relax.algorithms.advantages import ADVANTAGE_FNS
    from relax.algorithms.policy import POLICY_LOSS_FNS
    from relax.algorithms.rewards import REWARD_NORMALIZERS

    for name in list_algorithm_names():
        spec = get_algorithm(name)
        assert spec.reward_normalizer in REWARD_NORMALIZERS, name
        assert spec.advantage_fn in ADVANTAGE_FNS, name
        assert spec.policy_loss_fn in POLICY_LOSS_FNS, name


def test_every_spec_declares_a_kl_level_the_loss_knows_how_to_read():
    """``kl_level`` has no dispatch table to fail against.

    ``relax/backends/megatron/loss.py`` reads it as ``== "sequence"``, so a
    misspelled value does not raise -- it silently selects token-level KL and
    the run trains the wrong objective while reporting the right algorithm
    name. The three fields above cannot fail that way because a bad key raises
    on lookup; this one needs the check written out.
    """
    for name in list_algorithm_names():
        assert get_algorithm(name).kl_level in ("token", "sequence"), name


# ---------------- enum-like fields must fail loudly, not silently ----------------


@pytest.mark.parametrize(
    "field,bad",
    [
        ("kl_level", "Sequence"),  # right word, wrong case
        ("kl_level", "seq"),
        ("advantage_normalization", "token-global"),  # hyphen instead of underscore
        ("advantage_normalization", "none"),
    ],
)
def test_an_unsupported_enum_value_is_refused_when_the_spec_is_built(field, bad):
    """These two fields are compared for equality, so a typo picks a formula.

    `loss.py` asks `advantage_normalization == "token_global"` and
    `kl_level == "sequence"`; every other string takes the else branch. Unlike
    `advantage_fn`, which blows up with a KeyError the first time it is looked
    up, a misspelled value here starts training successfully and quietly uses
    the wrong KL level or the wrong advantage normalisation. The registry is a
    module-level literal, so validating in `__post_init__` moves that from a
    silent wrong-maths run to an import-time error.
    """
    from relax.algorithms.spec import AlgorithmSpec

    kwargs = dict(
        name="probe",
        reward_normalizer="none",
        advantage_fn="grpo_broadcast",
        policy_loss_fn="ppo_clip",
    )
    kwargs[field] = bad

    with pytest.raises(ValueError, match=field):
        AlgorithmSpec(**kwargs)


def test_the_supported_values_are_the_ones_the_shipped_specs_use():
    """Guards against the allow-list drifting away from the registry it guards."""
    from relax.algorithms.spec import ADVANTAGE_NORMALIZATIONS, ALGORITHM_SPECS, KL_LEVELS

    assert {s.kl_level for s in ALGORITHM_SPECS.values()} <= KL_LEVELS
    assert {s.advantage_normalization for s in ALGORITHM_SPECS.values()} <= ADVANTAGE_NORMALIZATIONS
