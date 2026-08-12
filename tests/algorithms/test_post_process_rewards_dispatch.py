# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""post_process_rewards must dispatch through the registry, not an if/elif
chain."""

import inspect
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

import relax.utils.utils as utils_mod  # noqa: E402


def _args(estimator="grpo", **overrides):
    base = dict(
        advantage_estimator=estimator,
        n_samples_per_prompt=4,
        rewards_normalization=True,
        grpo_std_normalization=True,
        custom_reward_post_process_path=None,
        reward_key=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _Sample:
    def __init__(self, group_index, reward):
        self.group_index = group_index
        self.reward = reward

    def get_reward_value(self, args):
        return self.reward if not args.reward_key else self.reward[args.reward_key]


def test_source_has_no_algorithm_name_literals():
    """The estimator whitelists must be gone from post_process_rewards."""
    src = inspect.getsource(utils_mod.post_process_rewards)
    for banned in ('"grpo"', '"gspo"', '"sapo"', '"cispo"', '"reinforce_plus_plus_baseline"'):
        assert banned not in src, f"post_process_rewards still hardcodes {banned}"


def test_debug_subsample_source_has_no_algorithm_name_literals():
    src = inspect.getsource(utils_mod.get_debug_data)
    for banned in ('"grpo"', '"gspo"', '"sapo"', '"cispo"', '"reinforce_plus_plus_baseline"'):
        assert banned not in src, f"get_debug_data still hardcodes {banned}"


def test_returns_raw_and_normalized():
    args = _args("grpo")
    samples = [_Sample(0, r) for r in (0.0, 1.0, 2.0, 3.0)]
    raw, normalized = utils_mod.post_process_rewards(args, samples)
    assert raw == [0.0, 1.0, 2.0, 3.0]
    assert normalized != raw
    assert abs(sum(normalized)) < 1e-5


def test_identity_path_returns_raw_twice():
    args = _args("reinforce_plus_plus")
    samples = [_Sample(0, r) for r in (0.0, 1.0, 2.0, 3.0)]
    raw, normalized = utils_mod.post_process_rewards(args, samples)
    assert normalized is raw


def test_rewards_normalization_off_returns_raw_twice():
    args = _args("grpo", rewards_normalization=False)
    samples = [_Sample(0, r) for r in (0.0, 1.0, 2.0, 3.0)]
    raw, normalized = utils_mod.post_process_rewards(args, samples)
    assert normalized is raw


def test_baseline_estimator_centres_without_dividing_by_std():
    args = _args("reinforce_plus_plus_baseline")
    samples = [_Sample(0, r) for r in (0.0, 1.0, 2.0, 3.0)]
    _, normalized = utils_mod.post_process_rewards(args, samples)
    assert normalized == pytest.approx([-1.5, -0.5, 0.5, 1.5])


def test_custom_path_still_short_circuits(monkeypatch):
    sentinel = (["raw"], ["norm"])
    monkeypatch.setattr(utils_mod, "load_function", lambda path: lambda a, s: sentinel)
    args = _args("grpo", custom_reward_post_process_path="pkg.mod.fn")
    assert utils_mod.post_process_rewards(args, []) is sentinel


def test_reward_key_selects_from_dict():
    args = _args("grpo", reward_key="score")
    samples = [_Sample(0, {"score": r, "other": 99.0}) for r in (0.0, 1.0, 2.0, 3.0)]
    raw, _ = utils_mod.post_process_rewards(args, samples)
    assert raw == [0.0, 1.0, 2.0, 3.0]


def test_unknown_estimator_raises_from_the_registry():
    args = _args("not_an_algorithm")
    samples = [_Sample(0, 1.0) for _ in range(4)]
    with pytest.raises(KeyError, match="Unknown advantage estimator"):
        utils_mod.post_process_rewards(args, samples)


# ---------------- raw_reward column stays scalar ----------------


def _real_sample(group_index, reward, metadata=None):
    """Use the production Sample so no field is accidentally missing."""
    from relax.utils.types import Sample

    return Sample(
        group_index=group_index,
        index=group_index,
        tokens=[1, 2, 3],
        response_length=2,
        reward=reward,
        metadata=metadata or {},
    )


def _train_data_args(**overrides):
    base = dict(
        advantage_estimator="grpo",
        n_samples_per_prompt=4,
        rewards_normalization=True,
        grpo_std_normalization=True,
        custom_reward_post_process_path=None,
        reward_key="score",
        debug_train_only=True,  # stop before dict_to_tensordict, keep plain lists
        use_opd=False,
        multimodal_keys=None,
        use_rollout_routing_replay=False,
        mask_offpolicy_in_partial_rollout=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_raw_reward_column_stays_scalar_when_metadata_overrides_some_samples():
    """Dict rewards plus a partial metadata override used to mix types.

    ``sample.reward`` is a dict for every run that uses --reward-key (and every
    run using --reward-key), so falling back to it produced ``[0.8, {...}]``,
    which blows up the TensorDict conversion.
    """
    samples = [
        _real_sample(0, {"score": 1.0, "format": 0.5}, {"raw_reward": 0.8}),
        _real_sample(0, {"score": 0.0, "format": 1.0}),
        _real_sample(0, {"score": 1.0, "format": 0.0}),
        _real_sample(0, {"score": 0.0, "format": 0.5}),
    ]

    train_data = utils_mod.convert_samples_to_train_data(_train_data_args(), samples)

    assert train_data["raw_reward"] == [0.8, 0.0, 1.0, 0.0]
    for value in train_data["raw_reward"]:
        assert isinstance(value, float), f"{value!r} is not a scalar"


def test_raw_reward_column_untouched_without_metadata_overrides():
    samples = [_real_sample(0, {"score": float(i)}) for i in range(4)]
    train_data = utils_mod.convert_samples_to_train_data(_train_data_args(), samples)
    assert train_data["raw_reward"] == [0.0, 1.0, 2.0, 3.0]


def test_metadata_override_still_wins_for_scalar_rewards():
    """The original purpose of the override must keep working."""
    samples = [_real_sample(0, float(i), {"raw_reward": 9.0} if i == 0 else None) for i in range(4)]
    train_data = utils_mod.convert_samples_to_train_data(_train_data_args(reward_key=None), samples)
    assert train_data["raw_reward"] == [9.0, 1.0, 2.0, 3.0]
