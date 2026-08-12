# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Argument parsing and validation must read the registry, not string lists."""

import argparse
import importlib
import pathlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


ARGS_PATH = pathlib.Path(__file__).resolve().parents[2] / "relax" / "utils" / "arguments.py"


@pytest.fixture()
def arguments_module(monkeypatch):
    """Import relax.utils.arguments with its heavy optional deps stubbed out.

    Mirrors tests/utils/test_arguments_opd_teacher_colocate.py.
    """
    router_pkg = ModuleType("sglang_router")
    launch_router = ModuleType("sglang_router.launch_router")
    launch_router.RouterArgs = object
    monkeypatch.setitem(sys.modules, "sglang_router", router_pkg)
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", launch_router)

    sglang_arguments = ModuleType("relax.backends.sglang.arguments")
    sglang_arguments.sglang_parse_args = lambda: None
    sglang_arguments.validate_args = lambda args: args
    monkeypatch.setitem(sys.modules, "relax.backends.sglang.arguments", sglang_arguments)

    device = ModuleType("relax.utils.device")
    device.get_dist_backend = lambda: "gloo"
    monkeypatch.setitem(sys.modules, "relax.utils.device", device)

    eval_config = ModuleType("relax.utils.training.eval_config")
    eval_config.EvalDatasetConfig = dict
    eval_config.build_eval_dataset_configs = lambda args, datasets_config, defaults: []
    eval_config.build_named_prompt_data_configs = lambda values: []
    eval_config.ensure_dataset_list = lambda values: values or []
    monkeypatch.setitem(sys.modules, "relax.utils.training.eval_config", eval_config)

    sys.modules.pop("relax.utils.arguments", None)
    module = importlib.import_module("relax.utils.arguments")
    yield module
    sys.modules.pop("relax.utils.arguments", None)


def _args(estimator="gdpo", **overrides):
    base = dict(
        advantage_estimator=estimator,
        normalize_advantages=False,
        rewards_normalization=True,
        custom_reward_post_process_path=None,
        n_samples_per_prompt=4,
        gdpo_reward_keys=["correctness", "format"],
        gdpo_reward_weights=None,
        reward_key="score",
        use_critic=False,
        fully_async=False,
        hybrid=False,
        dynamic_sampling_filter_path=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------- source-level: no hardcoded name lists ----------------


def test_choices_come_from_the_registry():
    assert "choices=list_algorithm_names()" in ARGS_PATH.read_text(encoding="utf-8")


def test_no_hardcoded_estimator_choice_list():
    src = ARGS_PATH.read_text(encoding="utf-8")
    assert '"reinforce_plus_plus_baseline",\n                    "ppo",' not in src


def test_no_estimator_name_comparisons_remain():
    src = ARGS_PATH.read_text(encoding="utf-8")
    for banned in (
        'args.advantage_estimator == "ppo"',
        'args.advantage_estimator in ["reinforce_plus_plus"',
    ):
        assert banned not in src, f"arguments.py still contains: {banned}"


def test_validation_reads_spec_fields():
    src = ARGS_PATH.read_text(encoding="utf-8")
    for field in (
        "needs_critic",
        "requires_normalize_advantages",
        "forbids_normalize_advantages",
        "requires_rewards_normalization",
        "allows_reward_post_process_hooks",
        "min_group_size",
        "uses_reward_components",
        "supports_fully_async",
    ):
        assert field in src, f"arguments.py does not consult spec.{field}"


def test_gdpo_arguments_declared():
    src = ARGS_PATH.read_text(encoding="utf-8")
    assert "--gdpo-reward-keys" in src
    assert "--gdpo-reward-weights" in src


# ---------------- behaviour ----------------


def test_parser_accepts_gdpo_and_its_options(arguments_module):
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)

    args = parser.parse_args(
        [
            "--advantage-estimator",
            "gdpo",
            "--gdpo-reward-keys",
            "correctness",
            "format",
            "--gdpo-reward-weights",
            "1.0",
            "0.5",
        ]
    )

    assert args.advantage_estimator == "gdpo"
    assert args.gdpo_reward_keys == ["correctness", "format"]
    assert args.gdpo_reward_weights == [1.0, 0.5]


def test_parser_rejects_an_unregistered_estimator(arguments_module):
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--advantage-estimator", "not_an_algorithm"])


def test_every_registered_algorithm_is_an_accepted_choice(arguments_module):
    from relax.algorithms import list_algorithm_names

    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)

    for name in list_algorithm_names():
        parsed = parser.parse_args(["--advantage-estimator", name])
        assert parsed.advantage_estimator == name


def test_gdpo_rejects_custom_reward_post_process(arguments_module):
    with pytest.raises(ValueError, match="custom-reward-post-process-path"):
        arguments_module.validate_algorithm_args(_args(custom_reward_post_process_path="pkg.mod.fn"))


def test_gdpo_rejects_normalize_advantages(arguments_module):
    with pytest.raises(ValueError, match="normalize-advantages"):
        arguments_module.validate_algorithm_args(_args(normalize_advantages=True))


def test_gdpo_rejects_group_size_below_two(arguments_module):
    with pytest.raises(ValueError, match="n-samples-per-prompt"):
        arguments_module.validate_algorithm_args(_args(n_samples_per_prompt=1))


def test_gdpo_rejects_disabled_rewards_normalization(arguments_module):
    with pytest.raises(ValueError, match="reward normalization"):
        arguments_module.validate_algorithm_args(_args(rewards_normalization=False))


def test_gdpo_requires_at_least_two_keys(arguments_module):
    with pytest.raises(ValueError, match="at least two reward keys"):
        arguments_module.validate_algorithm_args(_args(gdpo_reward_keys=["correctness"]))


def test_gdpo_rejects_duplicate_keys(arguments_module):
    with pytest.raises(ValueError, match="duplicates"):
        arguments_module.validate_algorithm_args(_args(gdpo_reward_keys=["a", "a"]))


def test_gdpo_rejects_weight_count_mismatch(arguments_module):
    with pytest.raises(ValueError, match="gdpo-reward-weights"):
        arguments_module.validate_algorithm_args(_args(gdpo_reward_weights=[1.0]))


def test_gdpo_requires_reward_key(arguments_module):
    with pytest.raises(ValueError, match="reward-key"):
        arguments_module.validate_algorithm_args(_args(reward_key=None))


def test_gdpo_happy_path(arguments_module):
    args = _args()
    arguments_module.validate_algorithm_args(args)
    assert args.use_critic is False


def test_reinforce_family_requires_normalize_advantages(arguments_module):
    for estimator in ("reinforce_plus_plus", "reinforce_plus_plus_baseline"):
        with pytest.raises(ValueError, match="normalize-advantages"):
            arguments_module.validate_algorithm_args(
                _args(estimator, normalize_advantages=False, gdpo_reward_keys=None)
            )


def test_reinforce_family_passes_with_normalize_advantages(arguments_module):
    args = _args("reinforce_plus_plus", normalize_advantages=True, gdpo_reward_keys=None)
    arguments_module.validate_algorithm_args(args)
    assert args.use_critic is False


def test_ppo_is_runnable_and_turns_on_the_critic(arguments_module):
    """PPO is enabled upstream again; `use_critic` is the switch that makes
    `relax/core/registry.py` bind the Critic component."""
    args = _args("ppo", gdpo_reward_keys=None, reward_key=None)
    arguments_module.validate_algorithm_args(args)
    assert args.use_critic is True


@pytest.mark.parametrize("estimator", ["grpo", "gspo", "sapo", "cispo"])
def test_grpo_family_passes_with_defaults(arguments_module, estimator):
    args = _args(estimator, gdpo_reward_keys=None, reward_key=None)
    arguments_module.validate_algorithm_args(args)
    assert args.use_critic is False


def test_non_gdpo_algorithms_ignore_reward_key_and_component_options(arguments_module):
    """Only multi-reward algorithms police --reward-key / --gdpo-*."""
    args = _args("grpo", gdpo_reward_keys=None, gdpo_reward_weights=None, reward_key=None)
    arguments_module.validate_algorithm_args(args)


def test_unknown_estimator_raises_from_the_registry(arguments_module):
    with pytest.raises(KeyError, match="Unknown advantage estimator"):
        arguments_module.validate_algorithm_args(_args("not_an_algorithm"))


# ---------------- --custom-config-path override timing ----------------


def _write_yaml(tmp_path, body):
    path = tmp_path / "override.yaml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def _overridable_args(tmp_path, body, **overrides):
    """Args as they look when the YAML merge runs: already validated once."""
    base = _args("grpo", gdpo_reward_keys=None, reward_key=None)
    base.loss_type = "policy_loss"
    base.custom_config_path = _write_yaml(tmp_path, body)
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_yaml_cannot_switch_to_an_estimator_that_needs_a_critic(arguments_module, tmp_path):
    """Role composition and the offload flags were derived from the pre-
    override `use_critic`, so accepting a YAML switch to PPO would leave the
    run neither fully critic nor fully critic-free."""
    args = _overridable_args(tmp_path, "advantage_estimator: ppo\n")

    with pytest.raises(ValueError, match="critic setup"):
        arguments_module.apply_custom_config_overrides(args)


def test_yaml_cannot_bypass_gdpo_requirements(arguments_module, tmp_path):
    """Switching to GDPO from YAML must still demand its reward keys."""
    args = _overridable_args(tmp_path, "advantage_estimator: gdpo\n")

    with pytest.raises(ValueError, match="at least two reward keys"):
        arguments_module.apply_custom_config_overrides(args)


def test_yaml_cannot_enable_conflicting_whitening_under_gdpo(arguments_module, tmp_path):
    """The dangerous case: silent double whitening rather than a crash."""
    args = _overridable_args(
        tmp_path,
        "normalize_advantages: true\n",
        advantage_estimator="gdpo",
        gdpo_reward_keys=["correctness", "format"],
        reward_key="score",
        n_samples_per_prompt=8,
    )

    with pytest.raises(ValueError, match="normalize-advantages"):
        arguments_module.apply_custom_config_overrides(args)


# The three validators below were split out of `validate_algorithm_args`
# because the main path has a derivation order, and only two of the four were
# wired back into the override path -- so a YAML file could select rloo and
# then move any value the missing three read. Each case here fails on the
# pre-fix code.


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param("advantage_estimator: rloo\nkl_coef: 0.01\n", "nonzero --kl-coef", id="reward-side-kl"),
        pytest.param(
            "advantage_estimator: rloo\nnum_steps_per_rollout: 4\n",
            "num-steps-per-rollout 1",
            id="update-schedule",
        ),
        pytest.param(
            "advantage_estimator: rloo\nglobal_batch_size: 999\n",
            "one optimizer update per rollout",
            id="batch-shape",
        ),
    ],
)
def test_yaml_cannot_bypass_the_late_running_validators(arguments_module, tmp_path, body, expected):
    args = _overridable_args(
        tmp_path,
        body,
        n_samples_per_prompt=8,
        rollout_batch_size=16,
        global_batch_size=128,
        num_steps_per_rollout=None,
        kl_coef=0.0,
        max_staleness=0,
        calculate_per_token_loss=True,
        rewards_normalization=True,
        normalize_advantages=False,
        partial_rollout=False,
        use_dynamic_global_batch_size=False,
        hybrid=False,
        fully_async=False,
    )

    with pytest.raises(ValueError, match=expected):
        arguments_module.apply_custom_config_overrides(args)


def test_yaml_that_changes_the_update_schedule_gets_a_rederived_batch(arguments_module, tmp_path):
    """Re-running the validator without its derivation rejected a legal config.

    `validate_batch_shape` reads `global_batch_size`, which the main path
    derives from `num_steps_per_rollout` *before* the merge. A YAML switching
    grpo@4-steps to rloo@1-step should get `rollout * n = 128`; the first
    version of this fix compared against the stale 32 and refused it.
    """
    args = _overridable_args(
        tmp_path,
        "advantage_estimator: rloo\nnum_steps_per_rollout: 1\n",
        n_samples_per_prompt=8,
        rollout_batch_size=16,
        global_batch_size=32,
        num_steps_per_rollout=4,
        kl_coef=0.0,
        max_staleness=0,
        calculate_per_token_loss=True,
        rewards_normalization=True,
        normalize_advantages=False,
        partial_rollout=False,
        use_dynamic_global_batch_size=False,
        hybrid=False,
        fully_async=False,
    )

    arguments_module.apply_custom_config_overrides(args)

    assert args.global_batch_size == 128, "the merge should re-derive it, not keep the pre-merge value"


def test_yaml_without_algorithm_changes_is_accepted(arguments_module, tmp_path):
    args = _overridable_args(tmp_path, "lr: 0.5\n")
    arguments_module.apply_custom_config_overrides(args)
    assert args.lr == 0.5
    assert args.advantage_estimator == "grpo"


def test_no_yaml_is_a_no_op(arguments_module):
    args = _args("grpo", gdpo_reward_keys=None, reward_key=None)
    args.loss_type = "policy_loss"
    args.custom_config_path = None
    arguments_module.apply_custom_config_overrides(args)


def test_sft_runs_skip_the_algorithm_recheck(arguments_module, tmp_path):
    """SFT never selects an estimator, so a stale one must not block it."""
    args = _overridable_args(tmp_path, "lr: 0.5\n", loss_type="sft", advantage_estimator="ppo")
    arguments_module.apply_custom_config_overrides(args)
    assert args.lr == 0.5


def test_slime_validate_args_applies_overrides_through_the_helper(arguments_module):
    """Guard the call site: the merge must go through the re-checking
    helper."""
    import inspect

    src = inspect.getsource(arguments_module.slime_validate_args)
    assert "apply_custom_config_overrides(args)" in src
    assert "yaml.safe_load" not in src, "the YAML merge was inlined again, skipping the re-check"


def test_spec_with_an_unregistered_implementation_is_rejected_at_startup(arguments_module, monkeypatch):
    """A registry typo must name itself, not KeyError inside a worker."""
    from dataclasses import replace

    from relax.algorithms.spec import ALGORITHM_SPECS

    broken = replace(ALGORITHM_SPECS["grpo"], advantage_fn="typo_does_not_exist")
    monkeypatch.setitem(ALGORITHM_SPECS, "grpo", broken)

    with pytest.raises(ValueError, match="typo_does_not_exist"):
        arguments_module.validate_algorithm_args(_args("grpo", gdpo_reward_keys=None, reward_key=None))


# ---------------- fully-async is not a supported execution mode for GDPO ----------------


def test_gdpo_is_rejected_under_fully_async(arguments_module):
    """Otherwise it trains on one slice at a time and, at slice size 1, on
    nothing."""
    with pytest.raises(ValueError, match="not supported under --fully-async"):
        arguments_module.validate_algorithm_args(_args(fully_async=True))


def test_gdpo_is_allowed_under_hybrid(arguments_module):
    """--hybrid sets fully_async later, but uses the colocate role set: advantages
    are computed in the Megatron worker, where the DP group exists."""
    arguments_module.validate_algorithm_args(_args(fully_async=True, hybrid=True))


def test_gdpo_is_allowed_under_colocate(arguments_module):
    arguments_module.validate_algorithm_args(_args(fully_async=False))


@pytest.mark.parametrize("estimator", ["grpo", "gspo", "sapo", "cispo"])
def test_other_estimators_are_unaffected_by_fully_async(arguments_module, estimator):
    args = _args(estimator, fully_async=True, gdpo_reward_keys=None, reward_key=None)
    arguments_module.validate_algorithm_args(args)


def test_supports_fully_async_defaults_to_true():
    from relax.algorithms import get_algorithm, list_algorithm_names

    unsupported = {n for n in list_algorithm_names() if not get_algorithm(n).supports_fully_async}
    assert unsupported == {"gdpo"}


def test_dynamic_sampling_filter_warns_for_multi_reward(arguments_module, caplog):
    """The built-in filter judges a group by the single --reward-key scalar."""
    import logging

    with caplog.at_level(logging.WARNING):
        arguments_module.validate_algorithm_args(_args(dynamic_sampling_filter_path="pkg.mod.fn"))

    assert any("dynamic-sampling-filter-path" in r.message for r in caplog.records)


def test_no_warning_without_a_filter(arguments_module, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        arguments_module.validate_algorithm_args(_args())

    assert not any("dynamic-sampling-filter-path" in r.message for r in caplog.records)
