# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The shipped GDPO example reward must produce the two configured
components."""

import importlib.util
import pathlib

import pytest


EXAMPLE_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "gdpo"
REWARD_PATH = EXAMPLE_DIR / "reward_gdpo.py"
SCRIPT_PATH = EXAMPLE_DIR / "run-qwen3-0.6B-1xgpu-gdpo.sh"


@pytest.fixture(scope="module")
def reward_module():
    spec = importlib.util.spec_from_file_location("reward_gdpo", REWARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_returns_all_three_keys(reward_module):
    out = reward_module.compute_gdpo_reward("<think>x</think><answer>42</answer>", "42")
    assert set(out) == {"score", "correctness", "format"}


def test_correct_and_well_formatted(reward_module):
    out = reward_module.compute_gdpo_reward("<think>reasoning</think><answer>42</answer>", "42")
    assert out["correctness"] == 1.0
    assert out["format"] == 1.0


def test_wrong_answer_but_well_formatted(reward_module):
    out = reward_module.compute_gdpo_reward("<think>reasoning</think><answer>7</answer>", "42")
    assert out["correctness"] == 0.0
    assert out["format"] == 1.0


def test_correct_answer_but_malformed(reward_module):
    """The case GDPO is designed for: the components disagree."""
    out = reward_module.compute_gdpo_reward("42", "42")
    assert out["correctness"] == 0.0  # no <answer> tag means the answer is unparseable
    assert out["format"] == 0.0


def test_partially_formatted_scores_half(reward_module):
    out = reward_module.compute_gdpo_reward("<answer>42</answer>", "42")
    assert out["format"] == 0.5
    assert out["correctness"] == 1.0


def test_thinking_without_an_answer_scores_half_format(reward_module):
    out = reward_module.compute_gdpo_reward("<think>reasoning</think>", "42")
    assert out["format"] == 0.5
    assert out["correctness"] == 0.0


def test_score_mirrors_correctness(reward_module):
    for response in ("<answer>42</answer>", "<answer>7</answer>", "nothing"):
        out = reward_module.compute_gdpo_reward(response, "42")
        assert out["score"] == out["correctness"]


def test_label_is_stringified_and_stripped(reward_module):
    assert reward_module.compute_gdpo_reward("<answer>42</answer>", 42)["correctness"] == 1.0
    assert reward_module.compute_gdpo_reward("<answer>42</answer>", " 42 ")["correctness"] == 1.0


def test_components_are_plain_floats(reward_module):
    out = reward_module.compute_gdpo_reward("<answer>42</answer>", "42")
    for key in ("score", "correctness", "format"):
        assert isinstance(out[key], float)
        assert not isinstance(out[key], bool)


def test_components_survive_the_gdpo_normalizer(reward_module):
    """End-to-end: example rewards feed the registered normalizer without error."""
    from types import SimpleNamespace

    from relax.algorithms.rewards import normalize_gdpo_decoupled

    responses = [
        "<think>a</think><answer>42</answer>",
        "<think>b</think><answer>7</answer>",
        "<answer>42</answer>",
        "nothing at all",
    ]
    samples = [
        SimpleNamespace(
            group_index=0,
            reward=reward_module.compute_gdpo_reward(r, "42"),
            get_reward_components=lambda keys, r=r: [reward_module.compute_gdpo_reward(r, "42")[k] for k in keys],
        )
        for r in responses
    ]
    args = SimpleNamespace(
        n_samples_per_prompt=4,
        gdpo_reward_keys=["correctness", "format"],
        gdpo_reward_weights=None,
    )

    out = normalize_gdpo_decoupled(args, samples, [0.0] * 4)
    assert len(out) == 4
    assert abs(sum(out)) < 1e-4  # both components are group-centred
    assert max(abs(v) for v in out) > 0.1  # and there is real signal


# ---------------- launch script ----------------


def test_launch_script_wires_the_reward_to_the_estimator():
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "--advantage-estimator gdpo" in src
    assert "--gdpo-reward-keys correctness format" in src
    assert "--custom-rm-path examples.gdpo.reward_gdpo.reward_func" in src
    assert "--reward-key score" in src


def test_launch_script_satisfies_the_gdpo_group_size_floor():
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "--n-samples-per-prompt 8" in src


def test_launch_script_does_not_enable_the_conflicting_whitening():
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "--normalize-advantages" not in src
    assert "--custom-reward-post-process-path" not in src


# ---------------- the label form the launch script actually feeds ----------------
#
# The script points --prompt-data at gsm8k/train.jsonl with --label-key answer,
# and GSM8K's `answer` is the whole worked solution ending in "#### 36". Every
# test above used a pre-cleaned "42", so nothing here noticed that comparing
# against the raw string makes correctness zero for every rollout -- collapsing
# the component in every group and quietly reducing the example to its single
# format reward.

RAW_GSM8K_LABEL = "Janet sells 16 - 3 - 4 = 9 duck eggs.\n9 * $2 = $18\n#### 18"


def test_correct_answer_scores_against_a_raw_gsm8k_label(reward_module):
    out = reward_module.compute_gdpo_reward("<think>work</think><answer>18</answer>", RAW_GSM8K_LABEL)
    assert out["correctness"] == 1.0, "raw GSM8K labels must not zero out correctness"
    assert out["format"] == 1.0


def test_a_precleaned_label_behaves_identically(reward_module):
    raw = reward_module.compute_gdpo_reward("<think>work</think><answer>18</answer>", RAW_GSM8K_LABEL)
    clean = reward_module.compute_gdpo_reward("<think>work</think><answer>18</answer>", "18")
    assert raw == clean, "normalising the label must not change a dataset that is already clean"


def test_a_wrong_answer_is_still_wrong_against_a_raw_label(reward_module):
    """Otherwise the normalisation could be making everything match."""
    out = reward_module.compute_gdpo_reward("<think>work</think><answer>99</answer>", RAW_GSM8K_LABEL)
    assert out["correctness"] == 0.0
    assert out["format"] == 1.0, "format is independent of correctness; that is the point of the example"


def test_the_two_components_can_disagree_on_a_raw_label(reward_module):
    """The case GDPO exists for, on the data the script actually loads."""
    correct_unformatted = reward_module.compute_gdpo_reward("18", RAW_GSM8K_LABEL)
    wrong_formatted = reward_module.compute_gdpo_reward("<think>w</think><answer>99</answer>", RAW_GSM8K_LABEL)

    assert correct_unformatted["format"] < wrong_formatted["format"]
    assert correct_unformatted["correctness"] == 0.0  # no <answer> tag, so unparseable
    assert wrong_formatted["correctness"] == 0.0


# ---------------- the docs must not contradict the script ----------------
#
# Three times in this branch a claim was fixed in one file and left stale in a
# sibling: the batch geometry, the "covers a full training batch" wording, and
# the algorithm-name lists. Prose cannot be diffed against code, so this checks
# the one number the prose actually depends on.


def _script_flag(name):
    import pathlib
    import re

    script = (
        pathlib.Path(__file__).resolve().parents[2] / "examples" / "gdpo" / "run-qwen3-0.6B-1xgpu-gdpo.sh"
    ).read_text(encoding="utf-8")
    match = re.search(rf"--{name}\s+(\d+)", script)
    assert match, f"--{name} not found in the launch script"
    return int(match.group(1))


def test_example_geometry_satisfies_the_paper_boundary():
    """rollout_batch_size * n_samples_per_prompt == global_batch_size."""
    product = _script_flag("rollout-batch-size") * _script_flag("n-samples-per-prompt")
    assert product == _script_flag("global-batch-size"), (
        f"the example must whiten exactly one training batch; got {product} samples per rollout "
        f"against a global batch of {_script_flag('global-batch-size')}"
    )


@pytest.mark.parametrize("doc", ["docs/zh/examples/algorithms.md", "docs/en/examples/algorithms.md"])
def test_docs_do_not_describe_the_example_as_violating_the_boundary(doc):
    """The known-deviation section used to cite the example as the
    counterexample.

    It stopped being one when global_batch_size moved to 32, and the text was
    left behind. Anything asserting num_rollout_minis is 2 for this example is
    now false.
    """
    import pathlib

    text = (pathlib.Path(__file__).resolve().parents[2] / doc).read_text(encoding="utf-8")
    assert "num_rollout_minis = 2" not in text, f"{doc} still calls the shipped example a counterexample"
