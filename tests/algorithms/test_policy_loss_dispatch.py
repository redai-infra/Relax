# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Policy loss selection must come from the registry."""

import pathlib
import re
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from relax.algorithms.policy import POLICY_LOSS_FNS, compute_policy_loss_for  # noqa: E402
from relax.algorithms.spec import get_algorithm, list_algorithm_names  # noqa: E402
from relax.utils.training.ppo_utils import (  # noqa: E402
    compute_cispo_loss,
    compute_policy_loss,
    compute_sapo_loss,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LOSS_PATH = REPO_ROOT / "relax" / "backends" / "megatron" / "loss.py"
SERVE_PATH = REPO_ROOT / "relax" / "components" / "advantages.py"


def _args(estimator, **overrides):
    base = dict(
        advantage_estimator=estimator,
        eps_clip=0.2,
        eps_clip_high=0.2,
        sapo_tau_pos=1.0,
        sapo_tau_neg=1.05,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _tensors():
    torch.manual_seed(0)
    return torch.randn(8), torch.randn(8), torch.randn(8)


# ---------------- registry ----------------


def test_every_registered_loss_is_reachable_from_some_spec():
    """No dead entries: a loss no spec names can never be dispatched to.

    The inverse direction is `test_every_spec_policy_loss_id_is_registered`.
    Together they pin the table to exactly what the registry uses, which is
    what a hard-coded inventory of names was doing before -- except this
    version fails for a reason instead of failing on every addition.
    """
    referenced = {get_algorithm(name).policy_loss_fn for name in list_algorithm_names()}
    assert set(POLICY_LOSS_FNS) == referenced


def test_every_spec_policy_loss_id_is_registered():
    for name in list_algorithm_names():
        assert get_algorithm(name).policy_loss_fn in POLICY_LOSS_FNS


# ---------------- adapters match their kernels ----------------


def test_ppo_clip_matches_the_underlying_kernel():
    log_probs, ppo_kl, advantages = _tensors()
    args = _args("grpo")
    got = compute_policy_loss_for(args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


def test_sapo_matches_the_underlying_kernel():
    log_probs, ppo_kl, advantages = _tensors()
    got = compute_policy_loss_for(_args("sapo"), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = compute_sapo_loss(ppo_kl=ppo_kl, advantages=advantages, tau_pos=1.0, tau_neg=1.05)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


def test_cispo_matches_the_underlying_kernel():
    log_probs, ppo_kl, advantages = _tensors()
    got = compute_policy_loss_for(_args("cispo"), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = compute_cispo_loss(
        log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages, eps_clip=0.2, eps_clip_high=0.2
    )
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


def test_sapo_defaults_when_args_lack_tau_fields():
    log_probs, ppo_kl, advantages = _tensors()
    args = SimpleNamespace(advantage_estimator="sapo", eps_clip=0.2, eps_clip_high=0.2)
    got = compute_policy_loss_for(args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = compute_sapo_loss(ppo_kl=ppo_kl, advantages=advantages, tau_pos=1.0, tau_neg=1.05)
    assert torch.equal(got[0], want[0])


def test_sapo_taus_are_read_from_args():
    log_probs, ppo_kl, advantages = _tensors()
    args = _args("sapo", sapo_tau_pos=2.0, sapo_tau_neg=3.0)
    got = compute_policy_loss_for(args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = compute_sapo_loss(ppo_kl=ppo_kl, advantages=advantages, tau_pos=2.0, tau_neg=3.0)
    assert torch.equal(got[0], want[0])


@pytest.mark.parametrize("estimator", ["grpo", "gspo", "ppo", "reinforce_plus_plus"])
def test_ppo_clip_family_share_one_loss(estimator):
    log_probs, ppo_kl, advantages = _tensors()
    reference = compute_policy_loss_for(_args("grpo"), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    actual = compute_policy_loss_for(_args(estimator), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    assert torch.equal(reference[0], actual[0])


# ---------------- call sites no longer branch on names ----------------


# Any comparison of the estimator against a literal name, in any of the shapes
# Python offers: `== "x"`, `!= "x"`, `in [...]`, `in {...}`, `in (...)`.
#
# This used to be a hand-written list of the exact strings the refactor had
# deleted. It missed a live one: the REINFORCE++ checks in loss.py were written
# `in {` and the list only banned `in [`, so two algorithm-name sets survived
# the refactor with a green test sitting on top of them.
ESTIMATOR_NAME_CHECK = re.compile(r"""advantage_estimator\s*(?:==|!=|\bin\b)\s*[\[{("']""")


def _name_checks(source: str) -> list[str]:
    return [line.strip() for line in source.splitlines() if ESTIMATOR_NAME_CHECK.search(line)]


def test_loss_py_no_longer_branches_on_estimator_names():
    found = _name_checks(LOSS_PATH.read_text(encoding="utf-8"))
    assert found == [], f"loss.py still compares the estimator to literal names: {found}"


def test_serve_path_no_longer_branches_on_estimator_names():
    found = _name_checks(SERVE_PATH.read_text(encoding="utf-8"))
    assert found == [], f"components/advantages.py still compares the estimator to literal names: {found}"


def test_the_name_check_pattern_catches_every_spelling():
    """Guard the guard: the previous version of this test was blind to `in.

    {`.
    """
    for spelling in (
        'if args.advantage_estimator == "gspo":',
        'if args.advantage_estimator != "ppo":',
        'if args.advantage_estimator in ["grpo", "gspo"]:',
        'x = args.advantage_estimator in {"reinforce_plus_plus"}',
        'if self.config.advantage_estimator in ("ppo",):',
    ):
        assert _name_checks(spelling) == [spelling], spelling
    for allowed in (
        "spec = get_algorithm(args.advantage_estimator)",
        'if get_algorithm(args.advantage_estimator).advantage_normalization == "token_global":',
    ):
        assert _name_checks(allowed) == [], allowed


def test_both_paths_delegate_to_the_shared_estimator():
    for path in (LOSS_PATH, SERVE_PATH):
        src = path.read_text(encoding="utf-8")
        assert "from relax.algorithms.advantages import" in src, f"{path.name} does not use the shared estimator"


def test_loss_py_reads_kl_level_and_full_log_probs_from_the_spec():
    src = LOSS_PATH.read_text(encoding="utf-8")
    assert 'kl_level == "sequence"' in src
    assert "needs_full_log_probs" in src


def test_neither_call_site_still_raises_not_implemented_for_estimators():
    for path in (LOSS_PATH, SERVE_PATH):
        src = path.read_text(encoding="utf-8")
        assert "advantage_estimator {" not in src, f"{path.name} still formats an estimator into NotImplementedError"
