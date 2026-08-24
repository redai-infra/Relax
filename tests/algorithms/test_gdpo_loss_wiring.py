# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GDPO through the production Megatron entry point, not through the pure
functions.

Every other step-3 test in this suite calls `advantage_gdpo` or `whiten_scalar`
directly. That leaves a hole the size of the whole wiring: deleting the
`mini_batch_sizes=` argument in `loss.py`, or routing `gdpo` to the wrong
`advantage_fn`, keeps all of them green while GDPO silently goes back to
whitening the merged rollout. `test_dispatch_parity_vs_main.py` covers part of
that with a regex over the source, which pins the text rather than the
behaviour.

These call `relax.backends.megatron.loss.compute_advantages_and_returns` with a
real `rollout_data` dict and check the numbers that come out.

Megatron is stubbed rather than skipped. The CPU runner has no `megatron.core`,
and the alternative -- an importorskip -- means this file never runs anywhere
the rest of the suite runs, which is the same hole in a different shape. Only
the handful of `mpu` entry points this path touches are faked, and each fake
returns the single-process answer.
"""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from relax.algorithms.advantages import whiten_scalar  # noqa: E402


# The `mpu` surface this file's call path touches, answered as a one-process
# run would. Kept as data so the import-time fake and the per-test patch cannot
# drift apart -- they are the same answers reaching the same call sites.
_SINGLE_PROCESS_ANSWERS = {
    "is_pipeline_last_stage": lambda *_a, **_k: True,
    "get_data_parallel_group": lambda *_a, **_k: None,
    "get_context_parallel_rank": lambda *_a, **_k: 0,
    "get_context_parallel_world_size": lambda *_a, **_k: 1,
}


def _single_process_mpu():
    mpu = ModuleType("megatron.core.mpu")
    for name, answer in _SINGLE_PROCESS_ANSWERS.items():
        setattr(mpu, name, answer)
    return mpu


def _load_loss_module():
    """Import the production loss module, with or without Megatron installed.

    Only the import needs the fake package, and only when Megatron is absent.
    Which `mpu` the module ends up bound to is decided per test by
    `single_process_mpu` below -- doing it here instead is what made this file
    pass on CI and fail in the project image.
    """
    try:
        import megatron.core  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        return importlib.import_module("relax.backends.megatron.loss")

    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    mpu = _single_process_mpu()
    core.mpu = mpu
    sys.modules.update({"megatron": megatron, "megatron.core": core, "megatron.core.mpu": mpu})
    try:
        return importlib.import_module("relax.backends.megatron.loss")
    finally:
        for name in ("megatron.core.mpu", "megatron.core", "megatron"):
            sys.modules.pop(name, None)


loss_module = _load_loss_module()


@pytest.fixture(autouse=True)
def single_process_mpu(monkeypatch):
    """Bind the loss module to a one-process `mpu`, installed or not.

    This file exists to run the real `loss.compute_advantages_and_returns`, and
    that function asks `mpu` which pipeline stage it is on. With Megatron
    absent the import-time fake answered; with Megatron present -- which is how
    `docker/Dockerfile` builds the project image, adding MCore to PYTHONPATH --
    the real `mpu` answered instead, and it cannot: no test here initialises a
    pipeline group, so all four cases died on `pipeline_model parallel group is
    not initialized`. GitHub's CPU runner has no MCore and so only ever
    exercised the other branch, which is why the suite looked green while
    `pytest -q tests/algorithms` in the image reported `4 failed, 805 passed`.

    Patching the functions *on* the `mpu` object, rather than rebinding
    `loss_module.mpu`, is what makes this cover the whole call path. `loss.py`
    is not the only module that does `from megatron.core import mpu`:
    `cp_utils.maybe_padded_total_lengths` holds its own reference and is
    reached from `compute_advantages_and_returns`, so rebinding one name left
    the other one talking to the uninitialised real thing. Every holder shares
    one module object, so patching its attributes reaches all of them at once.
    `monkeypatch` puts the originals back after each test.
    """
    for name, answer in _SINGLE_PROCESS_ANSWERS.items():
        monkeypatch.setattr(loss_module.mpu, name, answer, raising=False)


def _args(estimator="gdpo", **overrides):
    base = dict(
        advantage_estimator=estimator,
        use_rollout_logprobs=False,
        kl_coef=0.0,
        kl_loss_type="k1",
        gamma=1.0,
        lambd=1.0,
        qkv_format="thd",
        is_vl_model=False,
        uses_unsplit_forward=False,
        normalize_advantages=False,
        use_opd=False,
        opd_only_reward=False,
        opd_kl_coef=0.0,
        advantage_clip=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _rollout_data(rewards, lengths, mini_batch_sizes):
    return {
        "log_probs": [torch.zeros(n, dtype=torch.float32) for n in lengths],
        "ref_log_probs": None,
        "rewards": list(rewards),
        "values": None,
        "response_lengths": list(lengths),
        "loss_masks": [torch.ones(n, dtype=torch.float32) for n in lengths],
        "total_lengths": [n + 2 for n in lengths],
        "rollout_mini_local_sample_counts": mini_batch_sizes,
    }


REWARDS = [1.0, 3.0, 10.0, 14.0]
LENGTHS = [2, 3, 2, 3]


def test_gdpo_reaches_the_optimizer_whitened_per_training_batch():
    """The end-to-end number, against the two segments computed separately."""
    data = _rollout_data(REWARDS, LENGTHS, mini_batch_sizes=[2, 2])
    loss_module.compute_advantages_and_returns(_args(), data)

    first = whiten_scalar(torch.tensor(REWARDS[:2], dtype=torch.float32))
    second = whiten_scalar(torch.tensor(REWARDS[2:], dtype=torch.float32))
    expected = torch.cat([first, second])

    advantages = data["advantages"]
    assert len(advantages) == len(LENGTHS)
    for index, length in enumerate(LENGTHS):
        assert advantages[index].shape == (length,)
        assert torch.allclose(advantages[index], expected[index].expand(length))


def test_the_segmentation_is_load_bearing_not_decorative():
    """Whitening the merged rollout gives different numbers.

    This is the assertion that fails if `loss.py` stops forwarding
    `rollout_mini_local_sample_counts`. The two rewards groups are on different
    scales on purpose: a merged statistic puts the whole first batch below the
    mean and the whole second above it, which is exactly the objective drift
    per-batch whitening exists to avoid.
    """
    data = _rollout_data(REWARDS, LENGTHS, mini_batch_sizes=[2, 2])
    loss_module.compute_advantages_and_returns(_args(), data)
    per_batch = torch.stack([chunk[0] for chunk in data["advantages"]])

    merged = whiten_scalar(torch.tensor(REWARDS, dtype=torch.float32))

    assert not torch.allclose(per_batch, merged, atol=1e-3)
    assert (merged[:2] < 0).all() and (merged[2:] > 0).all(), "merged: the first batch is all-negative"
    assert per_batch[0] < 0 < per_batch[1], "per batch: each batch has both signs"


def test_a_missing_segmentation_fails_instead_of_defaulting():
    """No safe default exists, so the production path must not invent one."""
    data = _rollout_data(REWARDS, LENGTHS, mini_batch_sizes=None)
    with pytest.raises(ValueError, match="mini_batch_sizes is None"):
        loss_module.compute_advantages_and_returns(_args(), data)


def test_the_same_entry_point_routes_grpo_somewhere_else():
    """Guards the dispatch itself: `gdpo` must not be reaching a shared branch.

    GRPO broadcasts the raw scalar; GDPO whitens first. If the registry ever
    routed `gdpo` to `grpo_broadcast`, the test above would still pass for a
    batch whose whitening happens to be near-identity, but this one would not.
    """
    grpo_data = _rollout_data(REWARDS, LENGTHS, mini_batch_sizes=[2, 2])
    loss_module.compute_advantages_and_returns(_args("grpo"), grpo_data)

    for index, reward in enumerate(REWARDS):
        assert torch.allclose(grpo_data["advantages"][index], torch.full((LENGTHS[index],), reward))
