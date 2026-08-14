# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reward-stage replay adapters.

``reward.post_process`` restates the group normalization from
``relax.utils.utils.post_process_rewards``. That production function lives in a
module that imports Ray at module scope, so the offline runner cannot reuse it
directly without pulling Ray into the CPU path; instead we restate the (small,
stable) group-norm here and pin its semantics with the PR #65 parity fixture.
"""

from __future__ import annotations

from typing import Any

import torch

from relax.utils.logging_utils import get_logger
from relax.utils.replay.bundle import LoadedBundle
from relax.utils.replay.compare import compare_scalar, scalar_error, tolerance_for
from relax.utils.replay.report import FieldDivergence, StageResult, StageStatus
from relax.utils.replay.schema import StageId
from relax.utils.replay.stages import register_adapter


logger = get_logger(__name__)


@register_adapter(StageId.REWARD_RAW)
def replay_reward_raw(bundle: LoadedBundle, ctx: dict[str, Any]) -> StageResult:
    """Resolve the raw reward per sample.

    V1 covers scalar rewards only (no ``reward_key``), so the resolved reward
    is the sample's own scalar. Recompute equals the recorded value; divergence
    here signals a record/identity integrity problem.
    """
    result = StageResult(stage=StageId.REWARD_RAW.value, status=StageStatus.PASS)
    expected = bundle.expected.get(StageId.REWARD_RAW.value, {}).get("raw_rewards")
    actual = [record.raw_reward for record in bundle.index.samples]

    if expected is None:
        result.message = "no expected raw_rewards recorded"
        return result

    atol, rtol = tolerance_for(bundle.manifest, StageId.REWARD_RAW)
    for index, (exp, act) in enumerate(zip(expected, actual, strict=False)):
        if not compare_scalar(float(exp), float(act), atol=atol, rtol=rtol):
            result.status = StageStatus.FAIL
            result.divergences.append(
                FieldDivergence(
                    field="raw_reward",
                    sample_id=bundle.index.samples[index].sample_id,
                    expected=float(exp),
                    actual=float(act),
                    abs_error=scalar_error(float(exp), float(act)),
                )
            )
    if result.status == StageStatus.FAIL:
        result.message = f"raw reward mismatch in {len(result.divergences)} sample(s)"
    return result


@register_adapter(StageId.REWARD_POST_PROCESS)
def replay_reward_post_process(bundle: LoadedBundle, ctx: dict[str, Any]) -> StageResult:
    """Recompute group-normalized rewards (PR #65 semantics).

    Mirrors ``relax.utils.utils.post_process_rewards``: group by
    ``group_index`` (never by physical batch), subtract the group mean,
    optionally divide by the group std (``+1e-6`` floor), and reject groups
    whose size differs from ``n_samples_per_prompt``.
    """
    config = bundle.index.config
    raw_rewards = [record.raw_reward for record in bundle.index.samples]
    result = StageResult(stage=StageId.REWARD_POST_PROCESS.value, status=StageStatus.PASS)

    rewards = torch.tensor(raw_rewards, dtype=torch.float)
    positions_by_group: dict[int, list[int]] = {}
    for position, record in enumerate(bundle.index.samples):
        if record.group_index is None:
            result.status = StageStatus.FAIL
            result.message = f"sample {record.sample_id!r} has no group_index"
            return result
        positions_by_group.setdefault(record.group_index, []).append(position)

    normalized = torch.empty_like(rewards)
    for group_index, positions in positions_by_group.items():
        if len(positions) != config.n_samples_per_prompt:
            result.status = StageStatus.FAIL
            result.message = (
                f"reward group {group_index} has {len(positions)} samples, expected {config.n_samples_per_prompt}"
            )
            return result
        group_rewards = rewards[positions]
        group_rewards = group_rewards - group_rewards.mean()
        if config.grpo_std_normalization:
            group_rewards = group_rewards / (group_rewards.std() + 1e-6)
        normalized[positions] = group_rewards

    ctx[StageId.REWARD_POST_PROCESS.value] = normalized.tolist()

    expected = bundle.expected.get(StageId.REWARD_POST_PROCESS.value, {}).get("rewards")
    if expected is None:
        result.message = "no expected normalized rewards recorded"
        return result

    atol, rtol = tolerance_for(bundle.manifest, StageId.REWARD_POST_PROCESS)
    for index, (exp, act) in enumerate(zip(expected, normalized.tolist(), strict=False)):
        if not compare_scalar(float(exp), float(act), atol=atol, rtol=rtol):
            result.status = StageStatus.FAIL
            result.divergences.append(
                FieldDivergence(
                    field="reward",
                    sample_id=bundle.index.samples[index].sample_id,
                    expected=float(exp),
                    actual=float(act),
                    abs_error=scalar_error(float(exp), float(act)),
                )
            )
    if result.status == StageStatus.FAIL:
        result.message = f"normalized reward mismatch in {len(result.divergences)} sample(s)"
    return result
