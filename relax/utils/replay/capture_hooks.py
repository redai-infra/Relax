# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Train-side hook glue for production capture.

Import this module **only** from the hot-path files (``loss.py``, ``model.py``,
``actor.py``) — it lazily imports ``megatron.core.mpu`` to resolve the parallel
topology and is therefore *not* part of the offline replay path. The offline
modules (``runner``, ``adapters``, ``bundle``, ``schema``, ``layout``) never
import this file.

Each function is a thin, hot-path-safe wrapper: it short-circuits through
``capture.begin_step`` / ``capture.current_step`` / ``capture.end_step`` when
capture is disabled or the step is unselected.
"""

from __future__ import annotations

from typing import Any

import torch

from relax.utils.logging_utils import get_logger
from relax.utils.replay import capture
from relax.utils.replay.schema import ActorStepId, Identity, RecomputeConfig, StageId


logger = get_logger(__name__)


def build_config_from_args(args: Any) -> RecomputeConfig:
    """Build a :class:`RecomputeConfig` from the runtime ``args`` namespace."""
    return RecomputeConfig(
        advantage_estimator=str(getattr(args, "advantage_estimator", "grpo")),
        n_samples_per_prompt=int(getattr(args, "n_samples_per_prompt", 1)),
        grpo_std_normalization=bool(getattr(args, "grpo_std_normalization", False)),
        kl_loss_type=str(getattr(args, "kl_loss_type", "k1")),
        kl_coef=float(getattr(args, "kl_coef", 0.0)),
        eps_clip=float(getattr(args, "eps_clip", 0.2)),
        eps_clip_high=float(getattr(args, "eps_clip_high", 0.2)),
        entropy_coef=float(getattr(args, "entropy_coef", 0.0)),
    )


def build_identity(args: Any, rollout_id: int, step_id: int) -> Identity:
    """Resolve the parallel topology for the captured step via ``mpu``."""
    from megatron.core import mpu  # lazy: hot-path / megatron only

    return Identity(
        actor_step_id=ActorStepId(rollout_id=int(rollout_id), step_id=int(step_id)),
        rank={
            "dp": int(mpu.get_data_parallel_world_size()),
            "tp": int(mpu.get_tensor_model_parallel_world_size()),
            "pp": int(mpu.get_pipeline_model_parallel_world_size()),
            "cp": int(mpu.get_context_parallel_world_size()),
        },
    )


def build_rollout_identity(args: Any, rollout_id: int) -> Identity:
    """Resolve the parallel topology for a rollout-level cohort via ``mpu``."""
    from megatron.core import mpu  # lazy: hot-path / megatron only

    return Identity(
        rollout_id=int(rollout_id),
        rank={
            "dp": int(mpu.get_data_parallel_world_size()),
            "tp": int(mpu.get_tensor_model_parallel_world_size()),
            "pp": int(mpu.get_pipeline_model_parallel_world_size()),
            "cp": int(mpu.get_context_parallel_world_size()),
        },
    )


def begin_step_for(args: Any, rollout_id: int, step_id: int, bundle_id: str | None = None) -> None:
    """Open a per-step accumulator from the ``train_one_step`` loop.

    Short-circuits on the disabled/unselected guard *before* building the
    identity/config, so the disabled hot path pays only two global reads.
    """
    actor_step_id = (int(rollout_id), int(step_id))
    if not capture.should_capture(actor_step_id):
        return
    capture.begin_step(
        actor_step_id,
        identity=build_identity(args, rollout_id, step_id),
        config=build_config_from_args(args),
        bundle_id=bundle_id if bundle_id is not None else f"replay-{rollout_id}-{step_id}",
    )


def end_step_for() -> None:
    """Finalize and submit the current step's accumulator."""
    capture.end_step()


def begin_rollout_for(args: Any, rollout_id: int, bundle_id: str | None = None) -> None:
    """Open a per-rollout accumulator from the ``train_actor`` loop.

    Captures the reward → advantage chain (a rollout-level cohort). Short-
    circuits on the disabled/unselected guard before building the
    identity/config.
    """
    if not capture.should_capture_rollout(int(rollout_id)):
        return
    capture.begin_rollout(
        int(rollout_id),
        identity=build_rollout_identity(args, rollout_id),
        config=build_config_from_args(args),
        bundle_id=bundle_id if bundle_id is not None else f"replay-rollout-{rollout_id}",
    )


def end_rollout_for() -> None:
    """Finalize and submit the current rollout's accumulator."""
    capture.end_rollout()


def _cat(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Flatten a per-sample tensor list into one contiguous 1-D tensor
    (detached)."""
    return torch.cat([tensor.detach().reshape(-1) for tensor in tensors])


def _detach_1d_float(value: Any) -> torch.Tensor:
    """Return ``value`` as a detached 1-D float tensor (no GPU→CPU sync).

    A Python list is already CPU materialized, so ``torch.tensor(list)`` is
    cheap; a tensor is only detached here, its ``.tolist()`` deferred to the
    writer.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().reshape(-1).float()
    return torch.tensor(list(value), dtype=torch.float32)


def _detach_1d_long(value: Any) -> torch.Tensor:
    """Return ``value`` as a detached 1-D long tensor (no GPU→CPU sync)."""
    if isinstance(value, torch.Tensor):
        return value.detach().reshape(-1).long()
    return torch.tensor(list(value), dtype=torch.long)


def capture_rollout_advantage(*, rollout_data: Any, args: Any) -> None:
    """Record the reward → advantage chain from ``train_actor`` (rollout
    level).

    Called once per rollout after ``compute_advantages_and_returns`` has added
    ``advantages``/``returns``. Every value is stored as a detached tensor
    reference; per-sample metadata (group index / rewards / masks) is converted
    on the writer thread. The expected KL is recomputed here from the rollout
    vs reference log-probs because production keeps it a local variable.
    """
    step = capture.current_step()
    if step is None:
        return

    # Non-last pipeline stages return from compute_advantages_and_returns before
    # "advantages" is written; skip them (the last-PP-stage rank records once).
    if rollout_data.get("advantages") is None:
        return

    from relax.utils.training.ppo_utils import compute_approx_kl  # train-side only

    use_rollout = bool(getattr(args, "use_rollout_logprobs", False))
    kl_loss_type = str(getattr(args, "kl_loss_type", "k1"))
    kl_coef = float(getattr(args, "kl_coef", 0.0))

    step.stages = {
        StageId.REWARD_RAW,
        StageId.REWARD_POST_PROCESS,
        StageId.ADVANTAGE_KL,
        StageId.ADVANTAGE_ESTIMATE,
    }

    step.response_lengths = [int(length) for length in rollout_data["response_lengths"]]
    step.total_lengths = [int(length) for length in rollout_data["total_lengths"]]
    if rollout_data.get("loss_masks"):
        step.loss_masks_tensor = _cat(list(rollout_data["loss_masks"]))
    step.group_indices_tensor = _detach_1d_long(rollout_data["group_index"])
    step.raw_rewards_tensor = _detach_1d_float(rollout_data["raw_reward"])
    step.rewards_tensor = _detach_1d_float(rollout_data["rewards"])

    log_probs_key = "rollout_log_probs" if use_rollout else "log_probs"
    old_log_probs = list(rollout_data[log_probs_key])
    ref_log_probs = list(rollout_data["ref_log_probs"]) if rollout_data.get("ref_log_probs") else None

    step.tensors["old_log_probs"] = _cat(old_log_probs)
    step.tensors["advantages"] = _cat(list(rollout_data["advantages"]))

    if kl_coef == 0 or ref_log_probs is None:
        ref_log_probs = [torch.zeros_like(probs, dtype=torch.float32) for probs in old_log_probs]
        kl = [torch.zeros_like(probs, dtype=torch.float32) for probs in old_log_probs]
    else:
        kl = [
            compute_approx_kl(old_log_probs[index], ref_log_probs[index], kl_loss_type=kl_loss_type)
            for index in range(len(old_log_probs))
        ]
    step.tensors["ref_log_probs"] = _cat(ref_log_probs)
    step.tensors["kl"] = _cat(kl)


def capture_policy_loss(
    *,
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    entropy: torch.Tensor,
    advantages: torch.Tensor,
    loss_masks: list[torch.Tensor],
    response_lengths: list[int],
    total_lengths: list[int],
    reported_loss: dict[str, torch.Tensor],
) -> None:
    """Record the ``loss.policy`` stage from inside ``policy_loss_function``.

    Every value is stored as a detached tensor reference — no ``.cpu()`` /
    ``.item()`` / ``.tolist()`` on the training thread. The loss-mask list is
    concatenated into a single tensor and split back on the writer thread.
    """
    step = capture.current_step()
    if step is None:
        return

    step.stages = {StageId.LOSS_POLICY}
    step.tensors["old_log_probs"] = old_log_probs.detach()
    step.tensors["log_probs"] = log_probs.detach()
    step.tensors["entropy"] = entropy.detach()
    step.tensors["advantages"] = advantages.detach()
    step.response_lengths = [int(length) for length in response_lengths]
    step.total_lengths = [int(length) for length in total_lengths]
    step.loss_masks_tensor = torch.cat([mask.detach().reshape(-1) for mask in loss_masks]) if loss_masks else None

    policy_fields = ("loss", "pg_loss", "entropy_loss", "pg_clipfrac", "ppo_kl")
    step.expected[StageId.LOSS_POLICY.value] = {
        field: value.detach() for field, value in reported_loss.items() if field in policy_fields
    }
