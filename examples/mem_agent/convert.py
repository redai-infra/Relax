# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Expand MemAgent trajectories into independent ReLax training samples."""

from __future__ import annotations

from typing import Any

from relax.utils.types import Sample
from relax.utils.utils import dict_to_tensordict, post_process_rewards


def _get_turns(sample: Sample) -> list[dict[str, Any]]:
    metadata = sample.train_metadata or {}
    turns = metadata.get("mem_agent_turns", metadata.get("turns"))
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"Sample index={sample.index} has no MemAgent turns.")
    return turns


def _validate_turn(sample: Sample, turn: dict[str, Any]) -> None:
    required = {"tokens", "response_length", "loss_mask", "rollout_log_probs"}
    missing = required.difference(turn)
    if missing:
        raise ValueError(f"Sample index={sample.index} turn is missing fields: {sorted(missing)}")
    response_length = int(turn["response_length"])
    if response_length <= 0 or len(turn["tokens"]) < response_length:
        raise ValueError(f"Sample index={sample.index} has an invalid response_length.")
    if len(turn["loss_mask"]) != response_length:
        raise ValueError(f"Sample index={sample.index} has a misaligned loss_mask.")
    if len(turn["rollout_log_probs"]) != response_length:
        raise ValueError(f"Sample index={sample.index} has misaligned rollout_log_probs.")


def convert_samples(args: Any, samples: list[Sample]):
    """Normalize trajectory rewards first, then expand every saved turn."""
    if not samples:
        raise ValueError("MemAgent converter received an empty sample list.")
    if not getattr(args, "mem_agent_strict_alignment", True):
        raise ValueError("MemAgent training requires mem_agent_strict_alignment=true.")
    for sample in samples:
        if sample.status in (Sample.Status.ABORTED, Sample.Status.FAILED):
            raise ValueError(f"Cannot train from sample index={sample.index} with status={sample.status.value}.")

    # Normalize while there is still exactly one row per trajectory. Expanding
    # first would overweight long documents in the group statistics.
    raw_rewards, advantages = post_process_rewards(args, samples)
    credit_assignment = getattr(args, "mem_agent_credit_assignment", "split")
    if credit_assignment not in ("split", "share"):
        raise ValueError("mem_agent_credit_assignment must be either 'split' or 'share'.")

    # Every trajectory in one GRPO prompt group reads the same context, so it
    # must have the same number of turns. Besides catching partial trajectories,
    # this makes each expanded group n_samples_per_prompt * turn_count rows.
    turn_counts_by_group: dict[int, set[int]] = {}
    for sample in samples:
        if sample.group_index is None:
            raise ValueError("MemAgent samples require group_index.")
        turn_counts_by_group.setdefault(sample.group_index, set()).add(len(_get_turns(sample)))
    inconsistent_groups = {
        group_index: counts for group_index, counts in turn_counts_by_group.items() if len(counts) != 1
    }
    if inconsistent_groups:
        raise ValueError(f"MemAgent prompt groups have inconsistent turn counts: {inconsistent_groups}")

    train_data: dict[str, list[Any]] = {
        "tokens": [],
        "response_lengths": [],
        "loss_masks": [],
        "rollout_log_probs": [],
        "rewards": [],
        "raw_reward": [],
        "truncated": [],
        "sample_indices": [],
        "trajectory_indices": [],
        "turn_indices": [],
        "total_lengths": [],
    }

    for trajectory_position, (sample, raw_reward, advantage) in enumerate(
        zip(samples, raw_rewards, advantages, strict=True)
    ):
        turns = _get_turns(sample)
        turn_credit = float(advantage) / len(turns) if credit_assignment == "split" else float(advantage)
        # This expansion is deliberately lossless. Unlike the fixed VIME
        # helper, no tail rows are trimmed to a global-batch multiple.
        for fallback_turn_index, turn in enumerate(turns):
            _validate_turn(sample, turn)
            tokens = list(turn["tokens"])
            response_length = int(turn["response_length"])
            loss_mask = list(turn["loss_mask"])
            if sample.remove_sample:
                loss_mask = [0] * response_length
            train_data["tokens"].append(tokens)
            train_data["response_lengths"].append(response_length)
            train_data["loss_masks"].append(loss_mask)
            train_data["rollout_log_probs"].append(list(turn["rollout_log_probs"]))
            train_data["rewards"].append(turn_credit)
            train_data["raw_reward"].append(float(raw_reward))
            train_data["truncated"].append(int(turn.get("finish_reason") == Sample.Status.TRUNCATED.value))
            train_data["sample_indices"].append(sample.index)
            train_data["trajectory_indices"].append(trajectory_position)
            train_data["turn_indices"].append(int(turn.get("turn_index", fallback_turn_index)))
            train_data["total_lengths"].append(len(tokens))

    required_multiple = int(getattr(args, "mem_agent_train_rows_multiple", 1))
    if required_multiple <= 0:
        raise ValueError("mem_agent_train_rows_multiple must be positive.")
    if len(train_data["tokens"]) % required_multiple:
        raise ValueError(
            f"Expanded MemAgent row count {len(train_data['tokens'])} is not divisible by "
            f"mem_agent_train_rows_multiple={required_multiple}."
        )

    if getattr(args, "debug_train_only", False):
        return train_data
    return dict_to_tensordict(train_data, len(train_data["tokens"]))
