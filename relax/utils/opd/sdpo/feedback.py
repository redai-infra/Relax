# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SDPO feedback: privileged teacher prompts built from group outcomes."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

from relax.utils.opd.feedback import EnvironmentFeedback
from relax.utils.opd.opd_main_worker import TopkWorker
from relax.utils.opd.opd_opsd_worker import OpsdWorker
from relax.utils.opd.opd_utils import OPD_SAMPLE_MASK
from relax.utils.opd.sdpo.constants import SDPO_TOKEN_SELECTION
from relax.utils.opd.sdpo.validation import (
    validate_sdpo_student_topk_ids,
    validate_sdpo_text_only,
    validate_sdpo_topk_payload,
)
from relax.utils.types import Sample


def _has_sdpo_teacher_prompt(prompt: object) -> bool:
    if isinstance(prompt, str):
        return bool(prompt.strip())
    if isinstance(prompt, list):
        return bool(prompt)
    return False


def _clear_teacher_payload(sample: Sample) -> None:
    """Drop outputs from an earlier teacher request without touching rollout
    Top-K."""

    for field_name in (
        "teacher_log_probs",
        "teacher_topk_token_ids",
        "teacher_topk_log_probs",
        "teacher_at_student_topk_log_probs",
        "student_at_teacher_topk_log_probs",
        "opd_topk_token_ids",
        "opd_topk_student_log_probs",
        "opd_topk_teacher_log_probs",
        "opd_topk_ksz",
        "teacher_tokens",
        "teacher_prompt_length",
    ):
        setattr(sample, field_name, None)


def _render_sdpo_teacher_prompt(sample: Sample, additions: list[str]) -> str | list[dict[str, str]]:
    prompt = copy.deepcopy(sample.prompt)
    suffix = "\n\n".join(additions + (["Now produce the best answer to the original problem."] if additions else []))
    if isinstance(prompt, list):
        messages = prompt
        if not messages or messages[-1].get("role") != "user":
            messages.append({"role": "user", "content": ""})
        messages[-1]["content"] = f"{messages[-1].get('content', '')}\n\n{suffix}"
        return messages
    return f"{prompt}\n\n{suffix}"


def _set_sdpo_teacher_prompt(sample: Sample, additions: list[str]) -> None:
    sample.teacher_prompt = (
        _render_sdpo_teacher_prompt(sample, additions) if additions else copy.deepcopy(sample.prompt)
    )
    sample.opd_sample_mask = bool(additions)
    sample.teacher_tokens = None
    sample.teacher_prompt_length = None


def _is_successful_reward(reward: Any, threshold: float) -> bool:
    value = reward.get("score", reward.get("reward")) if isinstance(reward, dict) else reward
    try:
        return float(value) >= threshold
    except (TypeError, ValueError):
        return False


def _is_format_or_truncation_error(sample: Sample, reward: Any) -> bool:
    if isinstance(reward, dict) and reward.get("format_error"):
        return True
    status = getattr(sample, "status", None)
    if status is not None:
        return str(getattr(status, "value", status)).casefold() == "truncated"
    return False


def _sdpo_group_key(sample: Sample, position: int) -> Any:
    if sample.group_index is not None:
        return ("group", sample.group_index)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    uid = metadata.get("uid")
    return ("uid", uid) if uid is not None else ("singleton", position)


def _prepare_sdpo_teacher_prompts(group: list[Sample], rewards: list[Any], threshold: float) -> None:
    if len(group) != len(rewards):
        raise ValueError(f"feedback requires one reward per sample: {len(group)} != {len(rewards)}")
    by_group: dict[Any, list[Sample]] = defaultdict(list)
    for position, sample in enumerate(group):
        by_group[_sdpo_group_key(sample, position)].append(sample)
    reward_by_id = {id(sample): reward for sample, reward in zip(group, rewards, strict=True)}
    successful = {
        key: [sample for sample in samples if _is_successful_reward(reward_by_id[id(sample)], threshold)]
        for key, samples in by_group.items()
    }
    for key, samples in by_group.items():
        for sample in samples:
            reward = reward_by_id[id(sample)]
            is_success = _is_successful_reward(reward, threshold)
            peer = next((candidate for candidate in successful[key] if candidate is not sample), None)
            additions = []
            if is_success:
                source = peer or sample
                additions.append(f"<successful_attempt>\n{source.response}\n</successful_attempt>")
            elif peer is not None:
                additions.append(f"<successful_attempt>\n{peer.response}\n</successful_attempt>")
            elif _is_format_or_truncation_error(sample, reward):
                feedback = EnvironmentFeedback.feedback_text(sample, reward)
                if feedback:
                    additions.append(f"<feedback>\n{feedback}\n</feedback>")
            _set_sdpo_teacher_prompt(sample, additions)


class SDPOFeedback(EnvironmentFeedback):
    is_sdpo_feedback = True

    def prepare_teacher_prompts(self, group: list[Sample], rewards: list[Any]) -> None:
        for sample in group:
            validate_sdpo_text_only(sample)
        _prepare_sdpo_teacher_prompts(group, rewards, self.success_reward_threshold)
        for sample in group:
            _clear_teacher_payload(sample)
        for sample in group:
            if int(sample.response_length or 0) > 0 and not _has_sdpo_teacher_prompt(sample.teacher_prompt):
                raise ValueError(
                    f"SDPO requires a teacher prompt for every non-empty response; sample_index={sample.index}"
                )

    @staticmethod
    def create_opsd_worker(args: Any) -> OpsdWorker:
        return OpsdWorker(is_opsd=True)

    @classmethod
    def validate_launch_args(cls, args: Any) -> None:
        if args.opd_type != "sglang":
            raise ValueError("SDPO prompt routing requires --opd-type=sglang.")
        if int(getattr(args, "pipeline_model_parallel_size", 1)) != 1:
            raise ValueError("SDPO prompt routing does not support pipeline parallelism.")
        if getattr(args, "enable_mtp_training", False):
            raise ValueError("SDPO prompt routing does not support MTP auxiliary loss.")
        if args.opd_token_selection != SDPO_TOKEN_SELECTION:
            raise ValueError("SDPO prompt routing only supports --opd-token-selection=student_topk.")
        if args.opd_kl_type not in ("forward_kl", "reverse_kl", "jsd"):
            raise ValueError("SDPO prompt routing supports only forward_kl, reverse_kl, or jsd.")
        if getattr(args, "opd_norm_mode", "tail") == "trunc":
            raise ValueError("SDPO prompt routing requires --opd-norm-mode=tail or norm.")
        if not getattr(args, "calculate_per_token_loss", False):
            raise ValueError("SDPO prompt routing requires --calculate-per-token-loss.")
        if getattr(args, "multimodal_keys", None) or any(
            getattr(args, field_name, None) is not None
            for field_name in ("opd_teacher_image_key", "opd_teacher_video_key", "opd_teacher_audio_key")
        ):
            raise ValueError("SDPO prompt routing only supports text inputs; multimodal fields are not supported.")
        if not getattr(args, "group_rm", False):
            raise ValueError("SDPO requires --group-rm to build privileged teacher prompts from group outcomes.")
        opd_kl_coef = float(getattr(args, "opd_kl_coef", 0.0) or 0.0)
        opd_loss_coef = float(getattr(args, "opd_loss_coef", 0.0) or 0.0)
        if opd_kl_coef != 0.0 or opd_loss_coef <= 0.0:
            raise ValueError(
                "SDPO loss and prompt-routing teacher mode require --opd-kl-coef=0 and a positive --opd-loss-coef."
            )

    def extra_transfer_schema(self) -> list[str]:
        return [OPD_SAMPLE_MASK]

    def produce_extra_transfer(self, samples: list[Sample], train_data: dict) -> None:
        train_data[OPD_SAMPLE_MASK] = [bool(sample.opd_sample_mask) for sample in samples]

    def check_student_topk_ids(self, sample: Sample, top_k: int) -> None:
        validate_sdpo_student_topk_ids(
            token_ids=sample.student_topk_token_ids,
            response_rows=int(sample.response_length or 0),
            top_k=top_k,
            sample_index=int(sample.index) if sample.index is not None else -1,
        )

    def check_transfer_channels(self, sample: Sample, channels: dict, top_k: int) -> None:
        validate_sdpo_topk_payload(
            token_ids=channels.get(TopkWorker.TRANSFER_TOKEN_IDS),
            teacher_log_probs=channels.get(TopkWorker.TRANSFER_TEACHER_LOG_PROBS),
            response_rows=int(sample.response_length or 0),
            top_k=top_k,
            sample_index=int(sample.index) if sample.index is not None else -1,
        )


class GoldenAnswerSDPOFeedback(SDPOFeedback):
    """Static text datasets scored against a golden answer (MCQ, tool calls)."""


class CodeSDPOFeedback(SDPOFeedback):
    def record_sample_feedback(self, sample: Sample, reward: Any) -> None:
        raise NotImplementedError("CodeSDPOFeedback is a placeholder; the code-domain reward is not wired up yet")

    def prepare_teacher_prompts(self, group: list[Sample], rewards: list[Any]) -> None:
        raise NotImplementedError("CodeSDPOFeedback is a placeholder; the code-domain reward is not wired up yet")


__all__ = [
    "SDPOFeedback",
    "GoldenAnswerSDPOFeedback",
    "CodeSDPOFeedback",
]
