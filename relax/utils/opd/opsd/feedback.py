# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""OPSD feedback: the dataset-provided teacher prompt is the privilege.

The raw dataset column is rendered at ingestion and surfaced as
``metadata["opd_teacher_prompt"]``; this class owns the policy of assigning it
to ``sample.teacher_prompt``. Samples without the field fall back to the
student prompt via ``OpsdWorker``, matching plain OPD.
"""

from __future__ import annotations

from typing import Any

from relax.utils.opd.feedback import EnvironmentFeedback
from relax.utils.types import Sample


class OPSDFeedback(EnvironmentFeedback):
    def __init__(self, teacher_prompt_key: str | None = None, success_reward_threshold: float = 1.0) -> None:
        if not teacher_prompt_key:
            raise ValueError("OPSDFeedback requires {'teacher_prompt_key': ...} in --opd-feedback-kwargs.")
        super().__init__(teacher_prompt_key=teacher_prompt_key, success_reward_threshold=success_reward_threshold)

    def prepare_teacher_prompts(self, group: list[Sample], rewards: list[Any]) -> None:
        for sample in group:
            sample.teacher_prompt = None
            sample.opd_sample_mask = None
            privileged = sample.metadata.get("opd_teacher_prompt") if isinstance(sample.metadata, dict) else None
            if privileged is not None:
                sample.teacher_prompt = privileged

    @classmethod
    def validate_launch_args(cls, args: Any) -> None:
        if args.opd_type != "sglang":
            raise ValueError(
                "OPSD prompt routing currently only supports --opd-type=sglang "
                f"(got --opd-type={args.opd_type}). The megatron teacher path does not "
                "yet rebuild a teacher-side data_iterator from teacher_tokens."
            )


__all__ = [
    "OPSDFeedback",
]
