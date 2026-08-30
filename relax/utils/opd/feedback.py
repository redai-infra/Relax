# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Feedback strategies that own every OPD-flavored algorithm difference.

``EnvironmentFeedback`` is the polymorphism point of the OPD algorithm family
(OPD / MOPD / OPSD / SDPO). Every rollout runs the same shared two-step path
-- record what the environment said, then decide what privileged prompt the
teacher sees -- bound dynamically to a subclass. The base-class defaults
reproduce plain OPD (empty privilege), so a subclass only overrides the hooks
where its behavior differs. Hooks raise to hard-fail and return to degrade.

The constructor parameter schema is declared once on the interface; ``args``
carries it verbatim via ``--opd-feedback-kwargs`` and the selected subclass
binds it at construction time, consuming only the fields it uses.
"""

from __future__ import annotations

import importlib
from typing import Any

from relax.utils.opd.opd_opsd_worker import OpsdWorker
from relax.utils.types import Sample


class EnvironmentFeedback:
    def __init__(self, teacher_prompt_key: str | None = None, success_reward_threshold: float = 1.0) -> None:
        self.teacher_prompt_key = teacher_prompt_key
        self.success_reward_threshold = success_reward_threshold

    @staticmethod
    def record(sample: Sample, text: str | None) -> None:
        if text:
            sample.metadata.setdefault("env_feedback", []).append(str(text))

    def record_sample_feedback(self, sample: Sample, reward: Any) -> None:
        feedback = self._reward_feedback(reward)
        if feedback:
            self.record(sample, feedback)

    def prepare_teacher_prompts(self, group: list[Sample], rewards: list[Any]) -> None:
        """Assign the privileged teacher prompt for every sample in the
        group."""
        for sample in group:
            sample.teacher_prompt = None
            sample.opd_sample_mask = None

    @staticmethod
    def create_opsd_worker(args: Any) -> OpsdWorker:
        return OpsdWorker.from_args(args)

    @classmethod
    def validate_launch_args(cls, args: Any) -> None:
        return None

    def extra_transfer_schema(self) -> list[str]:
        return []

    def produce_extra_transfer(self, samples: list[Sample], train_data: dict) -> None:
        return None

    def check_student_topk_ids(self, sample: Sample, top_k: int) -> None:
        return None

    def check_transfer_channels(self, sample: Sample, channels: dict, top_k: int) -> None:
        return None

    @staticmethod
    def _reward_feedback(reward: Any) -> str | None:
        if isinstance(reward, dict):
            for key in ("feedback", "error", "feedback_raw"):
                value = reward.get(key)
                if value:
                    if isinstance(value, str):
                        return value if value.strip() else None
                    return str(value)
        return None

    @staticmethod
    def feedback_text(sample: Sample, reward: Any) -> str:
        reward_feedback = EnvironmentFeedback._reward_feedback(reward)
        if reward_feedback:
            return reward_feedback
        values = sample.metadata.get("env_feedback", []) if isinstance(sample.metadata, dict) else []
        return "\n".join(str(value) for value in values if value)


class OPDFeedback(EnvironmentFeedback):
    """Plain OPD/MOPD: the teacher sees exactly what the student saw."""


def load_feedback_class(path: str | None) -> type[EnvironmentFeedback]:
    if not path:
        return OPDFeedback
    module_name, separator, class_name = path.rpartition(".")
    if not separator:
        raise ValueError(f"Invalid feedback class path: {path!r}")
    cls = getattr(importlib.import_module(module_name), class_name, None)
    if not isinstance(cls, type) or not issubclass(cls, EnvironmentFeedback):
        raise TypeError(f"{path!r} must name an EnvironmentFeedback subclass")
    return cls


def load_feedback(path: str | None, kwargs: dict[str, Any] | None) -> EnvironmentFeedback:
    cls = load_feedback_class(path)
    try:
        return cls(**(kwargs or {}))
    except TypeError as e:
        raise TypeError(f"Invalid --opd-feedback-kwargs for {cls.__name__}: {kwargs or {}} ({e})") from e


__all__ = [
    "EnvironmentFeedback",
    "OPDFeedback",
    "load_feedback",
    "load_feedback_class",
]
