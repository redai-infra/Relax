# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from types import TracebackType
from typing import Any, Protocol

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class TelemetryBackend(Protocol):
    def mark(
        self,
        name: str,
        *,
        step: int | None = None,
        role: str | None = None,
        fields: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        """Record a telemetry event."""


_backend: TelemetryBackend | None = None
_hook_import_attempted = False


def _try_import_hook_once() -> None:
    global _hook_import_attempted
    if _backend is not None or _hook_import_attempted:
        return
    _hook_import_attempted = True
    hook = os.environ.get("RELAX_TELEMETRY_HOOK")
    if not hook:
        return
    try:
        importlib.import_module(hook)
    except Exception as e:
        logger.warning(f"Telemetry hook {hook!r} failed to import: {type(e).__name__}: {e}")


def _megatron_main_rank() -> bool | None:
    try:
        from megatron.core import mpu

        if hasattr(mpu, "model_parallel_is_initialized") and not mpu.model_parallel_is_initialized():
            return None
        return (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        )
    except Exception:
        return None


def _torch_global_rank_zero() -> bool | None:
    try:
        from torch import distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return None
        return dist.get_rank() == 0
    except Exception:
        return None


def _should_emit() -> bool:
    megatron_rank = _megatron_main_rank()
    if megatron_rank is not None:
        return megatron_rank
    torch_rank = _torch_global_rank_zero()
    if torch_rank is not None:
        return torch_rank
    return True


def register_backend(backend: TelemetryBackend | None) -> None:
    """Register a process-local telemetry backend.

    Passing ``None`` disables telemetry. The default state is disabled, so
    Relax can call these helpers unconditionally in open-source builds.
    """
    global _backend
    _backend = backend


def mark(
    name: str,
    *,
    step: int | None = None,
    role: str | None = None,
    fields: Mapping[str, Any] | None = None,
    **extra: Any,
) -> None:
    """Record a telemetry event if a backend has been registered."""
    logger.debug(f"telemetry mark: name={name}, step={step}, role={role}, fields={fields}, extra={extra}")
    _try_import_hook_once()
    backend = _backend
    if backend is None:
        return
    if not _should_emit():
        return

    try:
        backend.mark(name, step=step, role=role, fields=fields, **extra)
    except Exception:
        return


class TelemetrySpan:
    def __init__(
        self,
        begin_name: str,
        end_name: str,
        *,
        step: int | None = None,
        role: str | None = None,
        fields: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        self._begin_name = begin_name
        self._end_name = end_name
        self._step = step
        self._role = role
        self._fields = fields
        self._begin_extra = dict(extra)
        self._end_extra: dict[str, Any] = {}

    def update(self, **extra: Any) -> None:
        """Add fields that should be emitted on the end event."""
        self._end_extra.update(extra)

    def __enter__(self) -> "TelemetrySpan":
        mark(
            self._begin_name,
            step=self._step,
            role=self._role,
            fields=self._fields,
            **self._begin_extra,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        extra = dict(self._begin_extra)
        extra.update(self._end_extra)
        if exc_type is not None:
            extra.setdefault("status", "failed")
            extra.setdefault("error_type", exc_type.__name__)
            if exc is not None:
                extra.setdefault("error_message", str(exc))
        mark(self._end_name, step=self._step, role=self._role, fields=self._fields, **extra)
        return False


def span(
    name: str,
    *,
    step: int | None = None,
    role: str | None = None,
    fields: Mapping[str, Any] | None = None,
    **extra: Any,
) -> TelemetrySpan:
    return TelemetrySpan(
        f"{name}_begin",
        f"{name}_end",
        step=step,
        role=role,
        fields=fields,
        **extra,
    )


def mark_start(**extra: Any) -> None:
    fields = extra.get("fields")
    if isinstance(fields, Mapping):
        fields = dict(fields)
        if fields.get("model_name_or_path") is None:
            fields["model_name_or_path"] = (
                fields.get("hf_checkpoint")
                or fields.get("model_name_or_path")
                or fields.get("sglang_hf_checkpoint")
                or fields.get("model_name")
            )

        if fields.get("train_type") is None:
            loss_type = str(fields.get("loss_type") or "").lower()
            fields["train_type"] = "SFT" if "sft" in loss_type else "RL"

        if fields.get("train_method") is None:
            fields["train_method"] = "full"

        if fields.get("max_seq_length") is None:
            fields["max_seq_length"] = fields.get("max_tokens_per_gpu")

        extra = dict(extra)
        extra["fields"] = fields

    mark("start", **extra)


def mark_end(**extra: Any) -> None:
    mark("end", **extra)


def mark_setup_begin(**extra: Any) -> None:
    mark("setup_begin", **extra)


def mark_setup_end(**extra: Any) -> None:
    mark("setup_end", **extra)


def mark_checkpoint_load_begin(**extra: Any) -> None:
    mark("checkpoint_load_begin", **extra)


def mark_checkpoint_load_end(**extra: Any) -> None:
    mark("checkpoint_load_end", **extra)


def mark_step_begin(step: int, *, role: str | None = None, **extra: Any) -> None:
    mark("step_begin", step=step, role=role, **extra)


def mark_step_end(step: int, *, role: str | None = None, **extra: Any) -> None:
    mark("step_end", step=step, role=role, **extra)


def mark_save_begin(
    step: int | None = None,
    *,
    role: str | None = None,
    **extra: Any,
) -> None:
    mark("save_begin", step=step, role=role, **extra)


def mark_save_end(
    step: int | None = None,
    *,
    role: str | None = None,
    **extra: Any,
) -> None:
    mark("save_end", step=step, role=role, **extra)


def mark_eval_begin(
    step: int | None = None,
    *,
    role: str | None = None,
    **extra: Any,
) -> None:
    mark("eval_begin", step=step, role=role, **extra)


def mark_eval_end(
    step: int | None = None,
    *,
    role: str | None = None,
    **extra: Any,
) -> None:
    mark("eval_end", step=step, role=role, **extra)


def setup(**extra: Any) -> TelemetrySpan:
    return span("setup", **extra)


def checkpoint_load(**extra: Any) -> TelemetrySpan:
    return span("checkpoint_load", **extra)


def step(step: int, *, role: str | None = None, **extra: Any) -> TelemetrySpan:
    return span("step", step=step, role=role, **extra)


def save(
    step: int | None = None,
    *,
    role: str | None = None,
    **extra: Any,
) -> TelemetrySpan:
    return span("save", step=step, role=role, **extra)


def evaluation(
    step: int | None = None,
    *,
    role: str | None = None,
    **extra: Any,
) -> TelemetrySpan:
    return span("eval", step=step, role=role, **extra)
