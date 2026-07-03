# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import sys
import types

from relax.utils import telemetry


class RecordingBackend:
    def __init__(self) -> None:
        self.events = []

    def mark(self, name, *, step=None, role=None, fields=None, **extra):
        self.events.append(
            {
                "name": name,
                "step": step,
                "role": role,
                "fields": dict(fields or {}),
                "extra": dict(extra),
            }
        )


def test_mark_helpers_forward_structured_fields() -> None:
    backend = RecordingBackend()
    telemetry.register_backend(backend)
    try:
        telemetry.mark_step_begin(3, role="actor", mfu=0.5)
        telemetry.mark_step_end(3, role="actor", flops=123)
    finally:
        telemetry.register_backend(None)

    assert backend.events == [
        {
            "name": "step_begin",
            "step": 3,
            "role": "actor",
            "fields": {},
            "extra": {"mfu": 0.5},
        },
        {
            "name": "step_end",
            "step": 3,
            "role": "actor",
            "fields": {},
            "extra": {"flops": 123},
        },
    ]


def test_span_records_end_metrics_and_failure_status() -> None:
    backend = RecordingBackend()
    telemetry.register_backend(backend)
    try:
        try:
            with telemetry.step(7, role="critic", tokens=10) as span:
                span.update(mfu=0.6)
                raise RuntimeError("boom")
        except RuntimeError:
            pass
    finally:
        telemetry.register_backend(None)

    assert backend.events == [
        {
            "name": "step_begin",
            "step": 7,
            "role": "critic",
            "fields": {},
            "extra": {"tokens": 10},
        },
        {
            "name": "step_end",
            "step": 7,
            "role": "critic",
            "fields": {},
            "extra": {
                "tokens": 10,
                "mfu": 0.6,
                "status": "failed",
                "error_type": "RuntimeError",
                "error_message": "boom",
            },
        },
    ]


def test_mark_lazily_imports_hook_when_backend_is_unset(tmp_path, monkeypatch) -> None:
    hook_path = tmp_path / "lazy_telemetry_hook.py"
    hook_path.write_text(
        """
from relax.utils import telemetry


class Backend:
    def __init__(self):
        self.events = []

    def mark(self, name, *, step=None, role=None, fields=None, **extra):
        self.events.append({"name": name, "step": step, "role": role})


backend = Backend()
telemetry.register_backend(backend)
""",
        encoding="utf-8",
    )

    telemetry.register_backend(None)
    if hasattr(telemetry, "_hook_import_attempted"):
        monkeypatch.setattr(telemetry, "_hook_import_attempted", False)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("RELAX_TELEMETRY_HOOK", "lazy_telemetry_hook")
    monkeypatch.delitem(sys.modules, "lazy_telemetry_hook", raising=False)

    try:
        telemetry.mark_start()
        hook_module = sys.modules["lazy_telemetry_hook"]
        assert [event["name"] for event in hook_module.backend.events] == ["start"]
    finally:
        telemetry.register_backend(None)


class FakeMpu:
    def __init__(self, *, initialized: bool, dp_rank: int, tp_rank: int, pp_rank: int, pp_world_size: int) -> None:
        self.initialized = initialized
        self.dp_rank = dp_rank
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.pp_world_size = pp_world_size

    def model_parallel_is_initialized(self) -> bool:
        return self.initialized

    def get_data_parallel_rank(self, with_context_parallel=True):
        return self.dp_rank

    def get_tensor_model_parallel_rank(self):
        return self.tp_rank

    def get_pipeline_model_parallel_rank(self):
        return self.pp_rank

    def get_pipeline_model_parallel_world_size(self):
        return self.pp_world_size


class FakeDist:
    def __init__(self, *, available: bool, initialized: bool, rank: int) -> None:
        self.available = available
        self.initialized = initialized
        self.rank = rank

    def is_available(self) -> bool:
        return self.available

    def is_initialized(self) -> bool:
        return self.initialized

    def get_rank(self) -> int:
        return self.rank


def _install_fake_mpu(monkeypatch, mpu) -> None:
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    core.mpu = mpu
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)


def _install_fake_torch_dist(monkeypatch, dist) -> None:
    torch = types.ModuleType("torch")
    torch.distributed = dist
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)


def test_mark_allows_non_megatron_processes(monkeypatch) -> None:
    backend = RecordingBackend()
    telemetry.register_backend(backend)
    try:
        monkeypatch.delitem(sys.modules, "megatron", raising=False)
        monkeypatch.delitem(sys.modules, "megatron.core", raising=False)

        telemetry.mark_start()
    finally:
        telemetry.register_backend(None)

    assert [event["name"] for event in backend.events] == ["start"]


def test_mark_skips_non_main_megatron_rank(monkeypatch) -> None:
    backend = RecordingBackend()
    telemetry.register_backend(backend)
    try:
        _install_fake_mpu(
            monkeypatch,
            FakeMpu(initialized=True, dp_rank=1, tp_rank=0, pp_rank=1, pp_world_size=2),
        )

        telemetry.mark_step_begin(1, role="actor")
    finally:
        telemetry.register_backend(None)

    assert backend.events == []


def test_mark_allows_megatron_main_rank(monkeypatch) -> None:
    backend = RecordingBackend()
    telemetry.register_backend(backend)
    try:
        _install_fake_mpu(
            monkeypatch,
            FakeMpu(initialized=True, dp_rank=0, tp_rank=0, pp_rank=1, pp_world_size=2),
        )

        telemetry.mark_step_begin(1, role="actor")
    finally:
        telemetry.register_backend(None)

    assert [event["name"] for event in backend.events] == ["step_begin"]


def test_mark_falls_back_to_torch_dist_rank_when_megatron_is_not_initialized(monkeypatch) -> None:
    backend = RecordingBackend()
    telemetry.register_backend(backend)
    try:
        _install_fake_mpu(
            monkeypatch,
            FakeMpu(initialized=False, dp_rank=0, tp_rank=0, pp_rank=0, pp_world_size=1),
        )
        _install_fake_torch_dist(monkeypatch, FakeDist(available=True, initialized=True, rank=1))

        telemetry.mark_start()
    finally:
        telemetry.register_backend(None)

    assert backend.events == []


def test_mark_allows_torch_dist_global_rank_zero(monkeypatch) -> None:
    backend = RecordingBackend()
    telemetry.register_backend(backend)
    try:
        monkeypatch.delitem(sys.modules, "megatron", raising=False)
        monkeypatch.delitem(sys.modules, "megatron.core", raising=False)
        _install_fake_torch_dist(monkeypatch, FakeDist(available=True, initialized=True, rank=0))

        telemetry.mark_start()
    finally:
        telemetry.register_backend(None)

    assert [event["name"] for event in backend.events] == ["start"]


def test_mark_skips_torch_dist_nonzero_global_rank(monkeypatch) -> None:
    backend = RecordingBackend()
    telemetry.register_backend(backend)
    try:
        monkeypatch.delitem(sys.modules, "megatron", raising=False)
        monkeypatch.delitem(sys.modules, "megatron.core", raising=False)
        _install_fake_torch_dist(monkeypatch, FakeDist(available=True, initialized=True, rank=2))

        telemetry.mark_start()
    finally:
        telemetry.register_backend(None)

    assert backend.events == []


def test_mark_defaults_to_emit_when_torch_dist_is_not_initialized(monkeypatch) -> None:
    backend = RecordingBackend()
    telemetry.register_backend(backend)
    try:
        monkeypatch.delitem(sys.modules, "megatron", raising=False)
        monkeypatch.delitem(sys.modules, "megatron.core", raising=False)
        _install_fake_torch_dist(monkeypatch, FakeDist(available=True, initialized=False, rank=2))

        telemetry.mark_start()
    finally:
        telemetry.register_backend(None)

    assert [event["name"] for event in backend.events] == ["start"]
