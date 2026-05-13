# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from typing import Any


TORCH_MEMORY_SAVER_IMPORT_ERROR: Exception | None = None

try:
    from torch_memory_saver import torch_memory_saver as torch_memory_saver

    TORCH_MEMORY_SAVER_AVAILABLE = True
except Exception as exc:  # pragma: no cover - exercised in ROCm runtime
    TORCH_MEMORY_SAVER_IMPORT_ERROR = exc
    TORCH_MEMORY_SAVER_AVAILABLE = False

    class _TorchMemorySaverStub:
        _impl = None

        @property
        def memory_margin_bytes(self) -> int:
            return 0

        @memory_margin_bytes.setter
        def memory_margin_bytes(self, _: int) -> None:
            raise NotImplementedError(
                "torch_memory_saver is unavailable in the current runtime. "
                f"Import error: {TORCH_MEMORY_SAVER_IMPORT_ERROR}"
            )

        def get_cpu_backup(self, _: Any) -> None:
            return None

        def pause(self) -> None:
            return None

        def resume(self) -> None:
            return None

        def disable(self) -> None:
            return None

    torch_memory_saver = _TorchMemorySaverStub()
