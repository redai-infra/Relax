# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Compatibility guards for the MemAgent trajectory contract."""

from __future__ import annotations

from typing import Any


def require_strict_alignment(args: Any) -> None:
    """Preserve rejection of the retired false-valued compatibility option.

    Alignment validation is always strict and unconditional. The bundled
    recipes therefore no longer advertise a boolean switch, while older
    external configs that explicitly requested an unsupported relaxed mode
    continue to fail instead of silently changing behavior.
    """
    if not getattr(args, "mem_agent_strict_alignment", True):
        raise ValueError("MemAgent training requires mem_agent_strict_alignment=true.")
