# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Agentic rollout package."""

from typing import Any

from relax.utils.log_style import format_badge


AGENTIC_CHAT_API_SERVICE_NAME = "agentic_chat_api"
AGENTIC_CHAT_API_ROUTE_PREFIX = "/agentic_api"

_ANSI_RESET = "\033[0m"
_ANSI_EVENT = "\033[1;38;5;213m"


def format_agentic_event(component: str, event: str, **fields: Any) -> str:
    tokens = [format_badge(f"AGENTIC {component}"), f"{_ANSI_EVENT}event={event}{_ANSI_RESET}"]
    for key, value in fields.items():
        if value is None or key == "pool_target":
            continue
        if key == "pool" and "pool_target" in fields:
            value = f"{value}/{fields['pool_target']}"
        tokens.append(f"{key}={value}")
    return " ".join(tokens)
