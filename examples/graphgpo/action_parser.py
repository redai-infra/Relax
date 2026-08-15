# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reference-compatible action parsing helpers for the ALFWorld recipe."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


_CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_ACTION_PATTERN = re.compile(r"<action>(.*?)</action>", flags=re.DOTALL)


@dataclass(frozen=True)
class ParsedAction:
    """The environment action extracted from one model response."""

    action: str
    is_valid: bool
    source: Literal["tag", "fallback"]


def _contains_chinese(text: str) -> bool:
    return _CHINESE_PATTERN.search(text) is not None


def parse_action(response: str, *, fallback_chars: int = 30) -> ParsedAction:
    """Extract the first ``<action>`` block after lowercasing the response.

    This intentionally mirrors the frozen GraphGPO reference parser instead of
    adding stricter recipe-specific checks.  A missing tag falls back to the
    last ``fallback_chars`` characters for debugging, but that result is always
    marked invalid.  Duplicate tags and actions outside the admissible set are
    not rejected.  The reference validity flag additionally requires lowercase
    ``<think>`` tags in the original response and rejects Chinese text anywhere
    in that response.
    """

    if not isinstance(response, str):
        raise TypeError("response must be a string")
    if isinstance(fallback_chars, bool) or not isinstance(fallback_chars, int) or fallback_chars <= 0:
        raise ValueError("fallback_chars must be a positive integer")

    lowered_response = response.lower()
    match = _ACTION_PATTERN.search(lowered_response)
    if match is None:
        action = lowered_response[-fallback_chars:]
        source: Literal["tag", "fallback"] = "fallback"
        is_valid = False
    else:
        action = match.group(1).strip()
        source = "tag"
        is_valid = True

    if "<think>" not in response or "</think>" not in response:
        is_valid = False
    if _contains_chinese(response):
        is_valid = False
    return ParsedAction(action=action, is_valid=is_valid, source=source)
