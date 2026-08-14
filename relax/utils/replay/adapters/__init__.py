# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Replay adapters.

Each adapter recomputes one pipeline stage from bundle inputs (``bundle.index``
+ ``bundle.tensors``) and upstream recomputed outputs (``ctx``), then compares
against the bundle's expected outputs. Importing this package registers every
adapter into :mod:`relax.utils.replay.stages`.
"""

from __future__ import annotations

from relax.utils.replay.adapters import advantage, loss, reward, sample  # noqa: F401


def register_all() -> None:
    """Ensure every adapter module is imported (and thus registered)."""
    # Importing this package at module import time already registers adapters;
    # this function exists as an explicit, documented entry point.
    return None
