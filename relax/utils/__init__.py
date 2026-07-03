# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import logging
import os
from typing import Any


def try_import_telemetry_hook(logger: Any = None) -> None:
    """Import the optional telemetry hook without affecting training."""
    hook = os.environ.get("RELAX_TELEMETRY_HOOK")
    if not hook:
        return
    try:
        importlib.import_module(hook)
    except Exception as e:
        logger = logger or logging.getLogger(__name__)
        logger.warning(f"Telemetry hook {hook!r} failed to import: {type(e).__name__}: {e}")
