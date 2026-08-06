# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Node-group affinity helpers for the training entrypoint."""

import os

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _autoscaler_enabled(config_path: str) -> bool:
    """Return whether the autoscaler config is enabled."""
    try:
        from relax.utils.autoscaler.config import AutoscalerConfig

        return bool(AutoscalerConfig.from_yaml(config_path).enabled)
    except Exception as e:
        logger.warning(
            f"Could not read autoscaler config '{config_path}' to decide baseline pinning "
            f"({e}); treating as not enabled and leaving RELAX_INITIAL_NODE_GROUP unset."
        )
        return False


def _maybe_pin_baseline_to_stable(args) -> None:
    """Pin an elastic job's baseline roles to the stable worker group."""
    config_path = getattr(args, "autoscaler_config", None)
    if config_path and _autoscaler_enabled(config_path) and not os.environ.get("RELAX_INITIAL_NODE_GROUP", "").strip():
        os.environ["RELAX_INITIAL_NODE_GROUP"] = "stable"
