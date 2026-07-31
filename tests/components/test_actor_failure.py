# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Failure-propagation coverage for the actor background loop."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def test_actor_background_failure_propagates_with_health_check_enabled():
    from relax.components.actor import Actor

    actor_class = Actor.func_or_class
    actor = actor_class.__new__(actor_class)
    actor.config = SimpleNamespace(
        num_rollout=1,
        fully_async=False,
        colocate=False,
        debug_train_only=False,
        use_health_check=True,
    )
    actor.step = 0
    actor._lock = threading.RLock()
    actor._stop_event = threading.Event()
    actor._logger_instance = MagicMock()
    actor.healthy = MagicMock()
    actor.healthy.report_error.remote = MagicMock()
    actor._execute_training = MagicMock(side_effect=ValueError("invalid optimizer window"))

    with pytest.raises(ValueError, match="invalid optimizer window"):
        actor._background_run()

    actor.healthy.report_error.remote.assert_called_once()
