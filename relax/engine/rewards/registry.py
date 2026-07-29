# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Format-aware reward function registry.

Provides a decorator-based registration mechanism so that new reward
functions can be added with a single line without modifying any routing
branches.
"""

from typing import Callable

REWARD_REGISTRY: dict[str, Callable] = {}


def register_reward(name: str):
    """Decorator that registers a reward function under *name*.

    Usage::

        @register_reward("biology")
        def biology_reward(response, label, metadata=None):
            ...
    """

    def wrapper(func: Callable):
        REWARD_REGISTRY[name] = func
        return func

    return wrapper


def get_reward(name: str) -> Callable | None:
    """Look up a registered reward function by name.

    Returns ``None`` when no function is registered for *name*.
    """
    return REWARD_REGISTRY.get(name)


def list_registered() -> list[str]:
    """Return the names of all registered reward types."""
    return sorted(REWARD_REGISTRY.keys())
