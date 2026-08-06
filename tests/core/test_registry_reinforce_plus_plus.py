# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture()
def registry_module(monkeypatch):
    component_names = {
        "actor": "Actor",
        "actor_fwd": "ActorFwd",
        "advantages": "Advantages",
        "critic": "Critic",
        "rollout": "Rollout",
        "sft": "SFT",
    }
    classes = {}
    for module_name, class_name in component_names.items():
        module = ModuleType(f"relax.components.{module_name}")
        component_class = type(class_name, (), {})
        setattr(module, class_name, component_class)
        classes[class_name] = component_class
        monkeypatch.setitem(sys.modules, module.__name__, module)

    original_registry = sys.modules.get("relax.core.registry")
    sys.modules.pop("relax.core.registry", None)
    module = importlib.import_module("relax.core.registry")
    yield module, classes
    sys.modules.pop("relax.core.registry", None)
    # Restore the pre-test registry module so later tests in the same process
    # keep importing a single module instance. Popping without restoring made
    # ``process_role`` (old instance) return a ``ROLES_COLOCATE`` that fails
    # ``is`` comparison against the re-imported one in test_registry_sft.py /
    # test_registry_rloo.py.
    if original_registry is not None:
        sys.modules["relax.core.registry"] = original_registry


@pytest.mark.parametrize("estimator", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_reinforce_plus_plus_variants_are_registered_without_critic(registry_module, estimator):
    registry, classes = registry_module

    topology = registry.ALGOS[estimator]

    assert topology[registry.ROLES.actor] is classes["Actor"]
    assert topology[registry.ROLES.rollout] is classes["Rollout"]
    assert topology[registry.ROLES.advantages] is classes["Advantages"]
    assert topology[registry.ROLES.reference] is classes["ActorFwd"]
    assert topology[registry.ROLES.actor_fwd] is classes["ActorFwd"]
    assert registry.ROLES.critic not in topology


@pytest.mark.parametrize("estimator", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_reinforce_plus_plus_variants_use_sync_colocate_roles(registry_module, estimator):
    registry, _ = registry_module
    config = SimpleNamespace(
        debug_rollout_only=False,
        debug_train_only=False,
        loss_type="policy_loss",
        advantage_estimator=estimator,
        fully_async=False,
        hybrid=False,
    )

    assert registry.process_role(config) is registry.ROLES_COLOCATE
