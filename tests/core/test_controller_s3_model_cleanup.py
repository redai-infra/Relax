# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import transfer_queue


if not hasattr(transfer_queue, "StreamingTokenBudgetSampler"):
    pytest.skip(
        "controller cleanup tests require a TransferQueue build with StreamingTokenBudgetSampler",
        allow_module_level=True,
    )


_MISSING = object()
_REGISTRY_COMPONENTS = {
    "relax.components.actor": "Actor",
    "relax.components.actor_fwd": "ActorFwd",
    "relax.components.advantages": "Advantages",
    "relax.components.critic": "Critic",
    "relax.components.rollout": "Rollout",
    "relax.components.sft": "SFT",
}
_DCS_SERVICE_MODULES = (
    "relax.distributed.checkpoint_service",
    "relax.distributed.checkpoint_service.coordinator",
    "relax.distributed.checkpoint_service.coordinator.service",
)


def _load_controller_with_stubbed_registry_components():
    saved_modules = {
        name: sys.modules.get(name, _MISSING)
        for name in [*_REGISTRY_COMPONENTS, *_DCS_SERVICE_MODULES, "relax.core.registry"]
    }
    core_module = sys.modules.get("relax.core")
    registry_attr_was_set = core_module is not None and hasattr(core_module, "registry")
    original_registry_attr = getattr(core_module, "registry", None) if registry_attr_was_set else None
    distributed_module = sys.modules.get("relax.distributed")
    checkpoint_service_attr_was_set = distributed_module is not None and hasattr(
        distributed_module, "checkpoint_service"
    )
    original_checkpoint_service_attr = (
        getattr(distributed_module, "checkpoint_service", None) if checkpoint_service_attr_was_set else None
    )
    components_module = sys.modules.get("relax.components")
    saved_component_attrs = {}
    if components_module is not None:
        for module_name in _REGISTRY_COMPONENTS:
            attr_name = module_name.rsplit(".", 1)[1]
            saved_component_attrs[attr_name] = (
                hasattr(components_module, attr_name),
                getattr(components_module, attr_name, None),
            )

    module_name = "_test_controller_s3_model_cleanup_controller"
    try:
        for component_module_name, class_name in _REGISTRY_COMPONENTS.items():
            module = ModuleType(component_module_name)
            setattr(module, class_name, type(class_name, (), {}))
            sys.modules[component_module_name] = module
        checkpoint_service_module = ModuleType("relax.distributed.checkpoint_service")
        checkpoint_service_module.__path__ = []
        coordinator_module = ModuleType("relax.distributed.checkpoint_service.coordinator")
        coordinator_module.__path__ = []
        dcs_service_module = ModuleType("relax.distributed.checkpoint_service.coordinator.service")
        dcs_service_module.create_dcs_deployment = lambda *_args, **_kwargs: None
        checkpoint_service_module.coordinator = coordinator_module
        coordinator_module.service = dcs_service_module
        sys.modules["relax.distributed.checkpoint_service"] = checkpoint_service_module
        sys.modules["relax.distributed.checkpoint_service.coordinator"] = coordinator_module
        sys.modules["relax.distributed.checkpoint_service.coordinator.service"] = dcs_service_module
        if distributed_module is not None:
            distributed_module.checkpoint_service = checkpoint_service_module
        sys.modules.pop("relax.core.registry", None)
        if core_module is not None and registry_attr_was_set:
            delattr(core_module, "registry")

        controller_path = Path(__file__).resolve().parents[2] / "relax" / "core" / "controller.py"
        spec = importlib.util.spec_from_file_location(module_name, controller_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)
        for name, original_module in saved_modules.items():
            if original_module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module
        if core_module is not None:
            if registry_attr_was_set:
                setattr(core_module, "registry", original_registry_attr)
            elif hasattr(core_module, "registry"):
                delattr(core_module, "registry")
        if distributed_module is not None:
            if checkpoint_service_attr_was_set:
                setattr(distributed_module, "checkpoint_service", original_checkpoint_service_attr)
            elif hasattr(distributed_module, "checkpoint_service"):
                delattr(distributed_module, "checkpoint_service")
        if components_module is not None:
            for attr_name, (attr_was_set, original_attr) in saved_component_attrs.items():
                if attr_was_set:
                    setattr(components_module, attr_name, original_attr)
                elif hasattr(components_module, attr_name):
                    delattr(components_module, attr_name)


controller = _load_controller_with_stubbed_registry_components()
ROLES = controller.ROLES


class _FakeCleanupTask:
    def __init__(self):
        self.node_ids = []

    def options(self, *, scheduling_strategy):
        self.node_ids.append(scheduling_strategy.node_id)
        return self

    def remote(self, config):
        return f"cleanup-{self.node_ids[-1]}"


def test_controller_s3_cleanup_runs_once_on_each_alive_node(monkeypatch):
    monkeypatch.setenv("RELAX_S3_MODEL_CLEANUP_TASK_TIMEOUT_S", "17.5")
    config = SimpleNamespace(model_source=SimpleNamespace(), disable_s3_model_cleanup=False)
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = config
    instance.serve_dict = {}
    cleanup_task = _FakeCleanupTask()
    node_a = "a" * 56
    node_b = "b" * 56

    remote_options = {}

    def fake_remote(**kwargs):
        remote_options.update(kwargs)
        return lambda _func: cleanup_task

    monkeypatch.setattr(controller.ray, "remote", fake_remote)
    monkeypatch.setattr(
        controller.ray,
        "nodes",
        lambda: [
            {"Alive": True, "NodeID": node_a},
            {"Alive": False, "NodeID": "c" * 56},
            {"Alive": True, "NodeID": node_b},
        ],
    )
    monkeypatch.setattr(
        controller.ray,
        "wait",
        lambda refs, *, num_returns, timeout: (refs, []) if num_returns == 2 and timeout == 17.5 else None,
    )
    monkeypatch.setattr(controller.ray, "get", lambda refs: [(2, 10), (1, 5)] if len(refs) == 2 else None)

    instance._cleanup_s3_model_weights_after_init()
    assert cleanup_task.node_ids == [node_a, node_b]
    assert remote_options == {"num_cpus": 0, "max_retries": 0}


def test_controller_s3_cleanup_skips_shm_backed_rollout():
    config = SimpleNamespace(
        model_source=SimpleNamespace(),
        sglang_load_format="auto",
        rollout_external=False,
        sglang_config=None,
    )

    assert not controller._can_cleanup_s3_model_weights(config, {"rollout": object()})
    config.sglang_load_format = "dummy"
    assert controller._can_cleanup_s3_model_weights(config, {"rollout": object()})
    config.sglang_load_format = "runai_streamer"
    assert controller._can_cleanup_s3_model_weights(config, {"rollout": object()})
    config.sglang_load_format = "remote"
    assert not controller._can_cleanup_s3_model_weights(config, {"rollout": object()})
    config.sglang_load_format = "full"
    assert not controller._can_cleanup_s3_model_weights(config, {"rollout": object()})
    config.sglang_load_format = "dummy"
    config.sglang_config = "/tmp/sglang.yaml"
    assert not controller._can_cleanup_s3_model_weights(config, {"rollout": object()})
    config.sglang_config = None
    config.rollout_external = True
    assert not controller._can_cleanup_s3_model_weights(config, {"rollout": object()})


def test_controller_s3_cleanup_without_rollout_is_safe():
    config = SimpleNamespace(
        model_source=SimpleNamespace(),
        sglang_load_format="auto",
        rollout_external=False,
        sglang_config=None,
    )

    assert controller._can_cleanup_s3_model_weights(config, {"actor": object()})


def test_controller_s3_cleanup_skips_debug_rollout_only():
    config = SimpleNamespace(
        model_source=SimpleNamespace(),
        debug_rollout_only=True,
        sglang_load_format="dummy",
        rollout_external=False,
        sglang_config=None,
    )

    assert not controller._can_cleanup_s3_model_weights(config, {"rollout": object()})


def test_controller_s3_cleanup_requires_model_source():
    config = SimpleNamespace(
        model_source=None,
        sglang_load_format="runai_streamer",
        rollout_external=False,
        sglang_config=None,
    )

    assert not controller._can_cleanup_s3_model_weights(config, {})
    assert not controller._can_cleanup_s3_model_weights(config, {"rollout": object()})


def test_controller_s3_cleanup_timeout_cancels_pending_tasks(monkeypatch):
    monkeypatch.setenv("RELAX_S3_MODEL_CLEANUP_CANCEL_TIMEOUT_S", "2.5")
    config = SimpleNamespace(model_source=SimpleNamespace(), disable_s3_model_cleanup=False)
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = config
    instance.serve_dict = {}
    cleanup_task = _FakeCleanupTask()
    pending_ref = object()
    cancelled = []
    wait_calls = []

    monkeypatch.setattr(controller.ray, "remote", lambda **_kwargs: lambda _func: cleanup_task)
    monkeypatch.setattr(controller.ray, "nodes", lambda: [{"Alive": True, "NodeID": "a" * 56}])

    def fake_wait(refs, **kwargs):
        wait_calls.append((refs, kwargs))
        return ([], [pending_ref]) if len(wait_calls) == 1 else ([pending_ref], [])

    monkeypatch.setattr(controller.ray, "wait", fake_wait)
    monkeypatch.setattr(controller.ray, "cancel", lambda ref, *, force: cancelled.append((ref, force)))

    with pytest.raises(TimeoutError, match="1 of 1 node tasks pending"):
        instance._cleanup_s3_model_weights_after_init()
    assert cancelled == [(pending_ref, True)]
    assert wait_calls[1] == (
        [pending_ref],
        {
            "num_returns": 1,
            "timeout": 2.5,
        },
    )


def test_controller_s3_cleanup_timeout_raises_when_cancel_remains_pending(monkeypatch):
    config = SimpleNamespace(model_source=SimpleNamespace(), disable_s3_model_cleanup=False)
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = config
    instance.serve_dict = {}
    cleanup_task = _FakeCleanupTask()
    pending_ref = object()
    cancelled = []

    monkeypatch.setattr(controller.ray, "remote", lambda **_kwargs: lambda _func: cleanup_task)
    monkeypatch.setattr(controller.ray, "nodes", lambda: [{"Alive": True, "NodeID": "a" * 56}])
    monkeypatch.setattr(controller.ray, "wait", lambda _refs, **_kwargs: ([], [pending_ref]))
    monkeypatch.setattr(controller.ray, "cancel", lambda ref, *, force: cancelled.append((ref, force)))

    with pytest.raises(TimeoutError, match="cancellation did not reach 1 of 1 pending node tasks"):
        instance._cleanup_s3_model_weights_after_init()
    assert cancelled == [(pending_ref, True)]


@pytest.mark.parametrize("value", ["0", "-1"])
def test_controller_s3_cleanup_rejects_non_positive_task_timeout(monkeypatch, value):
    monkeypatch.setenv("RELAX_S3_MODEL_CLEANUP_TASK_TIMEOUT_S", value)
    config = SimpleNamespace(model_source=SimpleNamespace(), disable_s3_model_cleanup=False)
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = config
    instance.serve_dict = {}
    cleanup_task = _FakeCleanupTask()

    monkeypatch.setattr(controller.ray, "remote", lambda **_kwargs: lambda _func: cleanup_task)
    monkeypatch.setattr(controller.ray, "nodes", lambda: [{"Alive": True, "NodeID": "a" * 56}])

    with pytest.raises(ValueError, match="RELAX_S3_MODEL_CLEANUP_TASK_TIMEOUT_S must be greater than 0"):
        instance._cleanup_s3_model_weights_after_init()


def test_controller_s3_cleanup_disable_skips_node_discovery(monkeypatch):
    config = SimpleNamespace(model_source=SimpleNamespace(), disable_s3_model_cleanup=True)
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = config
    monkeypatch.setattr(controller.ray, "nodes", lambda: (_ for _ in ()).throw(AssertionError("must not run")))

    instance._cleanup_s3_model_weights_after_init()


def test_controller_s3_cleanup_runs_after_initial_sync_before_service_run(monkeypatch):
    events = []

    class Service:
        def __init__(self, role):
            self.role = role

        async def get_rollout_manager(self):
            return object()

        async def set_rollout_manager(self, _manager):
            events.append("set_rollout_manager")

        def update_weights_fully_async(self):
            async def update():
                events.append("update_weights")

            return update()

        def recv_weight_fully_async(self):
            async def receive():
                events.append(f"receive_{self.role.value}")

            return receive()

        async def get_step(self):
            return 0

        async def set_step(self, _step):
            events.append(f"set_step_{self.role.value}")

        def run(self):
            events.append(f"run_{self.role.value}")
            return None

    instance = controller.Controller.__new__(controller.Controller)
    instance.config = SimpleNamespace(
        debug_train_only=False,
        debug_rollout_only=False,
        fully_async=True,
        hybrid=False,
    )
    instance.serve_dict = {
        ROLES.actor: Service(ROLES.actor),
        ROLES.rollout: Service(ROLES.rollout),
        ROLES.actor_fwd: Service(ROLES.actor_fwd),
        ROLES.reference: Service(ROLES.reference),
    }
    instance._teacher_manager = None
    instance._pending_task_refs = []
    instance._pending_task_refs_lock = threading.Lock()
    instance._restarting = False
    monkeypatch.setattr(controller, "set_managed_opd_teacher_on_actor_service", lambda *_args: _async_noop())
    monkeypatch.setattr(instance, "_cleanup_s3_model_weights_after_init", lambda: events.append("cleanup"))

    instance.training_loop()

    cleanup_index = events.index("cleanup")
    assert events.index("update_weights") < cleanup_index
    assert events.index("set_step_actor") < cleanup_index
    assert cleanup_index < events.index("run_actor")


async def _async_noop():
    return None
