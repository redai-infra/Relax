# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


pytest.importorskip("ray", reason="Ray is an optional test dependency")
pytest.importorskip("transfer_queue", reason="TransferQueue is an optional test dependency")
pytest.importorskip("megatron", reason="Megatron is an optional test dependency")

from relax.core import controller as controller_module
from relax.engine.rollout import bootstrap


def test_controller_resolves_rollout_epoch_before_actor_service_creation(monkeypatch) -> None:
    config = Namespace(
        advantage_estimator="grpo",
        loss_type="grpo",
        colocate=False,
        hybrid=True,
        fully_async=True,
        resource={"actor": (1, 1), "rollout": (1, 1)},
        data_source_path="tests.fake_data_source",
        rollout_global_dataset=True,
        rollout_batch_size=16,
        num_epoch=3,
        num_rollout=20,
    )
    data_source = SimpleNamespace(lengths=SimpleNamespace(remote=Mock(return_value="length-ref")))

    class RemoteDataSource:
        def remote(self, remote_config):
            assert remote_config is config
            return data_source

    controller = controller_module.Controller.__new__(controller_module.Controller)
    controller.config = config
    controller.runtime_env = None
    controller.serve_dict = {}
    controller._health_manager = SimpleNamespace(mark_healthy=Mock())
    controller._validate_gpu_resources = Mock()

    creation_snapshots = []

    def create_service(role, cls, num_gpus, role_data_source, actor_rollout_pgs, actor_rollout_pg_roles):
        creation_snapshots.append((role, config.num_rollout_per_epoch, config.num_rollout))
        return role, SimpleNamespace(role=role), None

    controller._create_service_task = Mock(side_effect=create_service)

    monkeypatch.setattr(controller_module, "validate_ppo_config", Mock())
    monkeypatch.setattr(controller_module, "maybe_start_managed_opd_teacher", Mock(return_value=(None, None)))
    monkeypatch.setattr(controller_module, "resolve_sft_algo_key", Mock(return_value="grpo"))
    monkeypatch.setattr(controller_module, "ALGOS", {"grpo": {"actor": object(), "rollout": object()}})
    monkeypatch.setattr(controller_module, "process_role", Mock(return_value=["actor", "rollout"]))
    monkeypatch.setattr(controller_module, "validate_sft_resource", Mock())
    monkeypatch.setattr(controller_module, "register_extra_roles", Mock(return_value=[]))
    monkeypatch.setattr(controller_module, "load_function", Mock(return_value=object()))
    monkeypatch.setattr(controller_module.ray, "remote", lambda **kwargs: lambda cls: RemoteDataSource())
    monkeypatch.setattr(bootstrap.ray, "get", lambda ref: 64)

    controller.register_all_serve()

    assert sorted(creation_snapshots) == [("actor", 4, 12), ("rollout", 4, 12)]
