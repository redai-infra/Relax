# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
from contextlib import nullcontext
from types import SimpleNamespace

import pytest


class _FakeRemoteMethod:
    def __init__(self, event, name):
        self._event = event
        self._name = name

    def remote(self, *args, **kwargs):
        self._event.append(self._name)
        return self._name


class _FakeEngine:
    def __init__(self, event):
        self.pause_generation = _FakeRemoteMethod(event, "pause")
        self.flush_cache = _FakeRemoteMethod(event, "flush")
        self.continue_generation = _FakeRemoteMethod(event, "continue")


class _FakeIterator:
    def get_hf_weight_chunks(self, weights):
        yield [("weight", weights["weight"])]


def _nonzero_rank_failure_worker(rank, init_file):
    import torch
    import torch.distributed as dist

    from relax.backends.megatron.weight_update import update_weight_from_tensor as module
    from relax.utils.distributed_utils import init_gloo_group

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    init_gloo_group()

    class _RemoteMethod:
        def remote(self, *args, **kwargs):
            return None

    class _Engine:
        pause_generation = _RemoteMethod()
        flush_cache = _RemoteMethod()
        continue_generation = _RemoteMethod()

    updater = module.UpdateWeightFromTensor.__new__(module.UpdateWeightFromTensor)
    updater.args = SimpleNamespace()
    updater.model = []
    updater.weights_getter = lambda: {"weight": torch.tensor([1.0])}
    updater.model_name = "test"
    updater.quantization_config = None
    updater.weight_version = 0
    updater.lora_enabled = False
    updater.lora_adapter_mode = False
    updater._hf_weight_iterator = _FakeIterator()
    updater.rollout_engines = [_Engine()]
    updater.distributed_rollout_engines = []

    def send_hf_params(_named_tensors, *, weight_version=None):
        if rank == 1:
            raise RuntimeError("nonzero sender failed")
        return [], None

    updater._send_hf_params = send_hf_params
    module.ray.get = lambda refs: None
    module.device_utils.maybe_backend_barrier_on_weight_chunk = lambda **kwargs: None

    try:
        updater.update_weights()
    except RuntimeError as exc:
        assert "Weight update failed before commit" in str(exc)
        assert updater.weight_version == 0
    else:
        raise AssertionError("expected synchronized weight update failure")
    finally:
        dist.destroy_process_group()


def _make_updater(monkeypatch, event, *, send_error=None, versions=None):
    import torch

    from relax.backends.megatron.weight_update import update_weight_from_tensor as module

    updater = module.UpdateWeightFromTensor.__new__(module.UpdateWeightFromTensor)
    updater.args = SimpleNamespace()
    updater.model = []
    updater.weights_getter = lambda: {"weight": torch.tensor([1.0])}
    updater.model_name = "test"
    updater.quantization_config = None
    updater.weight_version = 0
    updater.lora_enabled = False
    updater.lora_adapter_mode = False
    updater._hf_weight_iterator = _FakeIterator()
    updater.rollout_engines = [_FakeEngine(event)]
    updater.distributed_rollout_engines = []

    def send_hf_params(_named_tensors, *, weight_version=None):
        event.append("send")
        if versions is not None:
            versions.append(weight_version)
        if send_error is not None:
            raise send_error
        return [], None

    updater._send_hf_params = send_hf_params
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "barrier", lambda **kwargs: event.append("barrier"))
    monkeypatch.setattr(module.dist, "all_reduce", lambda tensor, **kwargs: event.append("all_reduce"))
    monkeypatch.setattr(module, "get_gloo_group", lambda: None)
    monkeypatch.setattr(module.device_utils, "maybe_backend_barrier_on_weight_chunk", lambda **kwargs: None)
    monkeypatch.setattr(module.ray, "get", lambda refs: None)
    return updater


def test_sdpo_teacher_publish_pauses_flushes_transfers_and_resumes(monkeypatch):
    pytest.importorskip("megatron.core")
    event = []
    versions = []
    updater = _make_updater(monkeypatch, event, versions=versions)

    updater.update_weights()

    assert event[:3] == ["pause", "flush", "barrier"]
    assert event.count("all_reduce") == 8
    assert event.count("send") == 1
    assert event[-2:] == ["barrier", "all_reduce"]
    assert updater.weight_version == 1
    assert versions == [1]


def test_sdpo_teacher_publish_failure_does_not_resume_serving(monkeypatch):
    pytest.importorskip("megatron.core")
    event = []
    versions = []
    updater = _make_updater(monkeypatch, event, send_error=RuntimeError("chunk failed"), versions=versions)

    with pytest.raises(RuntimeError, match="chunk failed"):
        updater.update_weights()

    assert event[:3] == ["pause", "flush", "barrier"]
    assert event.count("all_reduce") == 5
    assert "continue" not in event
    assert updater.weight_version == 0
    assert versions == [1]


def test_sdpo_teacher_publish_retry_starts_from_uncommitted_version(monkeypatch):
    pytest.importorskip("megatron.core")
    event = []
    failed_updater = _make_updater(monkeypatch, event, send_error=RuntimeError("chunk failed"))

    with pytest.raises(RuntimeError, match="chunk failed"):
        failed_updater.update_weights()
    assert failed_updater.weight_version == 0

    retry_event = []
    retry_versions = []
    retry_updater = _make_updater(monkeypatch, retry_event, versions=retry_versions)
    retry_updater.update_weights()

    assert retry_updater.weight_version == 1
    assert retry_versions == [1]


def test_weight_update_coordinates_nonzero_sender_failure(tmp_path):
    pytest.importorskip("megatron.core")
    import torch.multiprocessing as mp

    mp.spawn(
        _nonzero_rank_failure_worker,
        args=(str(tmp_path / "weight-update-failure"),),
        nprocs=2,
        join=True,
    )


def test_torch_memory_saver_uses_cpu_weight_serialization(monkeypatch):
    pytest.importorskip("megatron.core")
    import torch

    from relax.backends.megatron.weight_update import update_weight_from_tensor as module

    captured = {}

    class _RemoteMethod:
        def remote(self, **kwargs):
            captured["request"] = kwargs
            return "ref"

    class _Engine:
        update_weights_from_tensor = _RemoteMethod()

    def fake_gather_object(obj, object_gather_list, **_kwargs):
        object_gather_list[0] = obj
        captured["serialized"] = obj

    monkeypatch.setenv("TMS_INIT_ENABLE", "1")
    monkeypatch.setattr(module, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "get_world_size", lambda _group=None: 1)
    monkeypatch.setattr(module.dist, "gather_object", fake_gather_object)

    from torch.multiprocessing import get_sharing_strategy

    previous_strategy = get_sharing_strategy()
    refs, long_lived_tensors = module._send_to_colocated_engine(
        [("weight", torch.ones(4))],
        ipc_engine=_Engine(),
        ipc_gather_src=0,
        ipc_gather_group=object(),
        weight_version=1,
    )

    assert refs == ["ref"]
    assert get_sharing_strategy() == previous_strategy
    assert long_lived_tensors[0]["flattened_tensor"].device.type == "cpu"
    decoded = module.MultiprocessingSerializer.deserialize(captured["serialized"][0])
    assert decoded["flattened_tensor"].device.type == "cpu"


@pytest.mark.skipif(
    importlib.util.find_spec("transfer_queue") is None, reason="transfer_queue runtime package unavailable"
)
def test_sdpo_actor_snapshot_refreshes_before_ema_update():
    pytest.importorskip("megatron.training.checkpointing")
    from relax.backends.megatron.actor import MegatronTrainRayActor

    events = []

    class _Backuper:
        def backup(self, tag):
            events.append(("backup", tag))

        def ema(self, **kwargs):
            events.append(("ema", kwargs))

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.weights_backuper = _Backuper()
    actor._sdpo_teacher_ema_enabled = True
    actor.args = type("Args", (), {"sdpo_teacher_ema_alpha": 0.01})()

    actor._snapshot_student_and_step_ema_teacher()

    assert events == [
        ("backup", "actor"),
        (
            "ema",
            {
                "source_tag": "actor",
                "target_tag": "actor_ema",
                "alpha": 0.01,
            },
        ),
    ]


@pytest.mark.skipif(
    importlib.util.find_spec("transfer_queue") is None, reason="transfer_queue runtime package unavailable"
)
def test_actor_snapshot_refresh_keeps_ordinary_opd_without_ema():
    pytest.importorskip("megatron.training.checkpointing")
    from relax.backends.megatron.actor import MegatronTrainRayActor

    events = []

    class _Backuper:
        def backup(self, tag):
            events.append(("backup", tag))

        def ema(self, **kwargs):
            events.append(("ema", kwargs))

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.weights_backuper = _Backuper()
    actor._sdpo_teacher_ema_enabled = False

    actor._snapshot_student_and_step_ema_teacher()

    assert events == [("backup", "actor")]


@pytest.mark.skipif(
    importlib.util.find_spec("transfer_queue") is None, reason="transfer_queue runtime package unavailable"
)
def test_sdpo_teacher_publish_hook_runs_after_student_publish(monkeypatch):
    pytest.importorskip("megatron.training.checkpointing")
    from relax.backends.megatron import actor as actor_module
    from relax.backends.megatron.actor import MegatronTrainRayActor

    events = []

    class _RemoteMethod:
        def remote(self):
            events.append("get_rollout_engines")
            return ([], None, 0, [], [])

    class _RolloutManager:
        get_rollout_engines_and_lock = _RemoteMethod()

    class _Updater:
        def update_weights(self):
            events.append("student_publish")

    actor = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
    actor.args = type(
        "Args",
        (),
        {
            "debug_train_only": False,
            "debug_rollout_only": False,
            "offload_train": False,
            "offload_rollout": False,
            "use_fault_tolerance": False,
            "ci_test": False,
            "keep_old_actor": False,
        },
    )()
    actor.rollout_manager = _RolloutManager()
    actor.weight_updater = _Updater()
    actor._torch_memory_saver_enabled = False
    actor.genrm_manager = None
    actor._train_state_offloader = SimpleNamespace(disable_during_update=nullcontext)
    actor._publish_sdpo_teacher_ema = lambda: events.append("teacher_publish")
    monkeypatch.setattr(actor_module.ray, "get", lambda value: value)
    monkeypatch.setattr(actor_module, "print_memory", lambda *args, **kwargs: None)

    actor.update_weights(publish_sdpo_teacher_ema=True)

    assert events == ["get_rollout_engines", "student_publish", "teacher_publish"]
