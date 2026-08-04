# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch

from relax.utils.training.tensor_backper import TensorBackuper


def test_tensor_backuper_keeps_actor_snapshot_on_source_device_and_preserves_tp_attrs(monkeypatch) -> None:
    megatron = pytest.importorskip("megatron.core.tensor_parallel")
    monkeypatch.setattr(megatron, "set_defaults_if_not_set_tensor_model_parallel_attributes", lambda tensor: None)

    source = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    source.tensor_model_parallel = True
    source.partition_dim = 0
    source.partition_stride = 1
    source.parallel_mode = "column"
    backuper = TensorBackuper.create(lambda: [("weight", source)], single_tag=None)

    backuper.backup("actor", on_device=True)
    snapshot = backuper.get("actor")["weight"]

    assert snapshot.device == source.device
    assert snapshot.data_ptr() != source.data_ptr()
    assert torch.equal(snapshot, source)
    assert snapshot.tensor_model_parallel is True
    assert snapshot.partition_dim == 0
    assert snapshot.partition_stride == 1
    assert snapshot.parallel_mode == "column"

    source.add_(10)
    backuper.backup("actor", on_device=True)
    assert torch.equal(snapshot, source)


def test_tensor_backuper_rejects_changing_snapshot_device_policy(monkeypatch) -> None:
    megatron = pytest.importorskip("megatron.core.tensor_parallel")
    monkeypatch.setattr(megatron, "set_defaults_if_not_set_tensor_model_parallel_attributes", lambda tensor: None)

    source = torch.ones(4)
    backuper = TensorBackuper.create(lambda: [("weight", source)], single_tag=None)
    backuper.backup("actor", on_device=True)

    with pytest.raises(RuntimeError, match="changed device policy"):
        backuper.backup("actor", on_device=False)
