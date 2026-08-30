# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch

from relax.utils.training.tensor_backper import TensorBackuper


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="backup synchronizes CUDA tensors")


def _make_backuper():
    state = {"weight": torch.tensor([1.0, 3.0]), "bias": torch.tensor([2.0])}
    backuper = TensorBackuper.create(source_getter=lambda: state.items(), single_tag=None)
    return state, backuper


def test_ema_initial_snapshot_matches_actor_without_mutating_source():
    state, backuper = _make_backuper()

    backuper.backup("actor")
    backuper.backup("actor_ema")

    assert torch.equal(backuper.get("actor")["weight"], state["weight"])
    assert torch.equal(backuper.get("actor_ema")["weight"], state["weight"])
    assert backuper.get("actor")["weight"].data_ptr() != backuper.get("actor_ema")["weight"].data_ptr()


def test_ema_matches_two_step_oracle_and_preserves_actor_snapshot():
    state, backuper = _make_backuper()
    backuper.backup("actor")
    backuper.backup("actor_ema")

    state["weight"] = torch.tensor([5.0, 7.0])
    state["bias"] = torch.tensor([6.0])
    backuper.backup("actor")
    actor_before = {key: value.clone() for key, value in backuper.get("actor").items()}
    backuper.ema(source_tag="actor", target_tag="actor_ema", alpha=0.25)

    assert torch.allclose(backuper.get("actor_ema")["weight"], torch.tensor([2.0, 4.0]))
    assert torch.allclose(backuper.get("actor_ema")["bias"], torch.tensor([3.0]))
    assert all(torch.equal(backuper.get("actor")[key], value) for key, value in actor_before.items())

    state["weight"] = torch.tensor([9.0, 11.0])
    state["bias"] = torch.tensor([10.0])
    backuper.backup("actor")
    backuper.ema(source_tag="actor", target_tag="actor_ema", alpha=0.25)

    assert torch.allclose(backuper.get("actor_ema")["weight"], torch.tensor([3.75, 5.75]))
    assert torch.allclose(backuper.get("actor_ema")["bias"], torch.tensor([4.75]))


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1])
def test_ema_rejects_invalid_alpha(alpha):
    _state, backuper = _make_backuper()
    backuper.backup("actor")
    backuper.backup("actor_ema")

    with pytest.raises(ValueError, match="EMA alpha"):
        backuper.ema(source_tag="actor", target_tag="actor_ema", alpha=alpha)


def test_ema_alpha_one_copies_source():
    state, backuper = _make_backuper()
    backuper.backup("actor")
    backuper.backup("actor_ema")
    state["weight"] = torch.tensor([8.0, 10.0])
    backuper.backup("actor")

    backuper.ema(source_tag="actor", target_tag="actor_ema", alpha=1.0)

    assert torch.equal(backuper.get("actor_ema")["weight"], torch.tensor([8.0, 10.0]))


def test_ema_copies_integral_and_boolean_buffers():
    state = {
        "weight": torch.tensor([1.0]),
        "counter": torch.tensor([2], dtype=torch.int64),
        "mask": torch.tensor([True, False], dtype=torch.bool),
    }
    backuper = TensorBackuper.create(source_getter=lambda: state.items(), single_tag=None)
    backuper.backup("actor")
    backuper.backup("actor_ema")

    state["weight"] = torch.tensor([5.0])
    state["counter"] = torch.tensor([9], dtype=torch.int64)
    state["mask"] = torch.tensor([False, True], dtype=torch.bool)
    backuper.backup("actor")
    backuper.ema(source_tag="actor", target_tag="actor_ema", alpha=0.25)

    assert torch.equal(backuper.get("actor_ema")["weight"], torch.tensor([2.0]))
    assert torch.equal(backuper.get("actor_ema")["counter"], torch.tensor([9]))
    assert torch.equal(backuper.get("actor_ema")["mask"], torch.tensor([False, True]))


def test_ema_rejects_snapshot_key_shape_and_dtype_mismatch():
    _state, backuper = _make_backuper()
    backuper.backup("actor")
    backuper.backup("actor_ema")

    del backuper.get("actor_ema")["bias"]
    with pytest.raises(ValueError, match="identical keys"):
        backuper.ema(source_tag="actor", target_tag="actor_ema", alpha=0.5)

    backuper.backup("actor_ema")
    backuper.get("actor_ema")["weight"] = torch.empty(3)
    with pytest.raises(ValueError, match="snapshot mismatch"):
        backuper.ema(source_tag="actor", target_tag="actor_ema", alpha=0.5)


def test_noop_backuper_rejects_ema():
    state = {"weight": torch.tensor([1.0])}
    backuper = TensorBackuper.create(source_getter=lambda: state.items(), single_tag="actor")
    backuper.backup("actor")

    with pytest.raises(RuntimeError, match="enable-weights-backuper"):
        backuper.ema(source_tag="actor", target_tag="actor_ema", alpha=0.5)
