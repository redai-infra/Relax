import importlib
import sys
import types
from argparse import Namespace

import pytest
import torch


def _load_stream_module(monkeypatch):
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    mpu = types.ModuleType("megatron.core.mpu")
    transfer_queue = types.ModuleType("transfer_queue")
    tq_dataloader = types.ModuleType("transfer_queue.dataloader")
    streaming_dataloader = types.ModuleType("transfer_queue.dataloader.streaming_dataloader")
    streaming_dataset = types.ModuleType("transfer_queue.dataloader.streaming_dataset")
    tensordict = types.ModuleType("tensordict")

    mpu.get_tensor_model_parallel_rank = lambda: 1
    mpu.get_context_parallel_rank = lambda: 0
    mpu.get_data_parallel_group = lambda with_context_parallel=True: object()
    core.mpu = mpu

    class _StreamingDataLoader:
        pass

    class _StreamingDataset:
        pass

    streaming_dataloader.StreamingDataLoader = _StreamingDataLoader
    streaming_dataset.StreamingDataset = _StreamingDataset
    tensordict.TensorDict = dict

    modules = {
        "megatron": megatron,
        "megatron.core": core,
        "megatron.core.mpu": mpu,
        "transfer_queue": transfer_queue,
        "transfer_queue.dataloader": tq_dataloader,
        "transfer_queue.dataloader.streaming_dataloader": streaming_dataloader,
        "transfer_queue.dataloader.streaming_dataset": streaming_dataset,
        "tensordict": tensordict,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop("relax.utils.data.stream_dataloader", None)
    return importlib.import_module("relax.utils.data.stream_dataloader")


def _make_iterator(stream_module, *, get_data_fn, all_consumed_fn, **kwargs):
    monkeypatch_kwargs = dict(
        args=Namespace(),
        tq_client=object(),
        data_fields=["tokens"],
        rollout_id=52,
        token_budget=1024,
        loss_scale=0.5,
        all_consumed_fn=all_consumed_fn,
        dp_rank=1,
        dp_size=2,
        window_quota=64,
        rollout_mini_index=0,
    )
    monkeypatch_kwargs.update(kwargs)
    return stream_module.StreamingTQIterator(**monkeypatch_kwargs)


def test_streaming_tq_iterator_sampling_config_carries_window_quota(monkeypatch):
    """The consumer forwards window_quota + rollout_mini_index (and NOT any
    per-DP count keys) so the sampler can enforce the per-window global cap."""
    stream_module = _load_stream_module(monkeypatch)
    monkeypatch.setattr(stream_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(stream_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)

    iterator = _make_iterator(
        stream_module,
        get_data_fn=lambda **k: (None, None),
        all_consumed_fn=lambda: True,
        window_quota=64,
        rollout_mini_index=3,
    )
    config = iterator._sampling_config(7)
    assert config["window_quota"] == 64
    assert config["rollout_mini_index"] == 3
    assert config["batch_index"] == 7
    # Removed per-DP count keys must be gone.
    assert "max_samples" not in config
    assert "remaining_samples" not in config
    assert "consumed_samples" not in config


def test_streaming_tq_iterator_finishes_on_window_drained_with_underfill(monkeypatch):
    """Regression for the fully-async DP-imbalance deadlock.

    Per-DP sample counts are uneven (token-budget driven). The iterator
    finishes when the per-window drained signal fires — NOT on a per-DP count
    target — so a DP may finish underfilled and the dummy-pad aligns cross-DP
    mb counts.
    """
    stream_module = _load_stream_module(monkeypatch)
    monkeypatch.setattr(stream_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(stream_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)

    batches = [{"tokens": [torch.tensor([0]), torch.tensor([1])]}]
    drained = {"flag": False}

    def fake_get_data_from_transfer_queue(**kwargs):
        if batches:
            return batches.pop(0), object()
        drained["flag"] = True  # window quota met → controller reports drained
        return None, None

    monkeypatch.setattr(stream_module, "get_data_from_transfer_queue", fake_get_data_from_transfer_queue)

    iterator = _make_iterator(
        stream_module,
        get_data_fn=fake_get_data_from_transfer_queue,
        all_consumed_fn=lambda: drained["flag"],
    )

    assert len(next(iterator)[0]["tokens"]) == 2
    # Underfilled (2 of 64 window quota) but window drained → clean StopIteration.
    with pytest.raises(StopIteration):
        next(iterator)
    assert iterator._sample_count == 2


def test_streaming_tq_iterator_polls_until_window_drained(monkeypatch):
    """Empty fetch while the window is NOT yet drained -> keep polling; once
    the per-window drained signal flips, finish (no per-DP count involved)."""
    stream_module = _load_stream_module(monkeypatch)
    monkeypatch.setattr(stream_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(stream_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)
    monkeypatch.setattr(stream_module.time, "sleep", lambda _s: None)

    # One real mb, then several empty polls, then drained.
    events = [("data", {"tokens": [torch.tensor([0])]})] + [("empty", None)] * 3
    drained = {"flag": False}

    def fake_get_data_from_transfer_queue(**kwargs):
        if events:
            kind, payload = events.pop(0)
            if kind == "data":
                return payload, object()
        if not events:
            drained["flag"] = True
        return None, None

    monkeypatch.setattr(stream_module, "get_data_from_transfer_queue", fake_get_data_from_transfer_queue)

    iterator = _make_iterator(
        stream_module,
        get_data_fn=fake_get_data_from_transfer_queue,
        all_consumed_fn=lambda: drained["flag"],
    )
    assert len(next(iterator)[0]["tokens"]) == 1
    with pytest.raises(StopIteration):
        next(iterator)


def test_streaming_tq_iterator_raises_on_stall_timeout(monkeypatch):
    """Deadlock guard: no data + window not drained past the stall timeout must
    raise (fast, diagnosable failure) instead of polling forever."""
    stream_module = _load_stream_module(monkeypatch)
    monkeypatch.setattr(stream_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(stream_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)
    monkeypatch.setattr(stream_module.time, "sleep", lambda _s: None)

    monkeypatch.setattr(stream_module, "get_data_from_transfer_queue", lambda **k: (None, None))

    iterator = _make_iterator(
        stream_module,
        get_data_fn=lambda **k: (None, None),
        all_consumed_fn=lambda: False,  # never drains → would hang forever
        max_stream_stall_s=0.0001,  # trip almost immediately
    )
    with pytest.raises(RuntimeError, match="stalled"):
        next(iterator)


def test_streaming_tq_iterator_rejects_nonpositive_window_quota(monkeypatch):
    stream_module = _load_stream_module(monkeypatch)
    monkeypatch.setattr(stream_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    with pytest.raises(ValueError, match="window_quota"):
        _make_iterator(
            stream_module,
            get_data_fn=lambda **k: (None, None),
            all_consumed_fn=lambda: True,
            window_quota=0,
        )


def test_streaming_tq_iterator_per_round_dummy_then_end(monkeypatch):
    """Train path (per_round_dummy): on an empty fetch the consumer reads the
    DUMMY-round flag from the fetched meta's extra_info and emits a zero-grad
    dummy, then StopIteration when the fetched meta carries STREAM_END.

    The end verdict rides the SAME empty fetch (stamped atomically by the
    controller), NOT a separate all_consumed_fn() RPC — so all_consumed_fn is
    pinned False here to prove the stop is driven by the fetch, not the RPC.
    Guarantees equal micro-batch counts across DP/PP without a cross-DP
    all_reduce (MoE lockstep) and without the two-RPC TOCTOU that let PP stages
    diverge across the drain boundary.
    """
    stream_module = _load_stream_module(monkeypatch)
    monkeypatch.setattr(stream_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(stream_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)
    monkeypatch.setattr(stream_module.time, "sleep", lambda _s: None)

    class _Meta:
        def __init__(self, extra_info):
            self.extra_info = extra_info

    # 2 real fetches, then 2 DUMMY-round empties (meta carries the flag), then an
    # end empty whose meta carries STREAM_END (window drained).
    events = [
        ({"tokens": [torch.tensor([0])]}, _Meta({})),
        ({"tokens": [torch.tensor([1])]}, _Meta({})),
        (None, _Meta({"dummy_round": True})),
        (None, _Meta({"dummy_round": True})),
        (None, _Meta({"stream_end": True})),
    ]

    def fake_get(**kwargs):
        return events.pop(0) if events else (None, _Meta({"stream_end": True}))

    monkeypatch.setattr(stream_module, "get_data_from_transfer_queue", fake_get)

    iterator = _make_iterator(
        stream_module,
        get_data_fn=fake_get,
        all_consumed_fn=lambda: False,  # never consulted on the train path anymore
        per_round_dummy=True,
    )

    assert not next(iterator)[0].get("__is_dummy__")  # real mb 0
    assert not next(iterator)[0].get("__is_dummy__")  # real mb 1
    assert next(iterator)[0]["__is_dummy__"] is True  # dummy round
    assert next(iterator)[0]["__is_dummy__"] is True  # dummy round
    with pytest.raises(StopIteration):  # empty, no dummy flag, stream_end → END
        next(iterator)
    assert iterator._mb_count == 4  # 2 real + 2 dummy → equal count with a busier DP
    assert iterator._sample_count == 2  # dummies don't count as samples
