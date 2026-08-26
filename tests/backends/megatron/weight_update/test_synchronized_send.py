# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch

from relax.backends.megatron.weight_update.synchronized_send import (
    raise_on_any_rank_failure,
    run_synchronized_phase,
    send_chunks_pipelined,
)


@contextmanager
def _single_rank_collectives(*, peer_failed: bool = False):
    """Stand in for the Gloo collectives on a one-rank world.

    ``peer_failed`` simulates another rank reporting a failure, which is what
    the all-reduce exists to propagate.
    """

    def fake_all_reduce(tensor, *args, **kwargs):
        if peer_failed:
            tensor.fill_(1)

    with (
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.all_reduce", side_effect=fake_all_reduce),
        patch("relax.backends.megatron.weight_update.synchronized_send.get_gloo_group", return_value=None),
    ):
        yield


def test_pipelined_send_defers_each_wait_until_the_next_chunk_is_in_flight():
    events = []

    def send_chunk(chunk):
        events.append(f"send:{chunk}")
        return [f"ref:{chunk}"], f"tensors:{chunk}"

    with (
        _single_rank_collectives(),
        patch("ray.get", side_effect=lambda refs: events.append(f"wait:{refs}")),
    ):
        send_chunks_pipelined(["a", "b"], send_chunk)

    assert events == ["send:a", "send:b", "wait:['ref:a']", "wait:['ref:b']"]


def test_pipelined_send_aborts_when_an_intermediate_chunk_fails_on_another_rank():
    """The IPC wait only raises on the gather-source rank.

    Every other rank has to learn about it from the all-reduce, otherwise it
    blocks in the next chunk's collectives.
    """

    send_chunk = MagicMock(side_effect=[(["ref:a"], "tensors:a"), (["ref:b"], "tensors:b"), (["ref:c"], "tensors:c")])

    with (
        _single_rank_collectives(peer_failed=True),
        patch("ray.get") as ray_get,
        pytest.raises(RuntimeError, match="failed on another rank"),
    ):
        send_chunks_pipelined(["a", "b", "c"], send_chunk)

    # The failure surfaces at the first chunk boundary, long before the final
    # drain that used to be the only synchronization point.
    assert send_chunk.call_count == 1
    ray_get.assert_not_called()


def test_pipelined_send_confirms_predecessors_before_a_marked_chunk():
    events = []

    def send_chunk(chunk):
        events.append(f"send:{chunk[0]}")
        return [f"ref:{chunk[0]}"], None

    with (
        _single_rank_collectives(),
        patch("ray.get", side_effect=lambda refs: events.append(f"wait:{refs}")),
    ):
        send_chunks_pipelined(
            [("a", None), ("b", 7)],
            send_chunk,
            confirm_before=lambda chunk: chunk[1] is not None,
        )

    assert events == ["send:a", "wait:['ref:a']", "send:b", "wait:['ref:b']"]


def test_synchronized_phase_reports_the_local_error_and_the_shared_flag():
    failure = RuntimeError("engine rejected the update")

    with _single_rank_collectives():
        local_error, failed = run_synchronized_phase(MagicMock(side_effect=failure))
        clean_error, clean_failed = run_synchronized_phase(MagicMock())

    assert local_error is failure
    assert failed is True
    assert clean_error is None
    assert clean_failed is False


def test_raise_on_any_rank_failure_reraises_the_local_error_unchanged():
    failure = ValueError("bad chunk")

    with _single_rank_collectives(), pytest.raises(ValueError) as excinfo:
        raise_on_any_rank_failure(MagicMock(side_effect=failure))

    assert excinfo.value is failure


def test_failure_flag_is_reduced_as_an_integer_tensor():
    reduced = []

    with (
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.all_reduce", side_effect=lambda tensor, *a, **k: reduced.append(tensor.clone())),
        patch("relax.backends.megatron.weight_update.synchronized_send.get_gloo_group", return_value=None),
    ):
        run_synchronized_phase(MagicMock())

    assert reduced[0].dtype == torch.int32
    assert reduced[0].tolist() == [0]
