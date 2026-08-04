# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch

from relax.backends.megatron.weight_update.live_weight_sync import (
    estimate_full_model_bytes,
    get_distributed_memory_status,
    required_live_weight_sync_bytes,
    run_live_weight_sync,
)


def test_estimate_full_model_bytes_accounts_for_tensor_and_expert_parallel_sizes():
    tensors = {
        "decoder.layers.0.linear.weight": torch.empty(25, dtype=torch.float32),
        "decoder.layers.0.mlp.experts.0.weight": torch.empty(10, dtype=torch.float32),
    }

    total_bytes, largest_param_bytes = estimate_full_model_bytes(
        tensors,
        tensor_parallel_size=2,
        expert_tensor_parallel_size=4,
    )

    assert total_bytes == 360
    assert largest_param_bytes == 200


def test_required_live_weight_sync_bytes_reserves_two_chunks():
    required_bytes = required_live_weight_sync_bytes(
        full_model_bytes=1_000,
        largest_param_bytes=600,
        update_weight_buffer_size=512,
        memory_margin_bytes=100,
    )

    assert required_bytes == 2_300


def test_get_distributed_memory_status_uses_global_worst_case(monkeypatch):
    group = object()

    def fake_all_reduce(values, *, op, group: object):
        assert op == torch.distributed.ReduceOp.MIN
        assert group is not None
        values[0] = 900
        values[1] = -950

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    status = get_distributed_memory_status(
        local_free_bytes=1_000,
        local_required_bytes=800,
        process_group=group,
    )

    assert status.min_free_bytes == 900
    assert status.max_required_bytes == 950
    assert status.can_sync is False


def test_run_live_weight_sync_orders_update_offload_sync_and_onload():
    calls = []

    run_live_weight_sync(
        update_weights=lambda: calls.append("update"),
        offload_actor=lambda: calls.append("offload"),
        synchronize_after_offload=lambda: calls.append("sync"),
        onload_rollout_kv=lambda: calls.append("onload_kv"),
    )

    assert calls == ["update", "offload", "sync", "onload_kv"]


def test_run_live_weight_sync_offloads_actor_when_update_fails():
    calls = []

    def fail_update():
        calls.append("update")
        raise RuntimeError("update failed")

    with pytest.raises(RuntimeError, match="update failed"):
        run_live_weight_sync(
            update_weights=fail_update,
            offload_actor=lambda: calls.append("offload"),
            synchronize_after_offload=lambda: calls.append("sync"),
            onload_rollout_kv=lambda: calls.append("onload_kv"),
        )

    assert calls == ["update", "offload"]


def test_run_live_weight_sync_does_not_onload_kv_when_offload_sync_fails():
    calls = []

    def fail_sync():
        calls.append("sync")
        raise RuntimeError("offload sync failed")

    with pytest.raises(RuntimeError, match="offload sync failed"):
        run_live_weight_sync(
            update_weights=lambda: calls.append("update"),
            offload_actor=lambda: calls.append("offload"),
            synchronize_after_offload=fail_sync,
            onload_rollout_kv=lambda: calls.append("onload_kv"),
        )

    assert calls == ["update", "offload", "sync"]
