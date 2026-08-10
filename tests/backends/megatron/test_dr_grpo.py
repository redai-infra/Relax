# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("megatron.core")

from relax.backends.megatron import cp_utils as cp_utils_module  # noqa: E402
from relax.backends.megatron import data as data_module  # noqa: E402
from relax.backends.megatron import loss as loss_module  # noqa: E402
from relax.backends.megatron.data import ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY  # noqa: E402
from relax.components import advantages as advantages_module  # noqa: E402
from relax.core.registry import ALGOS  # noqa: E402
from relax.utils.types import Sample  # noqa: E402
from relax.utils.utils import post_process_rewards  # noqa: E402


@pytest.fixture(autouse=True)
def _use_cpu_for_metadata_unit_tests(monkeypatch):
    monkeypatch.setattr(loss_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))


def test_dr_grpo_is_registered():
    assert ALGOS["dr_grpo"] == ALGOS["grpo"]


def test_dr_grpo_reward_centering_is_mandatory_and_does_not_divide_by_group_std():
    args = SimpleNamespace(
        advantage_estimator="dr_grpo",
        custom_reward_post_process_path=None,
        agentic_custom_advantage_path=None,
        rewards_normalization=False,
        grpo_std_normalization=True,
        n_samples_per_prompt=2,
        reward_key=None,
    )
    samples = [Sample(group_index=0, reward=1.0), Sample(group_index=0, reward=3.0)]

    _, rewards = post_process_rewards(args, samples)

    assert rewards == pytest.approx([-1.0, 1.0])


def test_dr_grpo_does_not_add_reward_side_kl_to_returns(monkeypatch):
    monkeypatch.setattr(loss_module.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(loss_module, "maybe_padded_total_lengths", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        loss_module,
        "compute_approx_kl",
        lambda *_args, **_kwargs: torch.tensor([0.25, 0.5]),
    )
    args = Namespace(
        use_rollout_logprobs=False,
        qkv_format="thd",
        is_vl_model=False,
        uses_unsplit_forward=False,
        kl_coef=0.4,
        kl_loss_type="low_var_kl",
        advantage_estimator="dr_grpo",
        use_opd=False,
        normalize_advantages=False,
    )
    rollout_data = {
        "log_probs": [torch.zeros(2)],
        "ref_log_probs": [torch.zeros(2)],
        "rewards": [2.0],
        "values": None,
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        "total_lengths": [2],
    }

    loss_module.compute_advantages_and_returns(args, rollout_data)

    expected = torch.tensor([2.0, 2.0])
    assert torch.allclose(rollout_data["returns"][0], expected)
    assert torch.allclose(rollout_data["advantages"][0], expected)


def test_dr_grpo_advantages_service_does_not_add_reward_side_kl_to_returns(monkeypatch):
    monkeypatch.setattr(
        advantages_module,
        "compute_approx_kl",
        lambda *_args, **_kwargs: torch.tensor([0.25, 0.5]),
    )
    advantages_class = advantages_module.Advantages.func_or_class
    service = object.__new__(advantages_class)
    service.config = Namespace(
        use_rollout_logprobs=False,
        kl_coef=0.4,
        kl_loss_type="low_var_kl",
        advantage_estimator="dr_grpo",
        use_opd=False,
    )
    rollout_data = {
        "log_probs": [torch.zeros(2)],
        "rollout_log_probs": [torch.zeros(2)],
        "ref_log_probs": [torch.zeros(2)],
        "rewards": [2.0],
        "values": None,
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        "total_lengths": [2],
    }

    result = service.compute_advantages_and_returns(rollout_data)

    expected = torch.tensor([2.0, 2.0])
    assert torch.allclose(result["returns"][0], expected)
    assert torch.allclose(result["advantages"][0], expected)


def test_dr_grpo_uses_response_tokens_for_gradient_normalizer(monkeypatch):
    monkeypatch.setattr(loss_module, "get_cp_local_num_tokens", lambda *args, **kwargs: torch.tensor(3.0))
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        loss_module,
        "get_sum_of_sample_mean",
        lambda *args, **kwargs: lambda value: value.sum(),
    )
    monkeypatch.setattr(
        loss_module,
        "policy_loss_function",
        lambda *_args, **_kwargs: (torch.tensor(6.0), {"loss": torch.tensor(6.0)}),
    )
    args = Namespace(
        loss_type="policy_loss",
        recompute_loss_function=False,
        qkv_format="thd",
        calculate_per_token_loss=True,
        allgather_cp=False,
        global_batch_size=2,
    )
    batch = {
        "total_lengths": [4, 3],
        "response_lengths": [2, 1],
        "loss_masks": [torch.ones(2), torch.ones(1)],
        "__loss_scale__": torch.tensor(7.0),
        "__dr_grpo_window_scale__": torch.tensor(0.3),
    }

    loss, normalizer, logging = loss_module.loss_function(args, batch, 1, torch.ones(1))

    assert torch.isclose(loss, torch.tensor(1.8))
    assert torch.isclose(normalizer, torch.tensor(3.0))
    assert torch.isclose(logging["values"][0], torch.tensor(3.0))
    assert torch.isclose(logging["values"][1], torch.tensor(1.8))


def test_non_dr_grpo_per_token_loss_ignores_streaming_loss_scale(monkeypatch):
    monkeypatch.setattr(loss_module, "get_cp_local_num_tokens", lambda *args, **kwargs: torch.tensor(3.0))
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        loss_module,
        "get_sum_of_sample_mean",
        lambda *args, **kwargs: lambda value: value.sum(),
    )
    monkeypatch.setattr(
        loss_module,
        "policy_loss_function",
        lambda *_args, **_kwargs: (torch.tensor(6.0), {"loss": torch.tensor(6.0)}),
    )
    args = Namespace(
        advantage_estimator="grpo",
        loss_type="policy_loss",
        recompute_loss_function=False,
        qkv_format="thd",
        calculate_per_token_loss=True,
        allgather_cp=False,
        global_batch_size=2,
    )
    batch = {
        "total_lengths": [4, 3],
        "response_lengths": [2, 1],
        "loss_masks": [torch.ones(2), torch.ones(1)],
        "__loss_scale__": torch.tensor(0.3),
    }

    loss, normalizer, logging = loss_module.loss_function(args, batch, 1, torch.ones(1))

    assert torch.isclose(loss, torch.tensor(6.0))
    assert torch.isclose(normalizer, torch.tensor(3.0))
    assert torch.isclose(logging["values"][1], torch.tensor(6.0))


def _distributed_dr_grpo_metadata_worker(rank: int, world_size: int, port: int, topology: str) -> None:
    import os

    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        if topology == "dp":
            process_group = dist.group.WORLD
            if rank == 0:
                loss_masks = [torch.ones(2)]
            else:
                loss_masks = [torch.ones(1), torch.ones(3)]
            expected_scale = 0.25
        else:
            process_groups = [dist.new_group([group_rank]) for group_rank in range(world_size)]
            process_group = process_groups[rank]
            if rank == 0:
                loss_masks = [torch.ones(2)]
                expected_scale = 0.25
            else:
                loss_masks = [torch.ones(1), torch.ones(1)]
                expected_scale = 0.125

        loss_module.mpu.get_data_parallel_group = lambda **_kwargs: process_group
        loss_module.device_utils.make_current_torch_device = lambda: torch.device("cpu")
        metadata = loss_module.prepare_policy_optimizer_window_metadata(
            Namespace(advantage_estimator="dr_grpo", rollout_max_response_len=8),
            {"loss_masks": loss_masks},
            [len(loss_masks)],
        )

        assert torch.isclose(metadata[0]["__dr_grpo_window_scale__"], torch.tensor(expected_scale))
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("topology", ["dp", "cp"])
def test_dr_grpo_optimizer_window_metadata_uses_real_process_groups(topology: str) -> None:
    import torch.multiprocessing as mp

    world_size = 2
    mp.spawn(
        _distributed_dr_grpo_metadata_worker,
        args=(world_size, _free_port(), topology),
        nprocs=world_size,
        join=True,
    )


def test_dr_grpo_cp_padding_preserves_token_count_and_reduction() -> None:
    total_lengths = [11, 7]
    response_lengths = [8, 4]
    loss_masks = [
        torch.tensor([1, 1, 0, 1, 1, 0, 1, 1], dtype=torch.float32),
        torch.tensor([1, 0, 1, 1], dtype=torch.float32),
    ]
    token_losses = [
        torch.arange(1, 9, dtype=torch.float64),
        torch.arange(11, 15, dtype=torch.float64),
    ]
    configurations = [
        ("thd", None, None),
        ("thd", None, [16, 12]),
        ("bshd", [16, 12], None),
    ]
    expected_tokens = sum(mask.sum() for mask in loss_masks)
    expected_loss = sum((value * mask).sum() for value, mask in zip(token_losses, loss_masks, strict=True))
    reduced_values = []

    for qkv_format, max_seq_lens, padded_total_lengths in configurations:
        total_tokens = torch.tensor(0.0)
        total_loss = torch.tensor(0.0, dtype=torch.float64)
        for cp_rank in range(2):
            local_losses = torch.cat(
                [
                    cp_utils_module.slice_log_prob_with_cp(
                        token_loss,
                        total_length,
                        response_length,
                        qkv_format,
                        max_seq_lens[i] if max_seq_lens is not None else None,
                        padded_total_lengths[i] if padded_total_lengths is not None else None,
                        dynamic_cp_size=2,
                        dynamic_cp_rank=cp_rank,
                    )
                    for i, (token_loss, total_length, response_length) in enumerate(
                        zip(token_losses, total_lengths, response_lengths, strict=True)
                    )
                ]
            )
            reducer = cp_utils_module.get_sum_of_sample_mean(
                total_lengths,
                response_lengths,
                loss_masks,
                calculate_per_token_loss=True,
                qkv_format=qkv_format,
                max_seq_lens=max_seq_lens,
                padded_total_lengths=padded_total_lengths,
                dynamic_cp_size=2,
                dynamic_cp_rank=cp_rank,
            )
            total_loss += reducer(local_losses)
            total_tokens += cp_utils_module.get_cp_local_num_tokens(
                total_lengths,
                response_lengths,
                loss_masks,
                qkv_format=qkv_format,
                max_seq_lens=max_seq_lens,
                padded_total_lengths=padded_total_lengths,
                dynamic_cp_size=2,
                dynamic_cp_rank=cp_rank,
            )

        assert torch.isclose(total_tokens, expected_tokens)
        assert torch.isclose(total_loss, expected_loss)
        reduced_values.append(total_loss)

    assert all(torch.equal(value, reduced_values[0]) for value in reduced_values[1:])


def test_dr_grpo_long_response_weight_is_proportional_to_valid_tokens(monkeypatch) -> None:
    monkeypatch.setattr(loss_module.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(loss_module.mpu, "get_data_parallel_group", lambda **kwargs: None)
    loss_masks = [torch.ones(1), torch.ones(4)]
    metadata = loss_module.prepare_policy_optimizer_window_metadata(
        Namespace(advantage_estimator="dr_grpo", rollout_max_response_len=4),
        {"loss_masks": loss_masks},
        [2],
    )
    loss_scale = metadata[0]["__dr_grpo_window_scale__"]

    def reduce_loss(split_microbatches: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        short_loss = torch.tensor(1.0, requires_grad=True)
        long_loss = torch.tensor(1.0, requires_grad=True)
        if split_microbatches:
            short_reducer = cp_utils_module.get_sum_of_sample_mean(
                [1], [1], loss_masks[:1], calculate_per_token_loss=True, dynamic_cp_size=1
            )
            long_reducer = cp_utils_module.get_sum_of_sample_mean(
                [4], [4], loss_masks[1:], calculate_per_token_loss=True, dynamic_cp_size=1
            )
            numerator = short_reducer(short_loss.expand(1)) + long_reducer(long_loss.expand(4))
        else:
            reducer = cp_utils_module.get_sum_of_sample_mean(
                [1, 4], [1, 4], loss_masks, calculate_per_token_loss=True, dynamic_cp_size=1
            )
            numerator = reducer(torch.cat([short_loss.expand(1), long_loss.expand(4)]))

        loss = numerator * loss_scale / 5
        loss.backward()
        return loss.detach(), short_loss.grad, long_loss.grad

    combined = reduce_loss(split_microbatches=False)
    split = reduce_loss(split_microbatches=True)

    assert torch.isclose(loss_scale, torch.tensor(5 / 8))
    assert torch.isclose(combined[0], torch.tensor(5 / 8))
    assert torch.isclose(combined[1], torch.tensor(1 / 8))
    assert torch.isclose(combined[2], torch.tensor(4 / 8))
    assert all(torch.isclose(actual, expected) for actual, expected in zip(split, combined, strict=True))


def test_dr_grpo_builds_fixed_microbatch_loss_scale_schedule(monkeypatch):
    reduce_groups = []
    monkeypatch.setattr(
        loss_module.dist,
        "all_reduce",
        lambda *args, **kwargs: reduce_groups.append(kwargs.get("group")),
    )
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_world_size", lambda **kwargs: 1)
    monkeypatch.setattr(
        loss_module.mpu,
        "get_data_parallel_group",
        lambda with_context_parallel=True: "dp_cp" if with_context_parallel else "dp_no_cp",
    )
    monkeypatch.setattr(data_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1)
    rollout_data = {
        "loss_masks": [torch.ones(2), torch.ones(1), torch.ones(4)],
        "total_lengths": [4, 3, 5],
        ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY: [2, 1],
    }
    args = Namespace(
        advantage_estimator="dr_grpo",
        rollout_max_response_len=5,
        global_batch_size=2,
        partial_rollout=False,
        use_dynamic_global_batch_size=False,
        balance_data=False,
        use_dynamic_batch_size=False,
        micro_batch_size=1,
    )
    data_iterators, num_microbatches = data_module.get_data_iterator(args, [SimpleNamespace()], rollout_data)
    data_module.bind_optimizer_window_metadata(
        data_iterators,
        num_microbatches,
        loss_module.prepare_policy_optimizer_window_metadata(
            args,
            rollout_data,
            rollout_data[ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY],
        ),
    )
    iterator = data_iterators[0]

    assert num_microbatches == [2, 1]
    assert torch.isclose(iterator.get_next(["loss_masks"])["__dr_grpo_window_scale__"], torch.tensor(0.3))
    assert torch.isclose(iterator.get_next(["loss_masks"])["__dr_grpo_window_scale__"], torch.tensor(0.3))
    assert torch.isclose(iterator.get_next(["loss_masks"])["__dr_grpo_window_scale__"], torch.tensor(0.8))
    iterator.reset()
    assert torch.isclose(iterator.get_next(["loss_masks"])["__dr_grpo_window_scale__"], torch.tensor(0.3))
    assert reduce_groups == ["dp_no_cp"]


def test_dr_grpo_metadata_moves_cpu_mask_stats_to_current_device(monkeypatch):
    reduced_devices = []
    monkeypatch.setattr(loss_module.device_utils, "make_current_torch_device", lambda: torch.device("meta"))
    monkeypatch.setattr(
        loss_module.dist,
        "all_reduce",
        lambda stats, **_kwargs: reduced_devices.append(stats.device),
    )
    monkeypatch.setattr(loss_module.mpu, "get_data_parallel_group", lambda **kwargs: None)

    metadata = loss_module.prepare_policy_optimizer_window_metadata(
        Namespace(advantage_estimator="dr_grpo", rollout_max_response_len=4),
        {"loss_masks": [torch.ones(2)]},
        [1],
    )

    assert reduced_devices == [torch.device("meta")]
    assert metadata[0]["__dr_grpo_window_scale__"].device == torch.device("meta")


def test_dr_grpo_metadata_rejects_zero_response_budget():
    with pytest.raises(ValueError, match="positive integer"):
        loss_module.prepare_policy_optimizer_window_metadata(
            Namespace(advantage_estimator="dr_grpo", rollout_max_response_len=0),
            {"loss_masks": [torch.ones(1)]},
            [1],
        )


def test_optimizer_window_metadata_binding_resets_metadata_offset():
    iterator = data_module.DataIterator(
        {"loss_masks": [torch.ones(1), torch.ones(1)]},
        micro_batch_size=1,
    )
    data_module.bind_optimizer_window_metadata(
        [iterator],
        [2],
        [{"__test_metadata__": torch.tensor(7.0)}],
    )
    iterator.get_next(["loss_masks"])

    data_module.bind_optimizer_window_metadata(
        [iterator],
        [2],
        [{"__test_metadata__": torch.tensor(9.0)}],
    )

    assert iterator.microbatch_offset == 0
    assert torch.isclose(iterator.get_next(["loss_masks"])["__test_metadata__"], torch.tensor(9.0))
