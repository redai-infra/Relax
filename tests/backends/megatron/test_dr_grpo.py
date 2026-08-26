# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import math
import os
import socket
from argparse import Namespace
from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("megatron.core")

import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

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


def test_dr_grpo_centers_rewards_without_dividing_by_group_std():
    """Dr.GRPO keeps the group mean subtraction but drops the std division.

    ``--disable-rewards-normalization`` is rejected for Dr.GRPO in
    ``_validate_dr_grpo_args``, so centering is guaranteed before this function
    runs; see ``test_dr_grpo_rejects_incompatible_semantics``. What is specific
    to Dr.GRPO here is that ``grpo_std_normalization`` no longer applies.
    """
    args = SimpleNamespace(
        advantage_estimator="dr_grpo",
        custom_reward_post_process_path=None,
        agentic_custom_advantage_path=None,
        rewards_normalization=True,
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
    exact_count_modes = []

    def get_num_tokens(*_args, **kwargs):
        exact_count_modes.append(kwargs["use_exact_loss_mask_count"])
        return torch.tensor(3.0)

    monkeypatch.setattr(loss_module, "get_cp_local_num_tokens", get_num_tokens)
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
        advantage_estimator="dr_grpo",
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
    assert exact_count_modes == [True]


def test_non_dr_grpo_per_token_loss_ignores_streaming_loss_scale(monkeypatch):
    exact_count_modes = []

    def get_num_tokens(*_args, **kwargs):
        exact_count_modes.append(kwargs["use_exact_loss_mask_count"])
        return torch.tensor(3.0)

    monkeypatch.setattr(loss_module, "get_cp_local_num_tokens", get_num_tokens)
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
    assert exact_count_modes == [False]


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
                use_exact_loss_mask_count=True,
            )

        assert torch.isclose(total_tokens, expected_tokens)
        assert torch.isclose(total_loss, expected_loss)
        reduced_values.append(total_loss)

    assert all(torch.equal(value, reduced_values[0]) for value in reduced_values[1:])


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_dr_grpo_token_count_counts_fully_masked_response_as_zero(cp_size: int) -> None:
    """Dr.GRPO counts real unmasked tokens at every CP degree.

    A filtered sample stays in the batch with an all-zero loss_mask. Clamping
    its count to 1 at CP=1 made that branch report one more token than CP>1 for
    the same data -- the asymmetry behind the CP-dependent Dr.GRPO denominator.
    """
    total_lengths = [16, 8]
    response_lengths = [8, 4]
    loss_masks = [torch.ones(8, dtype=torch.float64), torch.zeros(4, dtype=torch.float64)]

    total = sum(
        cp_utils_module.get_cp_local_num_tokens(
            total_lengths,
            response_lengths,
            loss_masks,
            dynamic_cp_size=cp_size,
            dynamic_cp_rank=cp_rank,
            use_exact_loss_mask_count=True,
        )
        for cp_rank in range(cp_size)
    )

    assert float(total) == 8.0, f"cp_size={cp_size}"


@pytest.mark.parametrize("cp_size", [1, 2])
def test_dr_grpo_token_count_is_zero_when_every_response_is_masked(cp_size: int) -> None:
    """A fully-filtered Dr.GRPO window yields a zero denominator.

    Callers must guard their own division; ``finalize_model_grads`` uses a safe
    denominator and may omit scaling or apply identity scaling.
    """
    loss_masks = [torch.zeros(8, dtype=torch.float64), torch.zeros(4, dtype=torch.float64)]

    total = sum(
        cp_utils_module.get_cp_local_num_tokens(
            [16, 8],
            [8, 4],
            loss_masks,
            dynamic_cp_size=cp_size,
            dynamic_cp_rank=cp_rank,
            use_exact_loss_mask_count=True,
        )
        for cp_rank in range(cp_size)
    )

    assert float(total) == 0.0, f"cp_size={cp_size}"


def test_non_dr_cp1_token_count_preserves_legacy_floor_for_mixed_masks() -> None:
    loss_masks = [torch.ones(8), torch.zeros(4)]

    total = cp_utils_module.get_cp_local_num_tokens(
        [16, 8],
        [8, 4],
        loss_masks,
        dynamic_cp_size=1,
        dynamic_cp_rank=0,
    )
    reducer = cp_utils_module.get_sum_of_sample_mean(
        [16, 8],
        [8, 4],
        loss_masks,
        calculate_per_token_loss=True,
        dynamic_cp_size=1,
        dynamic_cp_rank=0,
    )

    assert float(total) == 9.0
    assert float(reducer(torch.ones(12))) == 8.0


def test_non_dr_cp1_token_count_preserves_legacy_floor_for_all_zero_masks() -> None:
    loss_masks = [torch.zeros(8), torch.zeros(4)]

    total = cp_utils_module.get_cp_local_num_tokens(
        [16, 8],
        [8, 4],
        loss_masks,
        dynamic_cp_size=1,
        dynamic_cp_rank=0,
    )
    reducer = cp_utils_module.get_sum_of_sample_mean(
        [16, 8],
        [8, 4],
        loss_masks,
        calculate_per_token_loss=True,
        dynamic_cp_size=1,
        dynamic_cp_rank=0,
    )

    assert float(total) == 2.0
    assert float(reducer(torch.ones(12))) == 0.0


def test_dr_grpo_flags_fully_masked_window_and_leaves_others_alone(monkeypatch) -> None:
    """A fully filtered window must be flagged and zero-scaled.

    The gradient is then exactly zero, but stepping is not a no-op: Adam still
    decays its moments and the scheduler still burns a global batch of LR
    budget. ``train_one_step`` skips both when it sees this flag.
    """
    monkeypatch.setattr(loss_module.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(loss_module.mpu, "get_data_parallel_group", lambda **kwargs: None)

    metadata = loss_module.prepare_policy_optimizer_window_metadata(
        Namespace(advantage_estimator="dr_grpo", rollout_max_response_len=4),
        # Window 0: every response fully masked. Window 1: one masked, one kept.
        {"loss_masks": [torch.zeros(4), torch.zeros(4), torch.zeros(4), torch.ones(4)]},
        [2, 2],
    )

    assert float(metadata[0]["__dr_grpo_window_scale__"]) == 0.0
    assert bool(metadata[0]["__optimizer_window_empty__"]) is True
    # A partially masked window still trains, and must not be skipped.
    assert float(metadata[1]["__dr_grpo_window_scale__"]) == 0.5
    assert bool(metadata[1]["__optimizer_window_empty__"]) is False


def test_optimizer_window_empty_flag_reaches_every_micro_batch(monkeypatch):
    """Every micro-batch of a window must carry that window's own verdict.

    ``train_one_step`` reads the flag off the batch in ``forward_step`` rather
    than taking it as an argument, so this delivery path is what it relies on.
    """
    monkeypatch.setattr(loss_module.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(loss_module.mpu, "get_data_parallel_group", lambda **kwargs: None)

    num_microbatches = [2, 1]
    metadata = loss_module.prepare_policy_optimizer_window_metadata(
        Namespace(advantage_estimator="dr_grpo", rollout_max_response_len=4),
        # Window 0 (2 micro-batches) fully masked; window 1 (1 micro-batch) trains.
        {"loss_masks": [torch.zeros(4), torch.zeros(4), torch.ones(4)]},
        [2, 1],
    )
    iterator = data_module.DataIterator({"loss_masks": [torch.zeros(4)] * 3}, micro_batch_size=1)
    data_module.bind_optimizer_window_metadata([iterator], num_microbatches, metadata)

    flags = [bool(iterator.get_next(["loss_masks"])["__optimizer_window_empty__"]) for _ in range(3)]

    assert flags == [True, True, False]


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


def test_dr_grpo_metadata_all_reduces_large_counts_as_int64(monkeypatch) -> None:
    response_budget = 2**24 + 20
    global_response_tokens = 2**24 + 11
    reductions = []

    class SyntheticLossMask:
        def __init__(self, count: int) -> None:
            self.count = count

        def sum(self, dtype=None) -> torch.Tensor:
            return torch.tensor(self.count, dtype=dtype)

    def prepare(local_counts: list[int], remote_samples: int, remote_tokens: int) -> torch.Tensor:
        def all_reduce(stats: torch.Tensor, *, group) -> None:
            reductions.append((stats.clone(), group))
            stats.add_(torch.tensor([[remote_samples, remote_tokens]], dtype=torch.int64))

        monkeypatch.setattr(loss_module.dist, "all_reduce", all_reduce)
        metadata = loss_module.prepare_policy_optimizer_window_metadata(
            Namespace(advantage_estimator="dr_grpo", rollout_max_response_len=response_budget),
            {"loss_masks": [SyntheticLossMask(count) for count in local_counts]},
            [len(local_counts)],
        )
        return metadata[0]["__dr_grpo_window_scale__"]

    monkeypatch.setattr(
        loss_module.mpu,
        "get_data_parallel_group",
        lambda with_context_parallel=True: "dp_cp" if with_context_parallel else "dp_no_cp",
    )
    split_scale = prepare([2**24 + 1], remote_samples=1, remote_tokens=10)
    local_scale = prepare([2**24, 11], remote_samples=0, remote_tokens=0)
    expected_scale = torch.tensor(global_response_tokens, dtype=torch.float32) / torch.tensor(
        2 * response_budget, dtype=torch.float32
    )

    assert all(stats.dtype == torch.int64 for stats, _group in reductions)
    assert reductions[0][0].tolist() == [[1, 2**24 + 1]]
    assert reductions[1][0].tolist() == [[2, global_response_tokens]]
    assert [group for _stats, group in reductions] == ["dp_no_cp", "dp_no_cp"]
    assert torch.equal(split_scale, expected_scale)
    assert torch.equal(local_scale, expected_scale)


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


# ===========================================================================
# Frozen mixed-length reduction oracle
#
# The window holds 16 responses with lengths 1024/256/64/16 (four each),
# irregular prompt lengths and three loss-mask shapes, giving N = 16 responses
# and T = 4930 loss-contributing tokens. With ppo_kl == 0 (old log probs equal
# the current ones, as on the first optimizer step) the production PPO clip
# degenerates to -advantage per token, so the raw actor numerator is exactly
# S = 402 and the reductions are integer-exact:
#
#     alpha    = T / (N * B) = 4930 / 16384
#     Dr.GRPO  = S / (N * B) = 402 / 16384 = 0.0245361328125
#     GRPO     = S_grpo / T                = 0.031580906737842154
#
# Every DP/CP topology and micro-batch size must hit the same numbers.
#
# Production code under test: the optimizer-window metadata provider, the CP
# slice/reducer pair, compute_policy_loss, loss_function's per-token scaling and
# logging vector, and the metric reduction formula from model.py::train_one_step.
# The single stubbed step is get_log_probs_and_entropy (logits -> log probs),
# which needs a TP group and real model logits; ppo_kl == 0 pins its
# contribution analytically.
#
# Not covered: model forward numerics, Megatron DDP finalize_model_grads,
# optimizer step, and BF16/TE kernel behaviour.
# ===========================================================================

RESPONSE_LENGTH_BLOCKS = (1024, 256, 64, 16)
PROMPT_LENGTHS = (31, 63, 127, 255)
REWARD_VALUES = (-3.0, -1.0, 1.0, 3.0)
GROUP_SIZE = 4
NUM_RESPONSES = 16
RESPONSE_BUDGET = 1024

# Frozen oracle. Derived by hand from the window below, never from the reducers.
FROZEN_RESPONSE_TOKENS = 4930
FROZEN_DR_GRPO_NUMERATOR = 402
FROZEN_DR_GRPO_ALPHA = Fraction(FROZEN_RESPONSE_TOKENS, NUM_RESPONSES * RESPONSE_BUDGET)
FROZEN_DR_GRPO_LOSS = Fraction(FROZEN_DR_GRPO_NUMERATOR, NUM_RESPONSES * RESPONSE_BUDGET)
FROZEN_GRPO_LOSS = 0.031580906737842154
# GRPO divides by ``group_rewards.std() + 1e-6``; that epsilon alone shifts the
# token mean by ~1.2e-8, which is what the reported tolerance accounts for.
GRPO_ORACLE_ATOL = 1.7e-8

TOPOLOGIES = ((1, 1), (2, 1), (4, 1), (1, 2), (2, 2), (1, 4))
MICRO_BATCH_SIZES = (1, 2)

# OPSM with a negative delta masks every negative-advantage sample (seq_kl == 0
# here, and 0 > -1), leaving only the advantage >= 0 tokens.
OPSM_FORCING_DELTA = -1.0
FROZEN_OPSM_KEPT_TOKENS = 2390
FROZEN_OPSM_MASKED_TOKENS = FROZEN_RESPONSE_TOKENS - FROZEN_OPSM_KEPT_TOKENS  # 2540
FROZEN_OPSM_NUMERATOR = -4774
FROZEN_OPSM_LOSS = Fraction(FROZEN_OPSM_NUMERATOR, NUM_RESPONSES * RESPONSE_BUDGET)

# Emulated Qwen3.5 bridge layout: maybe_padded_total_lengths aligns to tp * cp * 2.
BRIDGE_TENSOR_PARALLEL_SIZE = 2

ENTROPY_COEF = 0.01
KL_LOSS_COEF = 0.01
REFERENCE_LOG_PROB = -0.5

# One response whose entire loss_mask is zeroed, as --rollout-sample-filter-path
# does. Index 1 carries 896 valid tokens, well above any rounding slack.
FULLY_MASKED_INDEX = 1
FROZEN_MASKED_RESPONSE_TOKENS = FROZEN_RESPONSE_TOKENS - 896  # 4034

# Four responses that share advantage -3 but differ ~73x in valid token count.
# Their gradients must stay proportional to those counts, which is the
# long-vs-short weighting claim at a far wider spread than a toy two-sample
# batch can express.
SAME_ADVANTAGE_INDICES = (0, 7, 10, 13)
SAME_ADVANTAGE_VALID_TOKENS = (1024, 224, 56, 14)

# With ppo_kl == 0 the per-token derivative is -advantage, so a response
# contributes -advantage * valid_tokens, and the window's total gradient is
# alpha * S. Freezing the absolute value matters: a ratio-only check still
# passes if every gradient is scaled by the same wrong constant (alpha applied
# twice, say).
FROZEN_DR_GRPO_GRADIENT_TOTAL = Fraction(FROZEN_DR_GRPO_NUMERATOR * FROZEN_RESPONSE_TOKENS, 16384)


def _build_loss_mask(response_length: int, variant: int) -> list[int]:
    """Tail / interior-span / strided masks, one shape per group member."""
    loss_mask = [1] * response_length
    if variant == 1:
        loss_mask[-max(response_length // 8, 1) :] = [0] * max(response_length // 8, 1)
    elif variant == 2:
        start = response_length // 3
        width = max(response_length // 8, 1)
        loss_mask[start : start + width] = [0] * width
    elif variant == 3:
        loss_mask[::8] = [0] * len(loss_mask[::8])
    return loss_mask


def _build_window(fully_masked_indices: tuple[int, ...] = ()) -> list[dict]:
    """16 responses ordered so contiguous DP shards are token-imbalanced.

    ``fully_masked_indices`` zeroes those responses' entire loss_mask, which is
    what ``--rollout-sample-filter-path`` does to a filtered sample (utils.py
    keeps the sample in the batch and sets ``loss_mask = [0] * response_length``).
    The default empty tuple reproduces the frozen fixture exactly.
    """
    window = []
    for length_block, response_length in enumerate(RESPONSE_LENGTH_BLOCKS):
        for group_index in range(GROUP_SIZE):
            offset = (length_block + group_index) % GROUP_SIZE
            index = length_block * GROUP_SIZE + group_index
            loss_mask = (
                [0] * response_length
                if index in fully_masked_indices
                else _build_loss_mask(response_length, group_index)
            )
            window.append(
                {
                    "index": index,
                    "group_index": group_index,
                    "prompt_length": PROMPT_LENGTHS[offset],
                    "response_length": response_length,
                    "reward": REWARD_VALUES[offset],
                    "loss_mask": loss_mask,
                }
            )
    return window


def _independent_oracle(advantage_estimator: str) -> tuple[int, float]:
    """Recompute ``(T, S)`` with plain Python arithmetic.

    Deliberately does not call ``post_process_rewards`` or any reducer, so a
    bug that is shared between the production advantage path and the reduction
    path cannot make the oracle agree with itself.
    """
    window = _build_window()
    groups: dict[int, list[dict]] = {}
    for response in window:
        groups.setdefault(response["group_index"], []).append(response)

    response_tokens = sum(sum(response["loss_mask"]) for response in window)
    numerator = 0.0
    for members in groups.values():
        rewards = [member["reward"] for member in members]
        mean = sum(rewards) / len(rewards)
        advantages = [reward - mean for reward in rewards]
        if advantage_estimator != "dr_grpo":
            variance = sum(advantage**2 for advantage in advantages) / (len(advantages) - 1)
            scale = math.sqrt(variance) + 1e-6
            advantages = [advantage / scale for advantage in advantages]
        for member, advantage in zip(members, advantages, strict=True):
            # compute_policy_loss yields -ratio * advantage, and ratio == 1 here.
            numerator += -advantage * sum(member["loss_mask"])
    return response_tokens, numerator


def _production_advantages(advantage_estimator: str) -> dict[int, float]:
    """Per-response advantage from the production reward post-processing."""
    window = _build_window()
    samples = [
        Sample(group_index=response["group_index"], index=response["index"], reward=response["reward"])
        for response in window
    ]
    args = Namespace(
        advantage_estimator=advantage_estimator,
        custom_reward_post_process_path=None,
        agentic_custom_advantage_path=None,
        rewards_normalization=True,
        grpo_std_normalization=True,
        n_samples_per_prompt=GROUP_SIZE,
        reward_key=None,
    )
    _, advantages = post_process_rewards(args, samples)
    return {response["index"]: advantage for response, advantage in zip(window, advantages, strict=True)}


def _loss_args(advantage_estimator: str, **overrides) -> Namespace:
    args = Namespace(
        advantage_estimator=advantage_estimator,
        loss_type="policy_loss",
        recompute_loss_function=False,
        qkv_format="thd",
        calculate_per_token_loss=True,
        allgather_cp=False,
        global_batch_size=NUM_RESPONSES,
        rollout_max_response_len=RESPONSE_BUDGET,
        use_rollout_logprobs=False,
        use_opsm=False,
        opsm_delta=0.0,
        opd_token_selection="none",
        eps_clip=0.2,
        eps_clip_high=0.2,
        get_mismatch_metrics=False,
        use_tis=False,
        entropy_coef=0.0,
        use_kl_loss=False,
        kl_loss_type="low_var_kl",
        use_unbiased_kl=False,
        kl_loss_coef=0.0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _install_cpu_parallel_state(cp_size, cp_rank, cp_group, dp_without_cp_group) -> None:
    """Point the Megatron accessors at the Gloo groups built by this worker."""

    def _data_parallel_group(with_context_parallel: bool = False):
        # Freezes the contract: Dr.GRPO's (N, T) must never be reduced on a group
        # that carries CP replicas, or every sample would be counted cp_size times.
        assert with_context_parallel is False, "Dr.GRPO metadata must use the DP-without-CP group"
        return dp_without_cp_group

    loss_module.device_utils.make_current_torch_device = lambda: torch.device("cpu")
    loss_module.mpu.get_data_parallel_group = _data_parallel_group
    cp_utils_module.mpu.get_context_parallel_world_size = lambda: cp_size
    cp_utils_module.mpu.get_context_parallel_rank = lambda: cp_rank
    cp_utils_module.mpu.get_context_parallel_group = lambda: cp_group


def _bridge_padded_total_lengths(total_lengths: list[int]) -> list[int] | None:
    """Explicit ``tp * cp * 2`` padded lengths, as the Qwen3.5 bridge path emits.

    The plain THD path leaves ``padded_total_lengths=None`` and lets the CP
    helpers derive ``ceil(total_length / (2 * cp))`` implicitly. The bridge path
    instead pads every sample up front, so both layouts must be exercised.
    """
    original = cp_utils_module.mpu.get_tensor_model_parallel_world_size
    cp_utils_module.mpu.get_tensor_model_parallel_world_size = lambda: BRIDGE_TENSOR_PARALLEL_SIZE
    try:
        return cp_utils_module.maybe_padded_total_lengths(total_lengths, "thd", is_vl_model=True)
    finally:
        cp_utils_module.mpu.get_tensor_model_parallel_world_size = original


def _cp_local_slice(tensor, total_length, response_length, padded_total_length):
    return cp_utils_module.slice_log_prob_with_cp(
        tensor, total_length, response_length, "thd", None, padded_total_length
    )


def _cp_local_response_shapes(total_lengths, response_lengths, padded_total_lengths) -> list[int]:
    """CP-local response lengths, taken from the production slicing helper."""
    padded = padded_total_lengths if padded_total_lengths is not None else [None] * len(total_lengths)
    return [
        _cp_local_slice(
            torch.zeros(response_length, dtype=torch.float64), total_length, response_length, padded_total_length
        ).numel()
        for total_length, response_length, padded_total_length in zip(
            total_lengths, response_lengths, padded, strict=True
        )
    ]


def _stub_log_probs_and_entropy(entropy_value: float, theta_sink: list | None = None):
    """Stand in for the logits -> log-prob step (needs a TP group and real
    logits).

    Without ``theta_sink`` it returns zeros, so ``ppo_kl == 0`` and the PPO ratio
    is exactly 1 -- the first-optimizer-step condition the oracle assumes. With a
    sink it returns ``theta_i * ones(...)`` per response, still zero-valued, so
    the ratio is unchanged but each response gains a differentiable handle for
    the per-response weighting assertions.
    """

    def _stub(logits, **kwargs):
        shapes = _cp_local_response_shapes(
            kwargs["total_lengths"], kwargs["response_lengths"], kwargs.get("padded_total_lengths")
        )
        if theta_sink is None:
            log_probs = [torch.zeros(size, dtype=torch.float64) for size in shapes]
        else:
            thetas = [torch.zeros((), dtype=torch.float64, requires_grad=True) for _ in shapes]
            theta_sink.clear()
            theta_sink.extend(thetas)
            log_probs = [
                theta * torch.ones(size, dtype=torch.float64) for theta, size in zip(thetas, shapes, strict=True)
            ]
        entropy = [torch.full((size,), entropy_value, dtype=torch.float64) for size in shapes]
        return None, {"log_probs": log_probs, "entropy": entropy}

    return _stub


def _build_micro_batch(members, advantages, extra_fields, use_bridge_padding, needs_reference):
    total_lengths = [member["prompt_length"] + member["response_length"] for member in members]
    response_lengths = [member["response_length"] for member in members]
    loss_masks = [torch.tensor(member["loss_mask"], dtype=torch.float64) for member in members]

    padded_total_lengths = _bridge_padded_total_lengths(total_lengths) if use_bridge_padding else None
    padded = padded_total_lengths if padded_total_lengths is not None else [None] * len(members)

    cp_local_advantages = []
    for member, total_length, response_length, padded_total_length in zip(
        members, total_lengths, response_lengths, padded, strict=True
    ):
        full = torch.full((response_length,), advantages[member["index"]], dtype=torch.float64)
        cp_local_advantages.append(_cp_local_slice(full, total_length, response_length, padded_total_length))

    batch = {
        "total_lengths": total_lengths,
        "response_lengths": response_lengths,
        "loss_masks": loss_masks,
        "advantages": cp_local_advantages,
        "log_probs": [torch.zeros_like(tensor) for tensor in cp_local_advantages],
        "unconcat_tokens": [None] * len(members),
    }
    if padded_total_lengths is not None:
        batch["padded_total_lengths"] = padded_total_lengths
    if needs_reference:
        batch["ref_log_probs"] = [torch.full_like(tensor, REFERENCE_LOG_PROB) for tensor in cp_local_advantages]
    batch.update(extra_fields)
    return batch


def _split_micro_batches(local, micro_batch_size, micro_batch_groups):
    """Fixed-size chunks, or an explicitly irregular dynamic-batching
    grouping."""
    if micro_batch_groups is None:
        return [local[start : start + micro_batch_size] for start in range(0, len(local), micro_batch_size)]
    grouped = [[local[position] for position in group] for group in micro_batch_groups]
    assert sorted(position for group in micro_batch_groups for position in group) == list(range(len(local)))
    return grouped


def _reduce_window(
    *,
    advantage_estimator,
    dp_size,
    dp_rank,
    cp_size,
    dp_with_cp_group,
    micro_batch_size=1,
    micro_batch_groups=None,
    entropy_value=0.0,
    arg_overrides=None,
    batch_fields=None,
    use_bridge_padding=False,
    collect_gradients=False,
    cp_group=None,
    fully_masked_indices=(),
):
    """Run one optimizer window end to end and return the reported metrics."""
    window = _build_window(fully_masked_indices)
    advantages = _production_advantages(advantage_estimator)
    shard = NUM_RESPONSES // dp_size
    local = window[dp_rank * shard : (dp_rank + 1) * shard]

    args = _loss_args(advantage_estimator, **(arg_overrides or {}))
    metadata = loss_module.prepare_policy_optimizer_window_metadata(
        args,
        {"loss_masks": [torch.tensor(member["loss_mask"], dtype=torch.float64) for member in local]},
        [len(local)],
    )

    micro_batches = _split_micro_batches(local, micro_batch_size, micro_batch_groups)
    theta_sink = [] if collect_gradients else None
    loss_module.get_log_probs_and_entropy = _stub_log_probs_and_entropy(entropy_value, theta_sink)

    keys = None
    values = None
    gradients: dict[int, torch.Tensor] = {}
    for members in micro_batches:
        extra = dict(batch_fields or {})
        if metadata is not None:
            extra["__dr_grpo_window_scale__"] = metadata[0]["__dr_grpo_window_scale__"]
        batch = _build_micro_batch(members, advantages, extra, use_bridge_padding, args.use_kl_loss)
        loss, _, logging = loss_module.loss_function(args, batch, len(micro_batches), torch.zeros(1))
        keys = logging["keys"]
        values = logging["values"] if values is None else values + logging["values"]
        if collect_gradients:
            loss.backward()
            for member, theta in zip(members, theta_sink, strict=True):
                gradients[member["index"]] = theta.grad.clone()

    # Same reduction as model.py::train_one_step: sum the logging vectors across
    # micro-batches, all-reduce over DP-with-CP, then divide by the global token
    # count that rides in slot 0.
    dist.all_reduce(values, group=dp_with_cp_group)
    values = values.tolist()
    reported = {key: value / values[0] for key, value in zip(keys, values[1:], strict=True)}
    reported["__response_tokens__"] = values[0]
    if metadata is not None:
        reported["__alpha__"] = float(metadata[0]["__dr_grpo_window_scale__"])
    if collect_gradients:
        # Each rank holds only its CP shard of every response; sum to recover the
        # per-response gradient the optimizer would actually see.
        for index, gradient in gradients.items():
            if cp_size > 1:
                dist.all_reduce(gradient, group=cp_group)
            reported.setdefault("__gradients__", {})[index] = gradient.item()
            # Replay finalize_model_grads' 1 / (all-reduced token count) scaling.
            # A pre-finalizer gradient cannot see a mismatch between that
            # denominator and the count folded into alpha -- where CP-topology
            # dependence hides.
            scaling = 1.0 / values[0] if values[0] > 0 else 1.0
            reported.setdefault("__gradients_after_finalizer__", {})[index] = gradient.item() * scaling
    return reported


def test_dr_grpo_window_fixture_matches_frozen_constants():
    window = _build_window()
    valid_token_counts = [sum(response["loss_mask"]) for response in window]

    assert len(window) == NUM_RESPONSES
    assert valid_token_counts == [1024, 896, 896, 896, 256, 224, 224, 224, 64, 56, 56, 56, 16, 14, 14, 14]
    assert sum(valid_token_counts) == FROZEN_RESPONSE_TOKENS

    dr_grpo_tokens, dr_grpo_numerator = _independent_oracle("dr_grpo")
    assert dr_grpo_tokens == FROZEN_RESPONSE_TOKENS
    assert dr_grpo_numerator == pytest.approx(FROZEN_DR_GRPO_NUMERATOR, abs=0.0)
    assert float(FROZEN_DR_GRPO_LOSS) == 0.0245361328125

    _, grpo_numerator = _independent_oracle("grpo")
    assert grpo_numerator / FROZEN_RESPONSE_TOKENS == pytest.approx(FROZEN_GRPO_LOSS, abs=GRPO_ORACLE_ATOL)


def test_dr_grpo_production_advantages_only_center_within_group():
    dr_grpo = _production_advantages("dr_grpo")
    grpo = _production_advantages("grpo")

    # Each group holds one of every reward, so centering alone leaves the raw values.
    assert sorted(round(value, 6) for value in dr_grpo.values()) == sorted(
        round(float(reward), 6) for reward in REWARD_VALUES * GROUP_SIZE
    )
    # GRPO additionally divides by the group std, shrinking every magnitude.
    assert all(abs(grpo[index]) < abs(dr_grpo[index]) for index in dr_grpo)


def _topology_worker(rank: int, dp_size: int, cp_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=dp_size * cp_size)
    try:
        dp_rank, cp_rank = divmod(rank, cp_size)

        # Every rank creates every group in the same order, or group creation
        # itself can mismatch or hang.
        cp_groups = [
            dist.new_group([replica * cp_size + index for index in range(cp_size)]) for replica in range(dp_size)
        ]
        dp_without_cp_groups = [
            dist.new_group([replica * cp_size + index for replica in range(dp_size)]) for index in range(cp_size)
        ]
        _install_cpu_parallel_state(cp_size, cp_rank, cp_groups[dp_rank], dp_without_cp_groups[cp_rank])

        for advantage_estimator in ("dr_grpo", "grpo"):
            expected = float(FROZEN_DR_GRPO_LOSS) if advantage_estimator == "dr_grpo" else FROZEN_GRPO_LOSS
            for micro_batch_size in MICRO_BATCH_SIZES:
                reported = _reduce_window(
                    advantage_estimator=advantage_estimator,
                    dp_size=dp_size,
                    dp_rank=dp_rank,
                    cp_size=cp_size,
                    micro_batch_size=micro_batch_size,
                    dp_with_cp_group=dist.group.WORLD,
                )
                context = f"{advantage_estimator} DP{dp_size}/CP{cp_size}/MBS{micro_batch_size}"

                assert reported["__response_tokens__"] == FROZEN_RESPONSE_TOKENS, context
                if advantage_estimator == "dr_grpo":
                    assert reported["__alpha__"] == float(FROZEN_DR_GRPO_ALPHA), context
                    assert reported["loss"] == expected, context
                    assert reported["pg_loss"] == expected, context
                else:
                    assert "__alpha__" not in reported, context
                    assert abs(reported["loss"] - expected) < GRPO_ORACLE_ATOL, context
                assert reported["pg_clipfrac"] == 0.0, context
                assert reported["ppo_kl"] == 0.0, context

            # The window's configured B happens to equal max(response_lengths), so
            # the numbers above cannot tell a configured budget from an observed
            # maximum. Doubling only the configured budget must halve Dr.GRPO
            # exactly and leave the GRPO token mean untouched.
            doubled = _reduce_window(
                advantage_estimator=advantage_estimator,
                dp_size=dp_size,
                dp_rank=dp_rank,
                cp_size=cp_size,
                micro_batch_size=1,
                dp_with_cp_group=dist.group.WORLD,
                arg_overrides={"rollout_max_response_len": 2 * RESPONSE_BUDGET},
            )
            context = f"{advantage_estimator} DP{dp_size}/CP{cp_size} with B={2 * RESPONSE_BUDGET}"
            if advantage_estimator == "dr_grpo":
                assert doubled["__alpha__"] == float(FROZEN_DR_GRPO_ALPHA) / 2, context
                assert doubled["loss"] == expected / 2, context
            else:
                assert abs(doubled["loss"] - expected) < GRPO_ORACLE_ATOL, context

        _assert_dynamic_batching(dp_size, dp_rank, cp_size)
        _assert_bridge_padding(dp_size, dp_rank, cp_size)
        _assert_entropy_and_kl_combination(dp_size, dp_rank, cp_size)
        _assert_opsm_forced_branch(dp_size, dp_rank, cp_size)
        _assert_response_weight_tracks_valid_tokens(dp_size, dp_rank, cp_size, cp_groups[dp_rank])
        _assert_fully_masked_response_keeps_denominators_aligned(dp_size, dp_rank, cp_size, cp_groups[dp_rank])
    finally:
        dist.destroy_process_group()


def _dynamic_micro_batch_groups(local_size: int) -> list[list[int]]:
    """Irregular groupings like the ``max_tokens_per_gpu`` dynamic batcher
    emits.

    Fixed-size chunking can hide a window scale that drifts with micro-batch
    membership, because every micro-batch then holds the same sample count.
    """
    groupings = {
        16: [[3], [2, 14], [5, 7, 8, 9, 13], [1, 11], [0, 15], [4, 6, 10, 12]],
        8: [[5], [0, 3, 6], [2, 7], [1, 4]],
        4: [[2], [0, 1, 3]],
    }
    return groupings[local_size]


def _assert_dynamic_batching(dp_size: int, dp_rank: int, cp_size: int) -> None:
    """Section 3.5: uneven micro-batch membership must not move the oracle."""
    reported = _reduce_window(
        advantage_estimator="dr_grpo",
        dp_size=dp_size,
        dp_rank=dp_rank,
        cp_size=cp_size,
        dp_with_cp_group=dist.group.WORLD,
        micro_batch_groups=_dynamic_micro_batch_groups(NUM_RESPONSES // dp_size),
    )
    context = f"dynamic batching DP{dp_size}/CP{cp_size}"
    assert reported["__response_tokens__"] == FROZEN_RESPONSE_TOKENS, context
    assert reported["__alpha__"] == float(FROZEN_DR_GRPO_ALPHA), context
    assert reported["loss"] == float(FROZEN_DR_GRPO_LOSS), context


def _assert_bridge_padding(dp_size: int, dp_rank: int, cp_size: int) -> None:
    """Section 3.3: explicit ``tp * cp * 2`` padding must hit the same oracle."""
    if cp_size == 1:
        return  # maybe_padded_total_lengths only applies to the CP>1 bridge path.
    reported = _reduce_window(
        advantage_estimator="dr_grpo",
        dp_size=dp_size,
        dp_rank=dp_rank,
        cp_size=cp_size,
        dp_with_cp_group=dist.group.WORLD,
        use_bridge_padding=True,
    )
    context = f"bridge padding DP{dp_size}/CP{cp_size}"
    # Padding tokens must not reach T, the numerator, or the reported loss.
    assert reported["__response_tokens__"] == FROZEN_RESPONSE_TOKENS, context
    assert reported["loss"] == float(FROZEN_DR_GRPO_LOSS), context


def _assert_entropy_and_kl_combination(dp_size: int, dp_rank: int, cp_size: int) -> None:
    """Section 3.6: entropy and explicit KL ride the same fixed-budget
    scale."""
    reported = _reduce_window(
        advantage_estimator="dr_grpo",
        dp_size=dp_size,
        dp_rank=dp_rank,
        cp_size=cp_size,
        dp_with_cp_group=dist.group.WORLD,
        entropy_value=0.75,
        arg_overrides={
            "entropy_coef": ENTROPY_COEF,
            "use_kl_loss": True,
            "kl_loss_coef": KL_LOSS_COEF,
        },
    )
    context = f"entropy+KL DP{dp_size}/CP{cp_size}"
    combined = reported["pg_loss"] - ENTROPY_COEF * reported["entropy_loss"] + KL_LOSS_COEF * reported["kl_loss"]
    assert abs(reported["loss"] - combined) < 1e-12, context
    # pg_loss keeps its own oracle, so entropy/KL did not leak into it.
    assert reported["pg_loss"] == float(FROZEN_DR_GRPO_LOSS), context
    # Every objective component carries alpha: an unscaled entropy mean would be
    # the raw 0.75, not 0.75 * alpha.
    assert abs(reported["entropy_loss"] - 0.75 * float(FROZEN_DR_GRPO_ALPHA)) < 1e-12, context
    # compute_approx_kl casts its inputs to float32, so this term cannot carry the
    # float64 tolerance used above. The margin still dwarfs the failure being
    # guarded against: a missing alpha would report the raw 0.1065, not 0.0321.
    reference_kl = math.exp(REFERENCE_LOG_PROB) - REFERENCE_LOG_PROB - 1
    assert abs(reported["kl_loss"] - reference_kl * float(FROZEN_DR_GRPO_ALPHA)) < 1e-6, context


def _assert_opsm_forced_branch(dp_size: int, dp_rank: int, cp_size: int) -> None:
    """Section 3.7: OPSM masking keeps the Dr.GRPO fixed denominator."""
    reported = _reduce_window(
        advantage_estimator="dr_grpo",
        dp_size=dp_size,
        dp_rank=dp_rank,
        cp_size=cp_size,
        dp_with_cp_group=dist.group.WORLD,
        arg_overrides={"use_opsm": True, "opsm_delta": OPSM_FORCING_DELTA},
    )
    context = f"OPSM DP{dp_size}/CP{cp_size}"
    # The denominator stays the full T even though only 2390 tokens survive.
    assert reported["__response_tokens__"] == FROZEN_RESPONSE_TOKENS, context
    assert reported["opsm_clipfrac"] == FROZEN_OPSM_MASKED_TOKENS / FROZEN_RESPONSE_TOKENS, context
    assert reported["pg_loss"] == float(FROZEN_OPSM_LOSS), context
    assert reported["loss"] == float(FROZEN_OPSM_LOSS), context


def _assert_response_weight_tracks_valid_tokens(dp_size, dp_rank, cp_size, cp_group) -> None:
    """A response's gradient weight must scale with its valid token count.

    Uses the four fixture responses that share advantage -3 but span 1024 -> 14
    valid tokens (~73x), instead of a two-sample batch whose lengths differ by
    a small factor. Only meaningful when one rank owns all four, i.e. DP=1.
    """
    if dp_size != 1:
        return
    reported = _reduce_window(
        advantage_estimator="dr_grpo",
        dp_size=dp_size,
        dp_rank=dp_rank,
        cp_size=cp_size,
        dp_with_cp_group=dist.group.WORLD,
        collect_gradients=True,
        cp_group=cp_group,
    )
    gradients = reported["__gradients__"]
    context = f"response weighting DP{dp_size}/CP{cp_size}"

    # Absolute oracle first: each response contributes -advantage * valid_tokens,
    # scaled by the window's alpha. Expected values come from the fixture, not
    # from a reducer.
    alpha = float(FROZEN_DR_GRPO_ALPHA)
    window = {response["index"]: response for response in _build_window()}
    for index, gradient in gradients.items():
        response = window[index]
        # Dr.GRPO centers within a group whose four rewards sum to zero, so the
        # advantage equals the raw reward.
        expected = -response["reward"] * sum(response["loss_mask"]) * alpha
        assert abs(gradient - expected) < 1e-9, f"{context} index={index}"
    assert abs(sum(gradients.values()) - float(FROZEN_DR_GRPO_GRADIENT_TOTAL)) < 1e-9, context

    # Then the relative claim, over responses that share an advantage but span
    # ~73x in valid token count.
    reference_index = SAME_ADVANTAGE_INDICES[0]
    reference_tokens = SAME_ADVANTAGE_VALID_TOKENS[0]
    for index, valid_tokens in zip(SAME_ADVANTAGE_INDICES, SAME_ADVANTAGE_VALID_TOKENS, strict=True):
        expected = gradients[reference_index] * valid_tokens / reference_tokens
        assert abs(gradients[index] - expected) < 1e-12, f"{context} index={index}"

    # Responses differing only in valid token count must not tie: a per-sample
    # mean would make all four gradients equal.
    assert len({round(gradients[index], 12) for index in SAME_ADVANTAGE_INDICES}) == len(SAME_ADVANTAGE_INDICES)


def _assert_fully_masked_response_keeps_denominators_aligned(dp_size, dp_rank, cp_size, cp_group) -> None:
    """A fully-masked response must not change the gradient at any CP degree.

    Dr.GRPO reaches ``S / (N * B)`` indirectly: it scales the loss by
    ``alpha = T_metadata / (N * B)`` and lets Megatron's finalizer divide the
    summed gradient by ``T_finalizer``. The target is only hit when those two
    independently computed counts are equal, so the post-finalizer gradient is
    ``-advantage * valid_tokens / (N * B)`` with T cancelling out -- an oracle
    that never mentions T, and so fails loudly if the two definitions drift.
    """
    reported = _reduce_window(
        advantage_estimator="dr_grpo",
        dp_size=dp_size,
        dp_rank=dp_rank,
        cp_size=cp_size,
        dp_with_cp_group=dist.group.WORLD,
        collect_gradients=True,
        cp_group=cp_group,
        fully_masked_indices=(FULLY_MASKED_INDEX,),
    )
    context = f"fully-masked response DP{dp_size}/CP{cp_size}"

    # The finalizer denominator and the count alpha was built from, both pinned
    # to the real token total. A reintroduced CP=1 clamp reports 4035 here.
    assert reported["__response_tokens__"] == FROZEN_MASKED_RESPONSE_TOKENS, context
    assert reported["__alpha__"] == FROZEN_MASKED_RESPONSE_TOKENS / (NUM_RESPONSES * RESPONSE_BUDGET), context

    window = {response["index"]: response for response in _build_window((FULLY_MASKED_INDEX,))}
    advantages = _production_advantages("dr_grpo")
    for index, gradient in reported["__gradients_after_finalizer__"].items():
        valid_tokens = sum(window[index]["loss_mask"])
        expected = -advantages[index] * valid_tokens / (NUM_RESPONSES * RESPONSE_BUDGET)
        assert abs(gradient - expected) < 1e-12, f"{context} index={index}"

    # The filtered response itself must contribute nothing, not one phantom token.
    if FULLY_MASKED_INDEX in reported["__gradients_after_finalizer__"]:
        assert reported["__gradients_after_finalizer__"][FULLY_MASKED_INDEX] == 0.0, context


@pytest.mark.parametrize(("dp_size", "cp_size"), TOPOLOGIES)
def test_dr_grpo_reduction_is_topology_and_micro_batch_invariant(dp_size: int, cp_size: int) -> None:
    world_size = dp_size * cp_size
    mp.spawn(_topology_worker, args=(dp_size, cp_size, _free_port()), nprocs=world_size, join=True)


def _train_one_step_spy_worker(rank: int, port: int, window_empty: bool, result_path: str) -> None:
    """Drive the real ``train_one_step`` and record optimizer/scheduler calls.

    The empty-window guard lives in ``train_one_step``, downstream of the
    metadata that :func:`prepare_policy_optimizer_window_metadata` attaches to
    each micro-batch. Asserting on the metadata alone cannot show that the
    optimizer and the LR scheduler are actually left alone, so this worker runs
    the production function with spies in place of Megatron's optimizer and
    ``OptimizerParamScheduler``.
    """
    import json

    from relax.backends.megatron import model as model_module

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=1)
    try:
        calls: dict[str, int] = {"optimizer_step": 0, "scheduler_step": 0, "zero_grad_buffer": 0}

        class _Optimizer:
            param_groups = [{"lr": 0.0}]

            def zero_grad(self) -> None:
                pass

            def step(self):
                calls["optimizer_step"] += 1
                return True, torch.tensor(1.5), None

        class _Scheduler:
            def step(self, increment: int) -> None:
                calls["scheduler_step"] += 1

        class _ModelChunk:
            def zero_grad_buffer(self) -> None:
                calls["zero_grad_buffer"] += 1

            def __call__(self, **_kwargs):
                return torch.zeros(1)

        batch = {
            "tokens": torch.zeros(1, dtype=torch.long),
            "packed_seq_params": None,
            "full_loss_masks": torch.zeros(1),
            "__optimizer_window_empty__": torch.tensor(window_empty),
        }
        # One logging vector shaped like loss_function's: [num_tokens, *metrics].
        losses_reduced = [{"keys": ["loss"], "values": torch.tensor([4.0, 8.0])}]

        args = Namespace(
            advantage_estimator="dr_grpo",
            calculate_per_token_loss=True,
            check_for_nan_in_loss_and_grad=True,
            ci_test=False,
            custom_megatron_before_train_step_hook_path=None,
            data_pad_size_multiplier=1,
            decoder_seq_length=None,
            dynamic_context_parallel=False,
            enable_mtp_training=False,
            fully_async=False,
            global_batch_size=8,
            is_vl_model=False,
            loss_type="policy_loss",
            micro_batch_size=1,
            qkv_format="thd",
            allgather_cp=False,
            seq_length=8,
            use_dynamic_batch_size=False,
            use_opd=False,
            uses_unsplit_forward=False,
        )

        def _forward_backward_func(*, forward_step_func, data_iterator, model, **_kwargs):
            # Run the production forward_step so the empty-window flag reaches
            # train_one_step the same way it does in training.
            forward_step_func(data_iterator, model[0])
            return losses_reduced

        model_module.get_args = lambda: args
        model_module.get_batch = lambda *_args, **_kwargs: batch
        model_module.get_forward_backward_func = lambda: _forward_backward_func
        model_module.mpu.is_pipeline_last_stage = lambda *_a, **_k: True
        model_module.mpu.get_data_parallel_group = lambda **_k: dist.group.WORLD
        model_module.mpu.get_virtual_pipeline_model_parallel_world_size = lambda: None

        loss_reduced, grad_norm = model_module.train_one_step(
            args=args,
            rollout_id=0,
            step_id=0,
            data_iterator=[iter([batch])],
            model=[_ModelChunk()],
            optimizer=_Optimizer(),
            opt_param_scheduler=_Scheduler(),
            num_microbatches=1,
        )

        with open(result_path, "w") as handle:
            json.dump({"calls": calls, "grad_norm": float(grad_norm), "loss": loss_reduced}, handle)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("window_empty", [True, False])
def test_train_one_step_skips_optimizer_and_scheduler_for_empty_window(tmp_path, window_empty: bool) -> None:
    import json

    result_path = str(tmp_path / "calls.json")
    mp.spawn(
        _train_one_step_spy_worker,
        args=(_free_port(), window_empty, result_path),
        nprocs=1,
        join=True,
    )
    with open(result_path) as handle:
        recorded = json.load(handle)

    if window_empty:
        assert recorded["calls"]["optimizer_step"] == 0
        assert recorded["calls"]["scheduler_step"] == 0
        # A skipped window must report a finite zero, not NaN, or downstream
        # grad-norm health checks would flag the step as diverged.
        assert recorded["grad_norm"] == 0.0
    else:
        assert recorded["calls"]["optimizer_step"] == 1
        assert recorded["calls"]["scheduler_step"] == 1
        assert recorded["grad_norm"] == 1.5


def _megatron_finalizer_worker(rank: int, port: int, num_tokens: float, result_path: str) -> None:
    """Scale a Dr.GRPO window's gradient with the real Megatron finalizer.

    ``FROZEN_DR_GRPO_GRADIENT_TOTAL`` is the gradient accumulated *before*
    Megatron normalizes it, i.e. the token-loss sum already carrying the
    ``T / (N * B)`` window scale. The fixed-denominator claim only holds if
    Megatron's own ``finalize_model_grads`` then divides by the all-reduced
    ``T``, so this worker calls that production function instead of repeating
    the division in test code.
    """
    import json

    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.transformer_config import TransformerConfig

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=1)
    try:
        config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1)
        config.timers = None
        config.sequence_parallel = False
        config.moe_router_enable_expert_bias = False

        class _ModelChunk(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
                self.weight.grad = torch.tensor([float(FROZEN_DR_GRPO_GRADIENT_TOTAL)], dtype=torch.float64)
                self.config = config
                self.scalings: list[float] = []

            def finish_grad_sync(self, force_all_reduce: bool = False) -> None:
                pass

            def scale_gradients(self, scaling: float) -> None:
                self.scalings.append(float(scaling))
                self.weight.grad *= scaling

        pg_collection = ProcessGroupCollection()
        for name in ("tp", "pp", "dp_cp", "dp", "cp", "mp"):
            setattr(pg_collection, name, dist.group.WORLD)
        pg_collection.embd = None
        pg_collection.pos_embd = None

        chunk = _ModelChunk()
        finalize_model_grads([chunk], num_tokens=torch.tensor(num_tokens), pg_collection=pg_collection)

        with open(result_path, "w") as handle:
            json.dump({"scalings": chunk.scalings, "gradient": chunk.weight.grad.tolist()}, handle)
    finally:
        dist.destroy_process_group()


def test_megatron_finalizer_reaches_dr_grpo_fixed_denominator(tmp_path) -> None:
    import json

    result_path = str(tmp_path / "finalized.json")
    mp.spawn(
        _megatron_finalizer_worker,
        args=(_free_port(), float(FROZEN_RESPONSE_TOKENS), result_path),
        nprocs=1,
        join=True,
    )
    with open(result_path) as handle:
        recorded = json.load(handle)

    # Megatron computes 1/T from the fp32 token counter, so compare on the
    # relative scale that dtype supports rather than bit-for-bit.
    assert recorded["scalings"] == pytest.approx([1.0 / FROZEN_RESPONSE_TOKENS], rel=1e-6)
    # Megatron's 1/T lands on the paper's sum(loss) / (N * B).
    assert recorded["gradient"][0] == pytest.approx(float(FROZEN_DR_GRPO_LOSS), rel=1e-6)


def test_megatron_finalizer_leaves_gradients_alone_for_empty_window(tmp_path) -> None:
    """A fully masked window reaches the finalizer with ``T == 0``.

    Megatron uses a safe denominator, so it may omit scaling or apply identity
    scaling. In either case the gradient must remain unchanged and finite.
    ``train_one_step`` then skips the optimizer, see
    :func:`test_train_one_step_skips_optimizer_and_scheduler_for_empty_window`.
    """
    import json

    result_path = str(tmp_path / "finalized_empty.json")
    mp.spawn(
        _megatron_finalizer_worker,
        args=(_free_port(), 0.0, result_path),
        nprocs=1,
        join=True,
    )
    with open(result_path) as handle:
        recorded = json.load(handle)

    assert recorded["gradient"][0] == pytest.approx(float(FROZEN_DR_GRPO_GRADIENT_TOTAL), abs=1e-12)
    assert math.isfinite(recorded["gradient"][0])
