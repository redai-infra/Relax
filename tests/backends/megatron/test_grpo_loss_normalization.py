from argparse import Namespace

import pytest


def _run_static_cp_dr_grpo_worker(rank: int, init_method: str, result_path: str) -> None:
    import torch
    import torch.distributed as dist

    from relax.backends.megatron import cp_utils

    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    try:
        local_values = torch.tensor([1.0] if rank == 0 else [2.0, 3.0, 4.0], requires_grad=True)
        reducer = cp_utils.get_sequence_loss_aggregator(
            "seq-mean-token-sum-norm",
            total_lengths=[6],
            response_lengths=[4],
            loss_masks=[torch.ones(4)],
            scale_factor=8,
            dynamic_cp_size=2,
            dynamic_cp_rank=rank,
        )
        local_loss = reducer(local_values)
        step_token_normalizer = torch.tensor(float(local_values.numel()))
        dist.all_reduce(step_token_normalizer, op=dist.ReduceOp.SUM)
        scale = cp_utils.get_per_token_loss_scale(
            num_microbatches=1,
            global_batch_size=1,
            data_parallel_world_size=2,
            step_token_normalizer=step_token_normalizer,
        )
        (local_loss * scale).backward()

        fixed_scale_gradient = local_values.grad / (step_token_normalizer * 2)
        assert torch.equal(fixed_scale_gradient, torch.full_like(local_values, 1.0 / 8.0))

        total_loss = local_loss.detach().clone()
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        if rank == 0:
            torch.save(
                torch.stack([total_loss, step_token_normalizer, fixed_scale_gradient.mean()]),
                result_path,
            )
    finally:
        dist.destroy_process_group()


@pytest.fixture()
def cp_utils_module():
    pytest.importorskip("torch")
    pytest.importorskip("megatron")
    from relax.backends.megatron import cp_utils

    return cp_utils


@pytest.fixture()
def megatron_data_environment(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("megatron.training")
    pytest.importorskip("tensordict")

    import torch.distributed as dist
    from megatron.core import mpu

    from relax.backends.megatron import data
    from relax.utils import device as device_utils

    if device_utils.get_device_name() != "cpu":
        pytest.skip("The real Megatron data-iterator integration test requires a CPU process group.")
    if dist.is_initialized() or mpu.model_parallel_is_initialized():
        pytest.skip("The test process already has Megatron parallel state initialized.")

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_path / 'megatron-data-test'}",
        rank=0,
        world_size=1,
    )
    try:
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            create_gloo_process_groups=False,
        )
        yield torch, data
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def test_real_megatron_static_iterator_reuses_step_normalizer(megatron_data_environment):
    torch, data_module = megatron_data_environment

    args = Namespace(
        balance_data=True,
        calculate_per_token_loss=True,
        pg_loss_aggregation="seq-mean-token-sum-norm",
        global_batch_size=5,
        micro_batch_size=1,
        use_dynamic_batch_size=False,
    )
    rollout_data = {
        "total_lengths": list(range(5)),
        "loss_masks": [torch.ones(1), torch.ones(2), torch.ones(3), torch.ones(4), torch.ones(5)],
        data_module.ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY: [2, 3],
    }

    data_iterators, num_microbatches = data_module.get_data_iterator(args, torch.nn.Module(), rollout_data)

    assert num_microbatches == [2, 3]
    iterator = data_iterators[0]
    batches = [iterator.get_next(["total_lengths"]) for _ in range(sum(num_microbatches))]
    assert [batch["total_lengths"] for batch in batches] == [[0], [1], [2], [3], [4]]
    assert torch.equal(
        torch.stack([batch["__per_token_loss_normalizer__"] for batch in batches]),
        torch.tensor([3.0, 3.0, 12.0, 12.0, 12.0]),
    )

    iterator.reset()
    assert torch.equal(iterator.get_next(["total_lengths"])["__per_token_loss_normalizer__"], torch.tensor(3.0))


def test_real_megatron_dynamic_iterator_reuses_step_normalizer(megatron_data_environment):
    torch, data_module = megatron_data_environment

    args = Namespace(
        balance_data=True,
        calculate_per_token_loss=True,
        pg_loss_aggregation="seq-mean-token-sum-norm",
        global_batch_size=5,
        max_tokens_per_gpu=4,
        use_dynamic_batch_size=True,
    )
    rollout_data = {
        "total_lengths": [2, 2, 3, 1, 1],
        "loss_masks": [torch.ones(1), torch.ones(2), torch.ones(3), torch.ones(4), torch.ones(5)],
        data_module.ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY: [2, 3],
    }

    data_iterators, num_microbatches = data_module.get_data_iterator(args, torch.nn.Module(), rollout_data)

    assert num_microbatches == [1, 2]
    iterator = data_iterators[0]
    batches = [iterator.get_next(["total_lengths"]) for _ in range(sum(num_microbatches))]
    assert torch.equal(
        torch.stack([batch["__per_token_loss_normalizer__"] for batch in batches]),
        torch.tensor([3.0, 12.0, 12.0]),
    )


def test_response_length_normalization_preserves_existing_behavior(cp_utils_module):
    import torch

    reducer = cp_utils_module.get_sum_of_sample_mean(
        total_lengths=[4, 5],
        response_lengths=[2, 3],
        loss_masks=[torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0, 0.0])],
        dynamic_cp_size=1,
    )

    value = reducer(torch.tensor([2.0, 11.0, 3.0, 5.0, 13.0]))

    assert torch.isclose(value, torch.tensor(6.0))


def test_seq_mean_token_sum_norm_uses_one_scale_factor_for_all_responses(cp_utils_module):
    import torch

    reducer = cp_utils_module.get_sequence_loss_aggregator(
        "seq-mean-token-sum-norm",
        total_lengths=[4, 5],
        response_lengths=[2, 3],
        loss_masks=[torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0, 0.0])],
        scale_factor=4,
        dynamic_cp_size=1,
    )
    values = torch.tensor([2.0, 11.0, 3.0, 5.0, 13.0], requires_grad=True)

    loss = reducer(values)
    loss.backward()

    assert torch.isclose(loss, torch.tensor(2.5))
    assert torch.equal(values.grad, torch.tensor([0.25, 0.0, 0.25, 0.25, 0.0]))


def test_seq_mean_token_sum_norm_requires_positive_scale_factor(cp_utils_module):
    import torch

    with pytest.raises(ValueError, match="scale_factor must be positive"):
        cp_utils_module.get_sequence_loss_aggregator(
            "seq-mean-token-sum-norm",
            total_lengths=[2],
            response_lengths=[1],
            loss_masks=[torch.tensor([1.0])],
            scale_factor=0,
            dynamic_cp_size=1,
        )


def test_per_token_finalizer_scale_recovers_fixed_dr_grpo_denominator(cp_utils_module):
    import torch

    first_microbatch = torch.tensor([2.0, 3.0], requires_grad=True)
    second_microbatch = torch.tensor([5.0], requires_grad=True)
    fixed_scale_factor = 8
    global_batch_size = 2
    step_token_normalizer = 3
    scale = cp_utils_module.get_per_token_loss_scale(
        num_microbatches=2,
        global_batch_size=global_batch_size,
        data_parallel_world_size=1,
        step_token_normalizer=step_token_normalizer,
    )

    megatron_loss = (
        (first_microbatch.sum() / fixed_scale_factor) * scale / 2
        + (second_microbatch.sum() / fixed_scale_factor) * scale / 2
    ) / step_token_normalizer
    megatron_loss.backward()

    assert torch.isclose(megatron_loss, torch.tensor(10.0 / (global_batch_size * fixed_scale_factor)))
    expected_gradient = 1.0 / (global_batch_size * fixed_scale_factor)
    assert torch.equal(first_microbatch.grad, torch.full_like(first_microbatch, expected_gradient))
    assert torch.equal(second_microbatch.grad, torch.full_like(second_microbatch, expected_gradient))


def test_per_token_finalizer_cp_shards_recover_fixed_dr_grpo_denominator(cp_utils_module):
    import torch

    fixed_scale_factor = 8
    step_token_normalizer = 4
    rank_zero = torch.tensor([1.0])
    rank_one = torch.tensor([2.0, 3.0, 4.0])
    scale = cp_utils_module.get_per_token_loss_scale(
        num_microbatches=1,
        global_batch_size=1,
        data_parallel_world_size=2,
        step_token_normalizer=step_token_normalizer,
    )

    final_loss = (
        (rank_zero.sum() / fixed_scale_factor + rank_one.sum() / fixed_scale_factor) * scale / 2
    ) / step_token_normalizer

    assert torch.isclose(final_loss, torch.tensor(10.0 / fixed_scale_factor))


def test_per_token_finalizer_requires_step_global_not_microbatch_normalizer(cp_utils_module):
    import torch

    fixed_scale_factor = 8
    first_microbatch = torch.tensor([2.0, 3.0])
    second_microbatch = torch.tensor([5.0])
    correct_scale = cp_utils_module.get_per_token_loss_scale(2, 2, 1, step_token_normalizer=3)
    incorrect_first_scale = cp_utils_module.get_per_token_loss_scale(2, 2, 1, step_token_normalizer=2)
    incorrect_second_scale = cp_utils_module.get_per_token_loss_scale(2, 2, 1, step_token_normalizer=1)

    correct_loss = (
        first_microbatch.sum() / fixed_scale_factor * correct_scale / 2
        + second_microbatch.sum() / fixed_scale_factor * correct_scale / 2
    ) / 3
    microbatch_weighted_loss = (
        first_microbatch.sum() / fixed_scale_factor * incorrect_first_scale / 2
        + second_microbatch.sum() / fixed_scale_factor * incorrect_second_scale / 2
    ) / 3

    assert torch.isclose(correct_loss, torch.tensor(10.0 / 16.0))
    assert not torch.isclose(microbatch_weighted_loss, correct_loss)


def test_static_cp_dr_grpo_matches_cp_one_fixed_scale_gradient(tmp_path, cp_utils_module):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo is required for the static CP process-group test.")

    cp_one_values = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    cp_one_reducer = cp_utils_module.get_sequence_loss_aggregator(
        "seq-mean-token-sum-norm",
        total_lengths=[6],
        response_lengths=[4],
        loss_masks=[torch.ones(4)],
        scale_factor=8,
        dynamic_cp_size=1,
    )
    cp_one_loss = cp_one_reducer(cp_one_values)
    cp_one_loss.backward()

    init_file = tmp_path / "gloo-init"
    result_path = tmp_path / "result.pt"
    torch.multiprocessing.spawn(
        _run_static_cp_dr_grpo_worker,
        args=(f"file://{init_file}", str(result_path)),
        nprocs=2,
        join=True,
    )

    total_loss, step_token_normalizer, fixed_scale_gradient = torch.load(result_path, weights_only=True)

    assert torch.isclose(total_loss, cp_one_loss.detach())
    assert torch.isclose(step_token_normalizer, torch.tensor(4.0))
    assert torch.isclose(fixed_scale_gradient, cp_one_values.grad.mean())


def test_padding_kwargs_preserve_fixed_sum_result(cp_utils_module):
    import torch

    kwargs = {
        "total_lengths": [6],
        "response_lengths": [4],
        "loss_masks": [torch.ones(4)],
        "scale_factor": 8,
        "dynamic_cp_size": 2,
    }
    base_zero = cp_utils_module.get_sequence_loss_aggregator("seq-mean-token-sum-norm", dynamic_cp_rank=0, **kwargs)
    base_one = cp_utils_module.get_sequence_loss_aggregator("seq-mean-token-sum-norm", dynamic_cp_rank=1, **kwargs)
    padded_zero = cp_utils_module.get_sequence_loss_aggregator(
        "seq-mean-token-sum-norm", dynamic_cp_rank=0, padded_total_lengths=[8], **kwargs
    )
    padded_one = cp_utils_module.get_sequence_loss_aggregator(
        "seq-mean-token-sum-norm", dynamic_cp_rank=1, padded_total_lengths=[8], **kwargs
    )
    bshd_zero = cp_utils_module.get_sequence_loss_aggregator(
        "seq-mean-token-sum-norm", dynamic_cp_rank=0, qkv_format="bshd", max_seq_lens=[8], **kwargs
    )
    bshd_one = cp_utils_module.get_sequence_loss_aggregator(
        "seq-mean-token-sum-norm", dynamic_cp_rank=1, qkv_format="bshd", max_seq_lens=[8], **kwargs
    )

    x_zero = torch.tensor([1.0], requires_grad=True)
    x_one = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)

    plain = base_zero(x_zero) + base_one(x_one)
    plain.backward()
    plain_grad = (x_zero.grad.clone(), x_one.grad.clone())
    x_zero.grad = None
    x_one.grad = None
    padded = padded_zero(x_zero) + padded_one(x_one)
    padded.backward()
    assert torch.equal(plain, padded)
    assert torch.equal(x_zero.grad, plain_grad[0])
    assert torch.equal(x_one.grad, plain_grad[1])
    x_zero.grad = None
    x_one.grad = None
    bshd = bshd_zero(x_zero) + bshd_one(x_one)
    bshd.backward()
    assert torch.equal(plain, bshd)
    assert torch.equal(x_zero.grad, plain_grad[0])
    assert torch.equal(x_one.grad, plain_grad[1])
    assert torch.isclose(plain, torch.tensor(10.0 / 8.0))


def test_sum_norm_reweights_short_vs_long_responses(cp_utils_module):
    import torch

    total_lengths = [8, 512]
    response_lengths = [8, 512]
    loss_masks = [torch.ones(8), torch.ones(512)]
    seq_values = torch.ones(520, requires_grad=True)
    sum_norm_values = torch.ones(520, requires_grad=True)

    seq_mean = cp_utils_module.get_sequence_loss_aggregator(
        "seq-mean-token-mean", total_lengths, response_lengths, loss_masks, dynamic_cp_size=1
    )
    sum_norm = cp_utils_module.get_sequence_loss_aggregator(
        "seq-mean-token-sum-norm",
        total_lengths,
        response_lengths,
        loss_masks,
        scale_factor=8,
        dynamic_cp_size=1,
    )

    seq_mean_loss = seq_mean(seq_values)
    sum_norm_loss = sum_norm(sum_norm_values)
    seq_mean_loss.backward()
    sum_norm_loss.backward()

    # Per-response mean is 1.0, so seq-mean weights the two responses equally (1:1).
    assert torch.isclose(seq_mean_loss, torch.tensor(2.0))
    # Fixed-sum weights every token equally: (8 + 512) / S.
    assert torch.isclose(sum_norm_loss, torch.tensor(520.0 / 8.0))
    # The long response's share of the total loss grows from 1/2 (seq-mean) to
    # 512/520 (sum-norm), proving the length weighting actually changed.
    seq_mean_long_share = torch.tensor(1.0) / seq_mean_loss
    sum_norm_long_share = torch.tensor(512.0 / 8.0) / sum_norm_loss
    assert torch.isclose(seq_mean_long_share, torch.tensor(0.5))
    assert sum_norm_long_share > seq_mean_long_share
    assert torch.equal(seq_values.grad[:8], torch.full((8,), 1.0 / 8.0))
    assert torch.equal(seq_values.grad[8:], torch.full((512,), 1.0 / 512.0))
    assert torch.equal(sum_norm_values.grad, torch.full((520,), 1.0 / 8.0))
    assert seq_values.grad[:8].sum() == seq_values.grad[8:].sum()
    assert sum_norm_values.grad[:8].sum() < sum_norm_values.grad[8:].sum()
