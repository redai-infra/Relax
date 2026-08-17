# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.backends.megatron.mixture_lora_modules import MixtureParallelLinearAdapter
from relax.utils.mixture_lora_common import MixtureLoraConfig


def _config() -> MixtureLoraConfig:
    return MixtureLoraConfig(
        num_experts=3,
        rank=2,
        top_k=2,
        temperature=0.7,
        aux_loss_coef=0.01,
        alpha=4.0,
        target_modules=("linear_qkv",),
    )


def _save_cpu_checkpoint(dist_checkpointing, sharded_state: dict, checkpoint_dir: str) -> None:
    """Avoid MCore's unconditional CUDA sync when every saved tensor is on
    CPU."""

    with patch.object(torch.cuda, "synchronize"):
        dist_checkpointing.save(sharded_state, checkpoint_dir)


def _pipeline_checkpoint_worker(
    rank: int,
    world_size: int,
    init_method: str,
    checkpoint_dir: str,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=90),
    )
    from megatron.core import dist_checkpointing, parallel_state
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.core.transformer.transformer_config import TransformerConfig

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=world_size,
    )
    try:
        transformer_config = TransformerConfig(
            num_layers=world_size,
            hidden_size=4,
            num_attention_heads=1,
            pipeline_model_parallel_size=world_size,
            pipeline_dtype=torch.float32,
            use_cpu_initialization=True,
        )
        model = torch.nn.Module()
        model.linear_qkv = ColumnParallelLinear(
            4,
            5,
            config=transformer_config,
            init_method=lambda weight: torch.nn.init.normal_(weight, mean=0.0, std=0.02),
            bias=False,
            gather_output=False,
            skip_bias_add=True,
        )
        expected_site_id = f"decoder.layers.{rank}.self_attention.linear_qkv"
        model.linear_qkv = MixtureParallelLinearAdapter(
            model.linear_qkv,
            _config(),
            expected_site_id,
            4,
            5,
            dropout=0.0,
            tp_group=parallel_state.get_tensor_model_parallel_group(),
            tp_rank=0,
            tp_world_size=1,
        )
        mixture = model.linear_qkv.mixture_lora
        assert mixture.site_id == expected_site_id

        generator = torch.Generator().manual_seed(7100 + rank)
        with torch.no_grad():
            for parameter in mixture.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator))
        expected_parameters = {name: parameter.detach().clone() for name, parameter in mixture.named_parameters()}

        prefix = f"{expected_site_id}."
        sharded_state = model.linear_qkv.sharded_state_dict(
            prefix=prefix,
            metadata={"dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True)},
        )
        local_mixture_keys = tuple(sorted(key for key in sharded_state if ".mixture_lora." in key))
        gathered_keys = [None] * world_size
        dist.all_gather_object(gathered_keys, local_mixture_keys)
        expected_keys = {
            f"decoder.layers.{stage}.self_attention.linear_qkv.mixture_lora.{suffix}"
            for stage in range(world_size)
            for suffix in ("_extra_state", "experts.lora_A", "experts.lora_B", "router.weight")
        }
        assert set().union(*map(set, gathered_keys)) == expected_keys
        assert set(gathered_keys[0]).isdisjoint(gathered_keys[1])

        if rank == 0:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        dist.barrier()
        _save_cpu_checkpoint(dist_checkpointing, sharded_state, checkpoint_dir)

        with torch.no_grad():
            for parameter in mixture.parameters():
                parameter.zero_()
        load_template = model.linear_qkv.sharded_state_dict(
            prefix=prefix,
            metadata={"dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True)},
        )
        loaded_state = dist_checkpointing.load(load_template, checkpoint_dir)
        mixture_prefix = f"{prefix}mixture_lora."
        local_mixture_state = {
            key.removeprefix(mixture_prefix): value
            for key, value in loaded_state.items()
            if key.startswith(mixture_prefix)
        }
        mixture.load_state_dict(local_mixture_state)
        for name, parameter in mixture.named_parameters():
            torch.testing.assert_close(parameter, expected_parameters[name])
        assert mixture.site_id == expected_site_id
        dist.barrier()
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def test_pipeline_checkpoint_restores_each_stage_parameters(tmp_path):
    pytest.importorskip("megatron.bridge.peft.base")
    checkpoint_dir = tmp_path / "mixture_lora_pp_checkpoint"
    init_method = f"file://{tmp_path / 'mixture-lora-pp-checkpoint-gloo-init'}"
    mp.spawn(
        _pipeline_checkpoint_worker,
        args=(2, init_method, str(checkpoint_dir)),
        nprocs=2,
        join=True,
    )


def _data_parallel_checkpoint_worker(
    rank: int,
    world_size: int,
    init_method: str,
    checkpoint_dir: str,
    save_checkpoint: bool,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=90),
    )
    from megatron.core import dist_checkpointing, parallel_state
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.core.transformer.transformer_config import TransformerConfig

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    try:
        torch.manual_seed(8100)
        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=4,
            num_attention_heads=1,
            use_cpu_initialization=True,
        )
        base = ColumnParallelLinear(
            4,
            6,
            config=transformer_config,
            init_method=lambda weight: torch.nn.init.normal_(weight, mean=0.0, std=0.02),
            bias=False,
            gather_output=False,
            skip_bias_add=True,
        )
        adapter = MixtureParallelLinearAdapter(
            base,
            _config(),
            "decoder.layers.0.self_attention.linear_qkv",
            4,
            6,
            dropout=0.0,
            tp_group=parallel_state.get_tensor_model_parallel_group(),
            tp_rank=0,
            tp_world_size=1,
        )
        generator = torch.Generator().manual_seed(8200)
        with torch.no_grad():
            for parameter in adapter.mixture_lora.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator))
        expected_parameters = {
            name: parameter.detach().clone() for name, parameter in adapter.mixture_lora.named_parameters()
        }
        prefix = "decoder.layers.0.self_attention.linear_qkv."
        metadata = {"dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True)}

        if save_checkpoint:
            if rank == 0:
                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            dist.barrier()
            _save_cpu_checkpoint(
                dist_checkpointing,
                adapter.sharded_state_dict(prefix=prefix, metadata=metadata),
                checkpoint_dir,
            )
        else:
            with torch.no_grad():
                for parameter in adapter.mixture_lora.parameters():
                    parameter.zero_()
            loaded_state = dist_checkpointing.load(
                adapter.sharded_state_dict(prefix=prefix, metadata=metadata),
                checkpoint_dir,
            )
            mixture_prefix = f"{prefix}mixture_lora."
            adapter.mixture_lora.load_state_dict(
                {
                    key.removeprefix(mixture_prefix): value
                    for key, value in loaded_state.items()
                    if key.startswith(mixture_prefix)
                }
            )
            for name, parameter in adapter.mixture_lora.named_parameters():
                torch.testing.assert_close(parameter, expected_parameters[name])
        dist.barrier()
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def test_checkpoint_reshards_mixture_parameters_when_data_parallel_size_shrinks(tmp_path):
    pytest.importorskip("megatron.bridge.peft.base")
    checkpoint_dir = tmp_path / "mixture_lora_dp_checkpoint"
    save_init_method = f"file://{tmp_path / 'mixture-lora-dp-save-gloo-init'}"
    load_init_method = f"file://{tmp_path / 'mixture-lora-dp-load-gloo-init'}"

    mp.spawn(
        _data_parallel_checkpoint_worker,
        args=(2, save_init_method, str(checkpoint_dir), True),
        nprocs=2,
        join=True,
    )
    mp.spawn(
        _data_parallel_checkpoint_worker,
        args=(1, load_init_method, str(checkpoint_dir), False),
        nprocs=1,
        join=True,
    )
