# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from relax.backends.megatron.weight_update.mixture_lora_sync import (
    MixtureLoraParamInfo,
    merge_mixture_lora_tp_shards,
)
from relax.backends.megatron.weight_update.update_weight_from_tensor import iter_mixture_weight_updates
from relax.utils.mixture_lora import MixtureLoraStateSpec


def _info(site, kind, global_shape, local_shape, shard_dim):
    return MixtureLoraParamInfo(
        state=MixtureLoraStateSpec(
            schema_version=1,
            site_id=f"decoder.layers.0.self_attention.{site}",
            parameter_kind=kind,
            global_shape=global_shape,
            dtype=torch.float32,
        ),
        local_shape=local_shape,
        tp_shard_dim=shard_dim,
        src_rank=0,
        weight_key="weight",
    )


def test_qkv_lora_b_tp_shards_are_converted_from_group_layout_to_qkv_blocks():
    # Two query groups, two query heads per group, then one K and one V head.
    grouped = torch.tensor([[[[10.0], [11.0], [20.0], [30.0], [12.0], [13.0], [21.0], [31.0]]]]).reshape(1, 8, 1)
    info = _info("linear_qkv", "experts.lora_B", (1, 8, 1), (1, 4, 1), 1)

    merged = merge_mixture_lora_tp_shards(
        info,
        grouped.chunk(2, dim=1),
        num_attention_heads=4,
        num_query_groups=2,
        head_dim=1,
    )

    expected = torch.tensor([10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 30.0, 31.0]).reshape(1, 8, 1)
    torch.testing.assert_close(merged, expected)


@pytest.mark.parametrize(
    ("site", "kind", "global_shape", "local_shape", "shard_dim"),
    [
        ("linear_qkv", "experts.lora_A", (2, 4, 6), (2, 2, 6), 1),
        ("linear_proj", "experts.lora_A", (2, 4, 6), (2, 4, 3), 2),
        ("linear_proj", "experts.lora_B", (2, 6, 4), (2, 3, 4), 1),
        ("linear_proj", "router.weight", (2, 6), (2, 3), 1),
    ],
)
def test_non_qkv_output_shards_concatenate_on_the_schema_axis(
    site,
    kind,
    global_shape,
    local_shape,
    shard_dim,
):
    info = _info(site, kind, global_shape, local_shape, shard_dim)
    shards = [torch.zeros(local_shape), torch.ones(local_shape)]

    merged = merge_mixture_lora_tp_shards(
        info,
        shards,
        num_attention_heads=4,
        num_query_groups=2,
        head_dim=1,
    )

    assert merged.shape == global_shape
    first, second = merged.chunk(2, dim=shard_dim)
    assert torch.equal(first, shards[0])
    assert torch.equal(second, shards[1])


def test_replicated_router_rejects_multiple_tp_copies():
    info = _info("linear_qkv", "router.weight", (2, 6), (2, 6), None)

    with pytest.raises(ValueError, match="expects one tensor"):
        merge_mixture_lora_tp_shards(
            info,
            [torch.zeros(2, 6), torch.zeros(2, 6)],
            num_attention_heads=4,
            num_query_groups=2,
            head_dim=1,
        )


def test_reconstructed_tensor_validates_shape_and_dtype():
    info = _info("linear_proj", "router.weight", (2, 6), (2, 3), 1)

    with pytest.raises(ValueError, match="TP shard shape mismatch"):
        merge_mixture_lora_tp_shards(
            info,
            [torch.zeros(2, 4), torch.zeros(2, 4)],
            num_attention_heads=4,
            num_query_groups=2,
            head_dim=1,
        )
    with pytest.raises(TypeError, match="has dtype"):
        merge_mixture_lora_tp_shards(
            info,
            [torch.zeros(2, 3, dtype=torch.float64), torch.zeros(2, 3, dtype=torch.float64)],
            num_attention_heads=4,
            num_query_groups=2,
            head_dim=1,
        )


def _named_parameters():
    base = torch.nn.Parameter(torch.zeros(4, 4))
    mixture = torch.nn.Parameter(torch.zeros(2, 2, 4))
    return [
        ("module.module.decoder.layers.0.self_attention.linear_qkv.weight", base),
        (
            "module.module.decoder.layers.0.self_attention.linear_qkv.mixture_lora.experts.lora_A",
            mixture,
        ),
    ]


def test_direct_hf_iterator_excludes_mixture_parameters():
    from relax.backends.megatron.weight_update.hf_weight_iterator_direct import _get_megatron_local_param_infos

    args = SimpleNamespace(update_weight_buffer_size=1024, mtp_num_layers=None)

    def gather_single_process(obj, object_list, group=None):
        del group
        object_list[0] = obj

    with (
        patch(
            "relax.backends.megatron.weight_update.hf_weight_iterator_direct.named_params_and_buffers",
            return_value=iter(_named_parameters()),
        ),
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_world_size", return_value=1),
        patch("torch.distributed.all_gather_object", side_effect=gather_single_process),
        patch(
            "relax.backends.megatron.weight_update.hf_weight_iterator_direct.get_gloo_group",
            return_value=None,
        ),
        patch("megatron.core.mpu.get_pipeline_model_parallel_world_size", return_value=1),
        patch("megatron.core.mpu.get_expert_model_parallel_world_size", return_value=1),
    ):
        infos = _get_megatron_local_param_infos(args, model=[])

    assert [info.name for info in infos] == ["module.module.decoder.layers.0.self_attention.linear_qkv.weight"]


def test_bridge_hf_iterator_excludes_mixture_parameters():
    from relax.backends.megatron.weight_update.hf_weight_iterator_bridge import _build_param_info_buckets

    vanilla = [(f"vp_stages.0.{name}", parameter) for name, parameter in _named_parameters()]
    args = SimpleNamespace(update_weight_buffer_size=1024, num_experts=None)
    with (
        patch(
            "relax.backends.megatron.weight_update.hf_weight_iterator_bridge.named_params_and_buffers",
            side_effect=[iter(vanilla), iter(_named_parameters())],
        ),
        patch("torch.distributed.get_rank", return_value=0),
        patch("megatron.core.mpu.get_pipeline_model_parallel_world_size", return_value=1),
        patch("megatron.core.mpu.get_expert_model_parallel_world_size", return_value=1),
        patch("megatron.core.mpu.get_tensor_model_parallel_world_size", return_value=1),
    ):
        expert_buckets, base_buckets, _, _ = _build_param_info_buckets(args, model=[])

    assert expert_buckets == []
    assert [[info.name for info in bucket] for bucket in base_buckets] == [
        ["module.module.decoder.layers.0.self_attention.linear_qkv.weight"]
    ]


def test_first_sync_sends_base_then_routes_and_versions_only_the_final_chunk():
    updates = list(
        iter_mixture_weight_updates(
            base_chunks=[["base-0"], ["base-1"]],
            mixture_chunks=[["mixture-0"], ["mixture-1"]],
            include_base=True,
            weight_version=7,
        )
    )

    assert updates == [
        (["base-0"], None),
        (["base-1"], None),
        (["mixture-0"], None),
        (["mixture-1"], 7),
    ]


def test_subsequent_sync_skips_base_and_sends_all_routed_parameters():
    def base_chunks_must_not_be_read():
        raise AssertionError("base chunks were read after the first sync")
        yield

    updates = list(
        iter_mixture_weight_updates(
            base_chunks=base_chunks_must_not_be_read(),
            mixture_chunks=[["mixture-0"], ["mixture-1"]],
            include_base=False,
            weight_version=8,
        )
    )

    assert updates == [(["mixture-0"], None), (["mixture-1"], 8)]


def test_sync_rejects_an_empty_mixture_parameter_set():
    with pytest.raises(ValueError, match="produced no routed tensors"):
        list(
            iter_mixture_weight_updates(
                base_chunks=[["base"]],
                mixture_chunks=[],
                include_base=True,
                weight_version=1,
            )
        )


def test_update_lifecycle_keeps_call_order_and_skips_base_after_first_sync():
    from relax.backends.megatron.weight_update.update_weight_from_tensor import UpdateWeightFromTensor

    events = []

    class RemoteMethod:
        def __init__(self, name):
            self.name = name

        def remote(self):
            events.append(self.name)
            return self.name

    engine = SimpleNamespace(
        pause_generation=RemoteMethod("pause"),
        flush_cache=RemoteMethod("flush"),
        continue_generation=RemoteMethod("continue"),
    )
    local_weights = {"cpu-mirror": torch.ones(1)}
    updater = UpdateWeightFromTensor.__new__(UpdateWeightFromTensor)
    updater.weight_version = 0
    updater.rollout_engines = [engine]
    updater.quantization_config = None
    updater.weights_getter = MagicMock(return_value=local_weights)
    updater._hf_weight_iterator = MagicMock()
    updater._hf_weight_iterator.get_hf_weight_chunks.return_value = iter([["base"]])
    updater._mixture_lora_sync = SimpleNamespace(base_sync_done=False)
    updater._mixture_lora_sync.get_weight_chunks = MagicMock(
        side_effect=[iter([["mixture-1"]]), iter([["mixture-2"]])]
    )
    sent_updates = []

    def record_updates(updates):
        events.append("update")
        sent_updates.append(list(updates))

    updater._send_weight_update_stream = record_updates

    with (
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.barrier"),
        patch("relax.backends.megatron.weight_update.update_weight_from_tensor.get_gloo_group", return_value=None),
        patch("ray.get", side_effect=lambda refs: refs),
    ):
        updater._update_weights_mixture_lora()
        updater._hf_weight_iterator.get_hf_weight_chunks.return_value = iter([["must-not-be-read"]])
        updater._update_weights_mixture_lora()

    assert events == ["pause", "flush", "update", "continue", "pause", "flush", "update", "continue"]
    assert sent_updates == [
        [(["base"], None), (["mixture-1"], 1)],
        [(["mixture-2"], 2)],
    ]
    assert updater.weights_getter.call_count == 2
    assert len(updater._mixture_lora_sync.get_weight_chunks.call_args_list) == 2
    assert all(
        recorded_call.args[0] is local_weights
        for recorded_call in updater._mixture_lora_sync.get_weight_chunks.call_args_list
    )
    assert updater.weight_version == 2
    assert updater._mixture_lora_sync.base_sync_done is True
