# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# These tests exercise the real Megatron Bridge weight conversion and
# distributed synchronization modules, which are not installed in CPU CI.
pytest.importorskip("megatron.bridge.peft.lora")

from relax.backends.megatron.weight_update.common import _maybe_get_cpu_backup  # noqa: E402
from relax.backends.megatron.weight_update.hf_weight_iterator_bridge import HfWeightIteratorBridge  # noqa: E402
from relax.backends.megatron.weight_update.mixture_lora_sync import (  # noqa: E402
    MixtureLoraParamInfo,
    _gather_pipeline_param_infos,
    merge_mixture_lora_tp_shards,
)
from relax.backends.megatron.weight_update.update_weight_from_tensor import (  # noqa: E402
    _send_to_colocated_engine,
    iter_mixture_weight_updates,
)
from relax.utils.mixture_lora_common import MixtureLoraStateSpec  # noqa: E402
from relax.utils.types import ParamInfo  # noqa: E402


@pytest.mark.parametrize(("use_host_tensors", "device_lookup_calls"), [(False, 1), (True, 0)])
def test_colocated_mixture_transfer_selects_device_by_sglang_pipeline_mode(
    use_host_tensors,
    device_lookup_calls,
):
    tensor = torch.ones(2)
    engine = MagicMock()
    engine.update_weights_from_tensor.remote.return_value = "ref"

    module = "relax.backends.megatron.weight_update.update_weight_from_tensor"
    with (
        patch(f"{module}.dist.get_rank", return_value=0),
        patch(f"{module}.dist.get_world_size", return_value=1),
        patch(
            f"{module}.dist.gather_object",
            side_effect=lambda value, object_gather_list, **_: object_gather_list.__setitem__(0, value),
        ),
        patch(f"{module}.make_current_torch_device", return_value=torch.device("cpu")) as current_device,
        patch(f"{module}.torch.multiprocessing.get_sharing_strategy", return_value="file_descriptor"),
        patch(f"{module}.torch.multiprocessing.set_sharing_strategy") as set_sharing_strategy,
    ):
        refs, long_lived = _send_to_colocated_engine(
            [("weight", tensor)],
            ipc_engine=engine,
            ipc_gather_src=0,
            ipc_gather_group="group",
            weight_version=1,
            use_host_tensors=use_host_tensors,
        )

    assert refs == ["ref"]
    assert long_lived
    assert current_device.call_count == device_lookup_calls
    expected_sharing_calls = [call("file_system"), call("file_descriptor")] if use_host_tensors else []
    assert set_sharing_strategy.call_args_list == expected_sharing_calls


def test_selective_offload_uses_cpu_copy_only_after_live_storage_is_released():
    cpu_copy = torch.arange(4, dtype=torch.float32)
    released = torch.empty(0)
    released._relax_cpu_offload_data = cpu_copy

    assert _maybe_get_cpu_backup(released) is cpu_copy

    resident = torch.ones(1)
    resident._relax_cpu_offload_data = cpu_copy

    assert _maybe_get_cpu_backup(resident) is resident


def _run_non_expert_bridge_iterator(*, current_rank: int, src_rank: int):
    name = "decoder.layers.0.self_attention.linear_proj.to_wrap.weight"
    info = ParamInfo(
        name=name,
        dtype=torch.float32,
        shape=torch.Size((4, 8)),
        attrs={},
        size=4 * 8 * 4,
        src_rank=src_rank,
    )
    parameter = torch.nn.Parameter(torch.ones(info.shape), requires_grad=False)
    converted = [("model.layers.0.self_attn.o_proj.weight", torch.full(info.shape, 2.0))]
    iterator = HfWeightIteratorBridge.__new__(HfWeightIteratorBridge)
    iterator.args = MagicMock()
    iterator._bridge_converter = MagicMock()
    iterator._bridge_converter.convert.return_value = converted
    iterator.lora_merge_mode = False
    iterator._expert_buckets = []
    iterator._non_expert_buckets = [[info]]
    iterator._vanilla_key_map = {info.name: info.name}

    def broadcast_owner_result(bucket_infos, all_converted, device):
        assert bucket_infos == [info]
        assert device == "cpu"
        assert all_converted == ([converted] if current_rank == src_rank else [None])
        return converted

    module = "relax.backends.megatron.weight_update.hf_weight_iterator_bridge"
    with (
        patch(f"{module}.dist.get_rank", return_value=current_rank),
        patch(f"{module}.device_utils.make_current_torch_device", return_value="cpu"),
        patch(f"{module}._load_to_gpu", return_value=[parameter]),
        patch(f"{module}.all_gather_param", return_value=parameter),
        patch(f"{module}._broadcast_converted_bucket", side_effect=broadcast_owner_result),
    ):
        result = list(iterator._iter_hf_params({info.name: parameter}))

    return iterator, result, converted


def test_non_expert_owner_stage_runs_bridge_conversion():
    iterator, result, converted = _run_non_expert_bridge_iterator(current_rank=0, src_rank=0)

    assert result == converted
    iterator._bridge_converter.convert.assert_called_once()


def test_non_expert_remote_stage_only_receives_converted_result():
    iterator, result, converted = _run_non_expert_bridge_iterator(current_rank=1, src_rank=0)

    assert result == converted
    iterator._bridge_converter.convert.assert_not_called()


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


def test_pipeline_metadata_gather_uses_output_list_first_and_merges_stages():
    stage_0 = _info("linear_qkv", "router.weight", (2, 6), (2, 6), None)
    stage_1 = MixtureLoraParamInfo(
        state=MixtureLoraStateSpec(
            schema_version=1,
            site_id="decoder.layers.1.self_attention.linear_qkv",
            parameter_kind="router.weight",
            global_shape=(2, 6),
            dtype=torch.float32,
        ),
        local_shape=(2, 6),
        tp_shard_dim=None,
        src_rank=1,
        weight_key="stage-1-weight",
    )

    def gather(output_list, input_object, group=None):
        assert input_object == (0, {stage_0.state.parameter_name: stage_0})
        assert group == "pp-group"
        output_list[:] = [input_object, (1, {stage_1.state.parameter_name: stage_1})]

    with (
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.all_gather_object", side_effect=gather),
    ):
        merged = _gather_pipeline_param_infos(
            {stage_0.state.parameter_name: stage_0},
            pipeline_group="pp-group",
            pipeline_world_size=2,
        )

    assert merged == {
        stage_0.state.parameter_name: stage_0,
        stage_1.state.parameter_name: stage_1,
    }


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


def _patch_bridge_mpu(**parallel_state):
    """Patch the ``mpu`` handle the bridge bound at import time.

    ``hf_weight_iterator_bridge`` does ``from megatron.core import mpu``, and
    sibling suites import it while a stub ``megatron.core`` sits in
    ``sys.modules``, so patching ``megatron.core.mpu`` can leave the object the
    bridge actually calls untouched.
    """

    from relax.backends.megatron.weight_update import hf_weight_iterator_bridge

    stub = SimpleNamespace(**{name: (lambda value=value: value) for name, value in parallel_state.items()})
    return patch.object(hf_weight_iterator_bridge, "mpu", stub)


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
        _patch_bridge_mpu(
            get_pipeline_model_parallel_world_size=1,
            get_expert_model_parallel_world_size=1,
            get_tensor_model_parallel_world_size=1,
        ),
    ):
        expert_buckets, base_buckets, _, _ = _build_param_info_buckets(args, model=[])

    assert expert_buckets == []
    assert [[info.name for info in bucket] for bucket in base_buckets] == [
        ["module.module.decoder.layers.0.self_attention.linear_qkv.weight"]
    ]


def test_bridge_ep_metadata_selects_one_owner_for_replicated_non_expert_params():
    from relax.backends.megatron.weight_update.hf_weight_iterator_bridge import _build_param_info_buckets

    non_expert_name = "module.module.decoder.layers.0.self_attention.linear_qkv.weight"
    local_expert_name = "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight1"
    remote_expert_name = "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight0"
    local_params = [
        (non_expert_name, torch.nn.Parameter(torch.zeros(4, 4))),
        (local_expert_name, torch.nn.Parameter(torch.zeros(4, 4))),
    ]
    args = SimpleNamespace(update_weight_buffer_size=1024, num_experts=2)

    def gather_ep(obj, object_list, group=None):
        rank, local_infos = obj
        assert rank == 3
        assert group == "ep-group"
        remote_infos = {
            non_expert_name: replace(local_infos[non_expert_name], src_rank=2),
            remote_expert_name: replace(local_infos[local_expert_name], name=remote_expert_name, src_rank=2),
        }
        object_list[:] = [obj, (2, remote_infos)]

    with (
        patch(
            "relax.backends.megatron.weight_update.hf_weight_iterator_bridge.named_params_and_buffers",
            side_effect=[iter(local_params), iter(local_params)],
        ),
        patch("torch.distributed.get_rank", return_value=3),
        patch("torch.distributed.all_gather_object", side_effect=gather_ep),
        _patch_bridge_mpu(
            get_pipeline_model_parallel_world_size=1,
            get_expert_model_parallel_world_size=2,
            get_expert_model_parallel_group="ep-group",
            get_tensor_model_parallel_world_size=1,
            get_expert_tensor_parallel_world_size=1,
        ),
    ):
        expert_buckets, non_expert_buckets, _, _ = _build_param_info_buckets(args, model=[])

    expert_infos = {info.name: info for bucket in expert_buckets for info in bucket}
    non_expert_infos = {info.name: info for bucket in non_expert_buckets for info in bucket}
    assert non_expert_infos[non_expert_name].src_rank == 2
    assert expert_infos[local_expert_name].src_rank == 3
    assert expert_infos[remote_expert_name].src_rank == 2


def _bridge_ep_owner_worker(rank: int, world_size: int, init_method: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    from megatron.core import parallel_state

    from relax.backends.megatron.weight_update.hf_weight_iterator_bridge import (
        _broadcast_converted_phase,
        _build_param_info_buckets,
    )

    parallel_state.initialize_model_parallel(expert_model_parallel_size=world_size)
    try:
        non_expert_name = "module.module.decoder.layers.0.self_attention.linear_qkv.weight"
        expert_name = f"module.module.decoder.layers.0.mlp.experts.linear_fc1.weight{rank}"
        local_params = [
            (non_expert_name, torch.nn.Parameter(torch.zeros(4, 4))),
            (expert_name, torch.nn.Parameter(torch.zeros(4, 4))),
        ]
        args = SimpleNamespace(update_weight_buffer_size=1024, num_experts=world_size)
        with patch(
            "relax.backends.megatron.weight_update.hf_weight_iterator_bridge.named_params_and_buffers",
            side_effect=[iter(local_params), iter(local_params)],
        ):
            expert_buckets, non_expert_buckets, _, _ = _build_param_info_buckets(args, model=[])

        expert_infos = {info.name: info for bucket in expert_buckets for info in bucket}
        non_expert_infos = {info.name: info for bucket in non_expert_buckets for info in bucket}
        assert set(expert_infos) == {
            "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight0",
            "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight1",
        }
        info = non_expert_infos[non_expert_name]
        assert info.src_rank == 0

        converted = [("model.layers.0.self_attn.qkv_proj.weight", torch.full((2, 2), 7.0))]
        local_converted = [converted if rank == info.src_rank else None]
        result = _broadcast_converted_phase(
            [info],
            local_converted,
            torch.device("cpu"),
            rank,
            parallel_state.get_expert_model_parallel_group(),
        )
        assert result[0][0][0] == converted[0][0]
        torch.testing.assert_close(result[0][0][1], converted[0][1])
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def test_bridge_ep_replicated_non_expert_has_one_real_collective_owner(tmp_path):
    init_method = f"file://{tmp_path / 'bridge-ep-owner-gloo-init'}"
    mp.spawn(_bridge_ep_owner_worker, args=(2, init_method), nprocs=2, join=True)


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
        patch("torch.distributed.all_reduce"),
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


def test_update_failure_keeps_generation_paused_and_preserves_version():
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
    updater = UpdateWeightFromTensor.__new__(UpdateWeightFromTensor)
    updater.weight_version = 4
    updater.rollout_engines = [engine]
    updater.quantization_config = None
    updater.weights_getter = MagicMock(return_value={"weight": torch.ones(1)})
    updater._hf_weight_iterator = MagicMock()
    updater._hf_weight_iterator.get_hf_weight_chunks.return_value = iter([[("base", torch.ones(1))]])
    updater._mixture_lora_sync = SimpleNamespace(base_sync_done=False)
    updater._mixture_lora_sync.get_weight_chunks = MagicMock(return_value=iter([[("router", torch.ones(1))]]))

    def fail_update(_updates):
        events.append("update")
        raise RuntimeError("engine rejected routed weights")

    updater._send_weight_update_stream = fail_update

    with (
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.all_reduce"),
        patch("relax.backends.megatron.weight_update.update_weight_from_tensor.get_gloo_group", return_value=None),
        patch("ray.get", side_effect=lambda refs: refs),
        pytest.raises(RuntimeError, match="engine rejected routed weights"),
    ):
        updater._update_weights_mixture_lora()

    assert events == ["pause", "flush", "update"]
    assert updater.weight_version == 4
    assert updater._mixture_lora_sync.base_sync_done is False


def test_stream_failure_does_not_publish_the_final_versioned_chunk():
    from relax.backends.megatron.weight_update.update_weight_from_tensor import UpdateWeightFromTensor

    updater = UpdateWeightFromTensor.__new__(UpdateWeightFromTensor)
    updater._send_hf_params = MagicMock(
        side_effect=[(["first-ref"], ["first-tensor"]), (["second-ref"], ["second-tensor"])]
    )

    with (
        patch("ray.get", side_effect=RuntimeError("engine update failed")),
        patch(
            "relax.backends.megatron.weight_update.update_weight_from_tensor."
            "device_utils.maybe_backend_barrier_on_weight_chunk"
        ) as chunk_barrier,
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.all_reduce"),
        patch("relax.backends.megatron.weight_update.update_weight_from_tensor.get_gloo_group", return_value=None),
        pytest.raises(RuntimeError, match="engine update failed"),
    ):
        updater._send_weight_update_stream([([("first", torch.ones(1))], None), ([("second", torch.ones(1))], 5)])

    updater._send_hf_params.assert_called_once()
    assert updater._send_hf_params.call_args.kwargs["weight_version"] is None
    assert chunk_barrier.call_count == 1
