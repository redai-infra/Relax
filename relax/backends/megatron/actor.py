# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import logging
import os
import random
import socket
import time
from argparse import Namespace
from contextlib import nullcontext
from functools import partial
from typing import List

import ray
import requests
import torch
import torch.distributed as dist
import transfer_queue as tq
from megatron.core import mpu
from tensordict import TensorDict
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig, AutoTokenizer

from relax.distributed.checkpoint_service.client.engine import create_client
from relax.distributed.ray.train_actor import TrainRayActor
from relax.utils import device as device_utils
from relax.utils import tracking_utils
from relax.utils.async_utils import run
from relax.utils.data.stream_dataloader import (
    create_stream_dataloader,
    get_data_from_transfer_queue,
    post_process_rollout_data,
)
from relax.utils.distributed_utils import get_gloo_group
from relax.utils.memory_utils import clear_memory, print_memory
from relax.utils.metrics.metric_utils import compute_rollout_step
from relax.utils.reloadable_process_group import destroy_process_groups, monkey_patch_torch_dist, reload_process_groups
from relax.utils.rotate_ckpt import rotate_ckpt
from relax.utils.timer import Timer, inverse_timer, timer, with_defer
from relax.utils.tracking_utils import init_tracking
from relax.utils.training import train_dump_utils
from relax.utils.training.routing_replay import RoutingReplay
from relax.utils.types import RolloutBatch
from relax.utils.utils import get_debug_data, get_serve_url, merge_dict_list, process_args

from ...utils.profile_utils import TrainProfiler
from ...utils.training.tensor_backper import TensorBackuper
from .checkpoint import load_checkpoint
from .cp_utils import slice_with_cp
from .data import (
    DataIterator,
    get_data_iterator,
    log_perf_data,
    log_perf_data_fwd,
    log_rollout_data,
    sync_actor_critic_data,
)
from .initialize import init, is_megatron_main_rank
from .loss import compute_advantages_and_returns, get_log_probs_and_entropy, get_values
from .model import forward_only, initialize_model_and_optimizer, save, train
from .weight_update.common import named_params_and_buffers
from .weight_update.update_weight_from_distributed import UpdateWeightFromDistributed
from .weight_update.update_weight_from_tensor import UpdateWeightFromTensor


logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class MegatronTrainRayActor(TrainRayActor):
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
        with timer("init_actor"):
            return self._init(args, role, with_ref, with_opd_teacher)

    @with_defer(lambda: Timer().start("train_wait"))
    def _init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
        monkey_patch_torch_dist(args)
        from relax.utils.checkpoint_write_patch import patch_checkpoint_write

        patch_checkpoint_write()
        if role == "reference" or role == "actor_fwd":
            process_args(args, role)
        super().init(args, role, with_ref, with_opd_teacher)

        self.genrm_manager = None

        init(args)
        tq.init(args.tq_config)
        self.data_system_client = tq.get_client()
        if is_megatron_main_rank():
            init_tracking(args, primary=False)

        self.prof = TrainProfiler(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(args.num_gpus_per_node):
            if i == dist.get_rank() % args.num_gpus_per_node:
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        self.train_parallel_config = {
            "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
        }
        dist.barrier(group=get_gloo_group())

        self._torch_memory_saver_enabled = False
        if args.offload_train:
            x = max(int(args.train_memory_margin_bytes), 0)
            try:
                torch_memory_saver.memory_margin_bytes = x
                self._torch_memory_saver_enabled = True
                if x > 0:
                    logger.info(f"Set torch_memory_saver.memory_margin_bytes to {x}")
            except (RuntimeError, NotImplementedError) as e:
                if "expandable_segments is not supported" in str(e) or "Only setter is supported" in str(e):
                    logger.warning(
                        "torch_memory_saver is unavailable in the current allocator mode; "
                        "skip memory saver hooks and continue with offload_train."
                    )
                else:
                    raise

        if role == "critic":
            self.args.load = self.args.critic_load
            self.args.save = self.args.critic_save
            self.args.lr = self.args.critic_lr
            self.args.lr_warmup_iters = self.args.critic_lr_warmup_iters

        self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id = initialize_model_and_optimizer(
            args, role
        )

        start_rollout_id = loaded_rollout_id + 1

        if role == "critic":
            if self.args.offload_train:
                self.sleep()
            return start_rollout_id

        if self.args.vocab_size is None:
            self.args.vocab_size = self.tokenizer.vocab_size
        if not self.args.fully_async:
            self.weights_backuper = TensorBackuper.create(
                source_getter=lambda: named_params_and_buffers(
                    self.args,
                    self.model,
                    convert_to_global_name=args.megatron_to_hf_mode == "raw",
                    translate_gpu_to_cpu=not self.args.enable_weights_backuper,
                ),
                single_tag=None if args.enable_weights_backuper else "actor",
            )
            self._active_model_tag: str | None = "actor"
            self.weights_backuper.backup("actor")

            if with_ref:
                self.load_other_checkpoint("ref", args.ref_load)

            # Load teacher model for Megatron-based on-policy distillation
            if with_opd_teacher:
                self.load_other_checkpoint("teacher", args.opd_teacher_load)

            if self.args.keep_old_actor:
                # Load old_actor checkpoint
                self.load_other_checkpoint("old_actor", args.load)
                # Create rollout_actor as a copy of current actor
                if args.update_weights_interval == 1:
                    self.weights_backuper.backup("rollout_actor")

            update_weight_cls = UpdateWeightFromTensor if self.args.colocate else UpdateWeightFromDistributed
            self.weight_updater = update_weight_cls(
                self.args,
                self.model,
                weights_getter=lambda: self.weights_backuper.get("actor"),
                model_name=type(self.hf_config).__name__.lower()
                if self.args.model_name is None
                else self.args.model_name,
                quantization_config=getattr(self.hf_config, "quantization_config", None),
            )
        else:
            is_pp_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
            )
            if is_pp_src_rank:
                master_address = ray._private.services.get_node_ip_address()
                with socket.socket() as sock:
                    sock.bind(("", 0))
                    master_port = sock.getsockname()[1]
            else:
                master_address = None
                master_port = None
            metadata = {
                "tp_size": mpu.get_tensor_model_parallel_world_size(),
                "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
                "pp_size": mpu.get_pipeline_model_parallel_world_size(),
                "ep_size": mpu.get_expert_model_parallel_world_size(),
                "cp_size": mpu.get_context_parallel_world_size(),
                "pp_rank": mpu.get_pipeline_model_parallel_rank(),
                "is_pp_src_rank": is_pp_src_rank,
                "master_address": master_address,
                "master_port": master_port,
            }
            self.checkpoint_engine_client = run(
                create_client(
                    args=self.args,
                    coordinator_url=self.args.coordinator_url,
                    role=role,
                    rank=dist.get_rank(),
                    model=self.model,
                    model_name=type(self.hf_config).__name__.lower()
                    if self.args.model_name is None
                    else self.args.model_name,
                    quantization_config=getattr(self.hf_config, "quantization_config", None),
                    backend_type=self.args.checkpoint_engine_backend,
                    metadata=metadata,
                    lock=self.lock,
                )
            )
        # empty cache after initialization
        clear_memory()

        if self.args.offload_train:
            # recover to actor in the end.
            self._switch_model("actor")
            self.sleep()

        self.rollout_engines = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from relax.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        self.prof.on_init_end()
        self.data_iterator = None

        return start_rollout_id

    @timer
    def sleep(self) -> None:
        assert self.args.offload_train

        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        # In disaggregate PPO (use_critic + not colocate), the actor's NCCL
        # weight-sync groups to rollout engines do not survive a sleep, so we
        # must explicitly tear them down before destroy_process_groups() and
        # reconnect on wake_up()/update_weights().
        if (
            self.role == "actor"
            and self.args.use_critic
            and not self.args.colocate
            and hasattr(self, "weight_updater")
            and hasattr(self.weight_updater, "disconnect_rollout_engines")
        ):
            self.weight_updater.disconnect_rollout_engines()
        destroy_process_groups()

        if self._torch_memory_saver_enabled:
            torch_memory_saver.pause()

        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        print_memory("before wake_up model")

        if self._torch_memory_saver_enabled:
            torch_memory_saver.resume()

        clear_memory()
        reload_process_groups(timeout_minutes=self.args.distributed_timeout_minutes)
        print_memory("after wake_up model")

    def _switch_model(self, target_tag: str) -> None:
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.weights_backuper.restore(target_tag)
        self._active_model_tag = target_tag

    def fill_routing_replay(self, data_iterator, num_microbatches, rollout_data):
        if "rollout_routed_experts" not in rollout_data:
            raise ValueError(
                "rollout_routed_experts is required in rollout_data when use_rollout_routing_replay is set."
            )

        from megatron.core.transformer.transformer_block import get_num_layers_to_build
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

        from relax.utils.training.routing_replay import RoutingReplay

        for iterator in data_iterator:
            iterator.reset()

        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        def pad_func(experts, pad):
            _, num_layers, topk = experts.shape
            pad = (
                torch.arange(
                    pad * num_layers * topk,
                    device=experts.device,
                    dtype=experts.dtype,
                ).reshape((pad, num_layers, topk))
                % self.args.num_experts
            )
            return torch.cat([experts, pad], dim=0)

        for _ in range(sum(num_microbatches)):
            batch = data_iterator[0].get_next(["rollout_routed_experts", "tokens"])
            rollout_routed_experts = batch["rollout_routed_experts"]
            tokens = batch["tokens"]
            # Reshape from flattened 2D (seq_i, num_layers*topk) back to 3D (seq_i, num_layers, topk)
            # because dict_to_tensordict flattened the last two dims for NestedTensor jagged layout.
            rollout_routed_experts = [
                r.reshape(r.shape[0], self.args.num_layers, self.args.moe_router_topk) for r in rollout_routed_experts
            ]
            assert len(rollout_routed_experts) == len(tokens)
            for a, b in zip(rollout_routed_experts, tokens, strict=False):
                assert a.shape[0] == b.shape[0] - 1, f"{a.shape}, {b.shape}"

            # We need to pad the experts to the last token. We won't calculate loss on this token so this should be fine.
            # TODO: fuse this padding with the following slice_with_cp to reduce memory copy.
            rollout_routed_experts = [pad_func(r, 1) for r in rollout_routed_experts]
            # TODO: maybe extract a common process function for here and get_batch?
            rollout_routed_experts = [slice_with_cp(r, pad_func) for r in rollout_routed_experts]
            rollout_routed_experts = torch.cat(rollout_routed_experts, dim=0)
            pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
            pad = (pad_size - rollout_routed_experts.size(0) % pad_size) % pad_size
            if pad != 0:
                rollout_routed_experts = pad_func(rollout_routed_experts, pad)

            if self.args.sequence_parallel:
                seqlen = rollout_routed_experts.size(0)
                assert seqlen % tp_size == 0
                start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
                rollout_routed_experts = rollout_routed_experts[start:end]

            routing_replay_offset = 0
            for vp_stage, model in enumerate(self.model):
                config = model.module.config
                num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
                offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
                for layer_id in range(offset, offset + num_layers_to_build):
                    # skip dense layer
                    if isinstance(config.moe_layer_freq, int):
                        if layer_id % config.moe_layer_freq != 0:
                            continue
                    elif isinstance(config.moe_layer_freq, list):
                        assert len(config.moe_layer_freq) == config.num_layers
                        if config.moe_layer_freq[layer_id] == 0:
                            continue
                    layer_routed_experts = rollout_routed_experts[:, layer_id]
                    RoutingReplay.all_routing_replays[routing_replay_offset].record(layer_routed_experts)
                    routing_replay_offset += 1
            assert routing_replay_offset == len(RoutingReplay.all_routing_replays), (
                f"{routing_replay_offset} vs {len(RoutingReplay.all_routing_replays)}"
            )

        del rollout_data["rollout_routed_experts"]

        for iterator in data_iterator:
            iterator.reset()

    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
        collect_topk: bool = False,
    ) -> dict[str, list[torch.Tensor]]:
        with timer(f"{store_prefix}log_probs"):
            log_prob_func = get_log_probs_and_entropy
            if collect_topk:
                log_prob_func = partial(log_prob_func, with_topk=True, topk_k=self.args.opd_log_prob_top_k)
            return forward_only(
                log_prob_func,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
            )

    def train(self, rollout_id: int) -> None:
        # offload genrm before train (rollout has already self-offloaded at end of _async_run)
        if self.args.offload_rollout and dist.get_rank() == 0 and self.genrm_manager is not None:
            ray.get(self.genrm_manager.offload.remote())

        if self.args.offload_train:
            self.wake_up()

        if self.args.debug_train_only:
            logger.info(f"start to get rollout_id: {rollout_id} data from transfer queue for debug with mcore.")
            batch_size = self.args.global_batch_size // mpu.get_data_parallel_world_size(with_context_parallel=False)
            rollout_data = get_debug_data(self.args, rollout_id, batch_size, dp_rank=mpu.get_data_parallel_rank())
            post_process_rollout_data(self.args, rollout_data)

            if self.role == "critic":
                return self.train_critic(rollout_id, rollout_data)
            else:
                return self.train_actor(rollout_id, rollout_data)
        else:
            logger.info(f"start to get rollout_id: {rollout_id} data from transfer queue for train with mcore.")
            batch_size = (
                self.args.rollout_batch_size
                * self.args.n_samples_per_prompt
                // mpu.get_data_parallel_world_size(with_context_parallel=False)
            )
            batch_index = 0
            while not self.all_consumed("train", rollout_id):
                data_fields = [
                    "tokens",
                    "total_lengths",
                    "response_lengths",
                    "loss_masks",
                    "rollout_log_probs",
                    "rewards",
                    "raw_reward",
                ]
                data_fields += ["rollout_routed_experts"] if self.args.use_rollout_routing_replay else []
                if self.args.multimodal_keys is not None:
                    data_fields.append("multimodal_train_inputs")
                if self.args.use_opd and self.args.opd_type == "sglang":
                    data_fields.append("teacher_log_probs")
                    if self.args.opd_log_prob_top_k > 0:
                        data_fields.append("teacher_topk_token_ids")
                        data_fields.append("teacher_topk_k")
                with timer("train_get_data"):
                    rollout_data, batch_meta = self._get_data_from_transfer_queue(
                        "train", rollout_id, data_fields, batch_size, batch_index
                    )
                if rollout_data is None:
                    continue
                batch_index += 1
                if self.role == "critic":
                    return self.train_critic(rollout_id, rollout_data)
                else:
                    return self.train_actor(rollout_id, rollout_data)

    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        rollout_data.update(
            forward_only(
                get_values,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
            )
        )

        if rollout_id >= self.args.num_critic_only_steps and not self.args.critic_train_only:
            sync_actor_critic_data(self.args, rollout_data, self._actor_critic_groups)

        compute_advantages_and_returns(self.args, rollout_data)

        self.args.loss_type = "value_loss"
        train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
        )
        tracking_utils.flush_metrics(self.args, compute_rollout_step(self.args, rollout_id))

    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)

        if self.args.use_rollout_routing_replay:
            self.fill_routing_replay(data_iterator, num_microbatches, rollout_data)

        with inverse_timer("train_wait"), timer("train"):
            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("ref")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="ref_",
                        )
                    )

                # Forward teacher model to get teacher_log_probs for Megatron-based OPD
                if "teacher" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("teacher")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="teacher_",
                            collect_topk=self.args.use_opd and self.args.opd_log_prob_top_k > 0,
                        )
                    )

                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics:
                    if self.args.use_routing_replay:
                        if self.args.use_rollout_routing_replay:
                            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
                        else:
                            os.environ["ROUTING_REPLAY_STAGE"] = "record"
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="",
                            collect_topk=self.args.use_opd and self.args.opd_log_prob_top_k > 0,
                        )
                    )
                    if self.args.use_rollout_routing_replay:
                        RoutingReplay.clear_all_forward()

                if self.args.use_critic:
                    sync_actor_critic_data(
                        self.args,
                        rollout_data,
                        self._actor_critic_groups,
                    )
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args)

            log_rollout_data(rollout_id, self.args, rollout_data)

            # Train
            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                )

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(
            self.args, rollout_id=rollout_id, rollout_data=rollout_data, tokenizer=self.tokenizer
        )

        if self.args.use_routing_replay:
            RoutingReplay.clear_all()

        # update the cpu actor weight to the latest model
        self.weights_backuper.backup("actor")

        # Update ref model if needed
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights_backuper.backup_tags
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.weights_backuper.backup("ref")

        total_lengths = rollout_data["total_lengths"]
        all_total_lengths = [None] * mpu.get_data_parallel_world_size(with_context_parallel=False)
        dist.all_gather_object(
            all_total_lengths, total_lengths, group=mpu.get_data_parallel_group(with_context_parallel=False)
        )
        all_total_lengths = sum(all_total_lengths, [])  # flatten
        Timer().seq_lens = all_total_lengths
        log_perf_data(rollout_id, self.args)
        is_train_done = (rollout_id + 1) == self.args.num_rollout
        if self.args.save is not None and (
            self.args.rotate_ckpt
            or self.args.save_interval is not None
            and ((rollout_id + 1) % self.args.save_interval == 0 or is_train_done)
        ):
            self.save_model(rollout_id, force_sync=is_train_done)
        if self.args.offload_train:
            self.sleep()
        self.update_weights()
        tracking_utils.flush_metrics(self.args, compute_rollout_step(self.args, rollout_id))
        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            try:
                rollout_serve_url = get_serve_url("rollout")
                response = requests.get(f"{rollout_serve_url}/evaluate", params={"train_step": rollout_id})
                response.raise_for_status()
            except Exception as e:
                logger.warning(f"Error triggering evaluation for rollout_id {rollout_id}: {e}")

        # On the final training step the rollout component has already exited
        # its main loop, so nothing else awaits the eval handler. Block here
        # until eval finishes; otherwise the controller's atexit shutdown
        # races with eval and tears down the SGLang engines mid-flight.
        if is_train_done:
            self._wait_for_previous_eval()

    def compute_ref_log_prob(self, rollout_id: int) -> None:
        if self.args.use_routing_replay:
            os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"

        logger.info(f"start to get rollout_id: {rollout_id} data from transfer queue for compute_ref_log_prob.")
        batch_size = (
            self.args.global_batch_size
            // mpu.get_data_parallel_world_size(with_context_parallel=False)
            // self.args.num_iters_per_train_update
        )
        batch_index = 0
        while not self.all_consumed("ref_log_probs", rollout_id):
            data_fields = ["tokens", "total_lengths", "response_lengths", "loss_masks", "rollout_log_probs"]
            if self.args.multimodal_keys is not None:
                data_fields.append("multimodal_train_inputs")
            data, batch_meta = self._get_data_from_transfer_queue(
                "ref_log_probs", rollout_id, data_fields, batch_size, batch_index
            )
            if data is None:
                continue
            batch_index += 1
            logger.info(f"Successfully got rollout_id: {rollout_id} data from transfer queue for compute_ref_log_prob")
            data_iterator, num_microbatches = get_data_iterator(self.args, self.model, data)

            output_dict = self.compute_log_prob(
                data_iterator,
                num_microbatches,
                store_prefix="ref_",
            )
            self._put_data_to_transfer_queue(output_dict, batch_meta)
        self.prof.step(rollout_id=rollout_id)

        self.recv_weight_fully_async(rollout_id)
        log_perf_data_fwd(self.args, rollout_id)

    def compute_actor_log_prob(self, rollout_id: int) -> None:
        if self.args.use_routing_replay:
            if self.args.use_rollout_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
            else:
                os.environ["ROUTING_REPLAY_STAGE"] = "record"
        logger.info(f"start to get rollout_id: {rollout_id} data from transfer queue for compute_actor_log_prob.")
        batch_size = (
            self.args.global_batch_size
            // mpu.get_data_parallel_world_size(with_context_parallel=False)
            // self.args.num_iters_per_train_update
        )
        batch_index = 0
        while not self.all_consumed("actor_log_probs", rollout_id):
            data_fields = ["tokens", "total_lengths", "response_lengths", "loss_masks", "rollout_log_probs"]
            if self.args.multimodal_keys is not None:
                data_fields.append("multimodal_train_inputs")
            data, batch_meta = self._get_data_from_transfer_queue(
                "actor_log_probs", rollout_id, data_fields, batch_size, batch_index
            )
            if data is None:
                continue
            batch_index += 1
            logger.info(
                f"Successfully got rollout_id: {rollout_id} data from transfer queue for compute_actor_log_prob"
            )
            data_iterator, num_microbatches = get_data_iterator(self.args, self.model, data)

            output_dict = self.compute_log_prob(
                data_iterator,
                num_microbatches,
                store_prefix="",
            )
            self._put_data_to_transfer_queue(output_dict, batch_meta)
            if self.args.use_rollout_routing_replay:
                RoutingReplay.clear_all_forward()
        self.prof.step(rollout_id=rollout_id)

        self.recv_weight_fully_async(rollout_id)
        log_perf_data_fwd(self.args, rollout_id)

    def train_async(self, rollout_id) -> None:
        if self.args.use_routing_replay:
            os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"

        logger.info(f"start to get rollout_id: {rollout_id} data from transfer queue for train step.")
        data_fields = [
            "tokens",
            "total_lengths",
            "response_lengths",
            "loss_masks",
            "advantages",
            "returns",
            "rollout_log_probs",
            "rewards",
            "raw_reward",
        ]
        # In true on-policy mode, actor_fwd is absent and old_log_probs is
        # recomputed inline from the train forward (see policy_loss_function).
        if not getattr(self.args, "true_on_policy_mode", False):
            data_fields.append("log_probs")
        if self.args.kl_coef != 0 or self.args.use_kl_loss:
            data_fields.append("ref_log_probs")
        if self.args.multimodal_keys is not None:
            data_fields.append("multimodal_train_inputs")
        if self.args.use_opd and self.args.opd_type == "sglang":
            data_fields.append("teacher_log_probs")
            data_fields.append("opd_reverse_kl")
            if self.args.opd_log_prob_top_k > 0:
                data_fields.append("teacher_topk_token_ids")
                data_fields.append("teacher_topk_k")
        if self.data_iterator is None:
            self.data_iterator, self.num_microbatches = create_stream_dataloader(
                self.args,
                rollout_id=rollout_id,
                task_name="actor_train",
                data_fields=data_fields,
                dp_rank=mpu.get_data_parallel_rank(),
            )
        else:
            for data_iterator in self.data_iterator:
                data_iterator.step(f"train_{rollout_id}")
        with inverse_timer("train_wait"), timer("train"):
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    self.data_iterator,
                    self.num_microbatches,
                )
            self.prof.step(rollout_id=rollout_id)

            data_for_log = self.data_iterator[0].get_buffer()
            rollout_data = merge_dict_list(data_for_log)
            log_rollout_data(rollout_id, self.args, rollout_data)
            train_dump_utils.save_debug_train_data(
                self.args, rollout_id=rollout_id, rollout_data=rollout_data, tokenizer=self.tokenizer
            )

            rollout_only, actor_fwd_only = self._check_services_health()

            # wait for last evaluation
            self._wait_for_previous_eval()

            self.update_weights_fully_async(rollout_id, rollout_only=rollout_only, actor_fwd_only=actor_fwd_only)
            dist.barrier(group=get_gloo_group())
            try:
                if dist.get_rank() == 0:
                    rollout_serve_url = get_serve_url("rollout")
                    response = requests.get(f"{rollout_serve_url}/evaluate", params={"train_step": rollout_id})
                    response.raise_for_status()

                    response = requests.get(f"{rollout_serve_url}/end_update_weight")
                    response.raise_for_status()
            except Exception as e:
                logger.warning(
                    f"Error during async weight update: {e}, maybe cause by rollout server failure. Will continue without async update for this step."
                )
            # On the final training step the rollout component has already
            # exited its main loop, so the eval just triggered above will not
            # be awaited anywhere. Block until it finishes; otherwise the
            # controller's atexit shutdown races with eval and tears down the
            # SGLang engines mid-flight.
            if (rollout_id + 1) == self.args.num_rollout:
                self._wait_for_previous_eval()
            if self.args.use_routing_replay:
                RoutingReplay.clear_all()
        total_lengths = rollout_data["total_lengths"]
        all_total_lengths = [None] * mpu.get_data_parallel_world_size(with_context_parallel=False)
        dist.all_gather_object(
            all_total_lengths, total_lengths, group=mpu.get_data_parallel_group(with_context_parallel=False)
        )
        all_total_lengths = sum(all_total_lengths, [])  # flatten
        Timer().seq_lens = all_total_lengths
        log_perf_data(rollout_id, self.args)
        tracking_utils.flush_metrics(self.args, compute_rollout_step(self.args, rollout_id))

    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return
        # torch dist may trigger nccl communication during saving; resume the
        # paused model (process groups + tms) so save can issue collectives and
        # touch GPU tensors.
        if self.args.offload_train:
            self.wake_up()

        if self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)

        if dist.get_rank() == 0:
            rotate_ckpt(self.args, global_step=rollout_id)

        dist.barrier(group=get_gloo_group())

        save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

        if force_sync and self.args.async_save:
            maybe_finalize_async_save(blocking=True)

        if self.args.save_hf is not None and self.role == "actor":
            from relax.backends.megatron.model import save_hf_model

            save_hf_model(self.args, rollout_id, self.model)

        if self.args.offload_train:
            self.sleep()

    @timer
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.offload_train:
            # CRITICAL: Barrier before onload_weights to ensure ALL ranks have
            # completed sleep() (and released GPU memory via tms.pause()) before
            # SGLang's resume_memory_occupation tries to reclaim GPU memory.
            # Without this, rank 0 may trigger SGLang resume while other ranks
            # still hold GPU memory, causing cuMemCreate OOM in SGLang schedulers.
            dist.barrier(group=get_gloo_group())

        if self.args.offload_rollout and dist.get_rank() == 0:
            # Onload rollout (weights) and genrm (KV resume only — genrm has no NCCL
            # weight sync since the reward model is static) in parallel so both engines
            # come back together before the next rollout step.
            onload_handles = [self.rollout_manager.onload_weights.remote()]
            if self.genrm_manager is not None:
                onload_handles.append(self.genrm_manager.onload.remote())
            ray.get(onload_handles)

        if self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_rollout_engines.remote())
            dist.barrier(group=get_gloo_group())

        rollout_engines, rollout_engine_lock, num_new_engines, engine_gpu_counts, engine_gpu_offsets = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote()
        )

        # Disaggregate PPO tears down the actor↔rollout NCCL groups on sleep(),
        # so we must wake_up() fully (not just reload_process_groups) and force
        # a reconnect here, then sleep() at the end.
        reconnect_rollout_engines = self.args.offload_train and self.args.use_critic and not self.args.colocate

        if reconnect_rollout_engines:
            self.wake_up()
        elif self.args.offload_train:
            reload_process_groups(timeout_minutes=self.args.distributed_timeout_minutes)

        if num_new_engines > 0 or reconnect_rollout_engines:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_num_new_engines.remote())

        with (
            torch_memory_saver.disable()
            if self.args.offload_train and self._torch_memory_saver_enabled
            else nullcontext()
        ):
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if self.args.ci_test and len(rollout_engines) > 0:
                engine = random.choice(rollout_engines)
                engine_version = ray.get(engine.get_weight_version.remote())
                if str(engine_version) != str(self.weight_updater.weight_version):
                    raise RuntimeError(
                        f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                    )

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                    # Then copy current actor to rollout_actor
                    self.weights_backuper.backup("rollout_actor")
                else:
                    self.weights_backuper.backup("old_actor")
        if reconnect_rollout_engines:
            self.sleep()
        elif self.args.offload_train:
            destroy_process_groups()

        if self.args.offload_rollout and dist.get_rank() == 0:
            ray.get(self.rollout_manager.onload_kv.remote())

    @timer("wait update_weights_fully_async")
    def _check_services_health(self) -> tuple[bool, bool]:
        """Check rollout and actor_fwd service health before weight update.

        Only rank 0 sends HTTP requests to check service availability, then
        the results are broadcast via allreduce to ensure all ranks have a
        consistent view.

        Returns:
            (rollout_only, actor_fwd_only): Flags indicating which services
            are unavailable and should be skipped during weight update.
        """
        # Default: both services healthy → update both
        rollout_only = False
        actor_fwd_only = False

        # When true_on_policy_mode is enabled, actor_fwd is intentionally absent
        # (its log_probs are recomputed inline by the train forward). Force
        # rollout-only weight update and skip the actor_fwd HTTP probe.
        actor_fwd_absent = getattr(self.args, "true_on_policy_mode", False)

        if dist.get_rank() == 0:
            # Check rollout service
            try:
                rollout_serve_url = get_serve_url("rollout")
                while True:
                    response = requests.get(f"{rollout_serve_url}/can_do_update_weight_for_async")
                    response.raise_for_status()
                    res = response.json()
                    if res:
                        response = requests.get(f"{rollout_serve_url}/recover_rollout_engines")
                        response.raise_for_status()
                        break
                    else:
                        time.sleep(1)
            except Exception as e:
                logger.warning(
                    f"Error checking rollout service: {e}, maybe caused by rollout server failure. "
                    "Will continue without rollout update for this step."
                )
                actor_fwd_only = True

            # Check actor_fwd service (skip when intentionally absent)
            if actor_fwd_absent:
                rollout_only = True
            else:
                try:
                    actor_fwd_serve_url = get_serve_url("actor_fwd")
                    response = requests.get(f"{actor_fwd_serve_url}/get_step")
                    response.raise_for_status()
                except Exception as e:
                    logger.warning(
                        f"Error checking actor_fwd service: {e}, maybe caused by actor_fwd server failure. "
                        "Will continue without actor_fwd update for this step."
                    )
                    rollout_only = True

        # Broadcast results from rank 0 to all ranks via allreduce
        # Encode booleans as integers: 1 = skip, 0 = healthy
        flags = torch.tensor(
            [int(rollout_only), int(actor_fwd_only)],
            dtype=torch.int32,
            device=device_utils.make_current_torch_device(),
        )
        dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=get_gloo_group())
        rollout_only = bool(flags[0].item())
        actor_fwd_only = bool(flags[1].item())

        return rollout_only, actor_fwd_only

    def _wait_for_previous_eval(self, max_wait_seconds: int = 1800) -> None:
        """Block until the rollout service's previous evaluation has finished.

        Only rank 0 polls the rollout HTTP endpoint; all ranks synchronise via
        a barrier afterwards so they proceed together.
        Args:
            max_wait_seconds: Maximum time (in seconds) to wait for eval
                completion before giving up and proceeding. Defaults to 300.
        """
        if dist.get_rank() == 0:
            try:
                rollout_serve_url = get_serve_url("rollout")
                start_time = time.monotonic()
                while True:
                    response = requests.get(f"{rollout_serve_url}/is_eval_done", timeout=10)
                    response.raise_for_status()
                    if response.json().get("done", True):
                        break
                    elapsed = time.monotonic() - start_time
                    if elapsed >= max_wait_seconds:
                        logger.warning(
                            f"Timed out waiting for previous evaluation after {elapsed:.1f}s "
                            f"(max_wait_seconds={max_wait_seconds}), proceeding with weight update."
                        )
                        break
                    logger.info("Waiting for previous evaluation to finish before updating weights...")
                    time.sleep(10)
            except Exception as e:
                logger.warning(f"Error checking eval status: {e}, proceeding with weight update.")
        dist.barrier(group=get_gloo_group())

    @timer
    def update_weights_fully_async(self, rollout_id, rollout_only=False, actor_fwd_only=False) -> None:
        """Fully async version of update_weights, which will be called by actor
        train engine after it train update.

        This sends weights to both rollout and actor_fwd nodes using pipelined
        design.
        """
        if rollout_only and actor_fwd_only:
            logger.warning("Both rollout_only and actor_fwd_only are True, skipping async weight update.")
            return

        dist.barrier(group=get_gloo_group())
        print_memory("before update_weights")

        weight_sync_lock = getattr(self, "_weight_sync_lock", None)
        if weight_sync_lock is not None and dist.get_rank() == 0:
            acquired = False
            while not acquired:
                acquired = ray.get(weight_sync_lock.acquire.remote())
                if not acquired:
                    time.sleep(1)

        try:
            if not rollout_only:
                run(self.checkpoint_engine_client.init_process_groups_for_actor_fwd_ref(rollout_id))
            run(self.checkpoint_engine_client.update_weights_for_rollout(rollout_only, actor_fwd_only))
        finally:
            if weight_sync_lock is not None and dist.get_rank() == 0:
                ray.get(weight_sync_lock.release.remote())

    @timer
    def recv_weight_fully_async(self, rollout_id) -> None:
        """Receive weights from actor (called by actor_fwd side)."""
        dist.barrier(group=get_gloo_group())
        print_memory("before update_weights")
        run(self.checkpoint_engine_client.init_process_groups_for_actor_fwd_ref(rollout_id))
        run(self.checkpoint_engine_client.recv_weight_fully_async())
        print_memory("after update_weights")

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        old_ckpt_step = None
        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step
        elif model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.opd_teacher_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if old_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag

    def all_consumed(self, task_name, rollout_id):
        # Only (TP=0, PP=0, CP=0) queries the transfer queue; otherwise different cp_ranks
        # may observe different consumption status due to concurrent fetches and diverge,
        # leaving some ranks idle while others enter the next collective and hang.
        if (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == 0
            and mpu.get_context_parallel_rank() == 0
        ):
            status = [run(self.data_system_client.async_check_consumption_status(task_name, f"train_{rollout_id}"))]
        else:
            status = [True]
        status = torch.tensor(status, device=device_utils.make_current_torch_device())
        dist.broadcast(status, group=mpu.get_context_parallel_group(), group_src=0)
        dist.broadcast(status, group=mpu.get_tensor_model_parallel_group(), group_src=0)
        dist.broadcast(status, group=mpu.get_pipeline_model_parallel_group(), group_src=0)

        return status[0]

    def _get_data_from_transfer_queue(self, task_name, rollout_id, data_fields, batch_size, batch_index):
        # Fetch data through ray on CPU, not sure if this will be performance bottleneck.
        # Both first pp stage and the last pp stage will recieve the data.
        partition_id = f"train_{rollout_id}"
        sampling_config = {
            "dp_rank": mpu.get_data_parallel_rank(),
            "task_name": task_name,
        }
        rollout_data, batch_meta = get_data_from_transfer_queue(
            self.args,
            self.data_system_client,
            data_fields,
            batch_size,
            partition_id,
            task_name,
            sampling_config,
            batch_index,
        )

        return rollout_data, batch_meta

    def _put_data_to_transfer_queue(self, output_dict=None, batch_meta=None):
        if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
            output_dict = {
                key: value.cpu()
                if not isinstance(value, List)
                else torch.nested.as_nested_tensor([item.cpu() for item in value], layout=torch.jagged)
                for key, value in output_dict.items()
            }
            output_dict = TensorDict(output_dict, batch_size=[len(batch_meta.samples)])
            run(self.data_system_client.async_put(data=output_dict, metadata=batch_meta))
