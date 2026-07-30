# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterable

import ray
import torch
import torch.distributed as dist
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
from torch_memory_saver import torch_memory_saver

from relax.utils import device as device_utils
from relax.utils.distributed_utils import get_gloo_group
from relax.utils.memory_utils import clear_memory
from relax.utils.timer import Timer
from relax.utils.weight_handoff import get_memory_preflight_error

from .update_weight_from_sglang import UpdateWeightFromSGLang


logger = logging.getLogger(__name__)

_TRAIN_MODEL_REGION = "colocate_train_model"


@contextmanager
def _nvtx_range(name: str):
    if torch.cuda.is_available():
        with torch.cuda.nvtx.range(name):
            yield
    else:
        yield


def _leaf_optimizers(optimizer: Any) -> Iterable[Any]:
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is None:
        yield optimizer
        return
    for child in chained:
        yield from _leaf_optimizers(child)


class ColocateWeightHandoff:
    """Own the opt-in phase transition between Megatron and SGLang."""

    def __init__(
        self,
        *,
        args,
        model,
        optimizer,
        weights_backuper,
        weight_updater,
        memory_saver_enabled: bool,
    ) -> None:
        if not memory_saver_enabled:
            raise RuntimeError("--colocate-weight-handoff requires torch_memory_saver")
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.weights_backuper = weights_backuper
        self.weight_updater = weight_updater
        self.reverse_updater = UpdateWeightFromSGLang(weight_updater.bridge_converter)
        self.rollout_manager = None
        self.has_live_train_weights = True
        self.initial_weight_sync = True
        self._optimizer_offloaded = False
        self._optimizer_offload_started: float | None = None
        self._preflight_done = False
        self._rollout_to_train_elapsed = 0.0
        self._nvtx_phase: str | None = None
        self._nonweight_offload_ref = None
        self._nonweight_offload_started: float | None = None
        self._train_region_bytes = int(getattr(args, "_colocate_train_model_bytes", 0))

        leaves = list(_leaf_optimizers(optimizer))
        unsupported = [type(opt).__name__ for opt in leaves if not hasattr(opt, "enable_state_offload")]
        if unsupported:
            raise RuntimeError(
                "--colocate-weight-handoff requires Megatron DistributedOptimizer; "
                f"unsupported optimizers: {unsupported}"
            )
        for opt in leaves:
            opt.enable_state_offload()

    def set_rollout_manager(self, rollout_manager) -> None:
        self.rollout_manager = rollout_manager

    def offload_initial_train_state(self) -> None:
        """Free training-only GPU state before SGLang starts for the first
        time."""
        self._submit_optimizer_offload()
        device_utils.synchronize()
        self._release_optimizer_gpu_state()
        torch_memory_saver.pause(_TRAIN_MODEL_REGION)
        self.has_live_train_weights = False
        clear_memory()

    def _connect_rollout_engines(self) -> tuple[list[Any], int]:
        if self.rollout_manager is None:
            raise RuntimeError("Rollout manager is not attached to the colocate handoff")
        rollout_engines, rollout_lock, num_new, gpu_counts, gpu_offsets = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote()
        )
        if num_new > 0 or self.weight_updater._ipc_engine is None:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_lock,
                engine_gpu_counts=gpu_counts,
                engine_gpu_offsets=gpu_offsets,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_num_new_engines.remote())
        return rollout_engines, num_new

    def _record_peak_memory(self) -> None:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        used_gib = (total_bytes - free_bytes) / 1024**3
        timer = Timer()
        timer.timers["colocate_peak_gpu_memory_gib"] = max(
            timer.timers.get("colocate_peak_gpu_memory_gib", 0.0), used_gib
        )

    def _push_phase(self, name: str) -> None:
        if not torch.cuda.is_available():
            return
        if self._nvtx_phase is not None:
            raise RuntimeError(f"NVTX phase {self._nvtx_phase!r} is already active")
        torch.cuda.nvtx.range_push(name)
        self._nvtx_phase = name

    def _pop_phase(self, expected: str) -> None:
        if not torch.cuda.is_available():
            return
        if self._nvtx_phase is None:
            return
        if self._nvtx_phase != expected:
            raise RuntimeError(f"Expected NVTX phase {expected!r}, found {self._nvtx_phase!r}")
        torch.cuda.nvtx.range_pop()
        self._nvtx_phase = None

    def _collect_preflight_errors(self, *, target: str, target_bytes: int) -> list[str]:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        bucket_bytes = int(self.args.update_weight_buffer_size)
        margin_bytes = max(int(self.args.train_memory_margin_bytes), 0)
        local_error = get_memory_preflight_error(
            rank=dist.get_rank(),
            target=target,
            free_bytes=free_bytes,
            total_bytes=total_bytes,
            target_bytes=target_bytes,
            bucket_bytes=bucket_bytes,
            margin_bytes=margin_bytes,
        )
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, local_error, group=get_gloo_group())
        return [error for error in errors if error is not None]

    def _preflight_allocation(self, *, target: str, target_bytes: int) -> None:
        errors = self._collect_preflight_errors(target=target, target_bytes=target_bytes)
        if errors:
            raise RuntimeError(
                "colocate weight handoff memory preflight failed before allocation: " + "; ".join(errors)
            )

    def _preflight_train_allocation(self) -> None:
        if self._preflight_done:
            return
        self._preflight_allocation(target="Megatron", target_bytes=self._train_region_bytes)
        self._preflight_done = True

    def _preflight_rollout_allocation(self) -> None:
        target_bytes = self.reverse_updater.last_source_bytes
        if target_bytes <= 0:
            raise RuntimeError("SGLang target weight size is unknown before train-to-rollout handoff")
        self._preflight_allocation(target="SGLang", target_bytes=target_bytes)

    def _begin_nonweight_offload(self) -> None:
        if dist.get_rank() != 0:
            return
        self._nonweight_offload_started = time.monotonic()
        self._nonweight_offload_ref = self.rollout_manager.offload.remote(
            tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]
        )

    def _complete_nonweight_offload(self) -> None:
        if dist.get_rank() == 0 and self._nonweight_offload_ref is not None:
            ray.get(self._nonweight_offload_ref)
            Timer().add("sglang_nonweight_offload", time.monotonic() - self._nonweight_offload_started)
            self._nonweight_offload_ref = None
            self._nonweight_offload_started = None

    def _submit_optimizer_offload(self) -> None:
        if self._optimizer_offloaded:
            return
        self._optimizer_offload_started = time.monotonic()
        for opt in _leaf_optimizers(self.optimizer):
            opt.offload_states()
        self._optimizer_offloaded = True

    def _release_optimizer_gpu_state(self) -> None:
        if not self._optimizer_offloaded:
            return
        for opt in _leaf_optimizers(self.optimizer):
            opt.release_offloaded_gpu_states()
        if self._optimizer_offload_started is not None:
            Timer().add("optimizer_offload", time.monotonic() - self._optimizer_offload_started)
            self._optimizer_offload_started = None

    def _reload_optimizer(self) -> None:
        if not self._optimizer_offloaded:
            return
        for opt in _leaf_optimizers(self.optimizer):
            opt.reload_offloaded_states()
        self._optimizer_offloaded = False
        self._optimizer_offload_started = None

    def prepare_train(self) -> None:
        """Release rollout-only state and allocate the Megatron target
        region."""
        started = time.monotonic()
        self._pop_phase("colocate_rollout")
        with _nvtx_range("colocate_rollout_to_train"):
            self._begin_nonweight_offload()
            try:
                errors = self._collect_preflight_errors(target="Megatron", target_bytes=self._train_region_bytes)
                if errors:
                    self._complete_nonweight_offload()
                    dist.barrier(group=get_gloo_group())
                    self._preflight_train_allocation()
                else:
                    self._preflight_done = True
                torch_memory_saver.resume(_TRAIN_MODEL_REGION)
                self._record_peak_memory()
            except Exception:
                self._complete_nonweight_offload()
                dist.barrier(group=get_gloo_group())
                if dist.get_rank() == 0:
                    ray.get(self.rollout_manager.onload_kv.remote())
                raise
        self._rollout_to_train_elapsed = time.monotonic() - started
        self._push_phase("colocate_train")

    def activate_train_weights(self) -> None:
        """Complete SGLang->Megatron restore after any reference forward."""
        started = time.monotonic()
        try:
            with _nvtx_range("sglang_to_megatron"):
                self._complete_nonweight_offload()
                dist.barrier(group=get_gloo_group())
                self._connect_rollout_engines()
                count = self.reverse_updater.update_weights(
                    self.weight_updater._ipc_engine,
                    expected_version=self.weight_updater.weight_version,
                )
                dist.barrier(group=get_gloo_group())
            Timer().add("sglang_to_megatron", time.monotonic() - started)

            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.offload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
            dist.barrier(group=get_gloo_group())
            self._reload_optimizer()
            self.has_live_train_weights = True
            Timer().add(
                "colocate_rollout_to_train",
                self._rollout_to_train_elapsed + time.monotonic() - started,
            )
            self._rollout_to_train_elapsed = 0.0
            logger.info("Restored %d Megatron weights from live SGLang tensors", count)
        except Exception:
            self._pop_phase("colocate_train")
            torch_memory_saver.pause(_TRAIN_MODEL_REGION)
            clear_memory()
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.onload_kv.remote())
            self._rollout_to_train_elapsed = 0.0
            self._push_phase("colocate_rollout")
            raise

    def to_rollout(self) -> None:
        """Push live Megatron weights, then release training state and resume
        KV."""
        started = time.monotonic()
        self._pop_phase("colocate_train")
        was_initial_sync = self.initial_weight_sync
        rollout_weights_resumed = False
        source_released = False
        with _nvtx_range("colocate_train_to_rollout"):
            try:
                self._connect_rollout_engines()
                if not was_initial_sync:
                    self._submit_optimizer_offload()
                    self._preflight_rollout_allocation()

                if dist.get_rank() == 0:
                    ray.get(self.rollout_manager.onload_weights.remote())
                dist.barrier(group=get_gloo_group())
                rollout_weights_resumed = True
                self._record_peak_memory()

                update_started = time.monotonic()
                # TMS-backed model storage cannot be exported through
                # PyTorch's CUDA IPC reducer.  Keep the model in its tagged
                # region, but allocate the existing flattened transfer
                # buckets from an ordinary CUDA mempool.
                with torch_memory_saver.disable():
                    self.weight_updater.update_weights()
                Timer().add("megatron_to_sglang", time.monotonic() - update_started)
                dist.barrier(group=get_gloo_group())

                if not was_initial_sync:
                    release_started = time.monotonic()
                    device_utils.synchronize()
                    self._release_optimizer_gpu_state()
                    torch_memory_saver.pause(_TRAIN_MODEL_REGION)
                    clear_memory()
                    Timer().add("train_weight_release", time.monotonic() - release_started)
                    self.has_live_train_weights = False
                    source_released = True

                # KV/graph restoration may consume most of the rollout GPU.
                # Wait until every colocated training rank has released its
                # model and optimizer allocations before rank 0 resumes it.
                dist.barrier(group=get_gloo_group())
                if dist.get_rank() == 0:
                    resume_started = time.monotonic()
                    ray.get(self.rollout_manager.onload_kv.remote())
                    Timer().add("rollout_kv_graph_resume", time.monotonic() - resume_started)
                dist.barrier(group=get_gloo_group())
                if was_initial_sync:
                    initial_cpu_weights = self.weights_backuper.get("actor")
                    initial_host_gib = (
                        sum(tensor.numel() * tensor.element_size() for tensor in initial_cpu_weights.values())
                        / 1024**3
                    )
                    Timer().add("colocate_weight_host_transfer_gib", initial_host_gib)
                    self.weights_backuper.discard("actor")
                    clear_memory(clear_host_memory=True)
                    self.initial_weight_sync = False
                else:
                    Timer().timers.setdefault("colocate_weight_host_transfer_gib", 0.0)
            except Exception:
                # Once the Megatron source is released, the successfully updated
                # SGLang weights are the only valid copy and must remain mapped.
                if rollout_weights_resumed and not source_released and dist.get_rank() == 0:
                    ray.get(self.rollout_manager.offload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
                if not was_initial_sync and not source_released:
                    self._reload_optimizer()
                self._push_phase("colocate_rollout" if source_released else "colocate_train")
                raise
        Timer().add("colocate_train_to_rollout", time.monotonic() - started)
        self._push_phase("colocate_rollout")
