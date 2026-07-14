# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import threading
import time
from argparse import Namespace
from typing import Any, Dict, Optional

import ray
import transfer_queue as tq
from fastapi import FastAPI
from ray import serve

from relax.components.base import Base
from relax.distributed.ray.placement_group import allocate_train_group
from relax.engine.sft.runtime import is_sft_mode, sft_partition_id, sft_task_name
from relax.utils.async_utils import run
from relax.utils.opd.opd_utils import set_managed_opd_teacher_on_train_group


app = FastAPI()


@serve.deployment(max_ongoing_requests=10, max_queued_requests=20)
@serve.ingress(app)
class Actor(Base):
    """Actor service for training the policy model.

    Supports two execution modes:
    - fully_async: Asynchronous training without waiting for rollout data
    - sync: Waits for rollout data before each training step
    """

    def __init__(
        self,
        healthy: Any,
        pgs: Any,
        num_gpus: int,
        config: Namespace,
        role: str,
        runtime_env: dict = None,
    ) -> None:
        super().__init__()

        self.config = config
        self._lock = threading.RLock()
        self.healthy = healthy
        self.role = role
        self.genrm_manager = None  # Set later via set_genrm_manager if genRM is enabled

        # Threading primitives so the training loop doesn't block the Serve/FastAPI thread
        self._stop_event = threading.Event()
        self._run_thread = None
        self._done_event: Optional[asyncio.Event] = None
        self._thread_error: Optional[Exception] = None

        self.actor_model = allocate_train_group(args=config, num_gpus=num_gpus, pg=pgs, runtime_env=runtime_env)

        tq.init(self.config.tq_config)
        self.data_system_client = tq.get_client()

        self.steps = ray.get(
            self.actor_model.async_init(
                config,
                role=self.role,
                with_ref=config.kl_coef != 0 or config.use_kl_loss,
                with_opd_teacher=self.config.opd_teacher_load,
            )
        )

        assert len(set(self.steps)) == 1
        if self.config.start_rollout_id is None:
            self.config.start_rollout_id = self.steps[0]
        self.step = self.config.start_rollout_id
        self._logger.info(f"Actor initialized with starting step {self.step}")

    def set_rollout_manager(self, rollout_manager: Any) -> None:
        """Set the rollout manager and initialize weights."""
        self.rollout_manager = rollout_manager
        self.actor_model.set_rollout_manager(self.rollout_manager)

        # Call update_weights when weight_updater exists (sync colocate or hybrid mode).
        # In pure fully_async mode weight_updater is not created and weights are synced via DCS.
        # SFT: skip the init-time weight sync. SFT only sync weights to SGLang
        # right before periodic predict (gated in `train_actor`); between
        # predicts SGLang stays fully offloaded. Sync-at-init would leave
        # SGLang with `weights` resumed but no follow-up offload, causing
        # the first predict-step `onload_weights` to crash on a non-idempotent
        # `set.remove`. NCCL group setup is lazy — `connect_rollout_engines`
        # fires on the first real `update_weights` instead.
        if (not self.config.fully_async or self.config.hybrid) and not is_sft_mode(self.config):
            self.actor_model.update_weights()

    def set_genrm_manager(self, genrm_manager: Any) -> None:
        """Set the genRM manager for coordinated offload/onload.

        In colocated mode, the genRM manager is used to offload genRM engines
        before training and onload them before rollout, since they share GPU
        resources.
        """
        self.genrm_manager = genrm_manager
        self.actor_model.set_genrm_manager(self.genrm_manager)
        self._logger.info("GenRM manager set on Actor for coordinated offload/onload")

    def set_teacher_manager(self, teacher_manager: Any) -> None:
        """Set the managed OPD teacher manager for coordinated
        offload/onload."""
        set_managed_opd_teacher_on_train_group(self.actor_model, teacher_manager)
        self._logger.info("Teacher manager set on Actor for coordinated offload/onload")

    def update_weights_fully_async(self, rollout_only: bool = False, actor_fwd_only: bool = False) -> None:
        self.actor_model.update_weights_fully_async(0, rollout_only=rollout_only, actor_fwd_only=actor_fwd_only)

    async def run(self) -> None:
        """Start the training loop in a background thread and async-wait until
        it completes.

        Uses an asyncio.Event so that this coroutine yields control back to the
        Ray Serve event loop while waiting. This keeps the Serve replica
        responsive to concurrent HTTP requests while the long-running training
        loop executes in a background thread.
        """
        if self._run_thread is not None and self._run_thread.is_alive():
            if self._done_event is not None:
                await self._done_event.wait()
            return
        self.data_system_client.reset_consumption(
            partition_id=sft_partition_id(self.config, self.step),
            task_name=sft_task_name(self.config, component="actor"),
        )
        # Create an asyncio.Event bound to the current event loop so the
        # background thread can signal completion without blocking the loop.
        loop = asyncio.get_running_loop()
        self._done_event = asyncio.Event()

        def _thread_target():
            try:
                self._background_run()
            except Exception as exc:
                self._thread_error = exc
            finally:
                # Thread-safe way to set the asyncio event from a non-async context
                loop.call_soon_threadsafe(self._done_event.set)

        self._thread_error = None
        self._run_thread = threading.Thread(target=_thread_target, daemon=True)
        self._run_thread.start()
        # Async-wait: yields control so other requests can be served
        await self._done_event.wait()
        if self._thread_error is not None:
            raise self._thread_error

    def _background_run(self) -> None:
        """The actual training loop running in a background thread.

        This is a near-direct translation of the original run() logic but uses
        thread-safe access to `self.step` and respects a stop event.
        """

        try:
            while True:
                if self._stop_event.is_set():
                    self._logger.info("Actor background loop stopping by request")
                    break

                with self._lock:
                    local_step = self.step

                if local_step >= self.config.num_rollout:
                    self._logger.info("All training steps finished")
                    break

                if not self.config.fully_async and self.config.colocate and not self.config.debug_train_only:
                    if not self._wait_for_rollout_data():
                        continue

                self._logger.info(f"Actor training step {local_step}/{self.config.num_rollout}")
                self._execute_training()

                self._logger.info(f"Actor training completed step {local_step}/{self.config.num_rollout}")

                run(
                    self.data_system_client.async_clear_partition(
                        partition_id=sft_partition_id(self.config, local_step)
                    )
                )
                self._logger.info(f"Actor cleared data for step {local_step}/{self.config.num_rollout}")

                try:
                    self.healthy.update_heartbeat.remote("actor", local_step + 1)
                except Exception:
                    pass

                # increment step with lock
                with self._lock:
                    self.step += 1

        except Exception as e:
            error_msg = f"Actor training failed at step {self.step}: {type(e).__name__}: {str(e)}"
            self._logger.exception(error_msg)
            self.healthy.report_error.remote("actor", error_msg)
            if not getattr(self.config, "use_health_check", False):
                raise

    def _wait_for_rollout_data(self) -> bool:
        """Wait for rollout data to be ready in async colocate mode.

        Returns:
            True if data is ready and training can proceed,
            False if should continue waiting (caller should skip this iteration)
        """
        partition_list = run(self.data_system_client.async_get_partition_list())
        if partition_list is None or sft_partition_id(self.config, self.step) not in partition_list:
            time.sleep(1)
            return False

        # RL: poll rollout status — RL rollout runs async on the same GPUs, so
        # we must wait for it to free them before training. SFT: skip the
        # poll. The Megatron actor's per-step `requests.get("/predict")` is
        # synchronous (rank 0 blocks until run_predict + offload returns), so
        # by the time `_execute_training` returns there's no in-flight rollout
        # work. Polling here would just deadlock if status ever drifted from
        # "offload" — which has happened repeatedly across this debug session.
        if is_sft_mode(self.config):
            return True
        if self.config.offload_rollout and ray.get(self.rollout_manager.get_status.remote()) == "onload":
            time.sleep(1)
            return False
        return True

    def _execute_training(self) -> None:
        """Execute training for the current step.

        Handles critic-only phase, training method selection (sync vs async),
        and model saving based on configuration.
        """
        # Skip training during critic-only phase
        if self.step < self.config.num_critic_only_steps:
            return

        # Use appropriate training method based on mode
        if self.config.hybrid:
            # hybrid mode: actor handles ref/actor_fwd/adv internally
            ray.get(self.actor_model.train_hybrid(self.step))
        elif self.config.fully_async:
            ray.get(self.actor_model.train_fully_async(self.step))
            # Save model checkpoint if needed
            self._maybe_save_model()
        else:
            ray.get(self.actor_model.async_train(self.step))

    def _maybe_save_model(self) -> None:
        """Save model checkpoint if save interval is reached."""
        if self.config.save is None or self.config.save_interval is None:
            return

        is_save_step = (self.step + 1) % self.config.save_interval == 0
        is_final_step = (self.step + 1) == self.config.num_rollout

        if self.config.rotate_ckpt or is_save_step or is_final_step:
            self.actor_model.save_model(self.step, force_sync=is_final_step)

    def train(self, step: int, clear_data: bool = True) -> Dict[str, Any]:
        """Execute a single training step (for external control).

        This method is called by ServiceController/RLSP for fine-grained control.
        It supports interactive debugging by allowing step-by-step execution.

        Args:
            step: The training step number to execute
            clear_data: Whether to clear data partition after training (default: True)

        Returns:
            Dict containing training metrics and status
        """
        import time

        self._logger.info(f"Actor.train called with step={step}, clear_data={clear_data}")
        self.step = step

        start_time = time.time()
        metrics = {}

        try:
            # Check if rollout data is available for this step
            partition_id = sft_partition_id(self.config, step)
            partition_list = run(self.data_system_client.async_get_partition_list())

            if partition_list is not None and partition_id in partition_list:
                self._logger.info(f"Data available for step {step}, executing training")

                # Execute training
                self._execute_training()

                # Only clear partition data if clear_data is True
                if clear_data:
                    run(self.data_system_client.async_clear_partition(partition_id=partition_id))
                    self._logger.info(f"Cleared data partition: {partition_id}")
                else:
                    self._logger.info(f"Keeping data partition (clear_data=False): {partition_id}")

                metrics["data_consumed"] = True
                metrics["elapsed_time"] = time.time() - start_time
                metrics["success"] = True
            else:
                self._logger.warning(f"No data available for step {step}, skipping training")
                metrics["data_consumed"] = False
                metrics["success"] = True
                metrics["message"] = f"No data in partition {partition_id}"

        except Exception as e:
            self._logger.error(f"Training failed at step {step}: {e}")
            metrics["success"] = False
            metrics["error"] = str(e)

        return metrics

    async def stop(self) -> None:
        """Signal the background training loop to stop and wait for the thread
        to join.

        This is optional but useful for graceful shutdown in tests or service
        stops.
        """
        self._stop_event.set()
        if self._run_thread is not None:
            self._run_thread.join(timeout=5)

    # --- HTTP endpoints for restart / recovery (bypass Ray Serve handle) ---

    @app.get("/get_step")
    def http_get_step(self) -> dict:
        return {"step": self.get_step()}

    @app.post("/set_step")
    def http_set_step(self, step: int) -> dict:
        self.set_step(step)
        return {"status": "ok"}

    @app.post("/stop_service")
    async def http_stop(self) -> dict:
        await self.stop()
        return {"status": "ok"}
