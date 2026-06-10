# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
import time
import uuid
from argparse import Namespace
from typing import Any, Dict, List, Optional, Union

import httpx
import ray
import transfer_queue as tq
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from ray import serve

from relax.components.base import Base
from relax.distributed.ray.placement_group import create_rollout_manager
from relax.utils.http_utils import _wrap_ipv6


app = FastAPI()


# ===================== Scale-Out API Models =====================


class ScaleOutAPIRequest(BaseModel):
    """API request model for scale-out operation.

    Idempotency guarantees:
    - If num_replicas > 0: ray_native mode. num_replicas is the target *absolute* total engine count.
      If the current engine count (including in-flight requests) already meets or
      exceeds the target, no scale-out occurs (returns immediately with noop status).
    - If engine_urls is provided: external mode. engine_urls are filtered to exclude engines already active
      in the system or currently being processed by in-flight scale-out requests (status in
      PENDING, CREATING, CONNECTING, HEALTH_CHECKING, WEIGHT_SYNCING, READY, ACTIVE).
    - num_gpus_per_engine is always taken from args.rollout_num_gpus_per_engine.
    """

    model_name: str = Field(default="default", description="Target model name")
    num_replicas: int = Field(
        default=0,
        ge=0,
        description=(
            "Target absolute total engine count for ray_native mode (num_replicas > 0). If current engines >= target, no-op."
        ),
    )
    engine_urls: List[str] = Field(
        default_factory=list,
        description=(
            "External engine URLs to add (external mode, when num_replicas == 0). "
            "URLs already active or in-flight are automatically filtered out (idempotent)."
        ),
    )
    timeout_secs: Optional[float] = Field(default=None, gt=0, description="Total timeout for the operation")


class ScaleOutResponse(BaseModel):
    """Response model for scale-out operation."""

    request_id: str
    status: str
    message: str = "Scale-out request accepted"


class ScaleOutStatusResponse(BaseModel):
    """Response model for scale-out status query."""

    request_id: str
    status: str
    model_name: str
    num_replicas: int
    engine_urls: List[str]
    engine_ids: List[str]
    failed_engines: List[str]
    created_at: float
    updated_at: float
    error_message: Optional[str]
    weight_version: Optional[str]


class EnginesInfoResponse(BaseModel):
    """Response model for engines info."""

    models: dict  # pyright: ignore[reportMissingTypeArgument]
    total_engines: int


class CancelResponse(BaseModel):
    """Response model for cancel operation."""

    request_id: str
    status: str
    message: str


class ListScaleOutRequestsResponse(BaseModel):
    """Response model for list scale-out requests endpoint."""

    requests: List[ScaleOutStatusResponse]
    total_count: int = Field(..., description="Total number of requests returned")
    model_name: Optional[str] = Field(None, description="Filter used: model_name")
    status_filter: Optional[str] = Field(None, description="Filter used: status")


class CancelAllScaleOutRequestsRequest(BaseModel):
    """Request model for cancel all scale-out requests."""

    model_name: Optional[str] = Field(None, description="Filter: only cancel requests for this model")
    status_filter: Optional[str] = Field(None, description="Filter: only cancel requests with this status")
    dry_run: bool = Field(False, description="If true, preview what would be cancelled without actually cancelling")


class CancelAllScaleOutRequestsResponse(BaseModel):
    """Response model for cancel all scale-out requests."""

    succeeded: List[str] = Field(..., description="List of cancelled request IDs")
    skipped: List[dict] = Field(  # pyright: ignore[reportMissingTypeArgument]
        ..., description="List of requests that couldn't be cancelled, with reasons"
    )
    total_count: int = Field(..., description="Total requests matching filters")
    dry_run: bool = Field(..., description="Whether this was a dry-run")
    filters: dict = Field(..., description="Filters that were applied")  # pyright: ignore[reportMissingTypeArgument]


# ===================== Scale-In API Models =====================


class ScaleInAPIRequest(BaseModel):
    """API request model for scale-in operation."""

    model_name: str = Field(default="default", description="Target model name")
    num_replicas: int = Field(
        default=0,
        ge=0,
        description="Target number of remaining replicas after scale-in. If > 0, takes priority over engine_urls.",
    )
    engine_urls: List[str] = Field(
        default_factory=list,
        description="Engine URLs to remove (e.g., ['http://localhost:30000', 'http://localhost:30001'])",
    )
    force: bool = Field(default=False, description="Force removal without waiting for drain")
    timeout_secs: Optional[float] = Field(default=None, gt=0, description="Total timeout for the operation")
    dry_run: bool = Field(default=False, description="Preview which engines would be removed without removing them")


class ScaleInResponse(BaseModel):
    """Response model for scale-in operation."""

    request_id: str
    status: str
    message: str = "Scale-in request accepted"


class ScaleInStatusResponse(BaseModel):
    """Response model for scale-in status query."""

    request_id: str
    status: str
    model_name: str
    num_replicas: int
    engine_urls: List[str] = Field(default_factory=list, description="Engine URLs to remove")
    timeout_secs: float
    force: bool
    dry_run: bool
    created_at: float
    updated_at: float
    selected_engines: List[str]
    removed_engines: List[str]
    failed_engines: List[str]
    error_message: Optional[str]


class ListScaleInRequestsResponse(BaseModel):
    """Response model for list scale-in requests endpoint."""

    requests: List[ScaleInStatusResponse]
    total_count: int = Field(..., description="Total number of requests returned")
    model_name: Optional[str] = Field(None, description="Filter used: model_name")
    status_filter: Optional[str] = Field(None, description="Filter used: status")


# ===================== OpenAI Chat Completion API Models =====================


class ChatMessage(BaseModel):
    """A single message in the conversation."""

    role: str = Field(..., description="The role of the message author (system, user, assistant, tool)")
    content: Optional[Union[str, List[Dict[str, Any]]]] = Field(None, description="The content of the message")
    name: Optional[str] = Field(None, description="An optional name for the participant")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls generated by the model")
    tool_call_id: Optional[str] = Field(None, description="Tool call that this message is responding to")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(default="default", description="Model to use for completion")
    messages: List[ChatMessage] = Field(..., description="A list of messages comprising the conversation")
    temperature: Optional[float] = Field(default=None, ge=0, le=2, description="Sampling temperature")
    top_p: Optional[float] = Field(default=None, ge=0, le=1, description="Nucleus sampling parameter")
    top_k: Optional[int] = Field(default=None, description="Top-k sampling parameter")
    n: Optional[int] = Field(default=1, ge=1, description="Number of completions to generate")
    max_tokens: Optional[int] = Field(default=None, description="Maximum number of tokens to generate")
    max_completion_tokens: Optional[int] = Field(default=None, description="Maximum number of completion tokens")
    stream: Optional[bool] = Field(default=False, description="Whether to stream partial results")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequences")
    presence_penalty: Optional[float] = Field(default=None, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(default=None, description="Frequency penalty")
    logprobs: Optional[bool] = Field(default=None, description="Whether to return log probabilities")
    top_logprobs: Optional[int] = Field(default=None, description="Number of top log probabilities to return")
    user: Optional[str] = Field(default=None, description="A unique identifier representing the end-user")
    seed: Optional[int] = Field(default=None, description="Random seed for deterministic generation")
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="A list of tools the model may call")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="Controls tool usage")
    response_format: Optional[Dict[str, Any]] = Field(default=None, description="Response format specification")

    model_config = {"extra": "allow"}


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""

    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[UsageInfo] = None
    system_fingerprint: Optional[str] = None


class ChatCompletionStreamChoice(BaseModel):
    """A single streaming completion choice (delta)."""

    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming chat completion response chunk."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]
    system_fingerprint: Optional[str] = None
    usage: Optional[UsageInfo] = None


class ModelObject(BaseModel):
    """OpenAI-compatible model object."""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "relax"


class ModelListResponse(BaseModel):
    """OpenAI-compatible model list response."""

    object: str = "list"
    data: List[ModelObject]


# ===================== Helper Functions =====================


def satisfy_staleness(partition_list: Optional[List[str]], current_rollout_id: int, max_staleness: int) -> bool:
    """Check if the current rollout remains within the allowed staleness.

    Args:
        partition_list: list of partition ids as strings like 'train_123' or None.
        current_rollout_id: current rollout index.
        max_staleness: maximum allowed staleness.

    Returns:
        True if within staleness bounds (or partition_list is empty/None), False otherwise.
    """
    if not partition_list:  # for None or []
        return True

    def split_partition(partition: str) -> int:
        return int(partition.split("_")[-1])

    return current_rollout_id + 1 - min(list(map(split_partition, partition_list))) <= max_staleness


@serve.deployment
@serve.ingress(app)
class Rollout(Base):
    """The class to run rollout and convert rollout data to training data."""

    def __init__(
        self,
        healthy: Any,
        pg: Optional[Any],
        config: Namespace,
        data_source: Optional[Any] = None,
        runtime_env: Optional[dict] = None,  # pyright: ignore[reportMissingTypeArgument]
    ) -> None:
        super().__init__()
        self.config = config
        self.healthy = healthy

        tq.init(self.config.tq_config)
        self.data_system_client = tq.get_client()
        self.rollout_manager, self.num_rollout_per_epoch = create_rollout_manager(
            config, pg, data_source=data_source, runtime_env=runtime_env
        )
        self.step = 0
        self.data_source = data_source

        self._stop_event = asyncio.Event()
        self._run_task: Optional[asyncio.Task] = None  # pyright: ignore[reportMissingTypeArgument]
        # Set by can_do_update_weight_for_async when it finishes; end_update_weight waits on it.
        self._weight_update_ready = asyncio.Event()
        self._weight_update_ready.set()
        self.eval_handler = None
        self.status = "running"

        self._sglang_base_url: Optional[str] = None
        self._proxy_client: Optional[httpx.AsyncClient] = None

    def _should_eval(self, local_step):
        if self.config.eval_interval is None or self.config.eval_prompt_data is None:
            return False

        step = local_step + 1

        should_eval = (step % self.config.eval_interval == 0) or (
            self.num_rollout_per_epoch is not None and step % self.num_rollout_per_epoch == 0
        )
        self._logger.info(f"Checking whether to evaluate rollout {step}, should_eval: {should_eval}")
        return should_eval

    async def run(self) -> None:
        if self._run_task is not None and not self._run_task.done():
            await self._run_task
            return
        if self.config.rollout_global_dataset:
            try:
                await self.rollout_manager.load.remote(self.step - 1)
            except Exception as e:
                self._logger.exception(f"Failed to load global dataset: {e}")

        self._run_task = asyncio.ensure_future(self._async_run())
        await self._run_task

    def get_rollout_manager(self) -> Any:
        return self.rollout_manager

    async def _async_run(self) -> None:
        from relax.engine.sft.runtime import is_sft_mode

        # SFT-with-rollout: Rollout is a passive SGLang server that responds to
        # HTTP /predict and /evaluate driven by the Actor. No RL rollout loop.
        if is_sft_mode(self.config):
            return
        try:
            if self.config.eval_interval is not None and self.step == 0 and not self.config.skip_eval_before_train:
                await self.rollout_manager.eval.remote(rollout_id=0)
            while True:
                local_step = self.step

                if self.eval_handler is not None:
                    await self.eval_handler
                    self.eval_handler = None

                if self._stop_event.is_set():
                    self._logger.info("Rollout loop stopping by request")
                    break

                while self.status == "paused":
                    self._logger.info("Rollout loop paused, waiting to resume...")
                    await asyncio.sleep(1)

                if local_step >= self.config.num_rollout:
                    if (
                        self.config.max_staleness > 0
                        and local_step == self.config.num_rollout
                        and await self.data_source.get_buffer_length.remote() != 0
                        and not await self.data_system_client.async_check_production_status(
                            ["tokens"], f"train_{local_step - 1}"
                        )
                    ):
                        self._logger.warning(
                            "Production status check failed for final rollout step, continuing anyway"
                        )
                    else:
                        self._logger.info("All rollouts finished")
                        break

                self._logger.info(f"Start rollout {local_step}/{self.config.num_rollout}")
                try:
                    await self.rollout_manager.generate.remote(rollout_id=local_step)
                    if self.config.offload_rollout:
                        await self.rollout_manager.offload.remote()
                except Exception as e:
                    error_msg = f"Rollout generation failed at step {local_step}: {type(e).__name__}: {str(e)}"
                    self._logger.exception(error_msg)
                    self.healthy.report_error.remote("rollout", error_msg)
                    if not getattr(self.config, "use_health_check", False):
                        raise
                    break
                self._logger.info(f"Finish rollout {local_step}/{self.config.num_rollout}")

                try:
                    self.healthy.update_heartbeat.remote("rollout", local_step + 1)
                except Exception:
                    pass

                wait_count = 0
                while True:
                    if self._stop_event.is_set():
                        self._logger.info("Rollout loop stopping during staleness wait")
                        return
                    partition_list = await self.data_system_client.async_get_partition_list()
                    rollout_done = local_step + 1 >= self.config.num_rollout
                    should_continue = rollout_done or satisfy_staleness(
                        partition_list, local_step, self.config.max_staleness
                    )
                    if not should_continue:
                        if wait_count % 10 == 0:
                            self._logger.warning(
                                f"Rollout {local_step}: waiting for data system to catch up. "
                                f"Current partitions: {partition_list}, waited {wait_count}s"
                            )
                        wait_count += 1
                        await asyncio.sleep(1)
                        continue
                    else:
                        if wait_count > 0:
                            self._logger.info(f"Rollout {local_step}: data system caught up after {wait_count}s")
                        break

                await self._maybe_save_data(local_step)
                self.step += 1
        except Exception as e:
            error_msg = f"Rollout failed at step {self.step}: {type(e).__name__}: {str(e)}"
            self._logger.exception(error_msg)
            self.healthy.report_error.remote("rollout", error_msg)
            if not getattr(self.config, "use_health_check", False):
                raise

    async def _maybe_save_data(self, local_step) -> None:
        if self.config.save is None or self.config.save_interval is None:
            return

        is_save_step = (local_step + 1) % self.config.save_interval == 0
        is_final_step = (local_step + 1) == self.config.num_rollout
        if self.config.rotate_ckpt or is_save_step or is_final_step:
            self._logger.info(
                f"Saving rollout data for step {local_step} (rotate_ckpt={self.config.rotate_ckpt}, save_step={is_save_step}, final_step={is_final_step})"
            )
            await self.data_source.save.remote(rollout_id=local_step)  # pyright: ignore[reportOptionalMemberAccess]

    async def stop(self) -> None:
        self._stop_event.set()
        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass

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

    @app.get("/evaluate")
    async def evaluate(self, train_step: int):
        self._logger.info(f"Received request to evaluate train_step {train_step}")
        try:
            if self._should_eval(train_step):
                self._logger.info(f"Evaluating train_step {train_step}")
                self.eval_handler = self.rollout_manager.eval.remote(rollout_id=train_step)
            return {"status": "ok", "rollout_id": train_step}
        except Exception as e:
            error_msg = f"Evaluation failed for train_step {train_step}: {type(e).__name__}: {str(e)}"
            self._logger.exception(error_msg)
            self.healthy.report_error.remote("rollout", error_msg)
            return {"status": "error", "message": error_msg}

    @app.get("/predict")
    async def predict(self, train_step: int):
        """Periodic SFT predict pass — symmetric with /evaluate.

        Body lives in ``relax.engine.sft.predict.runner.handle_predict``;
        this method is the thin HTTP-decoration shell.
        """
        from relax.engine.sft.predict.runner import handle_predict

        return await handle_predict(self, train_step)

    @app.get("/can_do_update_weight_for_async")
    async def can_do_update_weight_for_async(self):
        self._logger.debug("Handling can_do_update_weight_for_async request")
        step = self.step
        can_update = await self._async_check_production_for_update_weight(step)
        if can_update:
            self._weight_update_ready.clear()
            self.status = "paused"
            await self.rollout_manager.health_monitoring_pause.remote()
            await self.rollout_manager.set_weight_updating.remote(True)
            self._weight_update_ready.set()
            return 1
        return 0

    async def _async_check_production_for_update_weight(self, step: int) -> bool:
        return await self.data_system_client.async_check_production_status(
            ["tokens"], f"train_{step - 1}"
        ) or await self.data_system_client.async_check_production_status(["tokens"], f"train_{step}")

    @app.get("/is_eval_done")
    async def is_eval_done(self):
        """Check whether the previous evaluation has finished (success or
        failure)."""
        if self.eval_handler is None:
            return {"done": True}
        try:
            ready, _ = ray.wait([self.eval_handler], timeout=0)
        except Exception:
            # ObjectRef is invalid or already collected — treat as done
            return {"done": True}
        return {"done": len(ready) > 0}

    @app.get("/end_update_weight")
    async def end_update_weight(self):
        self._logger.info("Ending update weight, resuming rollout")
        await self._weight_update_ready.wait()

        self.status = "running"
        await self.rollout_manager.set_weight_updating.remote(False)

    @app.get("/recover_rollout_engines")
    async def recover_rollout_engines(self):
        self._logger.info("Recovering rollout engines")
        await self.rollout_manager.recover_rollout_engines.remote()
        return {"status": "ok"}

    @app.post("/scale_out", response_model=ScaleOutResponse)
    async def scale_out(self, request: ScaleOutAPIRequest):
        # Scale-out is incompatible with SlimeRouter (which uses fixed engine pool)
        if getattr(self.config, "use_slime_router", False):
            raise HTTPException(
                status_code=400,
                detail="Scale-out is not available when --use-slime-router is enabled. "
                "SlimeRouter uses a fixed engine pool that does not support dynamic scaling.",
            )
        # Idempotency is handled inside create_scale_out_request:
        #   - ray_native: computes effective delta from (num_replicas - current)
        #   - external: filters out addresses already active or in-flight
        # Auto-detect mode: if num_replicas > 0, use ray_native; otherwise use external
        result = await self.rollout_manager.create_scale_out_request.remote(
            model_name=request.model_name,
            num_replicas=request.num_replicas,
            engine_urls=request.engine_urls,
            timeout_secs=request.timeout_secs,
        )
        # Mutual exclusion: reject if another scale operation is in progress
        if result["status"] == "CONFLICT":
            raise HTTPException(status_code=409, detail=result["message"])
        # Step 2: If idempotency check resulted in a no-op, return immediately
        if result["status"] == "NOOP":
            return ScaleOutResponse(
                request_id=result["request_id"],
                status=result["status"],
                message=result.get("message", "Already at or above target engine count; no scale-out needed"),
            )
        # Step 3: Fire-and-forget execution (non-blocking)
        self.rollout_manager.execute_scale_out.remote(result["request_id"])
        return ScaleOutResponse(
            request_id=result["request_id"],
            status=result["status"],
            message="Scale-out request accepted",
        )

    @app.get("/scale_out", response_model=ListScaleOutRequestsResponse)
    async def list_scale_out_requests(self, model_name: Optional[str] = None, status: Optional[str] = None):
        """List all scale-out requests with optional filtering.

        Examples:
            # List all requests
            curl -X GET http://localhost:8000/rollout/scale_out

            # List PENDING requests
            curl -X GET "http://localhost:8000/rollout/scale_out?status=PENDING"

            # List 'actor' model's ACTIVE requests
            curl -X GET "http://localhost:8000/rollout/scale_out?model_name=actor&status=ACTIVE"
        """
        self._logger.info(f"Listing scale-out requests: model_name={model_name}, status={status}")

        try:
            requests = await self.rollout_manager.list_all_scale_out_requests.remote(
                model_name=model_name, status_filter=status
            )

            return ListScaleOutRequestsResponse(
                requests=[ScaleOutStatusResponse(**r) for r in requests],
                total_count=len(requests),
                model_name=model_name,
                status_filter=status,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self._logger.error(f"Error listing scale-out requests: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/scale_out_cancel", response_model=CancelAllScaleOutRequestsResponse)
    async def cancel_all_scale_out_requests(self, request: CancelAllScaleOutRequestsRequest):
        """Cancel all scale-out requests matching criteria.

        Status Codes:
            200: Success (some/all requests cancelled or previewed)
            400: Invalid filter values (e.g., bad status name)
            500: Internal server error

        Examples:
            # 1. Preview what would be cancelled (dry-run)
            curl -X POST http://localhost:8000/rollout/scale_out_cancel \\
              -H "Content-Type: application/json" \\
              -d '{"dry_run": true}'

            # 2. Cancel all PENDING requests
            curl -X POST http://localhost:8000/rollout/scale_out_cancel \\
              -H "Content-Type: application/json" \\
              -d '{"status_filter": "PENDING"}'

            # 3. Cancel all PENDING requests for 'actor' model
            curl -X POST http://localhost:8000/rollout/scale_out_cancel \\
              -H "Content-Type: application/json" \\
              -d '{"model_name": "actor", "status_filter": "PENDING"}'

            # 4. Cancel all requests for 'actor' model
            curl -X POST http://localhost:8000/rollout/scale_out_cancel \\
              -H "Content-Type: application/json" \\
              -d '{"model_name": "actor"}'
        """
        self._logger.info(
            f"Cancelling scale-out requests: "
            f"model_name={request.model_name}, "
            f"status_filter={request.status_filter}, "
            f"dry_run={request.dry_run}"
        )

        try:
            result = await self.rollout_manager.cancel_all_scale_out_requests.remote(
                model_name=request.model_name, status_filter=request.status_filter, dry_run=request.dry_run
            )

            return CancelAllScaleOutRequestsResponse(**result)
        except ValueError as e:
            self._logger.warning(f"Invalid request: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self._logger.error(f"Error cancelling scale-out requests: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

    # Parameterised routes AFTER fixed-path routes to avoid route conflicts.

    @app.get("/scale_out/{request_id}", response_model=ScaleOutStatusResponse)
    async def get_scale_out_status(self, request_id: str):
        result = await self.rollout_manager.get_scale_out_status.remote(request_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Scale-out request {request_id} not found")
        return ScaleOutStatusResponse(**result)

    @app.post("/scale_out/{request_id}/cancel", response_model=CancelResponse)
    async def cancel_scale_out(self, request_id: str):
        result = await self.rollout_manager.cancel_scale_out.remote(request_id)
        if result is None:
            raise HTTPException(
                status_code=404, detail=f"Scale-out request {request_id} not found or cannot be cancelled"
            )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return CancelResponse(
            request_id=result["request_id"],
            status=result["status"],
            message="Scale-out request cancelled",
        )

    @app.get("/engines")
    async def get_engines(self, model_name: Optional[str] = None):
        result = await self.rollout_manager.get_engines_info.remote(model_name)
        return result

    @app.post("/scale_in", response_model=ScaleInResponse)
    async def scale_in(self, request: ScaleInAPIRequest):
        # Scale-in is incompatible with SlimeRouter (which uses fixed engine pool)
        if getattr(self.config, "use_slime_router", False):
            raise HTTPException(
                status_code=400,
                detail="Scale-in is not available when --use-slime-router is enabled. "
                "SlimeRouter uses a fixed engine pool that does not support dynamic scaling.",
            )
        result = await self.rollout_manager.create_scale_in_request.remote(
            model_name=request.model_name,
            num_replicas=request.num_replicas,
            engine_urls=request.engine_urls,
            timeout_secs=request.timeout_secs,
            force=request.force,
            dry_run=request.dry_run,
        )
        if result["status"] == "CONFLICT":
            raise HTTPException(status_code=409, detail=result["message"])
        if result["status"] == "REJECTED":
            raise HTTPException(status_code=400, detail=result["message"])
        if result["status"] == "NOOP":
            return ScaleInResponse(
                request_id=result["request_id"],
                status=result["status"],
                message=result.get("message", "No engines available for scale-in"),
            )
        # Fire-and-forget: execute_scale_in runs asynchronously
        self.rollout_manager.execute_scale_in.remote(result["request_id"])
        return ScaleInResponse(
            request_id=result["request_id"],
            status=result["status"],
            message="Scale-in request accepted",
        )

    @app.get("/scale_in", response_model=ListScaleInRequestsResponse)
    async def list_scale_in_requests(self, model_name: Optional[str] = None, status: Optional[str] = None):
        """List all scale-in requests with optional filtering."""
        self._logger.info(f"Listing scale-in requests: model_name={model_name}, status={status}")

        try:
            requests = await self.rollout_manager.list_all_scale_in_requests.remote(
                model_name=model_name, status_filter=status
            )

            return ListScaleInRequestsResponse(
                requests=[ScaleInStatusResponse(**r) for r in requests],
                total_count=len(requests),
                model_name=model_name,
                status_filter=status,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self._logger.error(f"Error listing scale-in requests: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.get("/scale_in/{request_id}", response_model=ScaleInStatusResponse)
    async def get_scale_in_status(self, request_id: str):
        result = await self.rollout_manager.get_scale_in_status.remote(request_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Scale-in request {request_id} not found")
        return ScaleInStatusResponse(**result)

    # --- OpenAI-compatible Chat Completion API (proxied to SGLang router) ---

    def _get_proxy_client(self) -> httpx.AsyncClient:
        if self._proxy_client is None:
            self._proxy_client = httpx.AsyncClient(
                timeout=httpx.Timeout(None),
                limits=httpx.Limits(max_connections=256),
            )
        return self._proxy_client

    async def _ensure_sglang_base_url(self) -> str:
        if self._sglang_base_url is not None:
            return self._sglang_base_url
        addr = await self.rollout_manager.get_router_address.remote()
        router_ip, router_port = addr.get("router_ip"), addr.get("router_port")
        if not router_ip or not router_port:
            raise HTTPException(
                status_code=503,
                detail="SGLang router is not available. No router_ip/router_port found on RolloutManager.",
            )
        self._sglang_base_url = f"http://{_wrap_ipv6(router_ip)}:{router_port}"
        self._logger.info(f"Resolved SGLang router URL: {self._sglang_base_url}")
        return self._sglang_base_url

    async def _get_sglang_url(self, path: str) -> str:
        base = await self._ensure_sglang_base_url()
        return f"{base}{path}"

    @app.post("/v1/chat/completions")
    async def chat_completions(self, request: Request):
        body = await request.body()
        try:
            payload = ChatCompletionRequest.model_validate_json(body)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

        sglang_url = await self._get_sglang_url("/v1/chat/completions")
        client = self._get_proxy_client()

        if payload.stream:
            return await self._stream_chat_completions(client, sglang_url, body, dict(request.headers))
        return await self._non_stream_chat_completions(client, sglang_url, body, dict(request.headers))

    async def _non_stream_chat_completions(
        self,
        client: httpx.AsyncClient,
        url: str,
        body: bytes,
        headers: Dict[str, str],
    ) -> ChatCompletionResponse:
        forward_headers = self._build_forward_headers(headers)
        try:
            response = await client.post(url, content=body, headers=forward_headers)
            response.raise_for_status()
            data = response.json()
            return ChatCompletionResponse(**data)
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            self._logger.error(f"Failed to proxy chat completion to SGLang: {e}")
            raise HTTPException(status_code=502, detail=f"Failed to connect to SGLang router: {e}")

    async def _stream_chat_completions(
        self,
        client: httpx.AsyncClient,
        url: str,
        body: bytes,
        headers: Dict[str, str],
    ) -> StreamingResponse:
        forward_headers = self._build_forward_headers(headers)

        async def _event_generator():
            response = None
            try:
                req = client.build_request("POST", url, content=body, headers=forward_headers)
                response = await client.send(req, stream=True)
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        yield f"{line}\n\n"
            except httpx.HTTPStatusError as e:
                error_body = await e.response.aread()
                error_chunk = _make_error_chunk(e.response.status_code, error_body.decode(errors="replace"))
                yield f"data: {error_chunk}\n\n"
                yield "data: [DONE]\n\n"
            except httpx.RequestError as e:
                self._logger.error(f"Streaming connection to SGLang failed: {e}")
                error_chunk = _make_error_chunk(502, f"Failed to connect to SGLang router: {e}")
                yield f"data: {error_chunk}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if response is not None:
                    await response.aclose()

        return StreamingResponse(_event_generator(), media_type="text/event-stream")

    @staticmethod
    def _build_forward_headers(original_headers: Dict[str, str]) -> Dict[str, str]:
        hop_by_hop = {"host", "transfer-encoding", "connection", "keep-alive", "upgrade"}
        return {k: v for k, v in original_headers.items() if k.lower() not in hop_by_hop}

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models(self):
        sglang_url = await self._get_sglang_url("/v1/models")
        client = self._get_proxy_client()
        try:
            response = await client.get(sglang_url)
            response.raise_for_status()
            return ModelListResponse(**response.json())
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            self._logger.error(f"Failed to proxy model list to SGLang: {e}")
            raise HTTPException(status_code=502, detail=f"Failed to connect to SGLang router: {e}")


def _make_error_chunk(status_code: int, message: str) -> str:
    error_response = {
        "id": f"chatcmpl-error-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "error",
        "choices": [],
        "error": {"code": status_code, "message": message},
    }
    return json.dumps(error_response)
