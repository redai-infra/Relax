# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GenRM Service Implementation.

This module provides a Ray Serve deployment for Generative Reward Model
(genRM), which evaluates responses using LLM-based preference prediction.
"""

import asyncio
import time
from argparse import Namespace
from itertools import cycle
from typing import Any, List, Optional, Union

import httpx
import ray
from fastapi import FastAPI
from pydantic import BaseModel
from ray import serve
from ray.serve.schema import LoggingConfig

from relax.components.base import Base
from relax.distributed.ray.placement_group import create_genrm_manager
from relax.utils.data.processing_utils import load_tokenizer
from relax.utils.env import Envs


app = FastAPI()

# Max concurrent in-flight requests per GenRM Serve replica. Ray Serve's default
# of 5 throttles judge dispatch and leaves the SGLang engines idle. The replica
# is a pure-async CPU proxy (tokenize + forward), so a high cap lets one replica
# saturate the engines. Override via env for tuning.
GENRM_SERVE_MAX_ONGOING_REQUESTS = Envs.GENRM_SERVE_MAX_ONGOING_REQUESTS

# NOTE: GENRM_SERVE_MAX_ONGOING_REQUESTS above must stay module-level — it feeds
# the @serve.deployment decorator, which runs at import. The retry count has no
# such constraint, so it is read at the call site to stay lazy.


class Message(BaseModel):
    """Single chat message."""

    role: str
    content: str


class GenerateRequest(BaseModel):
    """Request model for genRM generation (OpenAI chat format).

    Accepts a list of messages in OpenAI format with optional sampling params.
    """

    messages: Union[List[Message], List[dict]]
    sampling_params: Optional[dict] = None


class GenerateResponse(BaseModel):
    """Response model for genRM generation.

    Returns the raw model response text.
    """

    response: str


@serve.deployment(
    max_ongoing_requests=GENRM_SERVE_MAX_ONGOING_REQUESTS,
    logging_config=LoggingConfig(
        log_level="WARNING",
        enable_access_log=False,  # 关闭 HTTP 访问日志
    ),
)
@serve.ingress(app)
class GenRM(Base):
    """GenRM Service for generative reward model evaluation.

    This service uses SGLang engines to perform preference evaluation by
    comparing model responses against ground truth or standards.
    """

    def __init__(
        self,
        healthy: Any,
        pg: Optional[Any],
        num_gpus: int,
        config: Namespace,
        role: str,
        runtime_env: Optional[dict] = None,
    ) -> None:
        """Initialize GenRM service.

        Args:
            healthy: Remote health manager actor handle.
            pg: Placement group for resource allocation.
            num_gpus: Number of GPUs allocated (used by Service framework).
            config: Runtime configuration namespace.
            role: Role name (should be "genrm").
            runtime_env: Optional Ray runtime environment dict.
        """
        super().__init__()
        self.config = config
        self.healthy = healthy
        self.role = role

        # Initialize GenRM Manager
        self.genrm_manager = create_genrm_manager(config, pg, runtime_env=runtime_env)

        # Store engine addresses for HTTP-based generation
        self._engine_hosts_ports = None
        self._engine_index = 0
        self._engine_cache_refreshed_at = 0.0
        self._logger.info("GenRM service initialized successfully")
        # Shared HTTP client for engine calls (avoids per-request connection overhead).
        # Raise pool limits well above httpx's default 100 so one replica can fan out
        # many concurrent engine requests; keepalive_expiry >> the default 5s so
        # idle-then-reused connections aren't reaped mid-burst (avoids ReadError/500).
        self._http_client = httpx.AsyncClient(
            timeout=1800,
            limits=httpx.Limits(max_connections=2048, max_keepalive_connections=2048, keepalive_expiry=600),
        )

        # Load tokenizer for prompt encoding
        self.tokenizer = load_tokenizer(config.genrm_model_path, trust_remote_code=True)

    def run(self):
        """GenRM is a passive HTTP service, no background loop needed.

        Unlike Actor or Rollout, GenRM only responds to incoming requests and
        does not actively produce work. Return None so the Controller training
        loop does not block on it.
        """
        return None

    @app.post("/generate")
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate response for given chat messages.

        Takes OpenAI-style messages as input, sends to SGLang engine,
        and returns the raw model response. The caller is responsible
        for formatting the prompt and parsing the response.

        Args:
            request: GenerateRequest containing messages list and optional sampling_params

        Returns:
            GenerateResponse containing raw model response text
        """
        try:
            # Call SGLang engine via GenRMManager
            output = await self._call_engine(request.messages, request.sampling_params)

            # Return raw response text
            response = output.get("text", "").strip()
            return GenerateResponse(response=response)

        except Exception as e:
            self._logger.error(f"GenRM generation failed: {e}")
            raise

    def _invalidate_engine_cache(self) -> None:
        """Force the next _pick_engine to re-read the live engine list.

        GenRMManager rebuilds a dead engine on a *fresh port*, so a cache held
        from before the rebuild points at nothing.

        Throttled: a dead engine fails many in-flight requests at once, and the
        refresh is a blocking ray.get on this replica's event loop. Without the
        cooldown a burst of N failures costs N serialized round-trips.
        """
        now = time.monotonic()
        if now - self._engine_cache_refreshed_at < Envs.GENRM_ENGINE_CACHE_REFRESH_COOLDOWN_S:
            return
        self._engine_hosts_ports = None

    def _needs_engine_refresh(self) -> bool:
        """Whether _pick_engine must re-read the engine list before serving.

        An empty list is never a valid cache. The manager reports ``[]`` for the
        whole window between offload() retiring every dead engine and the next
        onload() rebuilding them, so caching it would strand this replica on
        "No genRM engines available" forever — long after recovery finished,
        because nothing would ever re-read the list. The retry loop in
        _call_engine cannot rescue it either: _pick_engine raises before any
        HTTP request is made, so _invalidate_engine_cache is never reached.

        Re-reading is throttled by the same cooldown as an invalidation: while
        the list is empty *every* in-flight request lands here, and the read is
        a blocking ray.get on this replica's event loop.
        """
        if self._engine_hosts_ports is None:
            return True
        if self._engine_hosts_ports:
            return False
        return time.monotonic() - self._engine_cache_refreshed_at >= Envs.GENRM_ENGINE_CACHE_REFRESH_COOLDOWN_S

    def _pick_engine(self) -> tuple[int, str, int]:
        """Round-robin one live engine, refreshing the list if it was
        dropped."""
        if self._needs_engine_refresh():
            hosts_ports = ray.get(self.genrm_manager.get_engine_hosts_ports.remote())
            self._engine_cache_refreshed_at = time.monotonic()
            # Swap the list and its cycle together: the manager compacts the
            # list over dead engines, so a cycle built for the old length would
            # hand back an out-of-range index.
            self._engine_cycle = cycle(range(len(hosts_ports)))
            self._engine_hosts_ports = hosts_ports

        hosts_ports = self._engine_hosts_ports
        if not hosts_ports:
            raise RuntimeError("No genRM engines available")

        # Thread-safe round-robin via itertools.cycle (next() is atomic in CPython).
        # Re-read the local alias, not self._*, so a concurrent invalidation
        # can't make the index and the list disagree.
        idx = next(self._engine_cycle) % len(hosts_ports)
        host, port = hosts_ports[idx]
        return idx, host, port

    async def _call_engine(self, messages: list, sampling_params: Optional[dict] = None) -> dict:
        """Call an SGLang engine for text generation.

        Uses the engine addresses obtained from GenRMManager to send HTTP
        requests to the underlying SGLang server.

        Args:
            messages: List of chat messages in OpenAI format.
            sampling_params: Optional per-request sampling params that override defaults.

        Returns:
            Dict containing at least {"text": str} from the SGLang server.
        """
        idx, host, port = self._pick_engine()
        # ensure plain list — some tokenizers return BatchEncoding which is not JSON-serializable
        # Tokenization (chat-template render + encode) is synchronous CPU work; run it in a
        # worker thread so it does not block this replica's event loop. Fast (Rust) tokenizers
        # release the GIL during encode, so concurrent requests tokenize in parallel instead of
        # serializing — without this a single replica throttles dispatch and starves the engines.
        # Forward chat_template_kwargs from --genrm-sampling-config through to
        # the jinja template — e.g. `{"enable_thinking": false}` for Qwen3+ to
        # suppress the default <think> block. Keys unused by the template are
        # silently dropped by transformers, so this is safe across model families.
        chat_template_kwargs = self.config.genrm_sampling_config.get("chat_template_kwargs", {}) or {}
        input_ids = await asyncio.to_thread(
            self.tokenizer.apply_chat_template,
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )

        if not isinstance(input_ids, list):
            input_ids = (
                input_ids["input_ids"]
                if hasattr(input_ids, "__getitem__") and "input_ids" in input_ids
                else list(input_ids)
            )

        # Merge per-request sampling params with default config
        default_sampling = {
            "temperature": self.config.genrm_sampling_config.get("temperature", 0.2),
            "top_p": self.config.genrm_sampling_config.get("top_p", 1.0),
            "top_k": self.config.genrm_sampling_config.get("top_k", -1),
            "max_new_tokens": self.config.genrm_sampling_config.get("max_response_len", 1024),
        }
        # Override defaults with per-request params
        if sampling_params:
            default_sampling.update(sampling_params)

        payload = {
            "input_ids": input_ids,
            "sampling_params": default_sampling,
        }

        # Retry transient resets (transport-level or 5xx) with short backoff so
        # bursty colocate contention doesn't surface as a 500; 4xx is a client bug
        # (terminal) and a cancellation (caller timeout) is never retried.
        # Serve→engine retry attempts for transient resets (ReadError / 5xx under
        # bursty colocate contention), absorbing them before they surface as a 500
        # to the client. Set 1 to disable.
        retry_attempts = Envs.GENRM_ENGINE_RETRY_ATTEMPTS
        for _attempt in range(1, retry_attempts + 1):
            try:
                resp = await self._http_client.post(f"http://{host}:{port}/generate", json=payload)
                resp.raise_for_status()
                break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                status = int(getattr(getattr(e, "response", None), "status_code", 0) or 0)
                if (status == 0 or status >= 500) and _attempt < retry_attempts:
                    # A transport error means this engine may be gone and its
                    # replacement will come back on a different port, so drop the
                    # cache and re-pick — retrying the same dead host is useless.
                    if status == 0:
                        self._invalidate_engine_cache()
                    idx, host, port = self._pick_engine()
                    await asyncio.sleep(0.3 * _attempt)
                    continue
                raise
        return resp.json()

    @app.get("/health")
    async def health(self) -> dict:
        """Health check endpoint."""
        try:
            # Check if genRM engines are healthy
            is_healthy = ray.get(self.genrm_manager.health_check.remote())
            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "service": "genrm",
            }
        except Exception as e:
            self._logger.error(f"GenRM health check failed: {e}")
            return {
                "status": "unhealthy",
                "service": "genrm",
                "error": str(e),
            }

    @app.get("/metrics")
    async def metrics(self) -> dict:
        """Metrics endpoint."""
        return {
            "service": "genrm",
            "model_path": self.config.genrm_model_path,
            "num_gpus": self.config.genrm_num_gpus,
            "num_engines": self.config.genrm_num_gpus // self.config.genrm_num_gpus_per_engine,
        }

    def get_genrm_manager(self) -> Any:
        """Get the underlying GenRM manager."""
        return self.genrm_manager

    def onload(self) -> None:
        """Load genRM model weights to GPU."""
        self._logger.info("GenRM onload requested")
        ray.get(self.genrm_manager.onload.remote())

    def offload(self) -> None:
        """Offload genRM model weights from GPU."""
        self._logger.info("GenRM offload requested")
        ray.get(self.genrm_manager.offload.remote())


# ── Compatibility wrapper for old imports ─────────────────────────────────
GENRM_ROLE = "genrm"


def register_genrm(config, algo: dict) -> list[str]:
    """Compatibility wrapper; optional-role wiring lives in ``relax.core``."""
    from relax.core.optional_roles import register_genrm as _register_genrm

    return _register_genrm(config, algo)
