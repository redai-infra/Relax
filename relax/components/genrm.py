# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GenRM Service Implementation.

This module provides a Ray Serve deployment for Generative Reward Model
(genRM), which evaluates responses using LLM-based preference prediction.
"""

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


app = FastAPI()


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
        self._logger.info("GenRM service initialized successfully")
        # Shared HTTP client for engine calls (avoids per-request connection overhead)
        self._http_client = httpx.AsyncClient(timeout=1800)

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
        # Lazily fetch engine host/port information and build cycle iterator
        if self._engine_hosts_ports is None:
            self._engine_hosts_ports = ray.get(self.genrm_manager.get_engine_hosts_ports.remote())
            self._engine_cycle = cycle(range(len(self._engine_hosts_ports)))

        if not self._engine_hosts_ports:
            raise RuntimeError("No genRM engines available")

        # Thread-safe round-robin via itertools.cycle (next() is atomic in CPython)
        idx = next(self._engine_cycle)
        host, port = self._engine_hosts_ports[idx]

        url = f"http://{host}:{port}/generate"
        # ensure plain list — some tokenizers return BatchEncoding which is not JSON-serializable
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

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

        resp = await self._http_client.post(url, json=payload)
        resp.raise_for_status()
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
