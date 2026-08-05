# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Request-scoped model callback bridge from NeMo Gym to Relax."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import httpx

from .registry import GatewayRegistry
from .verbose_logging import log_verbose_payload


class CallbackRequestError(ValueError):
    pass


class CallbackUpstreamError(RuntimeError):
    pass


@dataclass(frozen=True)
class CallbackResponse:
    status_code: int
    payload: dict[str, Any]


class CallbackProvider:
    def __init__(
        self,
        registry: GatewayRegistry,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        proxy: str | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        self._registry = registry
        self._timeout_s = timeout_s
        self._client = httpx.AsyncClient(transport=transport, proxy=proxy, trust_env=False)

    async def close(self) -> None:
        await self._client.aclose()

    async def chat_completions(self, rollout_id: str, payload: Any) -> CallbackResponse:
        if not isinstance(payload, dict):
            raise CallbackRequestError("Callback body must be a JSON object")
        if payload.get("stream"):
            raise CallbackRequestError("Streaming callbacks are not supported by the Phase-2 provider")
        async with self._registry.callback_target(rollout_id) as target:
            callback_payload = copy.deepcopy(payload)
            sampling_params = target.generation.get("sampling_params", {})
            if isinstance(sampling_params, dict):
                callback_payload.update(copy.deepcopy(sampling_params))
            callback_payload["model"] = target.model
            return await self._post_chat(target, callback_payload)

    async def responses(self, rollout_id: str, payload: Any) -> CallbackResponse:
        if not isinstance(payload, dict):
            raise CallbackRequestError("Callback body must be a JSON object")
        if payload.get("stream"):
            raise CallbackRequestError("Streaming callbacks are not supported by the Phase-2 provider")

        try:
            from nemo_gym.openai_utils import (
                NeMoGymChatCompletion,
                NeMoGymResponseCreateParamsNonStreaming,
            )
            from nemo_gym.responses_converter import VLLMConverter
        except ImportError as exc:
            raise CallbackRequestError("The Responses callback route must run in the pinned NeMo Gym image") from exc

        async with self._registry.callback_target(rollout_id) as target:
            try:
                responses_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(payload)
                responses_params.model = target.model
                converter = VLLMConverter(
                    return_token_id_information=False,
                    uses_reasoning_parser=False,
                )
                chat_params = converter.responses_to_chat_completion_create_params(responses_params)
                chat_payload = chat_params.model_dump(exclude_unset=True, mode="json")
            except Exception as exc:
                raise CallbackRequestError("Invalid NeMo Gym Responses request") from exc

            sampling_params = target.generation.get("sampling_params", {})
            if isinstance(sampling_params, dict):
                chat_payload.update(copy.deepcopy(sampling_params))
            chat_payload["model"] = target.model
            upstream = await self._post_chat(target, chat_payload)
            if upstream.status_code >= 400:
                return upstream

            try:
                chat_completion = NeMoGymChatCompletion.model_validate(upstream.payload)
                response = converter.chat_completion_to_response(
                    responses_create_params=responses_params,
                    chat_completion=chat_completion,
                )
            except Exception as exc:
                raise CallbackUpstreamError(
                    "Upstream model returned an incompatible Chat Completions response"
                ) from exc
            return CallbackResponse(status_code=upstream.status_code, payload=response.model_dump(mode="json"))

    async def _post_chat(self, target: Any, payload: dict[str, Any]) -> CallbackResponse:
        url = _chat_completions_url(target.base_url)
        headers = copy.deepcopy(target.headers)
        headers[target.api_key_header] = f"{target.api_key_prefix}{target.api_key}"
        log_verbose_payload("request", payload, route="upstream/v1/chat/completions")
        timeout_s = min(target.remaining_s, self._timeout_s)
        timeout = httpx.Timeout(
            timeout_s,
            connect=min(timeout_s, 30.0),
            pool=min(timeout_s, 30.0),
            write=min(timeout_s, 30.0),
        )
        try:
            response = await self._client.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except httpx.RequestError as exc:
            raise CallbackUpstreamError("Upstream model callback transport failed") from exc
        try:
            response_payload = response.json()
        except ValueError as exc:
            raise CallbackUpstreamError("Upstream model callback returned a non-JSON response") from exc
        if not isinstance(response_payload, dict):
            raise CallbackUpstreamError("Upstream model callback response must be a JSON object")
        log_verbose_payload(
            "response",
            response_payload,
            route="upstream/v1/chat/completions",
            status=response.status_code,
        )
        return CallbackResponse(status_code=response.status_code, payload=response_payload)


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    suffix = "/chat/completions" if normalized.endswith("/v1") else "/v1/chat/completions"
    return f"{normalized}{suffix}"
