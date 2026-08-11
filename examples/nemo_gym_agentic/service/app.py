# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""FastAPI application for the reference shared NeMo Gym Gateway."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from ..app.protocol import PROTOCOL_VERSION, ProtocolValidationError
from .callback_provider import CallbackProvider, CallbackRequestError, CallbackUpstreamError
from .config import GatewayConfigError, GatewaySettings
from .registry import (
    AdmissionRejected,
    CallbackUnavailable,
    GatewayRegistry,
    TrialConflict,
    TrialNotFound,
)
from .run_adapter import HttpNemoGymRunAdapter, RunAdapter
from .verbose_logging import log_verbose_payload


def create_app(
    *,
    settings: GatewaySettings,
    adapter: RunAdapter | None = None,
    callback_transport: Any = None,
) -> FastAPI:
    run_adapter = adapter or HttpNemoGymRunAdapter(
        settings.environments.values(),
        artifact_root=settings.artifact_root,
    )
    registry = GatewayRegistry(settings=settings, adapter=run_adapter)
    callback_provider = CallbackProvider(
        registry,
        transport=callback_transport,
        proxy=settings.callback_proxy,
        timeout_s=settings.callback_timeout_s,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await registry.start()
        try:
            yield
        finally:
            await registry.close()
            await callback_provider.close()

    app = FastAPI(title="Relax NeMo Gym Gateway", version=PROTOCOL_VERSION, lifespan=lifespan)
    app.state.registry = registry

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return _service_metadata(registry, ready=None)

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        ready = await registry.ready()
        return JSONResponse(
            status_code=200 if ready else 503,
            content=_service_metadata(registry, ready=ready),
        )

    @app.post("/v1/trials")
    async def create_trial(request: Request) -> JSONResponse:
        payload = await _json_body(request)
        log_verbose_payload("request", payload, route="/v1/trials")
        try:
            result = await registry.create(payload)
        except ProtocolValidationError as exc:
            log_verbose_payload("response", {"detail": str(exc)}, route="/v1/trials", status=422)
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except GatewayConfigError as exc:
            log_verbose_payload("response", {"detail": str(exc)}, route="/v1/trials", status=400)
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except TrialConflict as exc:
            log_verbose_payload("response", {"detail": str(exc)}, route="/v1/trials", status=409)
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except AdmissionRejected as exc:
            log_verbose_payload("response", {"detail": str(exc)}, route="/v1/trials", status=429)
            raise HTTPException(status_code=429, detail=str(exc)) from None
        log_verbose_payload("response", result, route="/v1/trials", status=202)
        return JSONResponse(status_code=202, content=result)

    @app.get("/v1/trials/{request_id}")
    async def get_trial(request_id: str) -> dict[str, Any]:
        try:
            return await registry.get(request_id)
        except TrialNotFound:
            raise HTTPException(status_code=404, detail="trial not found") from None

    @app.post("/v1/trials/{request_id}/renew", status_code=204)
    async def renew_trial(request_id: str) -> Response:
        try:
            await registry.renew(request_id)
        except TrialNotFound:
            raise HTTPException(status_code=404, detail="trial not found") from None
        return Response(status_code=204)

    @app.post("/v1/trials/{request_id}/abort")
    async def abort_trial(request_id: str) -> JSONResponse:
        log_verbose_payload("request", {}, route="/v1/trials/{request_id}/abort", request_id=request_id)
        try:
            result = await registry.abort(request_id)
        except TrialNotFound:
            log_verbose_payload(
                "response",
                {"detail": "trial not found"},
                route="/v1/trials/{request_id}/abort",
                request_id=request_id,
                status=404,
            )
            raise HTTPException(status_code=404, detail="trial not found") from None
        log_verbose_payload(
            "response",
            result,
            route="/v1/trials/{request_id}/abort",
            request_id=request_id,
            status=202,
        )
        return JSONResponse(status_code=202, content=result)

    @app.post("/ng-rollout/{rollout_id}/v1/chat/completions")
    async def callback_chat_completions(rollout_id: str, request: Request) -> JSONResponse:
        payload = await _json_body(request)
        log_verbose_payload(
            "request",
            payload,
            route="/ng-rollout/{rollout_id}/v1/chat/completions",
            rollout_id=rollout_id,
        )
        try:
            result = await callback_provider.chat_completions(rollout_id, payload)
        except CallbackUnavailable:
            raise HTTPException(status_code=410, detail="callback capability is unavailable") from None
        except CallbackRequestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except CallbackUpstreamError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from None
        log_verbose_payload(
            "response",
            result.payload,
            route="/ng-rollout/{rollout_id}/v1/chat/completions",
            rollout_id=rollout_id,
            status=result.status_code,
        )
        return JSONResponse(status_code=result.status_code, content=result.payload)

    @app.post("/ng-rollout/{rollout_id}/v1/responses")
    async def callback_responses(rollout_id: str, request: Request) -> JSONResponse:
        payload = await _json_body(request)
        log_verbose_payload(
            "request",
            payload,
            route="/ng-rollout/{rollout_id}/v1/responses",
            rollout_id=rollout_id,
        )
        try:
            result = await callback_provider.responses(rollout_id, payload)
        except CallbackUnavailable:
            raise HTTPException(status_code=410, detail="callback capability is unavailable") from None
        except CallbackRequestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except CallbackUpstreamError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from None
        log_verbose_payload(
            "response",
            result.payload,
            route="/ng-rollout/{rollout_id}/v1/responses",
            rollout_id=rollout_id,
            status=result.status_code,
        )
        return JSONResponse(status_code=result.status_code, content=result.payload)

    return app


def create_app_from_env() -> FastAPI:
    return create_app(settings=GatewaySettings.from_env())


async def _json_body(request: Request) -> Any:
    try:
        return await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="request body must be valid JSON") from None


def _service_metadata(registry: GatewayRegistry, *, ready: bool | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "service_epoch": registry.service_epoch,
        "gym_commit": registry.settings.gym_commit,
        "config_fingerprint": registry.settings.config_fingerprint,
        **registry.stats(),
    }
    if ready is not None:
        payload["ready"] = ready
    return payload
