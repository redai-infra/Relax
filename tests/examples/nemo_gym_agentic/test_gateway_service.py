# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import copy

import httpx
import pytest
from relax_nemo_gym_example.app.protocol import (
    InterruptPolicy,
    ModelEndpoint,
    TrialRequest,
    TrialStatus,
)
from relax_nemo_gym_example.service.callback_provider import CallbackProvider
from relax_nemo_gym_example.service.config import EnvironmentSpec, GatewaySettings
from relax_nemo_gym_example.service.registry import (
    AdmissionRejected,
    CallbackUnavailable,
    GatewayRegistry,
    TrialConflict,
)
from relax_nemo_gym_example.service.run_adapter import AdapterResult, CleanupResult


class ControlledHandle:
    def __init__(self):
        self.result = asyncio.get_running_loop().create_future()
        self.abort_calls = 0
        self.force_cleanup_calls = 0

    async def wait(self):
        return await self.result

    async def abort(self):
        self.abort_calls += 1
        return CleanupResult(confirmed=True)

    async def force_cleanup(self):
        self.force_cleanup_calls += 1
        return CleanupResult(confirmed=True)

    async def probe_cleanup(self):
        return CleanupResult(confirmed=True)

    def complete(self, *, reward=1.0):
        if not self.result.done():
            self.result.set_result(
                AdapterResult(
                    status=TrialStatus.COMPLETED,
                    reward=reward,
                    metrics={"model_calls": 1},
                )
            )


class ControlledAdapter:
    def __init__(self):
        self.contexts = []
        self.handles = []
        self.started = asyncio.Event()
        self.closed = False

    async def start(self, context):
        handle = ControlledHandle()
        self.contexts.append(context)
        self.handles.append(handle)
        self.started.set()
        return handle

    async def ready(self):
        return True

    async def close(self):
        self.closed = True


class BlockingStartAdapter(ControlledAdapter):
    def __init__(self):
        super().__init__()
        self.start_entered = asyncio.Event()
        self.release_start = asyncio.Event()

    async def start(self, context):
        self.start_entered.set()
        await self.release_start.wait()
        return await super().start(context)


def _settings(*, max_concurrency=2, queue_capacity=4, lease_scan_interval_s=0.01):
    spec = EnvironmentSpec(
        environment="multi_step",
        config="multi-step-v1",
        agent_name="example_multi_step_simple_agent",
        agent_url="http://gym-agent.example",
        interrupt_policy=InterruptPolicy.PROTECTED,
        max_concurrency=max_concurrency,
        queue_capacity=queue_capacity,
        max_deadline_s=120.0,
        readiness_urls=("http://gym-agent.example",),
    )
    return GatewaySettings(
        environments={(spec.environment, spec.config): spec},
        callback_allowed_hosts=frozenset({"relax-one.example", "relax-two.example"}),
        gym_commit="test-commit",
        config_fingerprint="test-fingerprint",
        lease_scan_interval_s=lease_scan_interval_s,
        cleanup_grace_s=0.1,
    )


def _request(
    request_id,
    *,
    session_id="secret-one",
    callback_host="relax-one.example",
    callback_path="/agentic_api",
    api_key_header="Authorization",
    api_key_prefix="Bearer ",
    headers=None,
    lease_s=30.0,
    deadline_s=60.0,
):
    return TrialRequest(
        request_id=request_id,
        session_id=session_id,
        group_id="group-1",
        rollout_mode="train",
        environment="multi_step",
        config="multi-step-v1",
        task={"messages": [{"role": "user", "content": "solve"}], "metadata": {}},
        model_endpoint=ModelEndpoint(
            base_url=f"http://{callback_host}{callback_path}",
            api_key=session_id,
            model="policy-model",
            api_key_header=api_key_header,
            api_key_prefix=api_key_prefix,
            headers=headers or {},
        ),
        interrupt_policy=InterruptPolicy.PROTECTED,
        deadline_s=deadline_s,
        lease_s=lease_s,
    )


def test_callback_provider_configures_explicit_proxy_without_environment_lookup(monkeypatch):
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    captured = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def aclose(self):
            return None

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    provider = CallbackProvider(registry, proxy="http://proxy.example:3128")

    assert captured["proxy"] == "http://proxy.example:3128"
    assert captured["trust_env"] is False
    asyncio.run(provider.close())


async def _wait_for_status(registry, request_id, expected, *, timeout=1.0):
    async def poll():
        while True:
            payload = await registry.get(request_id)
            if payload["status"] == expected:
                return payload
            await asyncio.sleep(0.001)

    return await asyncio.wait_for(poll(), timeout=timeout)


async def test_registry_create_is_idempotent_and_rejects_payload_conflict():
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    payload = _request("request-one").to_payload()
    try:
        first = await registry.create(payload)
        second = await registry.create(copy.deepcopy(payload))

        assert first["request_id"] == "request-one"
        assert second["request_id"] == "request-one"
        assert len(registry._records) == 1

        conflicting = copy.deepcopy(payload)
        conflicting["environment"]["task"]["metadata"]["variant"] = 2
        with pytest.raises(TrialConflict):
            await registry.create(conflicting)
    finally:
        await registry.close()


async def test_registry_enforces_environment_queue_capacity_before_insert():
    adapter = ControlledAdapter()
    registry = GatewayRegistry(
        settings=_settings(max_concurrency=1, queue_capacity=0),
        adapter=adapter,
    )
    await registry.start()
    try:
        await registry.create(_request("request-one").to_payload())
        with pytest.raises(AdmissionRejected):
            await registry.create(_request("request-two", session_id="secret-two").to_payload())
        assert "request-two" not in registry._records
    finally:
        await registry.close()


async def test_abort_is_idempotent_revokes_callback_and_waits_for_confirmed_cleanup():
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    try:
        await registry.create(_request("request-one").to_payload())
        await asyncio.wait_for(adapter.started.wait(), timeout=1.0)
        rollout_id = adapter.contexts[0].rollout_id

        await registry.abort("request-one")
        await registry.abort("request-one")
        result = await _wait_for_status(registry, "request-one", "aborted")

        assert result["error"]["code"] == "aborted"
        assert adapter.handles[0].abort_calls == 1
        with pytest.raises(CallbackUnavailable):
            async with registry.callback_target(rollout_id):
                pass
    finally:
        await registry.close()


async def test_abort_during_start_cleans_the_registered_handle():
    adapter = BlockingStartAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    try:
        await registry.create(_request("request-one").to_payload())
        await asyncio.wait_for(adapter.start_entered.wait(), timeout=1.0)
        abort_task = asyncio.create_task(registry.abort("request-one"))
        await asyncio.sleep(0)
        assert not abort_task.done()

        adapter.release_start.set()
        await asyncio.wait_for(abort_task, timeout=1.0)
        await _wait_for_status(registry, "request-one", "aborted")

        assert adapter.handles[0].abort_calls == 1
    finally:
        await registry.close()


async def test_lease_expiry_cancels_a_running_trial():
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    try:
        await registry.create(_request("request-one", lease_s=0.03).to_payload())
        result = await _wait_for_status(registry, "request-one", "aborted")

        assert result["error"]["code"] == "lease_expired"
        assert adapter.handles[0].abort_calls == 1
    finally:
        await registry.close()


async def test_agent_failure_cleans_the_remote_run_before_marking_failed(caplog):
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    try:
        with caplog.at_level("ERROR", logger="uvicorn.error"):
            await registry.create(_request("request-one").to_payload())
            await asyncio.wait_for(adapter.started.wait(), timeout=1.0)
            adapter.handles[0].result.set_exception(RuntimeError("agent failed"))

            result = await _wait_for_status(registry, "request-one", "failed")

        assert result["error"]["code"] == "agent_error"
        assert adapter.handles[0].abort_calls == 1
        error_records = [record for record in caplog.records if "NeMo Gym trial failed" in record.message]
        assert len(error_records) == 1
        assert error_records[0].exc_info is not None
        assert "agent failed" in caplog.text
    finally:
        await registry.close()


async def test_two_callback_capabilities_forward_distinct_endpoints_and_tokens():
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    seen = []

    def handler(request):
        seen.append(
            {
                "host": request.url.host,
                "authorization": request.headers["authorization"],
                "body": request.content,
            }
        )
        return httpx.Response(
            200,
            json={
                "id": f"chat-{request.url.host}",
                "object": "chat.completion",
                "created": 1,
                "model": "policy-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    provider = CallbackProvider(registry, transport=httpx.MockTransport(handler))
    try:
        await registry.create(_request("request-one").to_payload())
        await registry.create(
            _request(
                "request-two",
                session_id="secret-two",
                callback_host="relax-two.example",
            ).to_payload()
        )
        while len(adapter.contexts) < 2:
            await asyncio.sleep(0.001)

        first, second = adapter.contexts
        await asyncio.gather(
            provider.chat_completions(first.rollout_id, {"messages": [], "model": "ignored"}),
            provider.chat_completions(second.rollout_id, {"messages": [], "model": "ignored"}),
        )

        assert {(item["host"], item["authorization"]) for item in seen} == {
            ("relax-one.example", "Bearer secret-one"),
            ("relax-two.example", "Bearer secret-two"),
        }
        assert all(b'"model":"policy-model"' in item["body"] for item in seen)
    finally:
        await provider.close()
        await registry.close()


async def test_callback_supports_custom_api_key_header_and_standard_v1_base_url():
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chat",
                "object": "chat.completion",
                "created": 1,
                "model": "policy-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
            },
        )

    provider = CallbackProvider(registry, transport=httpx.MockTransport(handler))
    try:
        await registry.create(
            _request(
                "request-custom-auth",
                callback_path="/v1",
                api_key_header="api-key",
                api_key_prefix="",
                headers={"x-user": "user@example.com", "x-app-id": "app"},
            ).to_payload()
        )
        while not adapter.contexts:
            await asyncio.sleep(0.001)

        await provider.chat_completions(adapter.contexts[0].rollout_id, {"messages": []})

        assert seen[0].url.path == "/v1/chat/completions"
        assert seen[0].headers["api-key"] == "secret-one"
        assert seen[0].headers["x-user"] == "user@example.com"
        assert "authorization" not in seen[0].headers
    finally:
        await provider.close()
        await registry.close()


async def test_callback_timeout_is_capped_by_gateway_setting():
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    seen_timeout = []

    def handler(request):
        seen_timeout.append(request.extensions["timeout"])
        return httpx.Response(
            200,
            json={
                "id": "chat",
                "object": "chat.completion",
                "created": 1,
                "model": "policy-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
            },
        )

    provider = CallbackProvider(
        registry,
        transport=httpx.MockTransport(handler),
        timeout_s=12.0,
    )
    try:
        await registry.create(_request("request-timeout-cap").to_payload())
        while not adapter.contexts:
            await asyncio.sleep(0.001)

        await provider.chat_completions(adapter.contexts[0].rollout_id, {"messages": []})

        assert seen_timeout == [{"connect": 12.0, "read": 12.0, "write": 12.0, "pool": 12.0}]
    finally:
        await provider.close()
        await registry.close()


async def test_one_hundred_concurrent_callbacks_keep_tokens_isolated():
    adapter = ControlledAdapter()
    registry = GatewayRegistry(
        settings=_settings(max_concurrency=100, queue_capacity=0),
        adapter=adapter,
    )
    await registry.start()
    seen_tokens = []

    def handler(request):
        seen_tokens.append(request.headers["authorization"])
        return httpx.Response(
            200,
            json={
                "id": "chat",
                "object": "chat.completion",
                "created": 1,
                "model": "policy-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
            },
        )

    provider = CallbackProvider(registry, transport=httpx.MockTransport(handler))
    try:
        for index in range(100):
            await registry.create(
                _request(
                    f"request-{index}",
                    session_id=f"secret-{index}",
                ).to_payload()
            )
        while len(adapter.contexts) < 100:
            await asyncio.sleep(0.001)

        await asyncio.gather(
            *(provider.chat_completions(context.rollout_id, {"messages": []}) for context in adapter.contexts)
        )

        assert set(seen_tokens) == {f"Bearer secret-{index}" for index in range(100)}
    finally:
        await provider.close()
        await registry.close()


async def test_abort_cancels_an_inflight_relax_callback():
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()

    class BlockingCallbackTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.entered = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def handle_async_request(self, request):
            self.entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    transport = BlockingCallbackTransport()
    provider = CallbackProvider(registry, transport=transport)
    try:
        await registry.create(_request("request-one").to_payload())
        await asyncio.wait_for(adapter.started.wait(), timeout=1.0)
        callback_task = asyncio.create_task(
            provider.chat_completions(adapter.contexts[0].rollout_id, {"messages": []})
        )
        await asyncio.wait_for(transport.entered.wait(), timeout=1.0)

        await registry.abort("request-one")
        with pytest.raises(asyncio.CancelledError):
            await callback_task
        await _wait_for_status(registry, "request-one", "aborted")

        assert transport.cancelled.is_set()
    finally:
        await provider.close()
        await registry.close()


async def test_terminal_completion_cannot_be_overwritten_by_late_abort():
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    await registry.start()
    try:
        await registry.create(_request("request-one").to_payload())
        await asyncio.wait_for(adapter.started.wait(), timeout=1.0)
        adapter.handles[0].complete(reward=0.75)
        completed = await _wait_for_status(registry, "request-one", "completed")

        await registry.abort("request-one")
        after_abort = await registry.get("request-one")

        assert completed["reward"] == 0.75
        assert after_abort == completed
        assert adapter.handles[0].abort_calls == 0
    finally:
        await registry.close()
