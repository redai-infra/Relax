# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import httpx
import pytest
from starlette.requests import Request

from relax.engine.router.router import SlimeRouter


def _args():
    return SimpleNamespace(
        slime_router_sticky=False,
        slime_router_sticky_idle_secs=600.0,
        slime_router_work_aware=True,
        slime_router_max_connections=8,
        slime_router_timeout=None,
        slime_router_middleware_paths=[],
        slime_router_health_check_failure_threshold=3,
        rollout_health_check_interval=60,
        sglang_server_concurrency=4,
        rollout_num_gpus=2,
        rollout_num_gpus_per_engine=1,
    )


def _request(*, estimated_tokens: str = "4096") -> Request:
    body = b"{}"
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/generate",
            "headers": [
                (b"x-relax-estimated-tokens", estimated_tokens.encode()),
                (b"x-relax-request-key", b"rid-test"),
                (b"x-relax-work-origin", b"fresh"),
            ],
        },
        receive,
    )


async def test_proxy_releases_work_reservation_after_downstream_failure() -> None:
    router = SlimeRouter(_args())
    router.work_ledger.add_worker("http://engine-a")

    async def fail(*_args, **_kwargs):
        raise httpx.ConnectError("downstream failed")

    router.client.request = fail
    try:
        with pytest.raises(httpx.ConnectError):
            await router.proxy(_request(), "generate")
    finally:
        await router.client.aclose()

    assert router.work_ledger.snapshot()["http://engine-a"] == {
        "active_requests": 0,
        "reserved_tokens": 0,
        "healthy": True,
    }
