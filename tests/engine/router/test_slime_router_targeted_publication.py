# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from types import SimpleNamespace

import httpx
from starlette.requests import Request

from relax.engine.router.router import SlimeRouter


def _args():
    return SimpleNamespace(
        slime_router_sticky=False,
        slime_router_sticky_idle_secs=600.0,
        hybrid_dcs_weight_sync=False,
        enable_cross_version_kv_continuation=False,
        slime_router_max_connections=8,
        slime_router_timeout=None,
        slime_router_middleware_paths=[],
        slime_router_health_check_failure_threshold=3,
        rollout_health_check_interval=60,
        sglang_server_concurrency=4,
        rollout_num_gpus=2,
        rollout_num_gpus_per_engine=1,
    )


def _request(*, rid: str | None = None) -> Request:
    body = json.dumps({"rid": rid} if rid is not None else {}).encode()
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
            "headers": [],
        },
        receive,
    )


async def test_targeted_publication_tracks_request_version_and_releases_on_failure() -> None:
    args = _args()
    args.hybrid_dcs_weight_sync = True
    args.enable_cross_version_kv_continuation = True
    router = SlimeRouter(args)
    router.worker_request_counts["http://engine-a"] = 0
    router.worker_failure_counts["http://engine-a"] = 0

    async def fail(*_args, **_kwargs):
        raise httpx.ReadError("downstream aborted request")

    router.client.request = fail
    try:
        response = await router.proxy(_request(rid="rid-targeted"), "generate")
        state = await router.request_version_ledger.snapshot()
    finally:
        await router.client.aclose()
        await router.control_client.aclose()

    assert response.status_code == 502
    assert state["active"] == {}
    assert router.worker_request_counts["http://engine-a"] == 0


async def test_targeted_publication_namespaces_cache_by_request_and_weight_epoch() -> None:
    args = _args()
    args.hybrid_dcs_weight_sync = True
    args.enable_cross_version_kv_continuation = True
    router = SlimeRouter(args)
    router.worker_request_counts["http://engine-a"] = 0
    router.worker_failure_counts["http://engine-a"] = 0
    router.request_version_ledger.current_version = 3
    captured = {}

    async def succeed(method, url, *, content, headers):
        captured.update(
            {
                "method": method,
                "url": url,
                "payload": json.loads(content),
                "headers": headers,
            }
        )
        return httpx.Response(
            200,
            json={"output_ids": [1], "meta_info": {"completion_tokens": 1}},
            request=httpx.Request(method, url),
        )

    router.client.request = succeed
    try:
        response = await router.proxy(_request(rid="rid-targeted"), "generate")
    finally:
        await router.client.aclose()
        await router.control_client.aclose()

    assert response.status_code == 200
    assert captured["payload"]["extra_key"] == ":weight-version:3"
    assert "content-length" not in captured["headers"]
    assert (await router.request_version_ledger.snapshot())["active"] == {}
