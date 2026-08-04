# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

from relax.engine.router.work_accounting import (
    ESTIMATED_TOKENS_HEADER,
    REQUEST_KEY_HEADER,
    WORK_ORIGIN_HEADER,
    WorkerWorkLedger,
    build_work_routing_headers,
)


def _ledger() -> WorkerWorkLedger:
    ledger = WorkerWorkLedger()
    ledger.add_worker("engine-a")
    ledger.add_worker("engine-b")
    return ledger


def test_work_headers_use_cold_start_bound_for_fresh() -> None:
    sample = SimpleNamespace(response_length=0, status="pending", metadata={})

    headers = build_work_routing_headers(
        sample,
        request_key="rid-1",
        recent_completed_response_lengths=[],
        max_response_length=8192,
    )

    assert headers == {
        ESTIMATED_TOKENS_HEADER: "8192",
        REQUEST_KEY_HEADER: "rid-1",
        WORK_ORIGIN_HEADER: "fresh",
    }


def test_work_headers_use_conditional_tail_and_origin() -> None:
    sample = SimpleNamespace(
        response_length=7000,
        status="aborted",
        metadata={"work_origin": "old_debt"},
    )

    headers = build_work_routing_headers(
        sample,
        request_key="rid-2",
        recent_completed_response_lengths=[7200, 7600, 8000, 8192],
        max_response_length=8192,
    )

    assert headers[ESTIMATED_TOKENS_HEADER] == "1000"
    assert headers[REQUEST_KEY_HEADER] == "rid-2"
    assert headers[WORK_ORIGIN_HEADER] == "old_debt"


def test_work_aware_router_balances_reserved_tokens_and_releases() -> None:
    ledger = _ledger()

    first = ledger.select_and_reserve(8192)
    second = ledger.select_and_reserve(8192)
    third = ledger.select_and_reserve(1024)

    assert (first, second, third) == ("engine-a", "engine-b", "engine-a")
    assert ledger.active_requests == {"engine-a": 2, "engine-b": 1}
    assert ledger.reserved_tokens == {"engine-a": 9216, "engine-b": 8192}

    ledger.release(first, 8192)
    ledger.release(third, 1024)
    ledger.release(second, 8192)

    assert ledger.active_requests == {"engine-a": 0, "engine-b": 0}
    assert ledger.reserved_tokens == {"engine-a": 0, "engine-b": 0}


def test_invalid_estimate_is_fail_open_zero() -> None:
    ledger = _ledger()
    assert ledger.select_and_reserve(-1) == "engine-a"
    assert ledger.reserved_tokens["engine-a"] == 0


def test_work_state_is_prompt_free() -> None:
    ledger = _ledger()
    ledger.select_and_reserve(4096)

    state = ledger.snapshot()

    assert state == {
        "engine-a": {
            "active_requests": 1,
            "reserved_tokens": 4096,
            "healthy": True,
        },
        "engine-b": {
            "active_requests": 0,
            "reserved_tokens": 0,
            "healthy": True,
        },
    }


def test_dead_worker_is_excluded_from_selection() -> None:
    ledger = _ledger()
    ledger.set_dead("engine-a", dead=True)

    assert ledger.select_and_reserve(8192) == "engine-b"
