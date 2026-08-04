# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from relax.utils.cross_version_kv import estimate_cross_version_kv_group_remaining_tokens


ESTIMATED_TOKENS_HEADER = "X-Relax-Estimated-Tokens"
REQUEST_KEY_HEADER = "X-Relax-Request-Key"
WORK_ORIGIN_HEADER = "X-Relax-Work-Origin"


class WorkerWorkLedger:
    def __init__(self) -> None:
        self.active_requests: dict[str, int] = {}
        self.reserved_tokens: dict[str, int] = {}
        self.dead_workers: set[str] = set()

    def add_worker(self, worker: str) -> None:
        self.active_requests.setdefault(worker, 0)
        self.reserved_tokens.setdefault(worker, 0)

    def set_dead(self, worker: str, *, dead: bool) -> None:
        if dead:
            self.dead_workers.add(worker)
        else:
            self.dead_workers.discard(worker)

    def select(self, estimated_tokens: int) -> str:
        estimated_tokens = max(int(estimated_tokens), 0)
        workers = [worker for worker in self.active_requests if worker not in self.dead_workers]
        if not workers:
            raise RuntimeError("No healthy workers available in the pool")
        return min(
            workers,
            key=lambda item: (
                self.reserved_tokens[item] + estimated_tokens,
                self.active_requests[item],
                item,
            ),
        )

    def select_and_reserve(self, estimated_tokens: int) -> str:
        estimated_tokens = max(int(estimated_tokens), 0)
        worker = self.select(estimated_tokens)
        self.reserve(worker, estimated_tokens)
        return worker

    def reserve(self, worker: str, estimated_tokens: int) -> None:
        if worker not in self.active_requests:
            raise KeyError(f"Unknown worker: {worker}")
        self.active_requests[worker] += 1
        self.reserved_tokens[worker] += max(int(estimated_tokens), 0)

    def release(self, worker: str, estimated_tokens: int) -> None:
        if worker not in self.active_requests:
            raise KeyError(f"Unknown worker: {worker}")
        self.active_requests[worker] -= 1
        if self.active_requests[worker] < 0:
            raise RuntimeError(f"Worker active request count became negative: {worker}")
        self.reserved_tokens[worker] = max(
            self.reserved_tokens[worker] - max(int(estimated_tokens), 0),
            0,
        )

    def snapshot(self) -> dict[str, dict[str, int | bool]]:
        return {
            worker: {
                "active_requests": self.active_requests[worker],
                "reserved_tokens": self.reserved_tokens[worker],
                "healthy": worker not in self.dead_workers,
            }
            for worker in self.active_requests
        }


def build_work_routing_headers(
    sample: Any,
    *,
    request_key: str,
    recent_completed_response_lengths: Sequence[int],
    max_response_length: int,
) -> dict[str, str]:
    estimated_tokens = estimate_cross_version_kv_group_remaining_tokens(
        [sample],
        recent_completed_response_lengths=recent_completed_response_lengths,
        max_response_length=max_response_length,
    )
    metadata = getattr(sample, "metadata", {})
    work_origin = metadata.get("work_origin") if isinstance(metadata, dict) else None
    if work_origin is None:
        work_origin = "resume" if int(getattr(sample, "response_length", 0) or 0) > 0 else "fresh"
    return {
        ESTIMATED_TOKENS_HEADER: str(estimated_tokens),
        REQUEST_KEY_HEADER: request_key,
        WORK_ORIGIN_HEADER: str(work_origin),
    }
