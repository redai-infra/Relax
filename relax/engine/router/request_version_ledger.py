# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any


class PublicationError(RuntimeError):
    pass


class PublicationFailed(PublicationError):
    pass


@dataclass(frozen=True)
class ActiveRequestVersion:
    rid: str
    worker: str
    kv_epoch_version: int
    registered_at: float


@dataclass(frozen=True)
class PublicationPlan:
    publication_id: str
    previous_version: int
    target_version: int
    max_gap: int
    expired_by_worker: dict[str, tuple[str, ...]]
    active_request_count: int

    @property
    def expired_rids(self) -> tuple[str, ...]:
        return tuple(rid for rids in self.expired_by_worker.values() for rid in rids)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["expired_count"] = len(self.expired_rids)
        value["safe_count"] = self.active_request_count - value["expired_count"]
        return value


class RequestVersionLedger:
    """Track active request KV epochs and serialize weight publications."""

    def __init__(self, *, initial_version: int = 0) -> None:
        if initial_version < 0:
            raise ValueError("initial_version must be non-negative")
        self.current_version = initial_version
        self.active: dict[str, ActiveRequestVersion] = {}
        self._condition = asyncio.Condition()
        self._publication: PublicationPlan | None = None
        self._failed_reason: str | None = None

    async def register(self, rid: str, worker: str) -> ActiveRequestVersion:
        return await self.register_selected(rid, lambda: worker)

    async def register_selected(
        self,
        rid: str,
        select_worker: Callable[[], str],
    ) -> ActiveRequestVersion:
        if not rid:
            raise ValueError("rid must be non-empty")
        async with self._condition:
            await self._condition.wait_for(lambda: self._publication is None or self._failed_reason is not None)
            if self._failed_reason is not None:
                raise PublicationFailed(self._failed_reason)
            if rid in self.active:
                raise PublicationError(f"duplicate active rid: {rid}")
            worker = select_worker()
            request = ActiveRequestVersion(
                rid=rid,
                worker=worker,
                kv_epoch_version=self.current_version,
                registered_at=time.monotonic(),
            )
            self.active[rid] = request
            return request

    async def unregister(self, rid: str) -> ActiveRequestVersion | None:
        async with self._condition:
            request = self.active.pop(rid, None)
            if request is not None:
                self._condition.notify_all()
            return request

    async def prepare(
        self,
        *,
        target_version: int,
        max_gap: int,
        abort_request: Callable[[str, str], Awaitable[None]],
        timeout_seconds: float,
    ) -> PublicationPlan:
        if max_gap < 1:
            raise ValueError("max_gap must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        async with self._condition:
            if self._failed_reason is not None:
                raise PublicationFailed(self._failed_reason)
            if self._publication is not None:
                raise PublicationError(f"publication already active: {self._publication.publication_id}")
            if target_version != self.current_version + 1:
                raise PublicationError(
                    f"target_version must advance exactly once: current={self.current_version}, target={target_version}"
                )

            expired: dict[str, list[str]] = defaultdict(list)
            for request in self.active.values():
                if target_version - request.kv_epoch_version > max_gap:
                    expired[request.worker].append(request.rid)
            plan = PublicationPlan(
                publication_id=uuid.uuid4().hex,
                previous_version=self.current_version,
                target_version=target_version,
                max_gap=max_gap,
                expired_by_worker={worker: tuple(sorted(rids)) for worker, rids in sorted(expired.items())},
                active_request_count=len(self.active),
            )
            self._publication = plan

        try:
            worker_by_rid = {rid: worker for worker, rids in plan.expired_by_worker.items() for rid in rids}
            deadline = time.monotonic() + timeout_seconds
            while True:
                async with self._condition:
                    pending_rids = [rid for rid in plan.expired_rids if rid in self.active]
                if not pending_rids:
                    break

                await asyncio.gather(*(abort_request(worker_by_rid[rid], rid) for rid in pending_rids))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"targeted retirement timed out with {len(pending_rids)} active request(s)")
                async with self._condition:
                    try:
                        await asyncio.wait_for(
                            self._condition.wait_for(lambda: all(rid not in self.active for rid in plan.expired_rids)),
                            timeout=min(0.25, remaining),
                        )
                    except (TimeoutError, asyncio.TimeoutError):
                        pass
        except BaseException as exc:
            async with self._condition:
                if self._publication == plan:
                    self._failed_reason = f"targeted retirement failed: {type(exc).__name__}: {exc}"
                    self._publication = None
                    self._condition.notify_all()
            raise
        return plan

    async def commit(self, publication_id: str, target_version: int) -> PublicationPlan:
        async with self._condition:
            plan = self._require_publication(publication_id, target_version)
            self.current_version = target_version
            self._publication = None
            self._condition.notify_all()
            return plan

    async def fail(self, publication_id: str, target_version: int, reason: str) -> PublicationPlan:
        async with self._condition:
            plan = self._require_publication(publication_id, target_version)
            self._failed_reason = reason or "weight publication failed"
            self._publication = None
            self._condition.notify_all()
            return plan

    def _require_publication(self, publication_id: str, target_version: int) -> PublicationPlan:
        plan = self._publication
        if plan is None:
            raise PublicationError("no publication is active")
        if plan.publication_id != publication_id:
            raise PublicationError(f"publication id mismatch: active={plan.publication_id}, supplied={publication_id}")
        if plan.target_version != target_version:
            raise PublicationError(
                f"publication target mismatch: active={plan.target_version}, supplied={target_version}"
            )
        return plan

    async def snapshot(self) -> dict[str, Any]:
        async with self._condition:
            publication = self._publication.to_dict() if self._publication is not None else None
            return {
                "current_version": self.current_version,
                "publication": publication,
                "failed_reason": self._failed_reason,
                "active": {
                    rid: {
                        "worker": request.worker,
                        "kv_epoch_version": request.kv_epoch_version,
                        "registered_at": request.registered_at,
                    }
                    for rid, request in sorted(self.active.items())
                },
            }
