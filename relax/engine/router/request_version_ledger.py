# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict, defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal


class PublicationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ActiveRequestVersion:
    rid: str
    worker: str
    kv_epoch_version: int
    kv_epoch_actor_step: int
    registered_at: float


@dataclass(frozen=True)
class PublicationPlan:
    publication_id: str
    previous_version: int
    target_version: int
    previous_actor_step: int
    target_actor_step: int
    max_gap: int
    max_actor_step_gap: int
    expired_by_worker: dict[str, tuple[str, ...]]
    publication_gap_expired_rids: tuple[str, ...]
    actor_step_gap_expired_rids: tuple[str, ...]
    active_request_count: int

    @property
    def expired_rids(self) -> tuple[str, ...]:
        return tuple(rid for rids in self.expired_by_worker.values() for rid in rids)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["expired_count"] = len(self.expired_rids)
        value["safe_count"] = self.active_request_count - value["expired_count"]
        return value


@dataclass(frozen=True)
class TerminalPublication:
    status: Literal["committed", "failed"]
    plan: PublicationPlan
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "reason": self.reason, **self.plan.to_dict()}


class RequestVersionLedger:
    """Track active request KV epochs and serialize weight publications."""

    _TERMINAL_HISTORY_LIMIT = 32

    def __init__(self, *, initial_version: int = 0, initial_actor_step: int = 0) -> None:
        if initial_version < 0:
            raise ValueError("initial_version must be non-negative")
        if initial_actor_step < 0:
            raise ValueError("initial_actor_step must be non-negative")
        self.current_version = initial_version
        self.current_actor_step = initial_actor_step
        self.active: dict[str, ActiveRequestVersion] = {}
        self._condition = asyncio.Condition()
        self._publication: PublicationPlan | None = None
        self._publication_prepared = False
        self._terminal_publications: OrderedDict[str, TerminalPublication] = OrderedDict()
        # Diagnostic only: publication failures never become a process-lifetime request gate.
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
            await self._condition.wait_for(lambda: self._publication is None)
            if rid in self.active:
                raise PublicationError(f"duplicate active rid: {rid}")
            worker = select_worker()
            request = ActiveRequestVersion(
                rid=rid,
                worker=worker,
                kv_epoch_version=self.current_version,
                kv_epoch_actor_step=self.current_actor_step,
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
        target_actor_step: int,
        max_gap: int,
        max_actor_step_gap: int,
        abort_request: Callable[[str, str], Awaitable[None]],
        timeout_seconds: float,
        publication_id: str | None = None,
    ) -> PublicationPlan:
        if max_gap < 1:
            raise ValueError("max_gap must be positive")
        if max_actor_step_gap < 0:
            raise ValueError("max_actor_step_gap must be non-negative")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if publication_id is not None and not publication_id:
            raise ValueError("publication_id must be non-empty when supplied")

        async with self._condition:
            if self._publication is not None:
                raise PublicationError(f"publication already active: {self._publication.publication_id}")
            if publication_id is not None and publication_id in self._terminal_publications:
                raise PublicationError(f"publication id was already finalized: {publication_id}")
            if target_version != self.current_version + 1:
                raise PublicationError(
                    f"target_version must advance exactly once: current={self.current_version}, target={target_version}"
                )
            if target_actor_step < self.current_actor_step:
                raise PublicationError(
                    "target_actor_step must not move backwards: "
                    f"current={self.current_actor_step}, target={target_actor_step}"
                )

            expired: dict[str, list[str]] = defaultdict(list)
            publication_gap_expired_rids: list[str] = []
            actor_step_gap_expired_rids: list[str] = []
            for request in self.active.values():
                publication_gap_expired = target_version - request.kv_epoch_version > max_gap
                actor_step_gap_expired = target_actor_step - request.kv_epoch_actor_step > max_actor_step_gap
                if publication_gap_expired:
                    publication_gap_expired_rids.append(request.rid)
                if actor_step_gap_expired:
                    actor_step_gap_expired_rids.append(request.rid)
                if publication_gap_expired or actor_step_gap_expired:
                    expired[request.worker].append(request.rid)
            plan = PublicationPlan(
                publication_id=publication_id or uuid.uuid4().hex,
                previous_version=self.current_version,
                target_version=target_version,
                previous_actor_step=self.current_actor_step,
                target_actor_step=target_actor_step,
                max_gap=max_gap,
                max_actor_step_gap=max_actor_step_gap,
                expired_by_worker={worker: tuple(sorted(rids)) for worker, rids in sorted(expired.items())},
                publication_gap_expired_rids=tuple(sorted(publication_gap_expired_rids)),
                actor_step_gap_expired_rids=tuple(sorted(actor_step_gap_expired_rids)),
                active_request_count=len(self.active),
            )
            self._publication = plan
            self._publication_prepared = False

        try:
            worker_by_rid = {rid: worker for worker, rids in plan.expired_by_worker.items() for rid in rids}
            deadline = time.monotonic() + timeout_seconds
            while True:
                async with self._condition:
                    pending_rids = [rid for rid in plan.expired_rids if rid in self.active]
                if not pending_rids:
                    break

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"targeted retirement timed out with {len(pending_rids)} active request(s)")
                try:
                    abort_results = await asyncio.wait_for(
                        asyncio.gather(
                            *(abort_request(worker_by_rid[rid], rid) for rid in pending_rids),
                            return_exceptions=True,
                        ),
                        timeout=remaining,
                    )
                except (TimeoutError, asyncio.TimeoutError) as exc:
                    raise TimeoutError(
                        f"targeted retirement timed out with {len(pending_rids)} active request(s)"
                    ) from exc
                failures = [
                    (rid, result)
                    for rid, result in zip(pending_rids, abort_results, strict=True)
                    if isinstance(result, BaseException)
                ]
                if failures:
                    rid, failure = failures[0]
                    raise PublicationError(
                        f"targeted retirement abort failed for rid={rid}: {type(failure).__name__}: {failure}"
                    ) from failure
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
                    self._publication_prepared = False
                    self._condition.notify_all()
            raise
        async with self._condition:
            if self._publication != plan:
                raise PublicationError(f"publication changed while prepare was completing: {plan.publication_id}")
            self._publication_prepared = True
        return plan

    async def commit(self, publication_id: str, target_version: int, target_actor_step: int) -> PublicationPlan:
        async with self._condition:
            terminal = self._require_terminal(publication_id, target_version, target_actor_step, "committed")
            if terminal is not None:
                return terminal.plan
            plan = self._require_publication(publication_id, target_version, target_actor_step)
            if not self._publication_prepared:
                raise PublicationError(f"publication prepare is not complete: {publication_id}")
            self.current_version = target_version
            self.current_actor_step = target_actor_step
            self._publication = None
            self._publication_prepared = False
            self._failed_reason = None
            self._remember_terminal(TerminalPublication(status="committed", plan=plan))
            self._condition.notify_all()
            return plan

    async def fail(
        self,
        publication_id: str,
        target_version: int,
        target_actor_step: int,
        reason: str,
    ) -> PublicationPlan:
        async with self._condition:
            terminal = self._require_terminal(publication_id, target_version, target_actor_step, "failed")
            if terminal is not None:
                return terminal.plan
            plan = self._require_publication(publication_id, target_version, target_actor_step)
            self._failed_reason = reason or "weight publication failed"
            self._publication = None
            self._publication_prepared = False
            self._remember_terminal(TerminalPublication(status="failed", plan=plan, reason=self._failed_reason))
            self._condition.notify_all()
            return plan

    def _remember_terminal(self, terminal: TerminalPublication) -> None:
        publication_id = terminal.plan.publication_id
        self._terminal_publications[publication_id] = terminal
        self._terminal_publications.move_to_end(publication_id)
        while len(self._terminal_publications) > self._TERMINAL_HISTORY_LIMIT:
            self._terminal_publications.popitem(last=False)

    def _require_terminal(
        self,
        publication_id: str,
        target_version: int,
        target_actor_step: int,
        expected_status: Literal["committed", "failed"],
    ) -> TerminalPublication | None:
        terminal = self._terminal_publications.get(publication_id)
        if terminal is None:
            return None
        plan = terminal.plan
        if plan.target_version != target_version or plan.target_actor_step != target_actor_step:
            raise PublicationError(
                "terminal publication target mismatch: "
                f"stored=({plan.target_version}, {plan.target_actor_step}), "
                f"supplied=({target_version}, {target_actor_step})"
            )
        if terminal.status != expected_status:
            raise PublicationError(
                f"publication {publication_id} is already {terminal.status}; cannot mark it {expected_status}"
            )
        return terminal

    def _require_publication(
        self,
        publication_id: str,
        target_version: int,
        target_actor_step: int,
    ) -> PublicationPlan:
        plan = self._publication
        if plan is None:
            raise PublicationError("no publication is active")
        if plan.publication_id != publication_id:
            raise PublicationError(f"publication id mismatch: active={plan.publication_id}, supplied={publication_id}")
        if plan.target_version != target_version:
            raise PublicationError(
                f"publication target mismatch: active={plan.target_version}, supplied={target_version}"
            )
        if plan.target_actor_step != target_actor_step:
            raise PublicationError(
                "publication actor-step target mismatch: "
                f"active={plan.target_actor_step}, supplied={target_actor_step}"
            )
        return plan

    async def snapshot(self) -> dict[str, Any]:
        async with self._condition:
            publication = self._publication.to_dict() if self._publication is not None else None
            last_publication = next(reversed(self._terminal_publications.values()), None)
            return {
                "current_version": self.current_version,
                "current_actor_step": self.current_actor_step,
                "publication": publication,
                "publication_prepared": self._publication_prepared,
                "last_publication": last_publication.to_dict() if last_publication is not None else None,
                "failed_reason": self._failed_reason,
                "active": {
                    rid: {
                        "worker": request.worker,
                        "kv_epoch_version": request.kv_epoch_version,
                        "kv_epoch_actor_step": request.kv_epoch_actor_step,
                        "registered_at": request.registered_at,
                    }
                    for rid, request in sorted(self.active.items())
                },
            }
