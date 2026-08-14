# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Production capture (PR B, data plane).

This module owns the capture half of the trajectory replay system: turning a
handful of post-forward tensors, identity and stage outputs from one training
step into a replay bundle that :mod:`relax.utils.replay.runner` can replay.

It contains no production math and touches no hot path itself. Training enables
it via :func:`maybe_enable_from_env` (``RELAX_REPLAY_CAPTURE``). The hot-path
hooks live in :mod:`relax.utils.replay.capture_hooks` and are called from:

- rollout-level reward / advantage: ``MegatronTrainRayActor.train_actor``;
- policy loss terms: ``relax.backends.megatron.loss.policy_loss_function``;
- actor-step identity: the ``train_one_step`` loop in
  ``relax.backends.megatron.model``.

The hot-path contract is strict:

- ``CaptureManager.submit`` only *detaches* tensors and enqueues them; it never
  calls ``.cpu()`` / ``.item()`` / ``.tolist()`` and never touches ``torch.distributed``.
- The GPU→CPU copy and serialization happen on a dedicated writer thread.
- The queue is bounded: when full, a step is dropped and a diagnostic is logged
  (never back-pressure into training).
- The writer thread isolates all bundle-build failures so a crash there cannot
  propagate into the training loop.
"""

from __future__ import annotations

import atexit
import queue
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger
from relax.utils.replay.bundle import BundleWriter
from relax.utils.replay.schema import (
    FORMAT_VERSION,
    STAGE_ORDER,
    BundleIndex,
    ComparisonPolicy,
    Identity,
    Manifest,
    ProducerInfo,
    RecomputeConfig,
    SampleRecord,
    StageContract,
    StageId,
)
from relax.utils.replay.stages import REIMPLEMENTED_STAGES, V1_STAGE_VERSIONS, capability_for


logger = get_logger(__name__)


@dataclass
class CaptureConfig:
    """Runtime configuration for production capture."""

    enabled: bool = False
    output_dir: str | Path = ""
    bundle_id_prefix: str = "replay"
    queue_capacity: int = 16
    # Actor steps (rollout_id, step_id) to capture; None captures every step.
    selected_steps: set[tuple[int, int]] | None = None
    # Rollouts to capture at rollout level (reward/advantage); None captures every rollout.
    selected_rollouts: set[int] | None = None
    redaction: dict[str, str] = field(default_factory=dict)
    commit: str = ""


@dataclass
class CaptureRecord:
    """Everything one capture cohort needs to be replayed.

    A record is anchored by exactly one of ``actor_step_id`` (per-step, loss)
    or ``rollout_id`` (per-rollout, reward/advantage). ``tensors`` holds the
    post-forward tensor payloads (inputs and expected outputs); ``expected``
    holds the JSON-serializable expected outputs.
    """

    identity: Identity
    samples: list[SampleRecord]
    config: RecomputeConfig
    tensors: dict[str, torch.Tensor]
    expected: dict[str, Any]
    bundle_id: str
    actor_step_id: tuple[int, int] | None = None
    rollout_id: int | None = None
    producer: ProducerInfo = field(default_factory=ProducerInfo)
    redaction: dict[str, str] = field(default_factory=dict)
    # Stages actually captured by this record; None means "derive from the
    # frozen matrix" (declare every matrix stage). A partial capture (e.g. loss
    # only) sets this explicitly so the manifest does not promise stages whose
    # payloads are absent.
    stages: set[StageId] | None = None
    # Raw per-sample metadata used to build ``SampleRecord`` on the writer
    # thread. The mask/reward/group tensors stay detached through the hot path;
    # their GPU→CPU conversion happens only in ``build_bundle_from_record``.
    response_lengths: list[int] | None = None
    total_lengths: list[int] | None = None
    loss_masks_tensor: torch.Tensor | None = None
    group_indices_tensor: torch.Tensor | None = None
    raw_rewards_tensor: torch.Tensor | None = None
    rewards_tensor: torch.Tensor | None = None

    @property
    def anchor(self) -> str:
        """Human-readable cohort anchor for logging."""
        if self.actor_step_id is not None:
            return f"step{self.actor_step_id}"
        if self.rollout_id is not None:
            return f"rollout-{self.rollout_id}"
        return "<unset>"


def build_manifest_for_record(record: CaptureRecord) -> Manifest:
    """Derive the manifest from the frozen V1 capability matrix.

    Each stage in ``STAGE_ORDER`` is declared with the capability the matrix
    grants for the bundle's topology (``record.config.advantage_estimator`` and
    ``record.identity.rank["cp"]``). Stages outside the matrix are
    ``unsupported`` and will be skipped — never silently recomputed. When
    ``record.stages`` is set, only those stages are declared (the rest get no
    contract and are skipped), so a partial capture never promises absent
    payloads.
    """
    context_parallel = record.identity.rank.get("cp", 1)
    declared = record.stages if record.stages is not None else set(STAGE_ORDER)
    stage_contracts: dict[StageId, StageContract] = {}
    for stage in STAGE_ORDER:
        if stage not in declared:
            continue
        capability = capability_for(
            stage,
            advantage_estimator=record.config.advantage_estimator,
            context_parallel=context_parallel,
        )
        stage_contracts[stage] = StageContract(
            stage=stage,
            version=V1_STAGE_VERSIONS[stage],
            capability=capability,
            implementation="reimplemented" if stage in REIMPLEMENTED_STAGES else "reuse",
        )
    return Manifest(
        format_version=FORMAT_VERSION,
        bundle_id=record.bundle_id,
        producer=record.producer,
        stage_contracts=stage_contracts,
        payloads={},
        comparison_policy=ComparisonPolicy(),
        redaction=dict(record.redaction),
    )


def _normalize_expected(expected: dict[str, Any]) -> dict[str, Any]:
    """Convert detached scalar tensors in ``expected`` to plain floats.

    Runs on the writer thread so the GPU→CPU sync (``.item()``) never happens
    in the training hot path. Handles nested dicts/lists (e.g. ``loss.policy``
    metric maps).
    """

    def _convert(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().item()
        if isinstance(value, dict):
            return {key: _convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_convert(item) for item in value]
        return value

    return {key: _convert(value) for key, value in expected.items()}


def _build_samples_from_raw(record: CaptureRecord) -> list[SampleRecord]:
    """Build the ``SampleRecord`` list from raw per-sample metadata.

    Runs on the writer thread; the flat ``loss_masks_tensor`` is split back per
    sample using ``response_lengths``, and the optional group/reward tensors
    are converted here so their GPU→CPU sync never happens in the hot path.
    When the group/reward tensors are absent (loss-only capture) placeholders
    are used.
    """
    if record.response_lengths is None or record.total_lengths is None or record.loss_masks_tensor is None:
        raise ValueError("capture record has no samples and incomplete raw sample metadata")

    count = len(record.response_lengths)
    flat_masks = record.loss_masks_tensor.detach().cpu().tolist()
    masks: list[list[int]] = []
    offset = 0
    for response_length in record.response_lengths:
        masks.append([int(value) for value in flat_masks[offset : offset + response_length]])
        offset += response_length

    group_indices = (
        [int(value) for value in record.group_indices_tensor.detach().cpu().tolist()]
        if record.group_indices_tensor is not None
        else [0] * count
    )
    raw_rewards = (
        [float(value) for value in record.raw_rewards_tensor.detach().cpu().tolist()]
        if record.raw_rewards_tensor is not None
        else [0.0] * count
    )
    rewards = (
        [float(value) for value in record.rewards_tensor.detach().cpu().tolist()]
        if record.rewards_tensor is not None
        else raw_rewards
    )
    micro_batch_id = _micro_batch_id_for(record)

    return [
        SampleRecord(
            sample_id=f"s-{index}",
            group_index=group_indices[index],
            response_length=int(record.response_lengths[index]),
            total_length=int(record.total_lengths[index]),
            loss_mask=masks[index],
            raw_reward=raw_rewards[index],
            reward=rewards[index],
            label_hash="",
            micro_batch_id=micro_batch_id,
        )
        for index in range(count)
    ]


def _micro_batch_id_for(record: CaptureRecord) -> str | None:
    """Stable micro-batch id for a per-step capture (``mb-<step_id>``).

    Rollout-level records have no actor step, so batch selection stays fail-
    closed.
    """
    if record.actor_step_id is None:
        return None
    return f"mb-{record.actor_step_id[1]:04d}"


def build_bundle_from_record(record: CaptureRecord, output_dir: str | Path) -> Path:
    """Serialize one :class:`CaptureRecord` into a replay bundle.

    Pure and synchronous; used directly by tests and by the async writer
    thread.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest_for_record(record)
    samples = record.samples if record.samples else _build_samples_from_raw(record)
    identity = record.identity
    micro_batch_id = _micro_batch_id_for(record)
    # When samples are reconstructed from raw step metadata, stamp a matching
    # identity.micro_batch_ids so --batch selection agrees with SampleRecord.
    if not record.samples and micro_batch_id is not None:
        identity = replace(identity, micro_batch_ids=[micro_batch_id])
    index = BundleIndex(bundle_id=record.bundle_id, identity=identity, samples=samples, config=record.config)

    expected = _normalize_expected(record.expected)
    # Derive per-sample reward expectations from the deferred tensors (writer
    # thread), so the reward adapters have JSON-serializable lists to compare.
    if record.raw_rewards_tensor is not None:
        expected.setdefault(StageId.REWARD_RAW.value, {})["raw_rewards"] = [
            float(value) for value in record.raw_rewards_tensor.detach().cpu().tolist()
        ]
    if record.rewards_tensor is not None:
        expected.setdefault(StageId.REWARD_POST_PROCESS.value, {})["rewards"] = [
            float(value) for value in record.rewards_tensor.detach().cpu().tolist()
        ]

    writer = BundleWriter(output_dir / record.bundle_id, manifest, index, expected)
    for name, tensor in record.tensors.items():
        writer.write_payload(name, tensor.detach().cpu().contiguous())
    writer.finalize(ranks=[0])
    return output_dir / record.bundle_id


class CaptureManager:
    """Bounded, asynchronous capture sink.

    Owns the writer thread and the bounded queue. Disabled by default: when
    ``config.enabled`` is False, :meth:`submit` is a no-op that never inspects
    tensors, so the disabled path is a single branch.
    """

    def __init__(self, config: CaptureConfig) -> None:
        self._config = config
        self._dropped_count = 0
        self._error_count = 0
        self._queue: queue.Queue[CaptureRecord | None] | None = None
        self._thread: threading.Thread | None = None
        if config.enabled:
            self._queue = queue.Queue(maxsize=config.queue_capacity)
            self._thread = threading.Thread(target=self._writer_loop, name="replay-capture-writer", daemon=True)
            self._thread.start()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def output_dir(self) -> str:
        return str(self._config.output_dir)

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    @property
    def error_count(self) -> int:
        return self._error_count

    def is_selected(self, actor_step_id: tuple[int, int]) -> bool:
        if self._config.selected_steps is None:
            return True
        return actor_step_id in self._config.selected_steps

    def is_rollout_selected(self, rollout_id: int) -> bool:
        if self._config.selected_rollouts is None:
            return True
        return rollout_id in self._config.selected_rollouts

    def _record_selected(self, record: CaptureRecord) -> bool:
        if record.actor_step_id is not None:
            return self.is_selected(record.actor_step_id)
        if record.rollout_id is not None:
            return self.is_rollout_selected(record.rollout_id)
        return False

    def submit(self, record: CaptureRecord) -> bool:
        """Enqueue a detached snapshot of ``record``; returns True if queued.

        Only ``detach()`` happens on the caller thread (no GPU→CPU sync, no
        collective). On queue overflow the record is dropped and False
        returned.
        """
        if not self.enabled or not self._record_selected(record):
            return False

        snapshot = CaptureRecord(
            identity=record.identity,
            samples=record.samples,
            config=record.config,
            tensors={name: tensor.detach() for name, tensor in record.tensors.items()},
            expected={
                key: value.detach() if isinstance(value, torch.Tensor) else value
                for key, value in record.expected.items()
            },
            bundle_id=record.bundle_id,
            actor_step_id=record.actor_step_id,
            rollout_id=record.rollout_id,
            producer=record.producer,
            redaction=record.redaction,
            stages=record.stages,
            response_lengths=record.response_lengths,
            total_lengths=record.total_lengths,
            loss_masks_tensor=record.loss_masks_tensor.detach() if record.loss_masks_tensor is not None else None,
            group_indices_tensor=record.group_indices_tensor.detach()
            if record.group_indices_tensor is not None
            else None,
            raw_rewards_tensor=record.raw_rewards_tensor.detach() if record.raw_rewards_tensor is not None else None,
            rewards_tensor=record.rewards_tensor.detach() if record.rewards_tensor is not None else None,
        )
        try:
            self._queue.put_nowait(snapshot)
            return True
        except queue.Full:
            self._dropped_count += 1
            logger.warning(
                "replay capture queue full (capacity=%d); dropping %s (dropped=%d)",
                self._config.queue_capacity,
                record.anchor,
                self._dropped_count,
            )
            return False

    def _writer_loop(self) -> None:
        assert self._queue is not None
        while True:
            record = self._queue.get()
            try:
                if record is None:  # shutdown sentinel
                    return
                build_bundle_from_record(record, self._config.output_dir)
            except Exception as exc:  # noqa: BLE001 — writer failure must never crash training
                self._error_count += 1
                logger.error("replay capture failed for %s: %s", record.anchor, exc)
            finally:
                self._queue.task_done()

    def flush(self, wait: bool = False) -> None:
        """Wait for all currently queued records to be written, if
        requested."""
        if self._queue is not None and wait:
            self._queue.join()

    def close(self, timeout: float | None = 10.0) -> None:
        """Signal the writer thread to stop and join it.

        The shutdown sentinel is enqueued with the same timeout as the join, so
        a stuck writer cannot hang process exit (atexit / actor teardown).
        """
        if self._thread is None or self._queue is None:
            return
        try:
            self._queue.put(None, timeout=timeout)
        except queue.Full:
            logger.warning("replay capture writer did not accept shutdown sentinel")
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning("replay capture writer still running after shutdown timeout; abandoning remaining records")
        self._thread = None


# ---------------------------------------------------------------------------
# Process-global capture state (the thin, hot-path-safe hook surface).
#
# Training files call the tiny functions below; everything heavier lives on the
# ``CaptureManager`` / writer thread. ``active()`` is a single global read and
# ``current_step()`` a single global read — no GPU→CPU sync, no collective.
# ---------------------------------------------------------------------------

_active: CaptureManager | None = None
_current_step: StepCapture | None = None
_atexit_registered: bool = False


def _parse_selected_steps(raw: str | None) -> set[tuple[int, int]] | None:
    """Parse ``RELAX_REPLAY_CAPTURE_STEPS`` (``ROLLOUT_ID:STEP_ID,...``)."""
    if raw is None or not raw.strip():
        return None
    steps: set[tuple[int, int]] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            rollout_id, step_id = token.split(":")
            steps.add((int(rollout_id), int(step_id)))
        except ValueError as exc:
            raise ValueError(
                f"invalid RELAX_REPLAY_CAPTURE_STEPS item {token!r} (expected ROLLOUT_ID:STEP_ID)"
            ) from exc
    return steps or None


def _parse_selected_rollouts(raw: str | None) -> set[int] | None:
    """Parse ``RELAX_REPLAY_CAPTURE_ROLLOUTS`` (``ID,ID,...``)."""
    if raw is None or not raw.strip():
        return None
    rollouts: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            rollouts.add(int(token))
        except ValueError as exc:
            raise ValueError(f"invalid RELAX_REPLAY_CAPTURE_ROLLOUTS item {token!r} (expected int)") from exc
    return rollouts or None


def config_from_env() -> CaptureConfig | None:
    """Build a :class:`CaptureConfig` from ``RELAX_REPLAY_CAPTURE*`` env vars.

    Returns ``None`` when capture is unset/false, so the caller can no-op.
    ``RELAX_REPLAY_CAPTURE=1`` without ``RELAX_REPLAY_CAPTURE_DIR`` logs a
    warning and also returns ``None`` — never write to an implicit path.
    """
    if not Envs.RELAX_REPLAY_CAPTURE:
        return None
    output_dir = Envs.RELAX_REPLAY_CAPTURE_DIR
    if not output_dir:
        logger.warning("RELAX_REPLAY_CAPTURE is set but RELAX_REPLAY_CAPTURE_DIR is empty; capture stays disabled")
        return None
    return CaptureConfig(
        enabled=True,
        output_dir=output_dir,
        selected_steps=_parse_selected_steps(Envs.RELAX_REPLAY_CAPTURE_STEPS),
        selected_rollouts=_parse_selected_rollouts(Envs.RELAX_REPLAY_CAPTURE_ROLLOUTS),
    )


def maybe_enable_from_env(*, rank: int | None = None) -> CaptureManager | None:
    """Enable capture from env vars. No-op when capture is not requested.

    ``rank`` scopes the output directory to ``<dir>/rank-<rank>`` so concurrent
    DP/TP writers do not clobber each other on a shared filesystem.
    """
    config = config_from_env()
    if config is None:
        return None
    if rank is not None:
        config = replace(config, output_dir=str(Path(config.output_dir) / f"rank-{rank}"))
    logger.info("enabling trajectory-replay capture under %s", config.output_dir)
    return enable(config)


def enable(config: CaptureConfig) -> CaptureManager:
    """Enable capture process-wide and return the active manager.

    Replaces any previously active manager (flushing it first). This is the
    Python-API injection point: the training actor calls it from
    :func:`maybe_enable_from_env` — no argument-parsing changes required.
    """
    global _active, _atexit_registered
    if _active is not None:
        _active.close()
    _active = CaptureManager(config)
    if not _atexit_registered:
        atexit.register(disable)
        _atexit_registered = True
    return _active


def disable() -> None:
    """Disable capture process-wide and flush the active manager."""
    global _active
    if _active is not None:
        _active.close()
        _active = None


def active() -> CaptureManager | None:
    """Return the active manager, or ``None`` when capture is disabled."""
    return _active


class StepCapture:
    """Accumulates the data captured for one cohort across hooks.

    Anchored by ``actor_step_id`` (per-step, loss) or ``rollout_id`` (per-
    rollout, reward/advantage). The training-side hooks write tensors /
    expected outputs into this object; ``end_step`` / ``end_rollout`` turn it
    into a :class:`CaptureRecord` and hand it to the manager. All tensor
    detaching happens in ``CaptureManager.submit``, so the hooks themselves
    only hold live references.
    """

    def __init__(
        self,
        identity: Identity,
        config: RecomputeConfig,
        bundle_id: str,
        *,
        actor_step_id: tuple[int, int] | None = None,
        rollout_id: int | None = None,
        producer: ProducerInfo | None = None,
    ) -> None:
        self.actor_step_id = actor_step_id
        self.rollout_id = rollout_id
        self.identity = identity
        self.config = config
        self.bundle_id = bundle_id
        self.producer = producer if producer is not None else ProducerInfo()
        self.samples: list[SampleRecord] = []
        self.tensors: dict[str, torch.Tensor] = {}
        self.expected: dict[str, Any] = {}
        self.stages: set[StageId] | None = None
        self.response_lengths: list[int] | None = None
        self.total_lengths: list[int] | None = None
        self.loss_masks_tensor: torch.Tensor | None = None
        self.group_indices_tensor: torch.Tensor | None = None
        self.raw_rewards_tensor: torch.Tensor | None = None
        self.rewards_tensor: torch.Tensor | None = None

    def to_record(self) -> CaptureRecord:
        return CaptureRecord(
            identity=self.identity,
            samples=self.samples,
            config=self.config,
            tensors=self.tensors,
            expected=self.expected,
            bundle_id=self.bundle_id,
            actor_step_id=self.actor_step_id,
            rollout_id=self.rollout_id,
            producer=self.producer,
            stages=self.stages,
            response_lengths=self.response_lengths,
            total_lengths=self.total_lengths,
            loss_masks_tensor=self.loss_masks_tensor,
            group_indices_tensor=self.group_indices_tensor,
            raw_rewards_tensor=self.raw_rewards_tensor,
            rewards_tensor=self.rewards_tensor,
        )


def begin_step(
    actor_step_id: tuple[int, int],
    *,
    identity: Identity,
    config: RecomputeConfig,
    bundle_id: str,
    producer: ProducerInfo | None = None,
) -> None:
    """Open a per-step capture accumulator (no-op when disabled/unselected)."""
    global _current_step
    if _active is None or not _active.is_selected(actor_step_id):
        _current_step = None
        return
    _current_step = StepCapture(identity, config, bundle_id, actor_step_id=actor_step_id, producer=producer)


def begin_rollout(
    rollout_id: int,
    *,
    identity: Identity,
    config: RecomputeConfig,
    bundle_id: str,
    producer: ProducerInfo | None = None,
) -> None:
    """Open a per-rollout capture accumulator (no-op when
    disabled/unselected)."""
    global _current_step
    if _active is None or not _active.is_rollout_selected(rollout_id):
        _current_step = None
        return
    _current_step = StepCapture(identity, config, bundle_id, rollout_id=rollout_id, producer=producer)


def should_capture(actor_step_id: tuple[int, int]) -> bool:
    """Cheap disabled/unselected guard (two global reads, no allocation).

    Callers use this *before* building an :class:`Identity` /
    :class:`RecomputeConfig` so the disabled hot path does no work at all.
    """
    return _active is not None and _active.is_selected(actor_step_id)


def should_capture_rollout(rollout_id: int) -> bool:
    """Cheap disabled/unselected guard for the rollout-level hot path."""
    return _active is not None and _active.is_rollout_selected(rollout_id)


def current_step() -> StepCapture | None:
    """Return the current cohort's accumulator, or ``None`` when disabled."""
    return _current_step


def end_step() -> None:
    """Finalize the current step's accumulator and submit it to the manager.

    Accumulators that never received a payload (non-last PP ranks, unselected
    hooks) are dropped instead of writing an incomplete bundle.
    """
    global _current_step
    step = _current_step
    _current_step = None
    if step is None or _active is None:
        return
    if not step.stages:
        return
    _active.submit(step.to_record())


def end_rollout() -> None:
    """Finalize the current rollout's accumulator and submit it to the
    manager."""
    end_step()
