# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pipelined weight transport with cross-rank failure synchronization.

Every weight-sync path sends chunk ``N`` while chunk ``N+1`` is still being
converted, then waits on chunk ``N``'s in-flight request before releasing its
tensors. That wait is *not* rank-symmetric: only the IPC gather-source rank
holds the object refs, so an engine-side failure raises there and nowhere else.
Without a shared failure flag the remaining ranks walk into the next chunk's
collectives and block until the distributed timeout.

The helpers here turn any such rank-local error into an all-rank abort by
all-reducing the failure flag over Gloo, and expose one pipelined send loop that
the base, adapter-mode and Mixture-of-LoRA paths all drive.
"""

from collections.abc import Callable, Iterable
from typing import Any

import ray
import torch
import torch.distributed as dist
from ray import ObjectRef

from relax.utils import device as device_utils
from relax.utils.distributed_utils import get_gloo_group
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def run_synchronized_phase(
    operation: Callable[[], Any],
    *,
    description: str = "weight update phase",
) -> tuple[Exception | None, bool]:
    """Run one phase and share its failure state across every rank.

    Returns the local exception (``None`` when this rank succeeded) and whether
    *any* rank failed. All ranks must call this in the same order.
    """

    local_error: Exception | None = None
    try:
        operation()
    except Exception as error:  # noqa: BLE001 - re-raised by the caller after the collective
        local_error = error
        logger.error(
            "%s failed on rank %d before failure synchronization",
            description,
            dist.get_rank(),
            exc_info=(type(error), error, error.__traceback__),
        )
    failed = torch.tensor([local_error is not None], dtype=torch.int32)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=get_gloo_group())
    return local_error, bool(failed.item())


def raise_on_any_rank_failure(
    operation: Callable[[], Any],
    *,
    description: str = "weight update phase",
) -> None:
    """Run ``operation`` and abort on every rank if it failed anywhere."""

    local_error, failed = run_synchronized_phase(operation, description=description)
    if not failed:
        return
    if local_error is not None:
        raise local_error
    raise RuntimeError(f"{description} failed on another rank")


def send_chunks_pipelined(
    chunks: Iterable[Any],
    send_chunk: Callable[[Any], tuple[list[ObjectRef], Any]],
    *,
    description: str = "weight chunk transfer",
    confirm_before: Callable[[Any], bool] | None = None,
) -> None:
    """Send weight chunks, overlapping each transfer with the next conversion.

    ``send_chunk`` converts and ships one chunk and returns its in-flight refs
    plus the tensors that must stay alive until those refs resolve. The wait on
    the previous chunk is deferred to this iteration so transfer and conversion
    overlap, and it is failure-synchronized so a gather-source-only error aborts
    all ranks instead of stranding them in the next collective.

    ``confirm_before`` marks chunks that must not be sent until every preceding
    chunk is confirmed on every rank — the Mixture-of-LoRA path uses it for the
    versioned chunk that publishes the update.

    Collective contract: all ranks must iterate the same number of chunks and
    agree on ``confirm_before``, which the shared chunk iterators guarantee.
    """

    pending_refs: list[ObjectRef] = []
    pending_tensors: Any = None

    def drain(refs: list[ObjectRef]) -> Callable[[], None]:
        def wait() -> None:
            if refs:
                ray.get(refs)

        return wait

    for chunk in chunks:
        if confirm_before is not None and confirm_before(chunk):
            raise_on_any_rank_failure(drain(pending_refs), description=f"preceding {description}")
            del pending_tensors
            pending_refs = []
            pending_tensors = None

        refs, long_lived_tensors = send_chunk(chunk)
        # Confirm the previous chunk on every rank before dropping the tensors
        # that back its shared-memory payload.
        raise_on_any_rank_failure(drain(pending_refs), description=description)
        del pending_tensors
        pending_refs = refs
        pending_tensors = long_lived_tensors
        # Backend-specific per-chunk synchronization lives in device utils so
        # this path stays hardware-agnostic.
        device_utils.maybe_backend_barrier_on_weight_chunk(group=get_gloo_group())

    raise_on_any_rank_failure(drain(pending_refs), description=f"final {description}")
    del pending_tensors
