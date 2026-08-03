# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from relax.core.service_plan import build_service_plan, resolve_algo_key


_SCALAR_TYPES = (str, int, float, bool, type(None))


def serialize_config(args: Any | None) -> dict[str, Any]:
    if args is None:
        return {}
    values = vars(args) if hasattr(args, "__dict__") else {}
    return {key: _to_jsonable(value) for key, value in sorted(values.items())}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, _SCALAR_TYPES):
        return value
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_jsonable(item) for item in value]
    return repr(value)


def build_topology_plan(args: Any | None) -> dict[str, Any]:
    if args is None:
        return {}

    topology = build_service_plan(args).to_dict()
    topology["data_system"] = _data_system_plan(args)
    return topology


def _data_system_plan(args: Any) -> dict[str, Any]:
    rollout_batch_size = getattr(args, "rollout_batch_size", None)
    over_sampling_batch_size = getattr(args, "over_sampling_batch_size", None) or rollout_batch_size
    batch_size_for_capacity = (
        over_sampling_batch_size
        if getattr(args, "partial_rollout", False) and getattr(args, "use_dynamic_global_batch_size", False)
        else rollout_batch_size
    )
    if getattr(args, "fully_async", False) and getattr(args, "use_dynamic_batch_size", False):
        sampler = "StreamingTokenBudgetSampler"
    elif resolve_algo_key(args) == "sft" or getattr(args, "balance_data", False):
        sampler = "SeqlenBalancedSampler"
    else:
        sampler = "GRPOGroupNSampler"

    storage_size = None
    if batch_size_for_capacity is not None:
        storage_size = (
            batch_size_for_capacity
            * (getattr(args, "max_staleness", 0) + 1)
            * getattr(args, "n_samples_per_prompt", 1)
        )
    return {
        "sampler": sampler,
        "batch_size_for_capacity": batch_size_for_capacity,
        "total_storage_size": storage_size,
        "num_data_storage_units": getattr(args, "num_data_storage_units", None),
    }
