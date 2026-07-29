# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_SCALAR_TYPES = (str, int, float, bool, type(None))

_RL_ALGOS = {"grpo", "gspo", "sapo", "cispo"}
_PPO_ALGO = "ppo"
_SFT_ALGO = "sft"
KNOWN_ALGOS = sorted([*_RL_ALGOS, _PPO_ALGO, _SFT_ALGO])


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


def resolve_algo_key(args: Any) -> str:
    if getattr(args, "loss_type", None) == _SFT_ALGO:
        return _SFT_ALGO
    return str(getattr(args, "advantage_estimator", "grpo"))


def expected_base_roles(args: Any) -> list[str]:
    if getattr(args, "debug_rollout_only", False):
        return ["rollout"]
    if getattr(args, "debug_train_only", False):
        return ["actor"]
    if getattr(args, "loss_type", None) == _SFT_ALGO:
        return ["sft", "actor"]

    algo_key = resolve_algo_key(args)
    if algo_key == _PPO_ALGO:
        if getattr(args, "fully_async", False):
            roles = ["actor", "critic", "rollout", "advantages", "reference"]
            if not getattr(args, "true_on_policy_mode", False):
                roles.append("actor_fwd")
            return roles
        return ["actor", "critic", "rollout"]

    if getattr(args, "hybrid", False):
        return ["actor", "rollout"]
    if getattr(args, "fully_async", False):
        roles = ["actor", "rollout", "advantages", "reference"]
        if not getattr(args, "true_on_policy_mode", False):
            roles.append("actor_fwd")
        return roles
    return ["actor", "rollout"]


def optional_roles(args: Any) -> list[str]:
    roles: list[str] = []
    if getattr(args, "genrm_model_path", None):
        roles.append("genrm")
    if getattr(args, "loss_type", None) == _SFT_ALGO and getattr(args, "sft_predict_interval", None) is not None:
        roles.append("rollout")
    return roles


def actor_rollout_pg_roles(args: Any) -> list[str]:
    roles = ["actor", "rollout", "genrm"]
    resource = _resource(args)
    if resolve_algo_key(args) == _PPO_ALGO and resource.get("critic") == resource.get("actor"):
        roles.append("critic")
    return roles


def build_topology_plan(args: Any | None) -> dict[str, Any]:
    if args is None:
        return {}

    resource = _resource(args)
    algo_key = resolve_algo_key(args)
    base_roles = expected_base_roles(args)
    extras = optional_roles(args)
    roles = _unique(base_roles + extras)
    role_names = set(roles)
    colocate = bool(getattr(args, "colocate", False) and {"actor", "rollout"}.issubset(role_names))
    pg_roles = actor_rollout_pg_roles(args)

    role_entries = []
    missing_roles = []
    invalid_resources = []
    for role in roles:
        spec = resource.get(role)
        if spec is None:
            missing_roles.append(role)
            role_entries.append({"role": role, "status": "missing_resource"})
            continue
        parsed = _parse_resource_spec(spec)
        if parsed is None:
            invalid_resources.append(role)
            role_entries.append({"role": role, "status": "invalid_resource", "raw": spec})
            continue
        num_serves, num_gpus = parsed
        role_entries.append(
            {
                "role": role,
                "status": "planned",
                "num_serves": num_serves,
                "num_gpus": num_gpus,
                "placement_group": _placement_group_for_role(role, colocate, pg_roles, args),
            }
        )

    resource_summary = _resource_summary(role_entries, colocate, pg_roles)
    data_system = _data_system_plan(args)
    startup = {
        "service_creation": "parallel" if getattr(args, "fully_async", False) else "serial",
        "will_start_ray": False,
        "will_start_sglang": False,
        "will_start_training_workers": False,
    }
    return {
        "algo_key": algo_key,
        "known_algorithms": KNOWN_ALGOS,
        "base_roles": base_roles,
        "optional_roles": extras,
        "roles": roles,
        "role_plan": role_entries,
        "missing_resource_roles": missing_roles,
        "invalid_resource_roles": invalid_resources,
        "colocate": colocate,
        "hybrid": bool(getattr(args, "hybrid", False)),
        "actor_rollout_pg_roles": pg_roles,
        "resource_summary": resource_summary,
        "data_system": data_system,
        "startup": startup,
    }


def _resource(args: Any) -> dict[str, Any]:
    resource = getattr(args, "resource", None)
    return resource if isinstance(resource, dict) else {}


def _parse_resource_spec(spec: Any) -> tuple[int, int] | None:
    if not isinstance(spec, Sequence) or isinstance(spec, (str, bytes, bytearray)):
        return None
    if len(spec) != 2:
        return None
    num_serves, num_gpus = spec
    if not isinstance(num_serves, int) or not isinstance(num_gpus, int):
        return None
    return num_serves, num_gpus


def _placement_group_for_role(role: str, colocate: bool, pg_roles: list[str], args: Any) -> str:
    if colocate and not getattr(args, "hybrid", False) and role in pg_roles:
        return "actor_rollout_shared"
    if role in {"advantages", "sft"}:
        return "none"
    return "dedicated"


def _resource_summary(role_entries: list[dict[str, Any]], colocate: bool, pg_roles: list[str]) -> dict[str, Any]:
    planned = [entry for entry in role_entries if entry.get("status") == "planned"]
    if colocate:
        shared_gpu = max((entry["num_gpus"] for entry in planned if entry["role"] in pg_roles), default=0)
        independent_gpu = sum(entry["num_gpus"] for entry in planned if entry["role"] not in pg_roles)
        total_required = shared_gpu + independent_gpu
    else:
        shared_gpu = 0
        independent_gpu = sum(entry["num_gpus"] for entry in planned)
        total_required = independent_gpu
    return {
        "shared_gpu": shared_gpu,
        "independent_gpu": independent_gpu,
        "total_required_gpus": total_required,
        "role_gpus": {entry["role"]: entry["num_gpus"] for entry in planned},
    }


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
    elif resolve_algo_key(args) == _SFT_ALGO or getattr(args, "balance_data", False):
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


def _unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
