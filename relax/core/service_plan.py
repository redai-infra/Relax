# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


KNOWN_ALGORITHMS = ("cispo", "grpo", "gspo", "ppo", "sapo", "sft")

MODE_TRAIN_ONLY = "train_only"
MODE_ROLLOUT_ONLY = "rollout_only"
MODE_COLOCATE = "colocate"
MODE_SFT = "sft"
MODE_FULLY_ASYNC = "fully_async"
MODE_FULLY_ASYNC_ON_POLICY = "fully_async_on_policy"
MODE_PPO_COLOCATE = "ppo_colocate"
MODE_PPO_FULLY_ASYNC = "ppo_fully_async"
MODE_PPO_FULLY_ASYNC_ON_POLICY = "ppo_fully_async_on_policy"

ROLE_NAMES_BY_MODE = {
    MODE_TRAIN_ONLY: ("actor",),
    MODE_ROLLOUT_ONLY: ("rollout",),
    MODE_COLOCATE: ("actor", "critic", "rollout"),
    MODE_SFT: ("sft", "actor"),
    MODE_FULLY_ASYNC: ("actor", "critic", "rollout", "advantages", "reference", "actor_fwd"),
    MODE_FULLY_ASYNC_ON_POLICY: ("actor", "critic", "rollout", "advantages", "reference"),
    MODE_PPO_COLOCATE: ("actor", "critic", "rollout"),
    MODE_PPO_FULLY_ASYNC: ("actor", "critic", "rollout", "advantages", "reference", "actor_fwd"),
    MODE_PPO_FULLY_ASYNC_ON_POLICY: ("actor", "critic", "rollout", "advantages", "reference"),
}

ALGORITHM_ROLES = {
    "cispo": frozenset(("actor", "rollout", "advantages", "reference", "actor_fwd")),
    "grpo": frozenset(("actor", "rollout", "advantages", "reference", "actor_fwd")),
    "gspo": frozenset(("actor", "rollout", "advantages", "reference", "actor_fwd")),
    "ppo": frozenset(("actor", "critic", "rollout", "advantages", "reference", "actor_fwd")),
    "sapo": frozenset(("actor", "rollout", "advantages", "reference", "actor_fwd")),
    "sft": frozenset(("sft", "actor", "rollout")),
}

CPU_ONLY_ROLES = frozenset(("advantages", "sft"))
ACTOR_ROLLOUT_PG_ROLES = ("actor", "rollout", "genrm")


@dataclass(frozen=True)
class PlanError:
    code: str
    message: str
    role: str | None = None
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        details = {"code": self.code, "message": self.message}
        if self.role is not None:
            details["role"] = self.role
        if self.value is not None:
            details["value"] = self.value
        return details


@dataclass(frozen=True)
class ServiceSpec:
    role: str
    status: str
    required: bool
    num_serves: int | None = None
    num_gpus: int | None = None
    placement_group: str | None = None
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "role": self.role,
            "status": self.status,
            "required": self.required,
        }
        if self.num_serves is not None:
            result["num_serves"] = self.num_serves
        if self.num_gpus is not None:
            result["num_gpus"] = self.num_gpus
        if self.placement_group is not None:
            result["placement_group"] = self.placement_group
        if self.raw is not None:
            result["raw"] = self.raw
        return result


@dataclass(frozen=True)
class ServicePlan:
    algo_key: str
    role_mode: str
    candidate_roles: tuple[str, ...]
    required_roles: tuple[str, ...]
    optional_roles: tuple[str, ...]
    roles: tuple[str, ...]
    service_specs: tuple[ServiceSpec, ...]
    errors: tuple[PlanError, ...]
    colocate: bool
    hybrid: bool
    shares_actor_rollout_pg: bool
    actor_rollout_pg_roles: tuple[str, ...]
    shared_gpu: int
    independent_gpu: int
    total_required_gpus: int
    service_creation: str

    @property
    def planned_specs(self) -> tuple[ServiceSpec, ...]:
        return tuple(spec for spec in self.service_specs if spec.status == "planned")

    def raise_for_errors(self) -> None:
        if not self.errors:
            return
        messages = "; ".join(error.message for error in self.errors)
        raise ValueError(f"Invalid service plan: {messages}")

    def to_dict(self) -> dict[str, Any]:
        missing_roles = [error.role for error in self.errors if error.code == "missing_role" and error.role]
        invalid_roles = [
            error.role
            for error in self.errors
            if error.code in {"resource_shape", "num_serves", "gpu_count", "gpu_required", "cpu_gpu_forbidden"}
            and error.role
        ]
        return {
            "algo_key": self.algo_key,
            "known_algorithms": list(KNOWN_ALGORITHMS),
            "role_mode": self.role_mode,
            "candidate_roles": list(self.candidate_roles),
            "required_roles": list(self.required_roles),
            "optional_roles": list(self.optional_roles),
            "roles": list(self.roles),
            "role_plan": [spec.to_dict() for spec in self.service_specs],
            "plan_errors": [error.to_dict() for error in self.errors],
            "missing_resource_roles": _unique(missing_roles),
            "invalid_resource_roles": _unique(invalid_roles),
            "colocate": self.colocate,
            "hybrid": self.hybrid,
            "shares_actor_rollout_pg": self.shares_actor_rollout_pg,
            "actor_rollout_pg_roles": list(self.actor_rollout_pg_roles),
            "resource_summary": {
                "shared_gpu": self.shared_gpu,
                "independent_gpu": self.independent_gpu,
                "total_required_gpus": self.total_required_gpus,
                "role_gpus": {spec.role: spec.num_gpus for spec in self.planned_specs if spec.num_gpus is not None},
            },
            "startup": {
                "service_creation": self.service_creation,
                "will_start_ray": False,
                "will_start_sglang": False,
                "will_start_training_workers": False,
            },
        }


def resolve_algo_key(config: Any) -> str:
    if getattr(config, "loss_type", None) == "sft":
        return "sft"
    return str(getattr(config, "advantage_estimator", "grpo"))


def resolve_role_mode(config: Any) -> str:
    if getattr(config, "debug_rollout_only", False):
        return MODE_ROLLOUT_ONLY
    if getattr(config, "debug_train_only", False):
        return MODE_TRAIN_ONLY
    if getattr(config, "loss_type", None) == "sft":
        return MODE_SFT
    if getattr(config, "advantage_estimator", None) == "ppo":
        if getattr(config, "fully_async", False):
            if getattr(config, "true_on_policy_mode", False):
                return MODE_PPO_FULLY_ASYNC_ON_POLICY
            return MODE_PPO_FULLY_ASYNC
        return MODE_PPO_COLOCATE
    if getattr(config, "hybrid", False):
        return MODE_COLOCATE
    if getattr(config, "fully_async", False):
        if getattr(config, "true_on_policy_mode", False):
            return MODE_FULLY_ASYNC_ON_POLICY
        return MODE_FULLY_ASYNC
    return MODE_COLOCATE


def resolve_base_roles(config: Any) -> list[str]:
    algo_key = resolve_algo_key(config)
    supported = ALGORITHM_ROLES.get(algo_key, frozenset())
    return [role for role in ROLE_NAMES_BY_MODE[resolve_role_mode(config)] if role in supported]


def resolve_optional_roles(config: Any) -> list[str]:
    roles = []
    if getattr(config, "genrm_model_path", None):
        roles.append("genrm")
    if getattr(config, "loss_type", None) == "sft" and getattr(config, "sft_predict_interval", None) is not None:
        roles.append("rollout")
    return roles


def actor_rollout_pg_roles(config: Any) -> list[str]:
    roles = list(ACTOR_ROLLOUT_PG_ROLES)
    resource = _resource(config)
    if resolve_algo_key(config) == "ppo" and resource.get("critic") == resource.get("actor"):
        roles.append("critic")
    return roles


def is_managed_teacher_enabled(config: Any) -> bool:
    return (
        getattr(config, "use_opd", False)
        and getattr(config, "opd_type", None) == "sglang"
        and (
            getattr(config, "teacher_hf_checkpoint", None) is not None
            or getattr(config, "opd_teacher_routes", None) is not None
        )
        and isinstance(getattr(config, "resource", None), Mapping)
        and "teacher" in config.resource
    )


def should_start_managed_teacher(config: Any) -> bool:
    return is_managed_teacher_enabled(config) and not getattr(config, "debug_train_only", False)


def is_managed_teacher_colocate(config: Any) -> bool:
    return (
        should_start_managed_teacher(config)
        and getattr(config, "colocate", False)
        and not getattr(config, "hybrid", False)
        and "actor" in config.resource
        and "rollout" in config.resource
    )


def build_service_plan(config: Any) -> ServicePlan:
    algo_key = resolve_algo_key(config)
    role_mode = resolve_role_mode(config)
    base_roles = resolve_base_roles(config)
    optional_roles = resolve_optional_roles(config)
    candidate_roles = _unique([*base_roles, *optional_roles])
    required_roles = list(candidate_roles)
    if "reference" in required_roles and not _needs_reference(config):
        required_roles.remove("reference")

    managed_teacher = should_start_managed_teacher(config)
    if managed_teacher:
        candidate_roles.append("teacher")
        required_roles.append("teacher")

    errors = []
    if algo_key not in ALGORITHM_ROLES:
        errors.append(
            PlanError(
                code="unsupported_algorithm",
                message=f"algorithm key {algo_key!r} is not registered; expected one of {list(KNOWN_ALGORITHMS)}.",
                value=algo_key,
            )
        )

    resource = _resource(config)
    if not isinstance(getattr(config, "resource", None), Mapping):
        errors.append(
            PlanError(
                code="resource_required",
                message="--resource is missing or is not a mapping.",
                value=getattr(config, "resource", None),
            )
        )

    roles = _unique([role for role in candidate_roles if role in resource or role in required_roles])
    role_names = set(roles)
    colocate = bool(getattr(config, "colocate", False) and {"actor", "rollout"}.issubset(role_names))
    hybrid = bool(getattr(config, "hybrid", False))
    shares_actor_rollout_pg = colocate and not (getattr(config, "fully_async", False) or hybrid)
    pg_roles = actor_rollout_pg_roles(config)

    specs = []
    for role in roles:
        required = role in required_roles
        raw = resource.get(role)
        if raw is None:
            if required:
                errors.append(
                    PlanError(
                        code="missing_role",
                        message=f"--resource is missing required role {role!r}.",
                        role=role,
                    )
                )
                specs.append(ServiceSpec(role=role, status="missing_resource", required=True))
            continue

        parsed = _parse_resource_spec(raw)
        if parsed is None:
            errors.append(
                PlanError(
                    code="resource_shape",
                    message=f"resource entry for role {role!r} must be [num_serves, num_gpus].",
                    role=role,
                    value=raw,
                )
            )
            specs.append(ServiceSpec(role=role, status="invalid_resource", required=required, raw=raw))
            continue

        num_serves, num_gpus = parsed
        role_errors = []
        if num_serves != 1:
            role_errors.append(
                PlanError(
                    code="num_serves",
                    message=f"role {role!r} requires num_serves=1, got {num_serves}.",
                    role=role,
                    value=raw,
                )
            )
        if num_gpus < 0:
            role_errors.append(
                PlanError(
                    code="gpu_count",
                    message=f"role {role!r} requires num_gpus >= 0, got {num_gpus}.",
                    role=role,
                    value=raw,
                )
            )
        elif num_gpus > 0 and role in CPU_ONLY_ROLES:
            role_errors.append(
                PlanError(
                    code="cpu_gpu_forbidden",
                    message=f"CPU-only role {role!r} requires num_gpus=0, got {num_gpus}.",
                    role=role,
                    value=raw,
                )
            )
        elif num_gpus == 0 and role not in CPU_ONLY_ROLES:
            role_errors.append(
                PlanError(
                    code="gpu_required",
                    message=f"model role {role!r} requires num_gpus > 0.",
                    role=role,
                    value=raw,
                )
            )

        if role_errors:
            errors.extend(role_errors)
            specs.append(
                ServiceSpec(
                    role=role,
                    status="invalid_resource",
                    required=required,
                    num_serves=num_serves,
                    num_gpus=num_gpus,
                    raw=raw,
                )
            )
            continue

        specs.append(
            ServiceSpec(
                role=role,
                status="planned",
                required=required,
                num_serves=num_serves,
                num_gpus=num_gpus,
                placement_group=_placement_group(
                    role,
                    shares_actor_rollout_pg=shares_actor_rollout_pg,
                    actor_rollout_pg_roles=pg_roles,
                    managed_teacher_colocate=is_managed_teacher_colocate(config),
                ),
            )
        )

    planned = [spec for spec in specs if spec.status == "planned"]
    shared_gpu = max(
        (spec.num_gpus or 0 for spec in planned if spec.placement_group == "actor_rollout_shared"),
        default=0,
    )
    independent_gpu = sum(spec.num_gpus or 0 for spec in planned if spec.placement_group == "dedicated")

    return ServicePlan(
        algo_key=algo_key,
        role_mode=role_mode,
        candidate_roles=tuple(candidate_roles),
        required_roles=tuple(required_roles),
        optional_roles=tuple(optional_roles),
        roles=tuple(roles),
        service_specs=tuple(specs),
        errors=tuple(errors),
        colocate=colocate,
        hybrid=hybrid,
        shares_actor_rollout_pg=shares_actor_rollout_pg,
        actor_rollout_pg_roles=tuple(pg_roles),
        shared_gpu=shared_gpu,
        independent_gpu=independent_gpu,
        total_required_gpus=shared_gpu + independent_gpu,
        service_creation=("parallel" if getattr(config, "fully_async", False) or hybrid else "serial"),
    )


def _resource(config: Any) -> dict[str, Any]:
    resource = getattr(config, "resource", None)
    return dict(resource) if isinstance(resource, Mapping) else {}


def _parse_resource_spec(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    if len(value) != 2:
        return None
    num_serves, num_gpus = value
    if type(num_serves) is not int or type(num_gpus) is not int:
        return None
    return num_serves, num_gpus


def _needs_reference(config: Any) -> bool:
    return bool(getattr(config, "use_kl_loss", False) or getattr(config, "kl_coef", 0) != 0)


def _placement_group(
    role: str,
    *,
    shares_actor_rollout_pg: bool,
    actor_rollout_pg_roles: list[str],
    managed_teacher_colocate: bool,
) -> str:
    if role in CPU_ONLY_ROLES:
        return "none"
    if role == "teacher" and managed_teacher_colocate:
        return "actor_rollout_shared"
    if shares_actor_rollout_pg and role in actor_rollout_pg_roles:
        return "actor_rollout_shared"
    return "dedicated"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
