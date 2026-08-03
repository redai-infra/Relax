# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import ast
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from relax.core.service_plan import CPU_ONLY_ROLES
from relax.utils.doctor.models import DiagnosticResult, DoctorContext


RuleFn = Callable[[DoctorContext], list[DiagnosticResult]]
_GENERALIZED_PATH_RE = re.compile(r"^(?P<path>.*)@\[-?\d*:-?\d*\]$")


@dataclass(frozen=True)
class DiagnosticRule:
    rule_id: str
    title: str
    check: RuleFn
    supports_partial: bool = False


_RULES: list[DiagnosticRule] = []


def diagnostic_rule(rule_id: str, title: str, *, supports_partial: bool = False) -> Callable[[RuleFn], RuleFn]:
    def decorator(func: RuleFn) -> RuleFn:
        _RULES.append(
            DiagnosticRule(
                rule_id=rule_id,
                title=title,
                check=func,
                supports_partial=supports_partial,
            )
        )
        return func

    return decorator


def get_rules() -> list[DiagnosticRule]:
    return list(_RULES)


def _result(
    rule_id: str,
    message: str,
    fix: str,
    *,
    details: dict[str, Any] | None = None,
    severity: str = "error",
) -> DiagnosticResult:
    return DiagnosticResult(
        rule_id=rule_id,
        severity=severity,  # type: ignore[arg-type]
        message=message,
        fix=fix,
        details=details or {},
    )


def _args(ctx: DoctorContext) -> Any | None:
    return ctx.args


def _resource(args: Any) -> dict[str, Any]:
    resource = getattr(args, "resource", None)
    return resource if isinstance(resource, dict) else {}


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and value > 0


def _has_path(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _looks_like_dotted_import(value: str) -> bool:
    parts = value.split(".")
    return len(parts) > 1 and all(part.isidentifier() for part in parts)


def _dataset_paths(value: Any) -> list[str]:
    values = value if isinstance(value, (list, tuple)) else [value]
    paths = []
    for item in values:
        text = str(item).strip()
        if match := _GENERALIZED_PATH_RE.match(text):
            text = match.group("path").strip()
        if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = [part.strip().strip("\"'") for part in text[1:-1].split(",") if part.strip()]
            if isinstance(parsed, (list, tuple)):
                paths.extend(str(path).strip() for path in parsed)
                continue
        paths.append(text)
    return paths


@diagnostic_rule("CONFIG_PARSE_ERROR", "Training arguments can be parsed", supports_partial=True)
def check_parse_error(ctx: DoctorContext) -> list[DiagnosticResult]:
    if ctx.parse_error is None:
        return []
    return [
        _result(
            "CONFIG_PARSE_ERROR",
            f"failed to parse or normalize training arguments: {ctx.parse_error}",
            "Fix the CLI arguments first. Run with the same arguments used by relax.entrypoints.train.",
        )
    ]


@diagnostic_rule("CONFIG_RESOURCE_REQUIRED", "Resource map is present", supports_partial=True)
def check_resource_required(ctx: DoctorContext) -> list[DiagnosticResult]:
    args = _args(ctx)
    if args is None:
        return []
    if not isinstance(getattr(args, "resource", None), dict):
        return [
            _result(
                "CONFIG_RESOURCE_REQUIRED",
                "--resource is missing or is not a JSON object.",
                'Pass --resource as JSON, for example: \'{"actor": [1, 8], "rollout": [1, 8]}\'.',
            )
        ]
    return []


@diagnostic_rule(
    "CONFIG_RESOURCE_SHAPE",
    "Each resource entry has one serve and non-negative GPUs",
    supports_partial=True,
)
def check_resource_shape(ctx: DoctorContext) -> list[DiagnosticResult]:
    args = _args(ctx)
    if args is None:
        return []
    if ctx.config_state == "validated":
        resource_errors = [
            error
            for error in ctx.topology.get("plan_errors", [])
            if error.get("code") in {"resource_shape", "num_serves", "gpu_count", "gpu_required"}
        ]
        return [
            _result(
                "CONFIG_RESOURCE_SHAPE",
                error["message"],
                "Use [1, 0] only for CPU roles; model roles require [1, <positive num_gpus>].",
                details=error,
            )
            for error in resource_errors
        ]

    diagnostics = []
    for role, spec in _resource(args).items():
        if not isinstance(spec, Sequence) or isinstance(spec, (str, bytes, bytearray)) or len(spec) != 2:
            diagnostics.append(
                _result(
                    "CONFIG_RESOURCE_SHAPE",
                    f"resource entry for role {role!r} must be [num_serves, num_gpus], got {spec!r}.",
                    'Use a two-element integer list for every role, e.g. "actor": [1, 8].',
                    details={"role": role, "value": spec},
                )
            )
            continue
        num_serves, num_gpus = spec
        if type(num_serves) is not int or type(num_gpus) is not int or num_gpus < 0:
            diagnostics.append(
                _result(
                    "CONFIG_RESOURCE_SHAPE",
                    f"resource entry for role {role!r} must contain integers and num_gpus >= 0.",
                    "Use integer values and keep CPU-only roles at [1, 0].",
                    details={"role": role, "value": spec},
                )
            )
        elif num_gpus == 0 and role not in CPU_ONLY_ROLES:
            diagnostics.append(
                _result(
                    "CONFIG_RESOURCE_SHAPE",
                    f"model role {role!r} requires num_gpus > 0.",
                    "Assign at least one GPU to model roles; only CPU roles may use [1, 0].",
                    details={"role": role, "value": spec},
                )
            )
        elif num_serves != 1:
            diagnostics.append(
                _result(
                    "CONFIG_RESOURCE_SHAPE",
                    f"role {role!r} has num_serves={num_serves}; Relax currently supports only one serve per role.",
                    "Set num_serves to 1 for this role.",
                    details={"role": role, "value": spec},
                )
            )
    return diagnostics


@diagnostic_rule("CONFIG_ALGORITHM_SUPPORTED", "Algorithm key is registered")
def check_algorithm_supported(ctx: DoctorContext) -> list[DiagnosticResult]:
    args = _args(ctx)
    if args is None:
        return []
    algo_key = ctx.topology.get("algo_key")
    known = set(ctx.topology.get("known_algorithms", []))
    if algo_key not in known:
        return [
            _result(
                "CONFIG_ALGORITHM_SUPPORTED",
                f"algorithm key {algo_key!r} is not registered in the dry-run topology table.",
                f"Use one of {sorted(known)} or add the new algorithm to the doctor topology registry.",
                details={"algo_key": algo_key, "known_algorithms": sorted(known)},
            )
        ]
    return []


@diagnostic_rule("CONFIG_REQUIRED_ROLES", "Required roles are present in resource map")
def check_required_roles(ctx: DoctorContext) -> list[DiagnosticResult]:
    args = _args(ctx)
    if args is None:
        return []
    missing = list(ctx.topology.get("missing_resource_roles", []))
    if not missing:
        return []
    return [
        _result(
            "CONFIG_REQUIRED_ROLES",
            f"--resource is missing required role(s): {missing}.",
            "Add all planned roles to --resource, or change the training mode so those roles are not required.",
            details={"missing_roles": missing, "planned_roles": ctx.topology.get("roles", [])},
        )
    ]


@diagnostic_rule("CONFIG_MODE_CONFLICT", "Training mode flags are mutually consistent", supports_partial=True)
def check_mode_conflict(ctx: DoctorContext) -> list[DiagnosticResult]:
    args = _args(ctx)
    if args is None:
        return []
    if getattr(args, "fully_async", False) and getattr(args, "colocate", False) and not getattr(args, "hybrid", False):
        return [
            _result(
                "CONFIG_MODE_CONFLICT",
                "--fully-async and --colocate cannot be combined directly.",
                "Use --hybrid for the supported hybrid mode, or remove one of --fully-async / --colocate.",
            )
        ]
    return []


@diagnostic_rule("CONFIG_DEBUG_MODE_CONFLICT", "Debug-only modes do not conflict", supports_partial=True)
def check_debug_mode_conflict(ctx: DoctorContext) -> list[DiagnosticResult]:
    args = _args(ctx)
    if args is None:
        return []
    if getattr(args, "debug_rollout_only", False) and getattr(args, "debug_train_only", False):
        return [
            _result(
                "CONFIG_DEBUG_MODE_CONFLICT",
                "--debug-rollout-only and --debug-train-only are both enabled.",
                "Pick only one debug mode.",
            )
        ]
    return []


@diagnostic_rule("CONFIG_PPO_TOPOLOGY", "PPO topology is supported")
def check_ppo_topology(ctx: DoctorContext) -> list[DiagnosticResult]:
    args = _args(ctx)
    if args is None or getattr(args, "advantage_estimator", None) != "ppo":
        return []
    diagnostics = []
    resource = _resource(args)
    if "critic" not in resource:
        diagnostics.append(
            _result(
                "CONFIG_PPO_TOPOLOGY",
                "--advantage-estimator ppo requires a 'critic' entry in --resource.",
                'Add "critic": [1, <num_gpus>] to --resource.',
            )
        )
    if getattr(args, "fully_async", False) or getattr(args, "hybrid", False):
        diagnostics.append(
            _result(
                "CONFIG_PPO_TOPOLOGY",
                "PPO does not currently support --fully-async or --hybrid.",
                "Run PPO in synchronous colocate mode.",
            )
        )
    if getattr(args, "colocate", False) and getattr(args, "max_staleness", 0) != 0:
        diagnostics.append(
            _result(
                "CONFIG_PPO_TOPOLOGY",
                "Synchronous colocate PPO requires --max-staleness 0.",
                "Set --max-staleness 0 for PPO colocate training.",
            )
        )
    return diagnostics


@diagnostic_rule("CONFIG_SFT_REQUIREMENTS", "SFT-specific options are complete")
def check_sft_requirements(ctx: DoctorContext) -> list[DiagnosticResult]:
    args = _args(ctx)
    if args is None or getattr(args, "loss_type", None) != "sft":
        return []
    diagnostics = []
    if not getattr(args, "custom_dataset_class_path", None) and not getattr(args, "prompt_data", None):
        diagnostics.append(
            _result(
                "CONFIG_SFT_REQUIREMENTS",
                "--loss-type sft requires --prompt-data unless --custom-dataset-class-path is set.",
                "Set --prompt-data to an SFT dataset path, or provide --custom-dataset-class-path.",
            )
        )
    if not getattr(args, "use_dynamic_batch_size", False):
        diagnostics.append(
            _result(
                "CONFIG_SFT_REQUIREMENTS",
                "--loss-type sft requires --use-dynamic-batch-size.",
                "Enable --use-dynamic-batch-size and set --max-tokens-per-gpu.",
            )
        )
    if getattr(args, "sft_oversize_strategy", None) == "custom" and not getattr(
        args, "sft_oversize_custom_function_path", None
    ):
        diagnostics.append(
            _result(
                "CONFIG_SFT_REQUIREMENTS",
                "--sft-oversize-strategy custom requires --sft-oversize-custom-function-path.",
                "Provide the custom oversize function path or use a built-in strategy.",
            )
        )
    if getattr(args, "sft_predict_interval", None) is not None:
        has_eval_source = bool(getattr(args, "eval_prompt_data", None)) or getattr(args, "eval_size", None) is not None
        if not getattr(args, "save", None) or not has_eval_source:
            diagnostics.append(
                _result(
                    "CONFIG_SFT_REQUIREMENTS",
                    "--sft-predict-interval requires --save and an eval source.",
                    "Set --save and either --eval-prompt-data or --eval-size.",
                )
            )
    return diagnostics


@diagnostic_rule("CONFIG_BATCH_SIZE", "Batch-size fields are internally consistent", supports_partial=True)
def check_batch_size(ctx: DoctorContext) -> list[DiagnosticResult]:
    args = _args(ctx)
    if args is None:
        return []
    diagnostics = []
    rollout_batch_size = getattr(args, "rollout_batch_size", None)
    global_batch_size = getattr(args, "global_batch_size", None)
    n_samples_per_prompt = getattr(args, "n_samples_per_prompt", 1)
    if rollout_batch_size is None and global_batch_size is None:
        diagnostics.append(
            _result(
                "CONFIG_BATCH_SIZE",
                "Either --rollout-batch-size or --global-batch-size must be set.",
                "Set --rollout-batch-size directly, or set --global-batch-size with --n-samples-per-prompt.",
            )
        )
    num_steps = getattr(args, "num_steps_per_rollout", None)
    if rollout_batch_size is not None and num_steps is not None:
        if not _is_positive_int(num_steps):
            diagnostics.append(
                _result(
                    "CONFIG_BATCH_SIZE",
                    "num_steps_per_rollout must be a positive integer.",
                    "Set --num-steps-per-rollout to an integer greater than zero.",
                    details={"num_steps_per_rollout": num_steps},
                )
            )
            return diagnostics
        expected = rollout_batch_size * n_samples_per_prompt // num_steps
        if global_batch_size is not None and global_batch_size != expected:
            diagnostics.append(
                _result(
                    "CONFIG_BATCH_SIZE",
                    f"global_batch_size={global_batch_size} does not match rollout batch formula {expected}.",
                    "Set global_batch_size = rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout.",
                    details={"expected_global_batch_size": expected},
                )
            )
    return diagnostics


@diagnostic_rule("CONFIG_ROLLOUT_COUNT", "Rollout count is bounded")
def check_rollout_count(ctx: DoctorContext) -> list[DiagnosticResult]:
    args = _args(ctx)
    if args is None:
        return []
    if getattr(args, "num_epoch", None) is None and getattr(args, "num_rollout", None) is None:
        return [
            _result(
                "CONFIG_ROLLOUT_COUNT",
                "Neither --num-rollout nor --num-epoch is set.",
                "Set at least one of --num-rollout or --num-epoch.",
            )
        ]
    if getattr(args, "num_epoch", None) is not None and getattr(args, "loss_type", None) != "sft":
        if not getattr(args, "rollout_global_dataset", True):
            return [
                _result(
                    "CONFIG_ROLLOUT_COUNT",
                    "--num-epoch requires rollout_global_dataset for RL training.",
                    "Remove --disable-rollout-global-dataset or use --num-rollout.",
                )
            ]
    return []


@diagnostic_rule("CONFIG_OVERSAMPLING", "Over-sampling batch size is valid")
def check_oversampling(ctx: DoctorContext) -> list[DiagnosticResult]:
    args = _args(ctx)
    if args is None:
        return []
    rollout_batch_size = getattr(args, "rollout_batch_size", None)
    over_sampling_batch_size = getattr(args, "over_sampling_batch_size", None)
    if rollout_batch_size is None or over_sampling_batch_size is None:
        return []
    if over_sampling_batch_size < rollout_batch_size:
        return [
            _result(
                "CONFIG_OVERSAMPLING",
                "over_sampling_batch_size must be greater than or equal to rollout_batch_size.",
                "Increase --over-sampling-batch-size or lower --rollout-batch-size.",
                details={
                    "over_sampling_batch_size": over_sampling_batch_size,
                    "rollout_batch_size": rollout_batch_size,
                },
            )
        ]
    if (
        over_sampling_batch_size > rollout_batch_size
        and not getattr(args, "fully_async", False)
        and not getattr(args, "partial_rollout", False)
    ):
        return [
            _result(
                "CONFIG_OVERSAMPLING",
                "over-sampled surplus will be discarded without fully_async or partial_rollout.",
                "Enable --fully-async / --partial-rollout, or set over_sampling_batch_size == rollout_batch_size.",
                severity="warning",
            )
        ]
    return []


@diagnostic_rule("CONFIG_DYNAMIC_BATCH", "Dynamic batch size has token budget")
def check_dynamic_batch(ctx: DoctorContext) -> list[DiagnosticResult]:
    args = _args(ctx)
    if args is None or not getattr(args, "use_dynamic_batch_size", False):
        return []
    if getattr(args, "max_tokens_per_gpu", None) is None:
        return [
            _result(
                "CONFIG_DYNAMIC_BATCH",
                "--use-dynamic-batch-size requires --max-tokens-per-gpu.",
                "Set --max-tokens-per-gpu to the per-GPU token budget.",
            )
        ]
    return []


@diagnostic_rule("CONFIG_CONTEXT_LENGTH", "Context-length settings can fit the token budget")
def check_context_length(ctx: DoctorContext) -> list[DiagnosticResult]:
    args = _args(ctx)
    if args is None:
        return []
    diagnostics = []
    max_context = getattr(args, "rollout_max_context_len", None)
    max_prompt = getattr(args, "rollout_max_prompt_len", None)
    if max_context is not None and max_prompt is not None and max_prompt > max_context - 1:
        diagnostics.append(
            _result(
                "CONFIG_CONTEXT_LENGTH",
                "rollout_max_prompt_len must leave at least one token for generation.",
                "Set rollout_max_prompt_len <= rollout_max_context_len - 1.",
                details={"rollout_max_prompt_len": max_prompt, "rollout_max_context_len": max_context},
            )
        )
    if getattr(args, "use_dynamic_batch_size", False) and max_context is not None:
        token_budget = getattr(args, "max_tokens_per_gpu", None)
        cp_size = getattr(args, "context_parallel_size", 1)
        if (
            token_budget is not None
            and token_budget * cp_size < max_context
            and not getattr(args, "dynamic_context_parallel", False)
        ):
            diagnostics.append(
                _result(
                    "CONFIG_CONTEXT_LENGTH",
                    "max_tokens_per_gpu * context_parallel_size is smaller than rollout_max_context_len.",
                    "Increase --max-tokens-per-gpu / --context-parallel-size, or reduce rollout_max_context_len.",
                    details={"token_budget": token_budget * cp_size, "rollout_max_context_len": max_context},
                )
            )
    return diagnostics


@diagnostic_rule("CONFIG_SGLANG_PARALLEL", "SGLang parallelism is internally consistent")
def check_sglang_parallel(ctx: DoctorContext) -> list[DiagnosticResult]:
    args = _args(ctx)
    if args is None or getattr(args, "debug_train_only", False):
        return []
    diagnostics = []
    pp_size = getattr(args, "sglang_pp_size", getattr(args, "sglang_pipeline_parallel_size", 1))
    rollout_num_gpus_per_engine = getattr(args, "rollout_num_gpus_per_engine", 1)
    if pp_size and rollout_num_gpus_per_engine % pp_size != 0:
        diagnostics.append(
            _result(
                "CONFIG_SGLANG_PARALLEL",
                "rollout_num_gpus_per_engine must be divisible by SGLang pipeline parallel size.",
                "Adjust --rollout-num-gpus-per-engine or --sglang-pipeline-parallel-size.",
                details={"rollout_num_gpus_per_engine": rollout_num_gpus_per_engine, "sglang_pp_size": pp_size},
            )
        )
    if getattr(args, "sglang_data_parallel_size", 1) > 1 and not getattr(args, "sglang_enable_dp_attention", False):
        diagnostics.append(
            _result(
                "CONFIG_SGLANG_PARALLEL",
                "sglang_data_parallel_size > 1 requires --sglang-enable-dp-attention.",
                "Enable --sglang-enable-dp-attention or set data parallel size to 1.",
            )
        )
    if getattr(args, "prefill_num_servers", None) is not None and getattr(args, "rollout_external", False):
        diagnostics.append(
            _result(
                "CONFIG_SGLANG_PARALLEL",
                "prefill_num_servers cannot be set when rollout_external is set.",
                "Remove --prefill-num-servers or disable external rollout.",
            )
        )
    if getattr(args, "sglang_config", None) is not None and getattr(args, "prefill_num_servers", None) is not None:
        diagnostics.append(
            _result(
                "CONFIG_SGLANG_PARALLEL",
                "sglang_config and prefill_num_servers are mutually exclusive.",
                "Use engine_groups in the SGLang YAML config instead of --prefill-num-servers.",
            )
        )
    return diagnostics


@diagnostic_rule("CONFIG_EVAL", "Evaluation configuration is complete")
def check_eval(ctx: DoctorContext) -> list[DiagnosticResult]:
    args = _args(ctx)
    if args is None:
        return []
    diagnostics = []
    if getattr(args, "loss_type", None) == "sft" and getattr(args, "eval_config", None) is not None:
        diagnostics.append(
            _result(
                "CONFIG_EVAL",
                "--loss-type sft uses --eval-prompt-data for eval; --eval-config is not supported.",
                "Replace --eval-config with --eval-prompt-data for SFT.",
            )
        )
    if getattr(args, "eval_size", None) is not None and getattr(args, "loss_type", None) != "sft":
        diagnostics.append(
            _result(
                "CONFIG_EVAL",
                "--eval-size is only meaningful under --loss-type sft.",
                "Use --eval-config / --eval-prompt-data for RL eval, or switch to SFT.",
            )
        )
    if getattr(args, "eval_interval", None) is not None and getattr(args, "loss_type", None) != "sft":
        if not getattr(args, "eval_datasets", None) and not getattr(args, "eval_prompt_data", None):
            diagnostics.append(
                _result(
                    "CONFIG_EVAL",
                    "Evaluation datasets must be configured when eval_interval is set.",
                    "Provide --eval-config or --eval-prompt-data.",
                )
            )
    return diagnostics


@diagnostic_rule("CONFIG_SAVE", "Save-related options have required paths")
def check_save(ctx: DoctorContext) -> list[DiagnosticResult]:
    args = _args(ctx)
    if args is None:
        return []
    if getattr(args, "save_interval", None) is not None and not getattr(args, "save", None):
        return [
            _result(
                "CONFIG_SAVE",
                "--save is required when --save-interval is set.",
                "Set --save to the checkpoint output directory, or remove --save-interval.",
            )
        ]
    return []


@diagnostic_rule("CONFIG_GENRM_COLOCATE", "GenRM colocate GPU split is valid")
def check_genrm_colocate(ctx: DoctorContext) -> list[DiagnosticResult]:
    args = _args(ctx)
    if args is None or not getattr(args, "colocate", False) or not getattr(args, "genrm_model_path", None):
        return []
    rollout_g = getattr(args, "rollout_num_gpus", None)
    genrm_g = getattr(args, "genrm_num_gpus", 1)
    actor_total = getattr(args, "actor_num_gpus_per_node", 1) * getattr(args, "actor_num_nodes", 1)
    if rollout_g is None:
        return [
            _result(
                "CONFIG_GENRM_COLOCATE",
                "When GenRM is enabled in colocated mode, --rollout-num-gpus must be set.",
                "Set --rollout-num-gpus and --genrm-num-gpus so they split or share actor GPUs.",
            )
        ]
    if not (rollout_g + genrm_g == actor_total or (rollout_g == actor_total and genrm_g == actor_total)):
        return [
            _result(
                "CONFIG_GENRM_COLOCATE",
                "Invalid GenRM colocate GPU allocation.",
                "Use split allocation rollout + genrm == actor total, or shared allocation rollout == genrm == actor total.",
                details={"rollout_num_gpus": rollout_g, "genrm_num_gpus": genrm_g, "actor_total_gpus": actor_total},
            )
        ]
    return []


@diagnostic_rule("CONFIG_PATHS", "Local paths referenced by the config exist", supports_partial=True)
def check_paths(ctx: DoctorContext) -> list[DiagnosticResult]:
    args = _args(ctx)
    if args is None:
        return []
    diagnostics = []
    filesystem_fields = [
        "custom_config_path",
        "eval_config",
    ]
    import_or_path_fields = [
        "data_source_path",
        "rollout_function_path",
        "eval_function_path",
        "custom_reward_function_path",
        "custom_dataset_class_path",
    ]
    for field in filesystem_fields:
        value = getattr(args, field, None)
        if not _has_path(value):
            continue
        path = os.path.expanduser(value)
        if not os.path.exists(path):
            diagnostics.append(
                _result(
                    "CONFIG_PATHS",
                    f"{field} points to a missing path: {value!r}.",
                    "Fix the path or mount the required data/config file.",
                    details={"field": field, "path": value},
                )
            )
    prompt_data = getattr(args, "prompt_data", None)
    if prompt_data:
        for path in _dataset_paths(prompt_data):
            expanded_path = os.path.expanduser(path)
            if not os.path.exists(expanded_path):
                diagnostics.append(
                    _result(
                        "CONFIG_PATHS",
                        f"prompt_data contains a missing path: {path!r}.",
                        "Fix the path or mount the required dataset file or directory.",
                        details={"field": "prompt_data", "path": path, "path_spec": prompt_data},
                    )
                )
    for field in import_or_path_fields:
        value = getattr(args, field, None)
        if not _has_path(value):
            continue
        if _looks_like_dotted_import(value):
            # Dotted import paths are resolved at runtime; do not treat them as filesystem paths.
            continue
        path = os.path.expanduser(value)
        if not os.path.exists(path):
            diagnostics.append(
                _result(
                    "CONFIG_PATHS",
                    f"{field} points to a missing path: {value!r}.",
                    "Fix the path, mount the required data/config file, or use a valid Python dotted import path.",
                    details={"field": field, "path": value},
                )
            )
    ref_load = getattr(args, "ref_load", None)
    if (
        (getattr(args, "kl_coef", 0) != 0 or getattr(args, "use_kl_loss", False))
        and ref_load
        and not os.path.exists(os.path.expanduser(ref_load))
    ):
        diagnostics.append(
            _result(
                "CONFIG_PATHS",
                f"ref_load {ref_load!r} does not exist while KL is enabled.",
                "Set --ref-load to an existing checkpoint path or disable KL.",
                details={"field": "ref_load", "path": ref_load},
            )
        )
    return diagnostics


@diagnostic_rule("CONFIG_LORA", "LoRA options are compatible")
def check_lora(ctx: DoctorContext) -> list[DiagnosticResult]:
    args = _args(ctx)
    if args is None or getattr(args, "lora_rank", 0) <= 0:
        return []
    diagnostics = []
    if getattr(args, "lora_merge_mode", False) and getattr(args, "lora_adapter_mode", False):
        diagnostics.append(
            _result(
                "CONFIG_LORA",
                "--lora-merge-mode and --lora-adapter-mode are mutually exclusive.",
                "Pick one LoRA rollout path.",
            )
        )
    if getattr(args, "lora_adapter_mode", False) and getattr(args, "sglang_dp_size", 1) != 1:
        diagnostics.append(
            _result(
                "CONFIG_LORA",
                "--lora-adapter-mode requires --sglang-dp-size 1.",
                "Set SGLang DP size to 1 or use LoRA merge mode.",
            )
        )
    return diagnostics


@diagnostic_rule("CONFIG_QKV_FORMAT", "QKV format is compatible with batching mode")
def check_qkv_format(ctx: DoctorContext) -> list[DiagnosticResult]:
    args = _args(ctx)
    if args is None or getattr(args, "qkv_format", None) != "bshd":
        return []
    diagnostics = []
    if getattr(args, "train_backend", "megatron") != "megatron":
        diagnostics.append(
            _result(
                "CONFIG_QKV_FORMAT",
                "bshd format is only supported for megatron backend.",
                "Use --train-backend megatron or switch qkv_format.",
            )
        )
    if getattr(args, "use_dynamic_batch_size", False):
        diagnostics.append(
            _result(
                "CONFIG_QKV_FORMAT",
                "Dynamic batch size is not supported for bshd format.",
                "Disable --use-dynamic-batch-size and set --micro-batch-size.",
            )
        )
    return diagnostics


@diagnostic_rule("CONFIG_ROTATE_CKPT", "Rotating checkpoints has required options")
def check_rotate_ckpt(ctx: DoctorContext) -> list[DiagnosticResult]:
    args = _args(ctx)
    if args is None or not getattr(args, "rotate_ckpt", False):
        return []
    missing = []
    if not getattr(args, "save", None):
        missing.append("--save")
    if getattr(args, "save_interval", None) is None:
        missing.append("--save-interval")
    if not getattr(args, "async_save", False):
        missing.append("--async-save")
    if not missing:
        return []
    return [
        _result(
            "CONFIG_ROTATE_CKPT",
            f"--rotate-ckpt is missing required option(s): {missing}.",
            "Set all required checkpoint rotation options or disable --rotate-ckpt.",
            details={"missing": missing},
        )
    ]
