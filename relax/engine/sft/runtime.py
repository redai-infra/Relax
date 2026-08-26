# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Mode predicates and naming helpers shared across the SFT path.

These are the bits previously duplicated as ``_sft_*`` private functions in
``backends/megatron/actor.py`` and ``components/actor.py``. Centralising them
here keeps the dispatchers in those files to one-line calls.
"""

import math
from argparse import Namespace


def is_sft_mode(args: Namespace) -> bool:
    """Single source of truth for the "are we training SFT?" check.

    ``args.loss_type == "sft"`` is the canonical signal across argparse,
    controller wiring, components, and the Megatron backend.
    """
    return getattr(args, "loss_type", None) == "sft"


def sft_objective(args: Namespace) -> str:
    """Return the offline objective while preserving causal-LM defaults."""
    return getattr(args, "sft_objective", "causal_lm")


def is_preference_mode(args: Namespace) -> bool:
    return is_sft_mode(args) and sft_objective(args) == "dpo"


def validate_preference_args(args: Namespace) -> None:
    """Reject unsupported preference configurations before Serve starts."""
    if not is_preference_mode(args):
        return
    if getattr(args, "custom_dataset_class_path", None):
        raise ValueError("preference objectives do not support --custom-dataset-class")
    if getattr(args, "multimodal_keys", None) is not None:
        raise ValueError("preference objectives v1 support pure text only")
    if int(getattr(args, "n_samples_per_prompt", 1)) != 1:
        raise ValueError("preference objectives require --n-samples-per-prompt 1")
    topology = {
        "tensor_model_parallel_size": int(getattr(args, "tensor_model_parallel_size", 1) or 1),
        "pipeline_model_parallel_size": int(getattr(args, "pipeline_model_parallel_size", 1) or 1),
        "context_parallel_size": int(getattr(args, "context_parallel_size", 1) or 1),
    }
    invalid = {name: size for name, size in topology.items() if size != 1}
    if invalid:
        raise ValueError(f"preference objectives v1 require TP=CP=PP=1, got {invalid}")
    if getattr(args, "dynamic_context_parallel", False):
        raise ValueError("preference objectives v1 do not support dynamic context parallelism")
    if getattr(args, "qkv_format", "thd") != "thd":
        raise ValueError("preference objectives v1 require --qkv-format thd")
    if getattr(args, "fully_async", False) or getattr(args, "hybrid", False):
        raise ValueError("preference objectives v1 support synchronous SFT topology only")
    if not getattr(args, "use_gloo_process_groups", False):
        raise ValueError("preference objectives require --use-gloo-process-groups for DP iterator control data")
    if getattr(args, "sft_chunked_logits", False) or getattr(args, "enable_mtp_training", False):
        raise ValueError("preference objectives v1 do not support SFT chunked logits or MTP")
    if getattr(args, "calculate_per_token_loss", False):
        raise ValueError("preference objectives use pair reduction and reject --calculate-per-token-loss")
    if int(getattr(args, "lora_rank", 0) or 0) > 0:
        raise ValueError("preference objectives v1 do not support LoRA")
    if (
        float(getattr(args, "hidden_dropout", 0.0) or 0.0) != 0.0
        or float(getattr(args, "attention_dropout", 0.0) or 0.0) != 0.0
    ):
        raise ValueError("preference objectives require hidden and attention dropout to be 0.0")
    if getattr(args, "sft_predict_interval", None) is not None:
        raise ValueError("preference objectives do not use SFT generation prediction")
    max_length = int(getattr(args, "preference_max_length", 0) or 0)
    max_completion_length = int(getattr(args, "preference_max_completion_length", 0) or 0)
    if max_length <= 0 or max_completion_length <= 0:
        raise ValueError("preference length limits must be positive")
    if max_completion_length > max_length:
        raise ValueError("--preference-max-completion-length must not exceed --preference-max-length")
    seq_length = int(getattr(args, "seq_length", max_length) or max_length)
    if max_length > seq_length:
        raise ValueError("--preference-max-length must not exceed --seq-length")
    if bool(getattr(args, "eval_prompt_data", None)) or getattr(args, "eval_size", None) is not None:
        raise ValueError("DPO held-out evaluation is delivered by the follow-up reward-modeling PR")
    beta = float(getattr(args, "dpo_beta", 0.1))
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError(f"--dpo-beta must be finite and positive, got {beta}")
    likelihood_temperature = float(getattr(args, "rollout_temperature", 1.0))
    if not math.isfinite(likelihood_temperature) or likelihood_temperature != 1.0:
        raise ValueError(
            "preference objectives require --rollout-temperature 1.0 so sampling temperature does not scale "
            "policy/reference likelihood logits"
        )
    if getattr(args, "ref_load", None) is not None:
        raise ValueError(
            "preference objectives do not use --ref-load: standard DPO snapshots the frozen reference "
            "from the pinned --dpo-reference-repository/--dpo-reference-revision snapshot"
        )
    if not getattr(args, "dpo_reference_free", False) and getattr(args, "ref_update_interval", None) is not None:
        raise ValueError("standard DPO requires a frozen reference and rejects --ref-update-interval")
    if not getattr(args, "dpo_reference_free", False) and not getattr(args, "enable_weights_backuper", False):
        raise ValueError("standard DPO requires --enable-weights-backuper for actor/ref snapshots")
    if not getattr(args, "dpo_reference_free", False):
        if not getattr(args, "dpo_reference_repository", None) or not getattr(args, "dpo_reference_revision", None):
            raise ValueError("standard DPO requires --dpo-reference-repository and --dpo-reference-revision")


def sft_partition_id(args: Namespace, step: int) -> str:
    return f"sft_{step}" if is_sft_mode(args) else f"train_{step}"


def sft_task_name(args: Namespace, *, component: str = "actor") -> str:
    """Return the TransferQueue task name.

    ``component`` distinguishes ``components/actor.py`` (uses ``train_actor``
    for RL reset/clear) from ``backends/megatron/actor.py`` (uses ``train``
    when consuming). Both collapse to ``sft_train`` under SFT.
    """
    if is_sft_mode(args):
        return "sft_train"
    if component == "actor":
        return "train_actor"
    return "train"


def should_run_sft_eval(args: Namespace, rollout_id: int) -> bool:
    """SFT PPL eval triggers every ``--eval-interval`` steps under SFT mode
    when an eval source is configured (either ``--eval-prompt-data`` or
    ``--eval-size``, mutually exclusive — see ``utils/arguments.py``).

    Pure Megatron path; no Rollout/SGLang involvement.
    """
    if not is_sft_mode(args):
        return False
    has_eval_source = bool(getattr(args, "eval_prompt_data", None)) or (getattr(args, "eval_size", None) is not None)
    if not has_eval_source:
        return False
    interval = getattr(args, "eval_interval", None)
    if interval is None or interval <= 0:
        return False
    return (rollout_id + 1) % interval == 0


def should_run_sft_predict(args: Namespace, rollout_id: int) -> bool:
    """SFT periodic predict triggers every ``--sft-predict-interval`` steps.

    Argparse already validated ``--loss-type sft``, ``--save``, and the eval
    data source, so we only need the interval check here.
    """
    interval = getattr(args, "sft_predict_interval", None)
    if interval is None or interval <= 0:
        return False
    return (rollout_id + 1) % interval == 0
