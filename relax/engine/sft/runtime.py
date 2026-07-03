# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Mode predicates and naming helpers shared across the SFT path.

These are the bits previously duplicated as ``_sft_*`` private functions in
``backends/megatron/actor.py`` and ``components/actor.py``. Centralising them
here keeps the dispatchers in those files to one-line calls.
"""

from argparse import Namespace


def is_sft_mode(args: Namespace) -> bool:
    """Single source of truth for the "are we training SFT?" check.

    ``args.loss_type == "sft"`` is the canonical signal across argparse,
    controller wiring, components, and the Megatron backend.
    """
    return getattr(args, "loss_type", None) == "sft"


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


def build_data_fields(args: Namespace) -> list[str]:
    """Decide which fields to pull from TQ based on training mode.

    SFT producer emits only tokens / loss_masks / total_lengths /
    response_lengths (+ multimodal_train_inputs if applicable). No log_probs,
    no rewards.
    """
    if is_sft_mode(args):
        fields = ["tokens", "total_lengths", "response_lengths", "loss_masks"]
        if args.multimodal_keys is not None:
            fields.append("multimodal_train_inputs")
        return fields

    fields = [
        "tokens",
        "total_lengths",
        "response_lengths",
        "loss_masks",
        "rollout_log_probs",
        "rewards",
        "raw_reward",
    ]
    if args.use_rollout_routing_replay:
        fields.append("rollout_routed_experts")
    if args.multimodal_keys is not None:
        fields.append("multimodal_train_inputs")
    if args.use_opd and args.opd_type == "sglang":
        fields.append("teacher_log_probs")
        if args.opd_log_prob_top_k > 0:
            fields.append("teacher_topk_token_ids")
            fields.append("teacher_topk_k")
    return fields
