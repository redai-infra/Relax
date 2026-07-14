# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

from relax.utils.opd.opd_utils import consume_opd_train_data


def build_data_fields(args: Namespace) -> list[str]:
    """Decide which fields to pull from TransferQueue for training."""
    if getattr(args, "loss_type", None) == "sft":
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
    if args.use_opd:
        consume_opd_train_data(fields, args)
    return fields
