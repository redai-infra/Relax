# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Controller-side rollout sizing helpers."""

from argparse import Namespace
from typing import Any

import ray

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def resolve_rl_num_rollout(config: Namespace, data_source: Any) -> None:
    """Resolve RL epoch sizing before Actor and Rollout services are
    created."""
    if getattr(config, "loss_type", None) == "sft":
        return

    if not config.rollout_global_dataset:
        config.num_rollout_per_epoch = None
        if config.num_epoch is not None:
            raise ValueError("num_epoch requires rollout_global_dataset for RL training")
        if config.num_rollout is None or config.num_rollout <= 0:
            raise ValueError("num_rollout must be positive when rollout_global_dataset is disabled")
        logger.info(
            f"RL num_rollout resolved before service creation: {config.num_rollout} "
            "(non-global dataset; no epoch boundary)"
        )
        return

    dataset_size = ray.get(data_source.lengths.remote())
    if dataset_size <= 0:
        raise ValueError("rollout_global_dataset requires a non-empty dataset")
    num_per_epoch = dataset_size // config.rollout_batch_size
    if num_per_epoch <= 0:
        if config.num_epoch is not None:
            raise ValueError(
                f"Dataset size {dataset_size} must be at least rollout_batch_size {config.rollout_batch_size} "
                "when num_epoch is configured"
            )
        if config.num_rollout is None or config.num_rollout <= 0:
            raise ValueError("num_rollout must be positive when a full rollout batch does not fit in one epoch")
        # The data source can wrap across epochs to fill a batch. There is no
        # step-aligned epoch boundary in this case, so disable that trigger.
        config.num_rollout_per_epoch = None
        logger.info(
            f"RL num_rollout resolved before service creation: {config.num_rollout} "
            f"(dataset_size={dataset_size} < rollout_batch_size={config.rollout_batch_size}; "
            "batches wrap across epochs)"
        )
        return

    if config.num_epoch is not None and dataset_size % config.rollout_batch_size != 0:
        raise ValueError(
            f"Dataset size {dataset_size} must be divisible by rollout_batch_size {config.rollout_batch_size} "
            "when num_epoch is configured; use explicit num_rollout for cross-epoch batches"
        )

    config.num_rollout_per_epoch = num_per_epoch
    if config.num_epoch is not None:
        epoch_rollout = num_per_epoch * config.num_epoch
        config.num_rollout = (
            min(config.num_rollout, epoch_rollout) if config.num_rollout is not None else epoch_rollout
        )
    if config.num_rollout is None or config.num_rollout <= 0:
        raise ValueError(
            f"num_rollout resolved to {config.num_rollout}; "
            f"num_rollout_per_epoch={num_per_epoch}, num_epoch={config.num_epoch}"
        )
    logger.info(
        f"RL num_rollout resolved before service creation: {config.num_rollout} "
        f"(num_rollout_per_epoch={num_per_epoch}, dataset_size={dataset_size})"
    )
