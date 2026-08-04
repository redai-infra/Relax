# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Controller-side rollout sizing helpers."""

from argparse import Namespace
from typing import Any

import ray

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def resolve_rl_num_rollout(config: Namespace, data_source: Any) -> None:
    """Resolve RL epoch sizing before Actor and Rollout services are created."""
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
    num_per_epoch = dataset_size // config.rollout_batch_size
    assert num_per_epoch > 0, f"Dataset size {dataset_size} < rollout_batch_size {config.rollout_batch_size}"

    config.num_rollout_per_epoch = num_per_epoch
    if config.num_epoch is not None:
        epoch_rollout = num_per_epoch * config.num_epoch
        config.num_rollout = (
            min(config.num_rollout, epoch_rollout) if config.num_rollout is not None else epoch_rollout
        )
    assert config.num_rollout is not None and config.num_rollout > 0
    logger.info(
        f"RL num_rollout resolved before service creation: {config.num_rollout} "
        f"(num_rollout_per_epoch={num_per_epoch}, dataset_size={dataset_size})"
    )
