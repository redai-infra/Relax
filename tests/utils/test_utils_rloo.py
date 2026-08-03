# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections import Counter
from types import SimpleNamespace

import torch

from relax.utils.types import Sample
from relax.utils.utils import get_debug_data


def test_get_debug_data_rloo_subsampling_preserves_complete_groups(tmp_path):
    group_size = 4
    samples = [
        Sample(
            group_index=group_index,
            index=group_index * group_size + sample_index,
            tokens=[1, 2],
            response_length=1,
            reward=float(sample_index),
            status=Sample.Status.COMPLETED,
        )
        for group_index in range(4)
        for sample_index in range(group_size)
    ]
    debug_path = tmp_path / "rollout-0.pt"
    torch.save({"samples": [sample.to_dict() for sample in samples]}, debug_path)
    args = SimpleNamespace(
        load_debug_rollout_data=str(tmp_path / "rollout-{rollout_id}.pt"),
        load_debug_rollout_data_subsample=0.5,
        custom_reward_post_process_path=None,
        advantage_estimator="rloo",
        rewards_normalization=False,
        n_samples_per_prompt=group_size,
        reward_key=None,
        multimodal_keys=None,
        use_opd=False,
        debug_train_only=True,
    )

    rollout_batch = get_debug_data(args, rollout_id=0, batch_size=2 * group_size, dp_rank=0)

    group_counts = Counter(rollout_batch["group_indices"])
    assert group_counts == {0: group_size, 3: group_size}
