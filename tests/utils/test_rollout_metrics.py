# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

from relax.utils.metrics.metric_utils import compute_response_length_metrics
from relax.utils.types import Sample


def test_compute_response_length_metrics_groups_numeric_rewards_by_sign():
    args = Namespace(reward_key=None)
    samples = [
        Sample(response_length=3, reward=2),
        Sample(response_length=5, reward=1),
        Sample(response_length=2, reward=-3),
        Sample(response_length=4, reward=0),
    ]

    assert compute_response_length_metrics(args, samples) == {
        "response_len/Correct/mean": 4.0,
        "response_len/Incorrect/mean": 3.0,
    }
