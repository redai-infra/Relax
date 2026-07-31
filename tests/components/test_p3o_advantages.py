# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""P3O advantage-path parity with GRPO."""

from types import SimpleNamespace

import torch

from relax.components.advantages import Advantages


def _compute(estimator: str):
    advantages_class = Advantages.func_or_class
    component = advantages_class.__new__(advantages_class)
    component.config = SimpleNamespace(
        advantage_estimator=estimator,
        kl_coef=0.0,
        use_kl_loss=False,
        use_rollout_logprobs=True,
        use_opd=False,
    )
    rollout_data = {
        "rollout_log_probs": [
            torch.tensor([-0.1, -0.2, -0.3]),
            torch.tensor([-0.4, -0.5]),
        ],
        "ref_log_probs": None,
        "rewards": [1.25, -0.75],
        "values": None,
        "response_lengths": [3, 2],
        "loss_masks": [torch.ones(3), torch.ones(2)],
        "total_lengths": [5, 4],
    }
    return component.compute_advantages_and_returns(rollout_data)


def test_p3o_advantages_match_grpo_shapes_and_values():
    p3o = _compute("p3o")
    grpo = _compute("grpo")

    for key in ("advantages", "returns"):
        p3o_values = p3o[key].unbind()
        grpo_values = grpo[key].unbind()
        assert [value.shape for value in p3o_values] == [torch.Size([3]), torch.Size([2])]
        assert len(p3o_values) == len(grpo_values)
        for p3o_value, grpo_value in zip(p3o_values, grpo_values, strict=True):
            torch.testing.assert_close(p3o_value, grpo_value)

    torch.testing.assert_close(p3o["advantages"].unbind()[0], torch.full((3,), 1.25))
    torch.testing.assert_close(p3o["advantages"].unbind()[1], torch.full((2,), -0.75))
