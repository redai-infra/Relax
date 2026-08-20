# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only one-step parity check for the GraphGPO GRPO adapter.

The comparison deliberately has two independent sides:

* the adapter side uses ``compute_group_advantages`` plus Relax's production
  GRPO token broadcast and clipped policy-loss definitions;
* the oracle side uses Relax's ordinary trajectory-level reward
  normalization, followed by a small, independently written PyTorch loss.

This catches changes in row payloads as well as changes that only become
visible after backpropagation.  Source definitions are loaded with ``ast`` so
the test does not import Ray or Megatron and remains runnable on a CPU host.
"""

from __future__ import annotations

import ast
import copy
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from examples.graphgpo.custom_advantage import compute_group_advantages
from examples.graphgpo.graph_credit import SUCCESS


REPO_ROOT = Path(__file__).resolve().parents[2]
RNG_SEED = 20260422


class _Sample:
    class Status:
        TRUNCATED = "truncated"

    def __init__(
        self,
        *,
        index: int,
        reward: float,
        group_index: int,
        tokens: list[int] | None = None,
        loss_mask: list[int] | None = None,
        rollout_log_probs: list[float] | None = None,
        custom_advantage: float | None = None,
    ) -> None:
        self.index = index
        self.reward = reward
        self.group_index = group_index
        self.tokens = tokens or [0, 1]
        self.response_length = len(loss_mask or [1])
        self.loss_mask = list(loss_mask or [1])
        self.rollout_log_probs = rollout_log_probs
        self.custom_advantage = custom_advantage
        self.status = "completed"
        self.remove_sample = False
        self.metadata = None
        self.train_metadata = None
        self.rollout_routed_experts = None
        self.multimodal_train_inputs = None

    def get_reward_value(self, _args: Namespace) -> float:
        return self.reward


@dataclass(frozen=True)
class _TurnRow:
    trajectory_id: str
    turn_index: int
    episode_return: float
    tokens: list[int]
    loss_mask: list[int]
    rollout_log_probs: list[float]


def _source_functions(torch: Any) -> Namespace:
    """Load only the production functions needed by this dependency-light
    test."""

    utils_path = REPO_ROOT / "relax" / "utils" / "utils.py"
    utils_tree = ast.parse(utils_path.read_text(encoding="utf-8"), filename=str(utils_path))
    utils_names = {"convert_samples_to_train_data", "post_process_rewards"}
    utils_nodes = [node for node in utils_tree.body if isinstance(node, ast.FunctionDef) and node.name in utils_names]
    assert {node.name for node in utils_nodes} == utils_names
    utils_module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *utils_nodes,
        ],
        type_ignores=[],
    )
    utils_namespace = {"Any": Any, "Sample": _Sample, "torch": torch}
    exec(compile(ast.fix_missing_locations(utils_module), str(utils_path), "exec"), utils_namespace)

    ppo_path = REPO_ROOT / "relax" / "utils" / "training" / "ppo_utils.py"
    ppo_tree = ast.parse(ppo_path.read_text(encoding="utf-8"), filename=str(ppo_path))
    ppo_names = {"compute_policy_loss", "get_grpo_returns"}
    ppo_nodes = []
    for node in ppo_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ppo_names:
            copied_node = copy.deepcopy(node)
            copied_node.decorator_list = []
            ppo_nodes.append(copied_node)
    assert {node.name for node in ppo_nodes} == ppo_names
    ppo_module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *ppo_nodes,
        ],
        type_ignores=[],
    )
    ppo_namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ppo_module), str(ppo_path), "exec"), ppo_namespace)

    return Namespace(
        convert_samples_to_train_data=utils_namespace["convert_samples_to_train_data"],
        post_process_rewards=utils_namespace["post_process_rewards"],
        compute_policy_loss=ppo_namespace["compute_policy_loss"],
        get_grpo_returns=ppo_namespace["get_grpo_returns"],
    )


def _response_log_probs(torch: Any, model: Any, tokens: list[int], response_length: int):
    token_tensor = torch.tensor(tokens, dtype=torch.long)
    logits = model(token_tensor[:-1])
    target_tokens = token_tensor[1:]
    all_log_probs = torch.log_softmax(logits, dim=-1).gather(1, target_tokens[:, None]).squeeze(1)
    response_start = len(tokens) - response_length - 1
    return all_log_probs[response_start:]


def _build_fixture(torch: Any):
    class TinyPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(31, 7, dtype=torch.float64)
            self.output = torch.nn.Linear(7, 31, bias=True, dtype=torch.float64)

        def forward(self, tokens):
            return self.output(torch.tanh(self.embedding(tokens)))

    torch.manual_seed(RNG_SEED)
    rollout_policy = TinyPolicy()

    trajectory_shapes = (
        ((2, [1, 1]), (3, [1, 0, 1])),
        ((4, [1, 1, 1, 0]),),
        ((1, [1]), (2, [0, 1]), (3, [1, 1, 1])),
    )
    success_flags = (True, False, False)
    invalid_turns = (frozenset(), frozenset(), frozenset({1}))
    metadata_by_slot: list[dict[str, dict[str, object]]] = []
    rows: list[_TurnRow] = []

    for trajectory_index, turns in enumerate(trajectory_shapes):
        trajectory_id = f"trajectory-{trajectory_index}"
        success = success_flags[trajectory_index]
        episode_return = 10.0 * float(success) - 0.1 * len(invalid_turns[trajectory_index])
        slot: dict[str, dict[str, object]] = {}
        for turn_index, (response_length, loss_mask) in enumerate(turns):
            is_final = turn_index == len(turns) - 1
            unit_name = f"turn_{turn_index:03d}"
            slot[unit_name] = {
                "row_id": f"row-{trajectory_index}-{turn_index}",
                "rollout_group_id": "rollout-group-0",
                "policy_version": "policy-version-7",
                "task_id": "shared-task",
                "trajectory_id": trajectory_id,
                "turn_index": turn_index,
                "state_key": f"state-{trajectory_index}-{turn_index}",
                "action": f"action-{trajectory_index}-{turn_index}",
                "next_state_key": (SUCCESS if success and is_final else f"state-{trajectory_index}-{turn_index + 1}"),
                "is_action_valid": turn_index not in invalid_turns[trajectory_index],
                "success": success,
                "terminal": success and is_final,
                "truncated": not success and is_final,
                "episode_return": episode_return,
            }

            prompt_length = 2 + ((trajectory_index + turn_index) % 3)
            tokens = torch.randint(0, 31, (prompt_length + response_length,), dtype=torch.long).tolist()
            with torch.no_grad():
                rollout_log_probs = _response_log_probs(
                    torch,
                    rollout_policy,
                    tokens,
                    response_length,
                ).tolist()
            rows.append(
                _TurnRow(
                    trajectory_id=trajectory_id,
                    turn_index=turn_index,
                    episode_return=episode_return,
                    tokens=tokens,
                    loss_mask=loss_mask,
                    rollout_log_probs=rollout_log_probs,
                )
            )
        metadata_by_slot.append(slot)

    initial_policy = copy.deepcopy(rollout_policy)
    torch.manual_seed(RNG_SEED + 1)
    with torch.no_grad():
        for parameter in initial_policy.parameters():
            parameter.add_(0.015 * torch.randn_like(parameter))
    return metadata_by_slot, rows, initial_policy


def _args(*, custom_advantage: bool) -> Namespace:
    return Namespace(
        custom_reward_post_process_path=None,
        agentic_custom_advantage_path="adapter" if custom_advantage else None,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        n_samples_per_prompt=3,
        multimodal_keys=None,
        use_opd=False,
        debug_train_only=True,
    )


def _loss_grad_and_delta(
    torch: Any,
    model: Any,
    rows: list[_TurnRow],
    token_advantages: list[Any],
    *,
    production_loss: Any | None,
):
    current_log_probs = torch.cat([_response_log_probs(torch, model, row.tokens, len(row.loss_mask)) for row in rows])
    rollout_log_probs = torch.cat([torch.tensor(row.rollout_log_probs, dtype=torch.float64) for row in rows])
    advantages = torch.cat(token_advantages).to(dtype=torch.float64)
    loss_mask = torch.cat([torch.tensor(row.loss_mask, dtype=torch.float64) for row in rows])

    if production_loss is not None:
        element_loss, _ = production_loss(
            rollout_log_probs - current_log_probs,
            advantages,
            0.2,
            0.2,
        )
    else:
        ratio = torch.exp(current_log_probs - rollout_log_probs)
        unclipped = -ratio * advantages
        clipped = -torch.clamp(ratio, 0.8, 1.2) * advantages
        element_loss = torch.maximum(unclipped, clipped)
    loss = (element_loss * loss_mask).sum() / loss_mask.sum()

    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    loss.backward()
    gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(parameter.grad, alpha=-0.05)
    deltas = {name: parameter.detach() - before[name] for name, parameter in model.named_parameters()}
    return loss.detach(), gradients, deltas


def test_grpo_adapter_matches_native_grpo_through_one_cpu_optimizer_step():
    torch = pytest.importorskip("torch")
    functions = _source_functions(torch)
    metadata_by_slot, rows, initial_policy = _build_fixture(torch)

    adapter_by_slot = compute_group_advantages(
        metadata_by_slot,
        method="grpo",
        expected_group_size=3,
        episode_weighting="trajectory_once",
    )
    adapter_by_turn = {
        (f"trajectory-{slot_index}", int(unit_name.removeprefix("turn_"))): value
        for slot_index, slot in enumerate(adapter_by_slot)
        for unit_name, value in slot.items()
    }

    trajectory_samples = [
        _Sample(index=index, reward=reward, group_index=0) for index, reward in enumerate((10.0, 0.0, -0.1))
    ]
    _, native_trajectory_advantages = functions.post_process_rewards(
        _args(custom_advantage=False),
        trajectory_samples,
    )
    native_by_trajectory = {
        f"trajectory-{index}": advantage for index, advantage in enumerate(native_trajectory_advantages)
    }

    adapter_samples = [
        _Sample(
            index=index,
            reward=row.episode_return,
            group_index=0,
            tokens=row.tokens,
            loss_mask=row.loss_mask,
            rollout_log_probs=row.rollout_log_probs,
            custom_advantage=adapter_by_turn[(row.trajectory_id, row.turn_index)],
        )
        for index, row in enumerate(rows)
    ]
    adapter_payload = functions.convert_samples_to_train_data(
        _args(custom_advantage=True),
        adapter_samples,
    )
    oracle_payload = {
        "tokens": [row.tokens for row in rows],
        "loss_masks": [row.loss_mask for row in rows],
        "rollout_log_probs": [row.rollout_log_probs for row in rows],
    }

    assert adapter_payload["tokens"] == oracle_payload["tokens"]
    assert adapter_payload["loss_masks"] == oracle_payload["loss_masks"]
    assert any(mask_value == 0 for mask in adapter_payload["loss_masks"] for mask_value in mask)
    for actual, expected in zip(
        adapter_payload["rollout_log_probs"],
        oracle_payload["rollout_log_probs"],
        strict=True,
    ):
        torch.testing.assert_close(
            torch.tensor(actual),
            torch.tensor(expected),
            rtol=0,
            atol=0,
        )

    zero_kl = [torch.zeros(len(row.rollout_log_probs), dtype=torch.float64) for row in rows]
    adapter_token_advantages = functions.get_grpo_returns(
        torch.tensor(adapter_payload["rewards"], dtype=torch.float64),
        zero_kl,
    )
    oracle_token_advantages = [
        torch.full_like(
            row_kl,
            native_by_trajectory[row.trajectory_id],
        )
        for row, row_kl in zip(rows, zero_kl, strict=True)
    ]
    for actual, expected in zip(
        adapter_token_advantages,
        oracle_token_advantages,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)

    adapter_model = copy.deepcopy(initial_policy)
    oracle_model = copy.deepcopy(initial_policy)
    adapter_loss, adapter_gradients, adapter_deltas = _loss_grad_and_delta(
        torch,
        adapter_model,
        rows,
        adapter_token_advantages,
        production_loss=functions.compute_policy_loss,
    )
    oracle_loss, oracle_gradients, oracle_deltas = _loss_grad_and_delta(
        torch,
        oracle_model,
        rows,
        oracle_token_advantages,
        production_loss=None,
    )

    torch.testing.assert_close(adapter_loss, oracle_loss, rtol=2e-6, atol=2e-6)
    assert set(adapter_gradients) == set(oracle_gradients)
    assert set(adapter_deltas) == set(oracle_deltas)
    for name in adapter_gradients:
        torch.testing.assert_close(
            adapter_gradients[name],
            oracle_gradients[name],
            rtol=2e-6,
            atol=2e-6,
        )
        torch.testing.assert_close(
            adapter_deltas[name],
            oracle_deltas[name],
            rtol=2e-6,
            atol=2e-6,
        )
    assert any(torch.count_nonzero(delta).item() > 0 for delta in adapter_deltas.values())
