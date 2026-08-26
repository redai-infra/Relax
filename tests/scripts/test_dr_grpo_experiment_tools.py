# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import math
from pathlib import Path

import pytest
import torch

from scripts.testing.summarize_dr_grpo_qwen35_gsm8k import (
    add_exact_loss_mask_tokens,
    add_outcome_lengths,
    find_training_log,
)


def _save_train_dump(path: Path, step: int, rank: int, masks: list[list[int]]) -> None:
    samples = [{"sample_index": index, "loss_masks": loss_mask} for index, loss_mask in enumerate(masks)]
    torch.save({"rollout_id": step, "rank": rank, "samples": samples}, path)


def test_find_training_log_prefers_complete_ray_job_log(tmp_path: Path) -> None:
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    recipe_log = log_dir / "recipe.log"
    recipe_log.write_text("submission only\n")
    ray_job_log = tmp_path / "ray_job_final.log"
    ray_job_log.write_text("complete metrics\n")

    assert find_training_log(tmp_path) == ray_job_log


def test_find_training_log_falls_back_to_single_recipe_log(tmp_path: Path) -> None:
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    recipe_log = log_dir / "recipe.log"
    recipe_log.write_text("complete metrics\n")

    assert find_training_log(tmp_path) == recipe_log


def test_exact_loss_mask_tokens_removes_model_parallel_replicas(tmp_path: Path) -> None:
    train_data_dir = tmp_path / "train_data"
    train_data_dir.mkdir()
    # TP ranks 0/1 hold the same DP0 shard; ranks 2/3 hold the same DP1 shard.
    rank_masks = {
        0: [[1, 0], [1]],
        1: [[1, 0], [1]],
        2: [[0, 0], [1, 1]],
        3: [[0, 0], [1, 1]],
    }
    for rank, masks in rank_masks.items():
        _save_train_dump(train_data_dir / f"0_{rank}.pt", 0, rank, masks)
    records: dict[int, dict[str, float]] = {0: {}}

    add_exact_loss_mask_tokens(
        tmp_path,
        records,
        num_steps=1,
        world_size=4,
        model_parallel_size=2,
        global_batch_size=4,
    )

    assert records[0]["train/loss_mask_tokens"] == 4
    assert isinstance(records[0]["train/loss_mask_tokens"], int)
    assert records[0]["train/fully_masked_responses"] == 1


def test_exact_loss_mask_tokens_rejects_non_binary_masks(tmp_path: Path) -> None:
    train_data_dir = tmp_path / "train_data"
    train_data_dir.mkdir()
    _save_train_dump(train_data_dir / "0_0.pt", 0, 0, [[0.5]])

    with pytest.raises(ValueError, match="Non-binary loss mask"):
        add_exact_loss_mask_tokens(
            tmp_path,
            {0: {}},
            num_steps=1,
            world_size=1,
            model_parallel_size=1,
            global_batch_size=1,
        )


def test_exact_loss_mask_tokens_rejects_disagreeing_model_parallel_dumps(tmp_path: Path) -> None:
    train_data_dir = tmp_path / "train_data"
    train_data_dir.mkdir()
    _save_train_dump(train_data_dir / "0_0.pt", 0, 0, [[1, 0]])
    _save_train_dump(train_data_dir / "0_1.pt", 0, 1, [[1, 1]])

    with pytest.raises(ValueError, match="rank dumps disagree"):
        add_exact_loss_mask_tokens(
            tmp_path,
            {0: {}},
            num_steps=1,
            world_size=2,
            model_parallel_size=2,
            global_batch_size=1,
        )


def test_exact_loss_mask_tokens_preserves_all_zero_window(tmp_path: Path) -> None:
    train_data_dir = tmp_path / "train_data"
    train_data_dir.mkdir()
    for rank in range(2):
        _save_train_dump(train_data_dir / f"0_{rank}.pt", 0, rank, [[0, 0], [0]])
    records: dict[int, dict[str, float]] = {0: {}}

    add_exact_loss_mask_tokens(
        tmp_path,
        records,
        num_steps=1,
        world_size=2,
        model_parallel_size=2,
        global_batch_size=2,
    )

    assert records[0]["train/loss_mask_tokens"] == 0
    assert records[0]["train/fully_masked_responses"] == 2


@pytest.mark.parametrize(
    ("run", "logged_kl", "loss_mask_tokens", "fully_masked_responses", "expected_reference_kl"),
    [
        ("grpo", 2.0, 3, 1, 8 / 3),
        ("dr_grpo", 1.5, 3, 1, 4.0),
        ("dr_grpo", 0.0, 0, 2, math.nan),
    ],
)
def test_reference_kl_uses_exact_training_denominator(
    tmp_path: Path,
    run: str,
    logged_kl: float,
    loss_mask_tokens: int,
    fully_masked_responses: int,
    expected_reference_kl: float,
) -> None:
    result_dir = tmp_path / "rollout_result" / "train"
    result_dir.mkdir(parents=True)
    samples = [
        {"reward": 1.0, "response_length": 2, "status": "completed"},
        {"reward": 0.0, "response_length": 4, "status": "truncated"},
    ]
    (result_dir / "0.jsonl").write_text(
        "".join(f"{json.dumps(sample)}\n" for sample in samples),
        encoding="utf-8",
    )
    records = {
        0: {
            "train/kl_loss": logged_kl,
            "train/loss_mask_tokens": loss_mask_tokens,
            "train/fully_masked_responses": fully_masked_responses,
        }
    }

    add_outcome_lengths(
        run,
        tmp_path,
        records,
        num_steps=1,
        response_budget=4,
        global_batch_size=2,
    )

    reference_kl = records[0]["train/reference_kl"]
    if math.isnan(expected_reference_kl):
        assert math.isnan(reference_kl)
    else:
        assert reference_kl == pytest.approx(expected_reference_kl)
