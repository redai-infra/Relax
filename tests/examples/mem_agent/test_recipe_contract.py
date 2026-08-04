# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Static checks for the frozen MemAgent launch and reproducibility recipe."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples" / "mem_agent"


def test_custom_config_freezes_model_memory_and_expanded_batch_contract():
    config = yaml.safe_load((EXAMPLE / "config.yaml").read_text(encoding="utf-8"))

    assert config["model_id"] == "Qwen/Qwen3-4B"
    assert config["model_revision"] == "1cfa9a7208912126459214e8b04321603b3df60c"
    assert config["mem_agent_chunk_tokens"] == 2048
    assert config["mem_agent_max_memory_tokens"] == 1024
    assert config["mem_agent_max_final_tokens"] == 256
    assert config["mem_agent_max_chunks"] == 64
    assert config["mem_agent_credit_assignment"] == "split"
    assert config["custom_train_sample_expansion_factor"] == 65
    assert config["custom_train_data_group_size"] == 1
    assert config["custom_train_expanded_batch"] is True


def test_train_script_keeps_trajectory_loss_and_real_turn_context_envelope():
    script = (EXAMPLE / "run-qwen3-4B-train.sh").read_text(encoding="utf-8")

    assert '--global-batch-size "${GLOBAL_BATCH_SIZE}"' in script
    assert "--rollout-max-context-len 8192" in script
    assert "--max-tokens-per-gpu 9216" in script
    assert "--custom-convert-samples-to-train-data-path examples.mem_agent.convert.convert_samples" in script
    assert "--use-dynamic-global-batch-size" not in script
    assert "--calculate-per-token-loss" not in script


def test_reward_exposes_a_step_level_raw_reward_metric():
    reward_source = (EXAMPLE / "reward.py").read_text(encoding="utf-8")
    assert '"mem_agent_raw_reward": score' in reward_source


def test_pipeline_and_acceptance_scripts_cover_required_stages_and_metrics():
    pipeline = (EXAMPLE / "run-pipeline.sh").read_text(encoding="utf-8")
    paired = (EXAMPLE / "run-paired-eval.sh").read_text(encoding="utf-8")
    evaluator = (EXAMPLE / "run-eval.sh").read_text(encoding="utf-8")

    for stage in (
        "prepare-data.sh",
        "run-qwen3-4B-train.sh",
        "summarize_reward.py",
        "convert-to-hf.sh",
        "run-eval.sh",
    ):
        assert stage in pipeline
    assert '--expected-steps "${NUM_ROLLOUT}"' in pipeline
    assert "training-reward.summary.json" in pipeline
    assert 'BASE_MODEL_PATH="${BASE_MODEL_PATH:?' in paired
    assert paired.count("MODE=recurrent") == 3
    assert '--pair "ruler-hqa-${length}"' in paired
    assert '--baseline-pair "ruler-hqa-${length}" sub_em_pct' in paired
    assert "--baseline-pair hotpotqa-dev boxed_em_pct" in paired
    assert 'LENGTHS="${LENGTHS:-50 200 800}"' in paired
    for value in (
        "--temperature 0.7",
        "--top-p 0.95",
        "--chunk-tokens 2048",
        "--max-memory-tokens 1024",
        "--max-chunks 64",
        '--server-max-model-len "${MAX_MODEL_LEN}"',
    ):
        assert value in evaluator
