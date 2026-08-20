# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import unittest
from pathlib import Path

import pytest

from examples.graphgpo.preflight import validate_batch_arithmetic


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RECIPE_ROOT = REPOSITORY_ROOT / "examples" / "graphgpo"


class RecipePackagingTest(unittest.TestCase):
    def test_isolated_agent_dependencies_are_pinned(self) -> None:
        content = (RECIPE_ROOT / "requirements-alfworld.txt").read_text(encoding="utf-8")
        self.assertEqual(
            content.splitlines(),
            [
                "alfworld==0.4.2",
                "httpx==0.28.1",
                "openai==2.46.0",
                "PyYAML==6.0.3",
            ],
        )

    def test_alfworld_config_is_text_only_and_path_portable(self) -> None:
        content = (RECIPE_ROOT / "configs" / "alfworld_qwen2_5_1_5b.yaml").read_text(encoding="utf-8")
        self.assertIn('type: "AlfredTWEnv"', content)
        self.assertIn("$ALFWORLD_DATA/json_2.1.1/train", content)
        self.assertIn("$ALFWORLD_DATA/json_2.1.1/valid_seen", content)
        self.assertIn("use_cuda: false", content)
        self.assertIn('training_method: "dqn"', content)
        self.assertIn("rl:", content)
        self.assertNotIn('training_method: "dagger"', content)
        self.assertIn("num_train_games: -1", content)
        self.assertIn("num_eval_games: -1", content)
        self.assertNotIn("/mnt/", content)
        self.assertNotIn("C:\\", content)

    def test_launcher_keeps_method_switch_and_variable_row_contract(self) -> None:
        content = (RECIPE_ROOT / "run_alfworld_qwen2_5_1_5b.sh").read_text(encoding="utf-8")
        for method in ("grpo", "gigpo", "graphgpo"):
            self.assertIn(method, content)
        self.assertIn(
            "--agentic-custom-advantage-path examples.graphgpo.custom_advantage.compute_custom_advantage",
            content,
        )
        self.assertIn(
            "--custom-rm-path examples.graphgpo.reward.reward_func",
            content,
        )
        self.assertIn("--group-rm", content)
        self.assertIn("--use-dynamic-batch-size", content)
        self.assertIn('--train-env-vars \'{"TORCH_COMPILE_DISABLE":"1"}\'', content)
        self.assertIn("--disable-jit-fuser", content)
        # The frozen Qwen2.5-1.5B config has tie_word_embeddings=true.
        self.assertNotIn("--untie-embeddings-and-output-weights", content)
        self.assertIn("RELAX_AGENTIC_MAX_EXPORTED_ROWS_PER_SAMPLE", content)
        self.assertIn(
            'EPISODE_WEIGHTING="${EPISODE_WEIGHTING:-trajectory_once}"',
            content,
        )
        self.assertIn(
            'export GRAPHGPO_EPISODE_WEIGHTING="${EPISODE_WEIGHTING}"',
            content,
        )
        self.assertIn('ENABLE_EVAL="${ENABLE_EVAL:-1}"', content)
        self.assertIn('if [ "${ENABLE_EVAL}" = "1" ]; then', content)
        self.assertIn('echo "enable_eval=${ENABLE_EVAL}"', content)
        self.assertIn(
            'SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.50}"',
            content,
        )
        self.assertIn(
            '--sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"',
            content,
        )
        self.assertIn(
            'echo "sglang_mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}"',
            content,
        )
        self.assertIn("python3 -m examples.graphgpo.preflight", content)
        self.assertIn('--model-lock "${MODEL_LOCK}"', content)
        self.assertIn('--checkpoint "${HF_CHECKPOINT}"', content)
        self.assertIn('--alfworld-data-root "${ALFWORLD_DATA}"', content)
        self.assertIn(
            "--custom-eval-rollout-log-function-path examples.graphgpo.eval_logger.log_eval_rollout_data",
            content,
        )
        self.assertIn('if [ "${DRY_RUN:-0}" = "1" ]', content)
        self.assertIn("image_verification_scope=external_executor_required", content)
        self.assertNotIn("image_digest=${IMAGE_DIGEST}", content)

    def test_launcher_default_batch_arithmetic_forms_one_complete_batch(self) -> None:
        content = (RECIPE_ROOT / "run_alfworld_qwen2_5_1_5b.sh").read_text(encoding="utf-8")
        self.assertIn('TASK_GROUPS="${TASK_GROUPS:-16}"', content)
        self.assertIn('GROUP_SIZE="${GROUP_SIZE:-8}"', content)
        self.assertIn('GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"', content)
        validate_batch_arithmetic(
            task_groups=16,
            group_size=8,
            global_batch_size=128,
        )
        with pytest.raises(ValueError, match="must be divisible"):
            validate_batch_arithmetic(
                task_groups=16,
                group_size=8,
                global_batch_size=256,
            )

    def test_launcher_temporarily_disables_nounset_while_sourcing_local_entrypoint(self) -> None:
        content = (RECIPE_ROOT / "run_alfworld_qwen2_5_1_5b.sh").read_text(encoding="utf-8")
        disable_index = content.index("    set +u\n")
        source_index = content.index('    source "${PROJECT_ROOT}/scripts/entrypoint/local.sh"\n')
        restore_index = content.index("    set -u\n", source_index)
        self.assertLess(disable_index, source_index)
        self.assertLess(source_index, restore_index)

    def test_managed_agent_entrypoint_uses_reserved_runtime_paths(self) -> None:
        content = (RECIPE_ROOT / "run_agent_app.sh").read_text(encoding="utf-8")
        self.assertIn("RELAX_INPUT_JSON", content)
        self.assertIn("RELAX_OUTPUT_JSON", content)
        self.assertIn("RELAX_BASE_URL", content)
        self.assertIn("ALFWORLD_PYTHON", content)
        self.assertIn("examples.graphgpo.rollout_agent", content)


if __name__ == "__main__":
    unittest.main()
