# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from examples.graphgpo.prepare_alfworld import (
    PINNED_ALFWORLD_VERSION,
    PREPARE_SCHEMA_VERSION,
    eligible_task_files,
    prepare_artifacts,
    prompt_rows_bytes,
)


class PrepareAlfWorldTest(unittest.TestCase):
    def _write_task(
        self,
        root: Path,
        *,
        split: str,
        task_dir: str,
        task_type: str = "pick_and_place_simple",
        solvable: bool = True,
    ) -> Path:
        path = root / "json_2.1.1" / split / task_dir / "trial_1" / "game.tw-pddl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"solvable": solvable}), encoding="utf-8")
        path.with_name("traj_data.json").write_text(
            json.dumps({"task_type": task_type}),
            encoding="utf-8",
        )
        return path

    def _write_shared(self, root: Path) -> None:
        logic = root / "logic"
        logic.mkdir()
        (logic / "alfred.pddl").write_text("domain", encoding="utf-8")
        (logic / "alfred.twl2").write_text("grammar", encoding="utf-8")

    def test_prepare_writes_stable_manifests_prompt_rows_and_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "prepared"
            self._write_shared(root)
            self._write_task(root, split="train", task_dir="pick_and_place_simple-B")
            self._write_task(root, split="train", task_dir="pick_and_place_simple-A")
            self._write_task(root, split="valid_seen", task_dir="look_at_obj_in_light-A")

            first = prepare_artifacts(
                data_root=root,
                output_dir=output,
                split_roots={
                    "train": root / "json_2.1.1" / "train",
                    "eval_in_distribution": root / "json_2.1.1" / "valid_seen",
                },
                limits={"eval_in_distribution": 1},
                max_steps=50,
            )
            second = prepare_artifacts(
                data_root=root,
                output_dir=output,
                split_roots={
                    "train": root / "json_2.1.1" / "train",
                    "eval_in_distribution": root / "json_2.1.1" / "valid_seen",
                },
                limits={"eval_in_distribution": 1},
                max_steps=50,
            )

            self.assertEqual(first, second)
            self.assertEqual(first["schema_version"], PREPARE_SCHEMA_VERSION)
            self.assertEqual(first["alfworld_version"], PINNED_ALFWORLD_VERSION)
            self.assertEqual(first["splits"]["train"]["task_count"], 2)
            lock = json.loads((output / "prepare.lock.json").read_text(encoding="utf-8"))
            self.assertEqual(lock, first)

            train_rows = [
                json.loads(line) for line in (output / "train.prompts.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(train_rows), 2)
            self.assertEqual(
                [row["metadata"]["manifest_index"] for row in train_rows],
                [0, 1],
            )
            self.assertTrue(
                train_rows[0]["metadata"]["manifest_task_id"] < train_rows[1]["metadata"]["manifest_task_id"]
            )
            self.assertEqual(
                {row["metadata"]["alfworld_train_eval"] for row in train_rows},
                {"train"},
            )
            self.assertEqual(
                {
                    (
                        row["metadata"]["temperature"],
                        row["metadata"]["top_p"],
                        row["metadata"]["max_tokens"],
                    )
                    for row in train_rows
                },
                {(1.0, 1.0, 512)},
            )
            self.assertTrue(all(row["metadata"]["task_id"].endswith("/game.tw-pddl") for row in train_rows))
            self.assertEqual(
                [row["metadata"]["task_id"] for row in train_rows],
                [row["metadata"]["manifest_task_id"] for row in train_rows],
            )
            self.assertEqual(
                first["splits"]["eval_in_distribution"]["sampling"],
                {"max_tokens": 512, "temperature": 0.4, "top_p": 1.0},
            )

    def test_prepare_filters_tasks_the_text_environment_would_skip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_shared(root)
            valid = self._write_task(
                root,
                split="train",
                task_dir="pick_and_place_simple-A",
            )
            self._write_task(
                root,
                split="train",
                task_dir="pick_and_place_simple-movable",
            )
            self._write_task(
                root,
                split="train",
                task_dir="pick_and_place_simple-Sliced",
            )
            self._write_task(
                root,
                split="train",
                task_dir="pick_and_place_simple-unsolvable",
                solvable=False,
            )
            self._write_task(
                root,
                split="train",
                task_dir="pick_and_place_simple-unknown",
                task_type="not_supported",
            )

            self.assertEqual(
                eligible_task_files(root / "json_2.1.1" / "train"),
                [valid],
            )

    def test_prepare_refuses_to_replace_changed_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "prepared"
            self._write_shared(root)
            game = self._write_task(
                root,
                split="train",
                task_dir="pick_and_place_simple-A",
            )
            arguments = {
                "data_root": root,
                "output_dir": output,
                "split_roots": {"train": root / "json_2.1.1" / "train"},
            }
            prepare_artifacts(**arguments)
            game.write_text(json.dumps({"solvable": True, "changed": True}), encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "refusing to replace"):
                prepare_artifacts(**arguments)

    def test_prompt_rows_reject_empty_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_shared(root)
            empty_split = root / "json_2.1.1" / "train"
            empty_split.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "no eligible tasks"):
                prepare_artifacts(
                    data_root=root,
                    output_dir=root / "prepared",
                    split_roots={"train": empty_split},
                )

            with self.assertRaisesRegex(ValueError, "positive integer"):
                prompt_rows_bytes(object(), max_steps=0)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
