# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from examples.graphgpo.preflight import (
    MODEL_LOCK_SCHEMA,
    MODEL_REPO_ID,
    verify_model_checkpoint,
    verify_prepared_artifacts,
)
from examples.graphgpo.prepare_alfworld import (
    PINNED_MODEL_REVISION,
    prepare_artifacts,
)


class PreflightTest(unittest.TestCase):
    def _write_task(self, root: Path, task_dir: str) -> None:
        task_root = root / "json_2.1.1" / "train" / task_dir / "trial_1"
        task_root.mkdir(parents=True)
        (task_root / "game.tw-pddl").write_text(json.dumps({"solvable": True}), encoding="utf-8")
        (task_root / "traj_data.json").write_text(
            json.dumps({"task_type": "pick_and_place_simple"}),
            encoding="utf-8",
        )

    def _prepare(self, root: Path) -> tuple[Path, Path, Path]:
        logic = root / "logic"
        logic.mkdir()
        (logic / "alfred.pddl").write_text("domain", encoding="utf-8")
        (logic / "alfred.twl2").write_text("grammar", encoding="utf-8")
        self._write_task(root, "pick_and_place_simple-B")
        self._write_task(root, "pick_and_place_simple-A")
        output = root / "prepared"
        prepare_artifacts(
            data_root=root,
            output_dir=output,
            split_roots={"train": root / "json_2.1.1" / "train"},
            max_steps=50,
        )
        return (
            output / "prepare.lock.json",
            output / "train.prompts.jsonl",
            output / "train.manifest.json",
        )

    def _verify(self, lock: Path, prompt: Path, manifest: Path) -> None:
        verify_prepared_artifacts(
            prepare_lock_path=lock,
            split_artifacts={"train": (prompt, manifest)},
            max_steps=50,
            model_revision=PINNED_MODEL_REVISION,
            alfworld_data_root=lock.parent.parent,
        )

    def test_preflight_accepts_matching_lock_manifest_and_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            lock, prompt, manifest = self._prepare(Path(temporary_directory))
            self._verify(lock, prompt, manifest)

    def test_preflight_rejects_model_max_steps_and_prompt_sha_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            lock, prompt, manifest = self._prepare(Path(temporary_directory))
            with self.assertRaisesRegex(ValueError, "model_revision"):
                verify_prepared_artifacts(
                    prepare_lock_path=lock,
                    split_artifacts={"train": (prompt, manifest)},
                    max_steps=50,
                    model_revision="0" * 40,
                    alfworld_data_root=lock.parent.parent,
                )
            with self.assertRaisesRegex(ValueError, "max_steps"):
                verify_prepared_artifacts(
                    prepare_lock_path=lock,
                    split_artifacts={"train": (prompt, manifest)},
                    max_steps=49,
                    model_revision=PINNED_MODEL_REVISION,
                    alfworld_data_root=lock.parent.parent,
                )
            prompt.write_text(prompt.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prompt SHA256"):
                self._verify(lock, prompt, manifest)

    def test_preflight_rejects_split_count_and_task_order_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            lock, prompt, manifest = self._prepare(Path(temporary_directory))
            lock_payload = json.loads(lock.read_text(encoding="utf-8"))
            lock_payload["splits"]["train"]["task_count"] = 3
            lock.write_text(json.dumps(lock_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "task count"):
                self._verify(lock, prompt, manifest)

        with tempfile.TemporaryDirectory() as temporary_directory:
            lock, prompt, manifest = self._prepare(Path(temporary_directory))
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            manifest_payload["split"] = "eval_in_distribution"
            manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
            lock_payload = json.loads(lock.read_text(encoding="utf-8"))
            lock_payload["splits"]["train"]["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            lock.write_text(json.dumps(lock_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest split"):
                self._verify(lock, prompt, manifest)

        with tempfile.TemporaryDirectory() as temporary_directory:
            lock, prompt, manifest = self._prepare(Path(temporary_directory))
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            manifest_payload["tasks"].reverse()
            manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
            lock_payload = json.loads(lock.read_text(encoding="utf-8"))
            lock_payload["splits"]["train"]["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            lock.write_text(json.dumps(lock_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stable order"):
                self._verify(lock, prompt, manifest)

    def test_preflight_rehashes_manifest_referenced_alfworld_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            lock, prompt, manifest = self._prepare(root)
            referenced_game = next((root / "json_2.1.1" / "train").rglob("game.tw-pddl"))
            referenced_game.write_text("changed", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "size does not match"):
                self._verify(lock, prompt, manifest)

    def test_model_lock_rehashes_all_nine_checkpoint_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            files: dict[str, dict[str, object]] = {}
            for index in range(9):
                name = f"file-{index}.bin"
                content = f"content-{index}".encode()
                (checkpoint / name).write_bytes(content)
                files[name] = {
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            model_lock = root / "model.lock.json"
            model_lock.write_text(
                json.dumps(
                    {
                        "schema": MODEL_LOCK_SCHEMA,
                        "repo_id": MODEL_REPO_ID,
                        "revision": PINNED_MODEL_REVISION,
                        "files": files,
                    }
                ),
                encoding="utf-8",
            )

            verify_model_checkpoint(
                model_lock_path=model_lock,
                checkpoint_path=checkpoint,
                model_revision=PINNED_MODEL_REVISION,
            )
            (checkpoint / "file-8.bin").write_bytes(b"changed-8")
            with self.assertRaisesRegex(ValueError, "SHA256 does not match"):
                verify_model_checkpoint(
                    model_lock_path=model_lock,
                    checkpoint_path=checkpoint,
                    model_revision=PINNED_MODEL_REVISION,
                )


if __name__ == "__main__":
    unittest.main()
