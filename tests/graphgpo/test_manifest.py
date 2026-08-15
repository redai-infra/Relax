# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import hashlib
import tempfile
import unittest
from pathlib import Path

from examples.graphgpo.manifest import (
    MANIFEST_VERSION,
    build_manifest,
    infer_task_type,
    manifest_bytes,
    manifest_sha256,
)


class ManifestTest(unittest.TestCase):
    def _make_task(self, root: Path, task_dir: str, content: bytes) -> Path:
        path = root / task_dir / "trial_1" / "game.tw-pddl"
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        path.with_name("traj_data.json").write_bytes(b'{"task":"metadata"}')
        return path

    def _make_shared_assets(self, root: Path) -> None:
        shared_root = root / "shared"
        shared_root.mkdir()
        (shared_root / "alfred.pddl").write_bytes(b"pddl")
        (shared_root / "alfred.twl2").write_bytes(b"grammar")

    def test_manifest_has_fixed_path_order_and_content_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._make_shared_assets(root)
            later = self._make_task(
                root,
                "pick_heat_then_place_in_recep-Z",
                b"later",
            )
            earlier = self._make_task(
                root,
                "pick_and_place_simple-A",
                b"earlier",
            )

            manifest = build_manifest(
                root,
                split="eval_in_distribution",
                task_files=[later, earlier],
            )

            self.assertEqual(manifest.schema_version, MANIFEST_VERSION)
            self.assertEqual(
                [task.task_type for task in manifest.tasks],
                ["pick_and_place_simple", "pick_heat_then_place_in_recep"],
            )
            self.assertEqual(
                manifest.tasks[0].game.sha256,
                hashlib.sha256(b"earlier").hexdigest(),
            )
            self.assertTrue(manifest.tasks[0].game.relative_path.endswith("game.tw-pddl"))
            self.assertTrue(manifest.tasks[0].trajectory.relative_path.endswith("traj_data.json"))
            self.assertEqual(
                {task.split for task in manifest.tasks},
                {"eval_in_distribution"},
            )
            self.assertEqual(
                [asset.asset_name for asset in manifest.shared_assets],
                ["alfred.pddl", "alfred.twl2"],
            )

    def test_discovery_and_serialization_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._make_shared_assets(root)
            self._make_task(root, "pick_cool_then_place_in_recep-B", b"b")
            self._make_task(root, "look_at_obj_in_light-A", b"a")

            first = build_manifest(root, split="train")
            second = build_manifest(root, split="train")

            self.assertEqual(first, second)
            self.assertEqual(manifest_bytes(first), manifest_bytes(second))
            self.assertEqual(manifest_sha256(first), manifest_sha256(second))

    def test_outside_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root_directory:
            with tempfile.TemporaryDirectory() as outside_directory:
                root = Path(root_directory)
                self._make_shared_assets(root)
                outside = self._make_task(
                    Path(outside_directory),
                    "pick_and_place_simple-A",
                    b"outside",
                )

                with self.assertRaisesRegex(ValueError, "outside root"):
                    build_manifest(root, split="train", task_files=[outside])

    def test_unknown_task_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot infer"):
            infer_task_type("unknown-task/trial/game.tw-pddl")

    def test_missing_trajectory_or_shared_asset_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            game = root / "pick_and_place_simple-A" / "trial_1" / "game.tw-pddl"
            game.parent.mkdir(parents=True)
            game.write_bytes(b"game")
            self._make_shared_assets(root)

            with self.assertRaises(FileNotFoundError):
                build_manifest(root, split="train", task_files=[game])

            game.with_name("traj_data.json").write_bytes(b"{}")
            (root / "shared" / "alfred.twl2").unlink()
            with self.assertRaisesRegex(ValueError, "expected exactly one"):
                build_manifest(root, split="train", task_files=[game])


if __name__ == "__main__":
    unittest.main()
