# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Command-line entry point for offline trajectory replay.

Usage:

    python -m relax.tools.trajectory_replay inspect  <bundle>
    python -m relax.tools.trajectory_replay validate <bundle>
    python -m relax.tools.trajectory_replay replay   <bundle> [--stage all]
        [--sample s-0 --group g-1 --batch mb-0007 --step 120:0]

inspect works on incomplete bundles (no COMPLETE required); validate
and replay require a complete, checksum-validated bundle. replay
supports selecting a single sample / group / micro-batch (each expands to its
semantic-group closure); cohort-level stages (loss) are skipped for a partial
selection. --step ROLLOUT_ID:STEP_ID selects that actor step: on a single
bundle it asserts identity (step bundles match actor_step_id; rollout bundles
match rollout_id); on a capture directory it picks the matching bundle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from relax.utils.logging_utils import get_logger
from relax.utils.replay.report import ReplayReport
from relax.utils.replay.runner import replay as run_replay
from relax.utils.replay.schema import index_from_dict, manifest_from_dict
from relax.utils.replay.validate import validate_bundle


logger = get_logger(__name__)


def _inspect(bundle_path: Path) -> int:
    try:
        manifest = manifest_from_dict(json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8")))
        index = index_from_dict(json.loads((bundle_path / "index.json").read_text(encoding="utf-8")))
    except FileNotFoundError as exc:
        logger.error("incomplete bundle: %s", exc)
        return 1
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error("invalid bundle metadata: %s", exc)
        return 1

    identity = index.identity
    logger.info("bundle %s (format %s)", manifest.bundle_id, manifest.format_version)
    if identity.actor_step_id is not None:
        logger.info(
            "actor step: rollout_id=%s step_id=%s", identity.actor_step_id.rollout_id, identity.actor_step_id.step_id
        )
    else:
        logger.info("rollout: rollout_id=%s", identity.rollout_id)
    logger.info("samples: %d", len(index.samples))
    logger.info("producer commit: %s", manifest.producer.commit or "<unset>")
    for stage, contract in sorted(manifest.stage_contracts.items()):
        logger.info("stage %-20s version=%-4s capability=%s", stage.value, contract.version, contract.capability.value)
    for name, spec in sorted(manifest.payloads.items()):
        logger.info("payload %-24s %s shape=%s bytes=%d", name, spec.dtype, spec.shape, spec.bytes)
    return 0


def _validate(bundle_path: Path) -> int:
    result = validate_bundle(bundle_path)
    for message in result.warnings:
        logger.warning("%s", message)
    for message in result.errors:
        logger.error("%s", message)
    if result.valid:
        logger.info("bundle valid: %s", bundle_path)
        return 0
    logger.error("bundle invalid: %s", bundle_path)
    return 1


def _format_report(report: ReplayReport) -> str:
    lines = [f"bundle {report.bundle_id} — first divergent stage: {report.first_divergent_stage or '<none>'}", ""]
    for result in report.stages:
        line = f"[{result.status.value:>7}] {result.stage}"
        if result.message:
            line += f" — {result.message}"
        if result.max_abs_error is not None:
            line += f" (max_abs_error={result.max_abs_error:.3e}, mismatches={result.mismatch_count})"
        lines.append(line)
        for divergence in result.divergences:
            where = divergence.sample_id or "-"
            offset = f"@{divergence.token_offset}" if divergence.token_offset is not None else ""
            lines.append(
                f"          {divergence.field} {where}{offset}: expected={divergence.expected} "
                f"actual={divergence.actual} abs_err={divergence.abs_error}"
            )
    return "\n".join(lines)


def _parse_step(step: str) -> tuple[int, int] | None:
    try:
        rollout_id, step_id = (int(part) for part in step.split(":"))
    except ValueError:
        return None
    return rollout_id, step_id


def _is_bundle(path: Path) -> bool:
    return (path / "index.json").is_file() and (path / "manifest.json").is_file()


def _iter_bundles(root: Path) -> list[Path]:
    """Return bundle directories under root (root itself, children, or rank-*
    children)."""
    if _is_bundle(root):
        return [root]
    if not root.is_dir():
        return []
    found: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if _is_bundle(child):
            found.append(child)
            continue
        if child.name.startswith("rank-"):
            found.extend(path for path in sorted(child.iterdir()) if path.is_dir() and _is_bundle(path))
    return found


def _load_index(bundle_path: Path):
    return index_from_dict(json.loads((bundle_path / "index.json").read_text(encoding="utf-8")))


def _identity_matches_step(index, rollout_id: int, step_id: int) -> str | None:
    """Return step, rollout, or None for how index matches the requested actor
    step."""
    actual = index.identity.actor_step_id
    if actual is not None:
        return "step" if (actual.rollout_id, actual.step_id) == (rollout_id, step_id) else None
    if index.identity.rollout_id == rollout_id:
        return "rollout"
    return None


def _resolve_bundle(bundle_path: Path, step: str | None) -> Path:
    """Resolve a bundle path or a capture directory to one bundle."""
    if _is_bundle(bundle_path):
        return bundle_path
    bundles = _iter_bundles(bundle_path)
    if not bundles:
        raise FileNotFoundError(f"{bundle_path} is not a replay bundle (missing index.json / manifest.json)")
    if step is None:
        if len(bundles) == 1:
            return bundles[0]
        raise ValueError(
            f"{bundle_path} contains {len(bundles)} bundles; pass a specific bundle or --step ROLLOUT_ID:STEP_ID"
        )
    parsed = _parse_step(step)
    if parsed is None:
        raise ValueError(f"invalid --step {step!r} (expected ROLLOUT_ID:STEP_ID)")
    rollout_id, step_id = parsed
    step_hits: list[Path] = []
    rollout_hits: list[Path] = []
    for path in bundles:
        kind = _identity_matches_step(_load_index(path), rollout_id, step_id)
        if kind == "step":
            step_hits.append(path)
        elif kind == "rollout":
            rollout_hits.append(path)
    hits = step_hits or rollout_hits
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        names = ", ".join(path.name for path in hits)
        raise ValueError(f"multiple bundles match actor step {rollout_id}:{step_id}: {names}")
    raise ValueError(f"no bundle in {bundle_path} matches actor step {rollout_id}:{step_id}")


def _replay(
    bundle_path: Path, stages: str, sample: list[str], group: list[str], batch: list[str], step: str | None
) -> int:
    if step is not None and _parse_step(step) is None:
        logger.error("invalid --step %r (expected ROLLOUT_ID:STEP_ID)", step)
        return 1
    try:
        bundle_path = _resolve_bundle(bundle_path, step)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        logger.error("%s", exc)
        return 1
    if step is not None:
        parsed = _parse_step(step)
        assert parsed is not None
        rollout_id, step_id = parsed
        if _identity_matches_step(_load_index(bundle_path), rollout_id, step_id) is None:
            logger.error("bundle is not the requested actor step %d:%d", rollout_id, step_id)
            return 1

    try:
        report = run_replay(bundle_path, sample_ids=sample or None, group_ids=group or None, batch_ids=batch or None)
    except Exception as exc:  # noqa: BLE001 — surface any replay failure to the user
        logger.error("replay failed: %s", exc)
        return 1

    if stages != "all":
        selected = set(stages.split(","))
        report.stages = [stage for stage in report.stages if stage.stage in selected]

    for line in _format_report(report).splitlines():
        logger.info("%s", line)
    return 0 if report.passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relax.tools.trajectory_replay", description="Offline trajectory replay")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="show identity, capability and payload summary")
    inspect_parser.add_argument("bundle", type=Path)

    validate_parser = subparsers.add_parser("validate", help="validate format, integrity and closure")
    validate_parser.add_argument("bundle", type=Path)

    replay_parser = subparsers.add_parser("replay", help="recompute stages and compare against expected outputs")
    replay_parser.add_argument("bundle", type=Path)
    replay_parser.add_argument("--stage", default="all", help="comma-separated stage filter (default: all)")
    replay_parser.add_argument("--sample", action="append", default=[], help="sample_id to replay (repeatable)")
    replay_parser.add_argument("--group", action="append", default=[], help="semantic group id to replay (repeatable)")
    replay_parser.add_argument("--batch", action="append", default=[], help="micro-batch id to replay (repeatable)")
    replay_parser.add_argument(
        "--step",
        help="select actor step ROLLOUT_ID:STEP_ID (assert on a bundle; pick from a capture directory)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        return _inspect(args.bundle)
    if args.command == "validate":
        return _validate(args.bundle)
    if args.command == "replay":
        return _replay(args.bundle, args.stage, args.sample, args.group, args.batch, args.step)
    return 1


if __name__ == "__main__":
    sys.exit(main())
