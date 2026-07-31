# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Sequence

from relax.utils.logging_utils import get_logger
from relax.utils.reproducibility import (
    ManifestError,
    execute_manifest,
    inspect_environment,
    load_manifest,
    replay_command,
)


logger = get_logger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check or replay a Relax experiment manifest.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    check_parser = subparsers.add_parser("check", help="Compare the current environment with a manifest.")
    check_parser.add_argument("manifest", type=Path)

    rerun_parser = subparsers.add_parser("rerun", help="Preview or execute the recorded command.")
    rerun_parser.add_argument("manifest", type=Path)
    rerun_parser.add_argument(
        "--execute", action="store_true", help="Execute after validation; default is preview only."
    )
    rerun_parser.add_argument("--allow-drift", action="store_true", help="Execute even if environment drift is found.")
    return parser


def _log_differences(differences: list[dict]) -> None:
    for difference in differences:
        logger.warning(
            "%s differs: expected=%r actual=%r. %s",
            difference["field"],
            difference["expected"],
            difference["actual"],
            difference["suggestion"],
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        differences = inspect_environment(manifest)
        if args.action == "check":
            if differences:
                _log_differences(differences)
                return 1
            logger.info("Environment matches the manifest.")
            return 0

        command = replay_command(manifest)
        logger.info("Recorded command: %s", shlex.join(command))
        if differences:
            _log_differences(differences)
        if not args.execute:
            logger.info("Preview only. Add --execute to rerun the command.")
            return 0
        if differences and not args.allow_drift:
            logger.error("Replay stopped because environment drift was found. Use --allow-drift to override.")
            return 1
        return execute_manifest(manifest)
    except ManifestError as error:
        logger.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
