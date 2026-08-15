# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Command-line entry point for immutable GraphGPO diagnostic summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from examples.graphgpo.diagnostics import write_graph_diagnostics_summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    write_graph_diagnostics_summary(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
