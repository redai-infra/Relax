#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Entrypoint for the Relax rollout result viewer.

Thin wrapper around :func:`relax.utils.visualize.main` so the viewer can
be launched the same way as other Relax entrypoints::

    python -m relax.entrypoints.visualize <save>/rollout_result --port 8080
"""

from relax.utils.visualize import main


if __name__ == "__main__":
    main()
