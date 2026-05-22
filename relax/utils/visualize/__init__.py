# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Relax rollout result viewer.

A lightweight web UI for browsing the per-step JSONL files written by
:func:`relax.utils.training.train_dump_utils.save_rollout_result_jsonl`
and :func:`save_eval_summary_jsonl`.

Usage::

    python -m relax.utils.visualize <save>/rollout_result --port 8080
"""

from relax.utils.visualize.server import create_app, main


__all__ = ["create_app", "main"]
