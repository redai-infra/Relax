# AMD ROCm Validation

This directory tracks the AMD/ROCm validation work for Relax.

## Current Target

- Task: Qwen3.5-9B DAPO-Math
- Hardware: AMD Instinct MI355X, single node, 8 GPUs
- Launch script: `scripts/training/text/run-qwen35-9B-8xgpu-async.sh`
- Mode: fully async

## Files

- `qwen35-9b-dapo-math.md`: runbook and experiment notes for the current validation.
