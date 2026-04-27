# Qwen3-4B DAPO-Math on AMD ROCm

## Goal

Validate a small dense Qwen3 model on AMD ROCm before returning to Qwen3.5 backend enablement.

## Choice

- Model: `Qwen/Qwen3-4B`
- Task: DAPO-Math
- Mode: fully async
- GPUs: 4
- Base recipe: `scripts/training/text/run-qwen3-4B-4xgpu-async.sh`

This is preferred over Qwen3.5 for the first AMD smoke because it is dense and does not require Qwen3.5 GatedDeltaNet / experimental attention support.

## Assets

Assets are stored under `/data/models/minimax/Relax/amd/assets/exps`:

```text
Qwen3-4B/
dapo-math-17k/dapo-math-17k.jsonl
aime-2024/aime-2024.jsonl
```

## Runner

Use:

```bash
/data/models/minimax/Relax/amd/run-qwen3-4b-dapo-math-direct.sh
```

The runner connects directly to a Ray cluster through `RAY_ADDRESS`, avoiding Ray Jobs API issues observed during the Qwen3.5 run.

## Experiment Log

### 2026-04-26

- Downloaded `Qwen/Qwen3-4B` to `amd/assets/exps/Qwen3-4B`.
- Verified config and tokenizer load as `Qwen3Config`.
- Added `amd/run-qwen3-4b-dapo-math-direct.sh`.
- Patched `relax.utils.utils.get_serve_url()` to respect `RELAX_SERVE_PORT` instead of hard-coding `8000`.
- Started an independent 4-GPU Ray cluster:
  - GCS: `10.235.26.199:6380`
  - Dashboard/job server: `http://10.235.26.199:8266`
  - Dashboard agent HTTP/gRPC: `8267` / `8268`
  - Worker ports: `30000-65000`
  - Visible GPUs: `0,1,2,3`
- Verified Ray reports `4.0 GPU` and `ray job list` is empty.
- Launched `amd/run-qwen3-4b-dapo-math-direct.sh`.
- The run progressed through:
  - Ray initialization
  - Ray Serve startup on `RELAX_SERVE_PORT=18081`
  - DCSCoordinator deployment
  - 4-GPU resource validation
  - placement groups for actor, rollout, reference, actor_fwd
  - streaming index build for `dapo-math-17k` (`17398` lines)
- The run failed during rollout service deployment:
  - `ModuleNotFoundError: No module named 'sgl_kernel'`
  - The failure happens while SGLang imports its quantization/MoE runner modules during `ServerArgs` / model config initialization.
- Cleanup:
  - Stopped the independent 4-GPU Ray cluster on `10.235.26.199:6380`.
  - Existing 2-GPU Ray cluster on `127.0.0.1:6379` is still running.
  - GPU utilization returned to 0%.

## Current Blocker

Qwen3-4B avoids the Qwen3.5 Megatron experimental-attention blocker. The current blocker is the missing SGLang runtime package `sgl_kernel`.

`pip index versions sgl-kernel` shows the package is available, latest `0.3.21`, but it is not installed in this ROCm image.

## Dockerfile Update

`docker/Dockerfile.rocm` has been updated to follow SGLang's ROCm build flow for the minimum required kernel package:

1. Build `sgl-kernel` from the checked-out SGLang source.
2. Use `sgl-kernel/pyproject_rocm.toml`.
3. Run `AMDGPU_TARGET=<arch> python setup_rocm.py install`.
4. Install SGLang Python from `python[srt_hip]` after the kernel build.

This should be rebuilt into the image before rerunning Qwen3-4B.

## Current Container Hotfix

Applied the same minimal SGLang ROCm kernel flow directly in the current container:

```bash
cd /sgl-workspace/sglang
python -m pip install --upgrade pip setuptools wheel setuptools_scm scikit-build-core pybind11
cd sgl-kernel
rm -f pyproject.toml
cp pyproject_rocm.toml pyproject.toml
AMDGPU_TARGET=gfx950 python setup_rocm.py install
cd ..
cp python/pyproject_other.toml python/pyproject.toml
pip install -e "python[srt_hip]" --no-build-isolation
```

Verified:

- `sgl-kernel`: `0.3.21.post1`
- `sglang`: `0.5.6.post3.dev2790+g476d371a4`
- `sgl_kernel` imports from `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`
- SGLang ROCm runtime env vars are exported from `/root/.bashrc`
- Reran Qwen3-4B after installing `sgl-kernel`; this passed the previous SGLang import failure.
- New failure:
  - `ValueError: apply_rope_fusion is not available. Please install TE >= 1.4.`
  - Cause: Megatron Bridge provider enabled RoPE fusion, but TransformerEngine is unavailable in this ROCm image.
  - Fix: added `--no-rope-fusion` to `amd/run-qwen3-4b-dapo-math-direct.sh`.
- Reran with `--no-rope-fusion`; RoPE fusion error passed.
- New failure:
  - `NameError: name 'TESpecProvider' is not defined`
  - Cause: Bridge still selected TransformerEngine layer spec while TE is unavailable.
  - Fix: added `--transformer-impl local` to the runner.
- Reran with `--transformer-impl local`; local spec was selected successfully.
- New failures:
  - Actor side: `TypeError: '>=' not supported between instances of 'NoneType' and 'Version'` from Megatron optimizer TE version check.
  - Rollout side: `ModuleNotFoundError: No module named 'aiter'`.
  - Fixes:
    - Removed `--use-precision-aware-optimizer` from the runner.
    - Install AITER in the current container following SGLang ROCm Dockerfile.
- Installed AITER in the current container:
  - Source: `https://github.com/ROCm/aiter.git`
  - Commit/tag: `v0.1.11.post1`
  - Build env: `PREBUILD_KERNELS=1 GPU_ARCHS=gfx950`
  - Verified `import aiter` from `/sgl-workspace/aiter/aiter/__init__.py`
- Reran after AITER install; AITER import worked.
- New validation failure:
  - `--optimizer-cpu-offload` requires `--use-precision-aware-optimizer`
  - But `--use-precision-aware-optimizer` is incompatible with this no-TE environment.
  - Fix: removed `--optimizer-cpu-offload` and `--overlap-cpu-optimizer-d2h-h2d` from the 4B smoke runner.
- Reran after removing optimizer CPU offload:
  - SGLang rollout service deployed successfully.
  - SGLang `/health`, `/server_info`, and `/model_info` responded.
  - DAPO streaming dataset and rollout manager initialized.
- Current blocker:
  - Actor/reference/actor_fwd fail while loading HF weights through Megatron Bridge.
  - Error: `AttributeError: 'NoneType' object has no attribute 'megatron_module'`.
  - Preceded by repeated warnings like `No mapping found for megatron_param: decoder.layers.*.pre_mlp_layernorm.weight`.
  - This indicates a Megatron Bridge conversion mapping issue for Qwen3-4B under the no-TE local spec path.
- Cleanup:
  - Stopped the independent 4-GPU Ray cluster.
