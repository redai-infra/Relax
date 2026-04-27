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

## ROCm TransformerEngine Submodule

Added ROCm TransformerEngine as a submodule:

```text
third_party/TransformerEngine -> https://github.com/ROCm/TransformerEngine.git (branch: dev)
```

ROCm TE docs describe two install paths.

Wheel install for ROCm 7.2:

```bash
wget -r -l1 -nd -A 'transformer_engine*' https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/
pip install ./transformer_engine* --no-build-isolation
```

Source install from the submodule:

```bash
cd third_party/TransformerEngine
git submodule update --init --recursive
export NVTE_FRAMEWORK=pytorch
export NVTE_ROCM_ARCH=gfx950
export NVTE_USE_ROCM=1
pip install --no-build-isolation .
```

If the HIP compiler cannot detect the platform, also export:

```bash
export HIP_PLATFORM=amd
```

## Current Container TransformerEngine Install

Installed ROCm TE wheels for ROCm 7.2 into the current container:

```bash
mkdir -p /tmp/te-rocm72
cd /tmp/te-rocm72
wget -r -l1 -nd -A 'transformer_engine*' https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/
pip install ./transformer_engine-2.4.0-py3-none-any.whl \
  ./transformer_engine_rocm-2.4.0-py3-none-manylinux_2_28_x86_64.whl \
  ./transformer_engine_torch-2.4.0.tar.gz \
  --no-build-isolation
```

Verified:

- `transformer_engine`: `2.4.0`
- `transformer_engine_torch`: `2.4.0`
- `transformer_engine.pytorch.LayerNormLinear`: available
- `transformer_engine.pytorch.RMSNorm`: available
- `transformer_engine.pytorch.DotProductAttention`: available

After TE install, removed the no-TE runner flags:

- `--no-rope-fusion`
- `--transformer-impl local`
- Reran Qwen3-4B with TE installed:
  - Megatron Bridge mapping issue passed.
  - `actor`, `actor_fwd`, `reference`, `rollout`, and `advantages` all registered successfully.
  - Rollout completed and transferred data.
  - Actor-to-rollout weight update completed.
  - Training reached actor/reference log-prob and actor train step.
- New blocker:
  - `ValueError: No dot product attention backend is available for the provided inputs.`
  - Raised from `transformer_engine.pytorch.attention.dot_product_attention.DotProductAttention.forward`.
  - Reference service became unhealthy and triggered global restart.
  - Next debugging direction: run with `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2` to see why TE disables all attention backends, or switch training attention backend away from TE fused attention for this smoke.
- NVTE debug / static backend selection showed the likely backend issue:
  - Current training args use `qkv_format=thd`.
  - TE ROCm selector reports `thd_thd_thd + causal` has no backend.
  - `thd` disables `UnfusedDotProductAttention`.
  - `sbhd` / `bshd` layouts have available fused or unfused backends.
  - Fix under test: set `--qkv-format bshd` in the 4B smoke runner.
- Reran with `--qkv-format bshd`:
  - The TE dot-product attention backend error did not recur during initial log-prob/training.
  - All 5 services registered successfully.
  - Rollout generation completed the first 16 samples and transferred rollout batches.
  - Actor training entered `MegatronTrainRayActor.train_async`.
  - Run was still active at the time of this note.
- Adjusted `relax/backends/megatron/arguments.py` so `qkv_format=bshd` keeps `variable_seq_lengths=False`.
- Reran `bshd + variable_seq_lengths=False`:
  - CLI args confirmed `qkv_format=bshd` and `variable_seq_lengths=False`.
  - This run ended early with Ray/GCS disconnect (`Failed to connect to GCS within 60 seconds`) before producing a useful training-side result.
  - GPU utilization returned to 0% after cleanup.
- Added temporary TransformerEngine attention diagnostics in the current container:
  - `dot_product_attention/utils.py` now warns when backend selection ends in `NoBackend`, including `run_config` and backend candidates.
  - `dot_product_attention/dot_product_attention.py` now logs q/k/v tensor metadata, qkv layout, mask type, sequence lengths, and selected backend flags before raising the `No dot product attention backend` error.
  - The direct runner now defaults `NVTE_DEBUG=1`, `NVTE_DEBUG_LEVEL=2`, and `RAY_DEDUP_LOGS=0` so TE and Ray preserve the backend rejection details in logs.
- Added `amd/run_qwen3-4b.sh` as the one-command smoke runner:
  - Kills stale Relax, Ray, SGLang, Megatron worker, and Ray dashboard/worker processes at startup.
  - Recreates `/tmp/ray-qwen3-4b` and starts a fresh local Ray head on `10.235.26.199:6380`.
  - Exports all Qwen3-4B DAPO-Math, ROCm TE debug, Ray, and Serve settings before invoking `amd/run-qwen3-4b-dapo-math-direct.sh`.
  - Intended usage: edit variables inside the script, then run `bash amd/run_qwen3-4b.sh`.
- TE backend root cause from `qwen3-4b-dapo-math-te-debug-20260427-153023`:
  - The launch used `--attention-backend flash`, and Megatron set `NVTE_FUSED_ATTN=0` plus `NVTE_UNFUSED_ATTN=0`.
  - ROCm TE reported `flash_attn_version='not installed'`, so FlashAttention was unavailable.
  - With fused/unfused forced off and flash missing, backend selection ended as `available_backends=[False, 0, 0]`.
  - Changed the smoke runner to `--attention-backend auto` so TE can fall back to fused or unfused on ROCm.
- Reran with `--attention-backend auto`:
  - TE selected `FusedAttention backend (sub-backend 1)`.
  - Step 0 and step 1 logprob/training passed the previous attention blocker.
  - New blocker at step 2: TorchDynamo fake tensor failure in Megatron `fused_cross_entropy.py` (`torch.split(..., SymInt)`).
  - Added `TORCHDYNAMO_DISABLE=1`, `--disable-jit-fuser`, and `--train-env-vars '{"TORCHDYNAMO_DISABLE": "1"}'` to avoid the Dynamo compiled fused CE path for ROCm smoke.
- Reran with Dynamo/JIT fuser disabled:
  - All 5 services registered successfully.
  - TE selected `FusedAttention backend (sub-backend 1)`.
  - Rollout, reference logprob, actor_fwd logprob, advantages, and actor training completed all 4 smoke steps.
  - Checkpoint saved successfully at iteration 3 under `Qwen3-4B_mcore_4xgpu/`.
  - Process exited with code 0.
