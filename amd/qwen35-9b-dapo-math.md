# Qwen3.5-9B DAPO-Math on AMD ROCm

## Goal

Validate that Relax can run Qwen3.5-9B DAPO-Math on AMD Instinct MI355X with the ROCm image.

## Initial Plan

1. Use the existing ROCm container; a single-node 8-GPU run does not need multiple Docker containers.
2. Activate the image environment from `/root/.bashrc`.
3. Verify ROCm, PyTorch, Ray, and Relax import paths.
4. Verify model and dataset assets.
5. Launch `scripts/training/text/run-qwen35-9B-8xgpu-async.sh`.
6. Record failures and fixes here as the source of truth for this validation.

## Expected Assets

The launch script expects `MODEL_DIR` to contain:

```text
Qwen3.5-9B/
dapo-math-17k/dapo-math-17k.jsonl
aime-2024/aime-2024.jsonl
Qwen3-9B_mcore_8xgpu/        # created or reused for checkpoints
```

## Environment Notes

- ROCm GPU visibility should be checked with both `rocm-smi` and PyTorch.
- On this image, AMD devices are exposed through PyTorch's `torch.cuda` API.
- If non-interactive shells skip `/root/.bashrc` setup, run commands with `PS1` set or use `/opt/venv/bin/*` directly.

## Experiment Log

### 2026-04-26

- Created this AMD validation directory.
- Environment check passed:
  - Python: `/opt/venv/bin/python`
  - Ray: `/opt/venv/bin/ray`, version 2.55.1
  - PyTorch: 2.9.1 ROCm 7.2
  - GPUs: 8 x AMD Instinct MI355X visible through `torch.cuda`
  - Relax import path: `/data/models/minimax/Relax/relax/__init__.py`
- Downloaded assets under `/data/models/minimax/Relax/amd/assets/exps`:
  - `Qwen/Qwen3.5-9B` -> `Qwen3.5-9B/`
  - `zhuzilin/dapo-math-17k` -> `dapo-math-17k/`
  - `zhuzilin/aime-2024` -> `aime-2024/`
- Processed AIME in place with `scripts/tools/process_aime.py`.
- Asset check passed for model config/tokenizer and both dataset JSONL files.
- Launch run directory: `/data/models/minimax/Relax/amd/runs/qwen35-9b-dapo-math-20260426-153106`
- Launch command environment:
  - `MODEL_DIR=/data/models/minimax/Relax/amd/assets/exps`
  - `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
  - `NUM_GPUS=8`
  - `HOST_IP=127.0.0.1`
  - `RAY_NO_WAIT=1`
  - `RAY_TMPDIR=<run-dir>/ray_tmp`
- First launch failed before job submission:
  - Failure: Ray plasma store Unix socket path exceeded the 107-byte AF_UNIX limit.
  - Cause: `RAY_TMPDIR` under the deep `amd/runs/...` path made Ray's session socket path too long.
  - Fix: keep logs in `amd/runs`, but use a short Ray temp path such as `/tmp/ray-q35-153106`.
- Second launch also failed before job submission:
  - Failure: Ray GCS could not bind port `6379`.
  - Evidence: Ray `gcs_server.err` reports `Address already in use` for `0.0.0.0:6379`.
  - Container tools did not expose the owning PID, so this appears to be occupied outside this run.
  - Fix: manually start Ray on a non-default GCS port (`6380`) and submit through `scripts/entrypoint/ray-job.sh`.
- Found an existing Ray cluster on `6379/8265`, but it only exposes 2 GPUs (`CUDA_VISIBLE_DEVICES=1,6`), so it cannot run this 8-GPU recipe.
- Started an independent Ray head:
  - GCS: `10.235.26.199:6380`
  - Dashboard/job server: `http://10.235.26.199:8266`
  - Resources: 8 GPUs
- Patched the Qwen3.5-9B launch script so the Ray job server port can be overridden with `RAY_DASHBOARD_PORT`.
- First submission to the independent Ray dashboard failed because the dashboard agent tried to bind port `52365`, which was already used by the existing 2-GPU Ray cluster.
- Restarted the independent Ray head with explicit non-conflicting ports:
  - GCS: `6380`
  - dashboard: `8266`
  - dashboard agent HTTP: `8267`
  - dashboard agent gRPC: `8268`
  - node manager: `6381`
  - object manager: `6382`
  - metrics: `6383`
  - worker ports: `20000-20199`
- Verified `ray job list --address=http://10.235.26.199:8266` works and reports no jobs.
- Resubmitting through Ray Jobs still hung before a job appeared in `ray job list`.
- Stopped the stuck `ray job submit` process; no training job had been created.
- Added `amd/run-qwen35-9b-dapo-math-direct.sh` to bypass Ray Jobs and run the Relax driver directly against `RAY_ADDRESS=10.235.26.199:6380`.
- First direct driver run reached Relax/Megatron argument parsing, then failed because `CUDA_DEVICE_MAX_CONNECTIONS=1` was missing in the direct runner environment.
- Added `CUDA_DEVICE_MAX_CONNECTIONS=1` to the direct runner.
- Second direct driver run passed argument validation and connected to Ray, then failed during worker registration:
  - Failure: `No available ports. Please specify a wider port range using --min-worker-port and --max-worker-port.`
  - Cause: the independent Ray head was started with a too-narrow worker range (`20000-20199`).
  - Fix: restart the independent Ray head with a wider worker range (`30000-65000`).
- Restarted Ray with the wider worker range and relaunched the direct driver.
- Third direct run progressed through:
  - Ray initialization
  - Ray Serve startup
  - DCSCoordinator and MetricsService deployment
  - 8-GPU resource validation
  - placement group allocation for actor/rollout/reference/actor_fwd
- Third direct run failed during service deployment:
  - Rollout/SGLang failure: `ModuleNotFoundError: No module named 'sgl_kernel'`.
  - Actor/Megatron failure: `NotImplementedError: Experimental attention variant is not supported with local spec yet.`
- Megatron expert conclusion:
  - Qwen3.5-9B uses GatedDeltaNet / experimental attention (`gated_delta_net`) plus attention output gate.
  - Current ROCm image does not provide `transformer_engine`, `fla`, `causal_conv1d`, or `sgl_kernel`.
  - Megatron local spec explicitly does not support `experimental_attention_variant`, so this is not a config-only failure.
  - Removing `--use-gated-attention` or `--attention-output-gate` would invalidate the Qwen3.5 architecture and checkpoint shapes.
  - Running Qwen3.5 on ROCm requires real backend enablement: ROCm-compatible TE/FLA/GDN support or local Megatron GDN implementation, plus SGLang ROCm kernel dependency handling.
- Cleanup:
  - Stopped the independent 8-GPU Ray cluster on `10.235.26.199:6380`.
  - The pre-existing 2-GPU Ray cluster on `127.0.0.1:6379` is still running.
  - GPU utilization returned to 0%.

## Current Conclusion

Qwen3.5-9B DAPO-Math cannot currently be run to training on this ROCm image with only launch-script changes. The run reached Ray/Serve placement and model service initialization, then failed on missing Qwen3.5 backend support in Megatron/SGLang.

For AMD validation now, use a supported Qwen3/Qwen3-MoE recipe first. For Qwen3.5 specifically, the next work item is model-backend integration rather than launch orchestration.

## Follow-up Runner

Added `amd/run_qwen35-9b.sh` as the one-command Qwen3.5-9B smoke runner:

- Cleans stale Relax, Ray, SGLang, Megatron worker, and Ray dashboard/worker processes at startup.
- Recreates `/tmp/ray-qwen35-9b` and starts a fresh local Ray head on `10.235.26.199:6380` with 8 GPUs.
- Exports the ROCm TE debug settings validated in the Qwen3-4B run.
- Aligns the direct runner with the Qwen3-4B ROCm fixes:
  - `--attention-backend auto`
  - `TORCHDYNAMO_DISABLE=1`
  - `--disable-jit-fuser`
  - `--train-env-vars '{"TORCHDYNAMO_DISABLE": "1"}'`

Intended usage:

```bash
bash amd/run_qwen35-9b.sh
```

### 2026-04-27 Follow-up

- Ran `bash amd/run_qwen35-9b.sh`.
- The wrapper correctly restarted Ray with 8 GPUs and launched the direct runner with:
  - `attention_backend=auto`
  - `disable_jit_fuser=True`
  - `train_env_vars={'TORCHDYNAMO_DISABLE': '1'}`
- The run passed Ray/Serve startup and began deploying actor, rollout, reference, actor_fwd, and advantages services.
- New root blocker:
  - `ImportError: FLA is not installed. Please install it with pip install flash-linear-attention.`
  - Raised while instantiating Megatron `GatedDeltaNet`, then `TransformerLayer`.
  - This confirms Qwen3.5-9B requires FLA/GatedDeltaNet backend support before training can proceed.
- Stopped the run after capturing the error to avoid Serve restart loops.
- Checked SGLang's ROCm Dockerfile:
  - It installs AITER, SGLang ROCm extras, `sgl-kernel`, TileLang, and related ROCm serving dependencies.
  - It does not explicitly install `flash-linear-attention`.
  - The missing FLA error is from Megatron's `GatedDeltaNet`, not directly from SGLang.
- Installed `flash-linear-attention==0.5.0` in the current container and verified Megatron's required imports:
  - `fla.modules.convolution.causal_conv1d`
  - `fla.modules.l2norm.l2norm`
  - `fla.ops.gated_delta_rule.chunk_gated_delta_rule`
- Updated `docker/Dockerfile.rocm` to install ROCm TransformerEngine 2.4.0 wheels and `flash-linear-attention==0.5.0` for future images.
- Reran after installing FLA:
  - Megatron `GatedDeltaNet` initialization passed the previous FLA import blocker.
  - Megatron actor/reference began loading the Qwen3.5 HF checkpoint.
  - New rollout blocker: SGLang TP=2 tried to use device ordinal 1 while the Ray actor saw only `CUDA_VISIBLE_DEVICES='4'`.
  - Changed the 9B smoke runner to `--rollout-num-gpus-per-engine 1` so the two rollout GPUs run as two 1-GPU SGLang engines for the next smoke attempt.
- Reran with two 1-GPU rollout engines:
  - The invalid device ordinal issue did not recur.
  - All 5 services registered successfully.
  - Actor, rollout, reference, actor_fwd, and advantages entered step 0.
  - TE selected `FusedAttention backend (sub-backend 1)` for Megatron logprob/training.
  - New blocker: one SGLang engine hit HIP OOM in logits allocation while full token usage reached ~1.0.
  - This was with the original full-style rollout settings (`num_rollout=1000`, `rollout_batch_size=32`, `response_len=8192`).
  - Changed the default 9B runner to a smaller smoke profile:
    - `NUM_ROLLOUT=4`
    - `--num-iters-per-train-update 2`
    - `--rollout-batch-size 2`
    - `--rollout-max-response-len 2048`
    - `--global-batch-size 16`
