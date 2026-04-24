# Docker Release Guidelines

## Overview

We publish two types of Docker images for Relax:

### 1. Stable Version

- Based on specific SGLang releases
- Patches are stored and maintained for these versions
- Recommended for production use

### 2. Latest Version

- Aligns with `lmsysorg/sglang:latest`
- Contains the most recent features and improvements
- Recommended for development and testing

## Pre-Release Testing

Before each update, we perform comprehensive testing on the following models using H100 GPUs:

| Model              | Sync | Async |
| ------------------ | ---- | ----- |
| Qwen3-4B           | ✓    | ✓     |
| Qwen3-30B-A3B      | ✓    | ✓     |
| Qwen3-omni-30B-A3B | ✓    | ✓     |
| Qwen3.5-35B-A3B    | ✓    | ✓     |

## Testing Modes

- **Sync**: Synchronous training mode
- **Async**: Asynchronous training mode

All models are tested in both modes to ensure stability and compatibility across different training scenarios.

## Experimental ROCm Build

For AMD Instinct MI355/MI350 class systems, use `docker/Dockerfile.rocm`.

This image is intentionally separate from the CUDA path because Relax's default image depends on NVIDIA-only packages such as Apex and `nvidia-modelopt`. The ROCm Dockerfile uses an AMD base image and keeps the same Relax patch flow for Megatron-LM and SGLang.

Build the ROCm training image with:

```bash
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile.rocm \
  --target relax \
  -t relax:rocm-relax-smoke \
  .
```

The current ROCm Dockerfile is validated on MI355/MI350 systems with
`rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1` as the base
image.

For day-to-day development, start a bind-mounted container so `/root/Relax`
inside the container points to the host checkout:

```bash
chmod +x docker/run-rocm-bind.sh
CONTAINER_NAME=relax_rocm_bind docker/run-rocm-bind.sh
docker exec -it relax_rocm_bind bash
```

Inside the bind-mounted container, a real-model 2-GPU smoke can be launched
with:

```bash
cd /root/Relax
REAL_HF_MODEL_DIR=/mnt/dcgpuval/models/meta-llama/Meta-Llama-3-8B-Instruct \
NUM_GPUS=2 \
bash scripts/training/text/run-llama3-8b-2xgpu-debug.sh
```
