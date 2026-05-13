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

For AMD Instinct MI300X / MI355 / MI350 systems, use `docker/Dockerfile.rocm`
(base image `rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1`).

Build with the helper script — defaults to `gfx942` (MI300X), set `GPU_ARCH`
for MI355/MI350:

```bash
# MI300X (default)
docker/build-rocm.sh

# MI355 / MI350
GPU_ARCH=gfx950 docker/build-rocm.sh
```

The image is tagged `relax:rocm-${GPU_ARCH}`. AITER is prebuilt for both
archs, so a `gfx942` image still runs on `gfx950` hosts (just slower).

For day-to-day dev, start a bind-mounted container so `/root/Relax` inside
the container points to the host checkout:

```bash
docker/run-rocm-bind.sh                      # uses relax:rocm-gfx942
IMAGE=relax:rocm-gfx950 docker/run-rocm-bind.sh
docker exec -it relax_rocm_bind bash
```
