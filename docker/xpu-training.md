# XPU 训练指导

## 概述

本文档介绍在昆仑芯 XPU 算力节点上使用 Relax 框架训练业界主流开源大模型的完整流程。当前算力规格为昆仑芯 klx（8 卡单机）。

## 模型支持

| 模型            | 训练场景 | Sync | 训练所需最小卡数 | 参考脚本                                             |
| --------------- | -------- | ---- | ---------------- | ---------------------------------------------------- |
| Qwen3-4B        | DAPO     | √    | P800 8 卡        | `scripts/training/text/run-qwen3-4B-8xklx.sh`        |
| Qwen3.5-9B      | DAPO     | √    | P800 8 卡        | `scripts/training/text/run-qwen35-9B-8xklx.sh`       |
| Qwen3.5-35B-A3B | DAPO     | √    | P800 16 卡       | `scripts/training/text/run-qwen35-35B-A3B-16xklx.sh` |

## 环境准备

### 前置准备

- 资源类型：`P800 (XPU)`
- 当前可用镜像：`iregistry.baidu-int.com/xpu/xrelax_torch29_ubuntu2204_xsgl0510_dev:20260630_12`
- 代码路径约定：**Relax 代码库必须放在容器内 `/workspace/Relax`**，文档及补丁中的路径均基于此前提

### 环境检查

```bash
xpu_smi                              # 查看 XPU 卡状态
xpu_smi -L | grep -c "XPU"           # 确认挂载卡数
```

## 启动配置

### 容器启动

```bash
CONTAINER_NAME="<自定义容器名>"
PROJECT="iregistry.baidu-int.com/xpu/xrelax_torch29_ubuntu2204_xsgl0510_dev:20260630_12"

# 拼接 8 张 XPU + xpuctrl
DOCKER_DEVICE_CONFIG=""
for ((idx=0; idx<8; idx++)); do
  DOCKER_DEVICE_CONFIG+=" --device=/dev/xpu${idx}:/dev/xpu${idx}"
done
DOCKER_DEVICE_CONFIG+=" --device=/dev/xpuctrl:/dev/xpuctrl"

docker run --privileged -it ${DOCKER_DEVICE_CONFIG} \
  --net=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --tmpfs /dev/shm:rw,nosuid,nodev,exec,size=32g \
  --name ${CONTAINER_NAME} \
  ${PROJECT} /bin/bash
```

> - `--device=/dev/xpuX` / `/dev/xpuctrl`：XPU 卡及管理设备节点
> - `--tmpfs /dev/shm:size=32g`：BKCL / 多进程通信所需共享内存
> - 进入容器后将 Relax 代码库及相关数据 / 权重放置在 `/workspace` 下，本文档及 patch 中的路径均默认 Relax 位于 `/workspace/Relax`

### 训练启动

```bash
#qwen3-4B脚本
bash scripts/training/text/run-qwen3-4B-8xklx.sh
#qwen3.5-9B脚本
bash scripts/training/text/run-qwen35-9B-8xklx.sh
#qwen3.5-35B-A3B脚本
bash scripts/training/text/run-qwen35-35B-A3B-16xklx.sh
```

## 下一步

- [ ] Qwen3.5-9B 性能优化
- [ ] Qwen3.5-35B-A3B 性能优化
