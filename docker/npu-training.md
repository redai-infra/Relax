# NPU 训练指导

## 概述

本文档介绍了在华为昇腾的算力节点上使用Relax训练框架，对业界主流的开源三方大模型进行训练的详细过程。训练使用的算力资源是910C。

## 模型支持

| 模型     | 训练场景 | Sync | Async | 训练所需最小卡数 | 参考脚本                                                |
| -------- | -------- | ---- | ----- | ---------------- | ------------------------------------------------------- |
| Qwen3-4B | DAPO     | ×    | √     | 910C 4卡         | `scripts/training/text/run-qwen3-4B-8xgpu-async-npu.sh` |

## 环境准备

### 前置准备

- 资源类型：`Ascend910 Snt9b23`
- 驱动版本：`Software Version 25.2.1`
- 固件版本：`Firmware Version 7.7.0.9.220`
- 基础镜像：`quay.io/ascend/vllm-ascend:v0.18.0-a3`

### 环境检查

`npu-smi info`                    # 在每个实例节点上运行此命令可以看到NPU卡状态
`npu-smi info -l | grep Total`    # 在每个实例节点上运行此命令可以看到总卡数，用来确认对应卡数已经挂载
`npu-smi info -t board -i 1 | egrep -i "software|firmware"`   #查看驱动和固件版本

### 安装方法

（推荐）基于 Dockerfile 构建镜像：`docker build -f docker/Dockerfile.npu -t npu-2026 .`

## 启动配置

### 容器启动

- 启动命令：

```
export work_dir="自定义挂载的工作目录"     # 容器内挂载的目录，如果挂载SFS可使用挂载目录
export container_work_dir="自定义挂载到容器内的工作目录"
export container_name="自定义容器名称"
export image_name="镜像名称"

docker run \
--shm-size=200gb   --cap-add=SYS_PTRACE   \
--device=/dev/davinci0   \
--device=/dev/davinci1   \
--device=/dev/davinci2   \
--device=/dev/davinci3   \
--device=/dev/davinci4   \
--device=/dev/davinci5   \
--device=/dev/davinci6   \
--device=/dev/davinci7   \
--device=/dev/davinci_manager   \
--device=/dev/devmm_svm   \
--device=/dev/hisi_hdc   \
--name ${container_name}   \
-v /usr/local/dcmi:/usr/local/dcmi   \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi   \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /etc/ascend_install.info:/etc/ascend_install.info   \
-v /var/log/npu/:/usr/slog   \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi   \
-v ${work_dir}:${container_work_dir} \
-u root \
-itd ${image_name}  /bin/bash
```

- 参数说明：

  > ⦁	--shm-size：表示共享内存，用于多进程间通信

  > ⦁	--device=/dev/davinciX：表示容器挂载的卡号

  > ⦁	driver及npu-smi 需同时挂载至容器

### 训练启动

- 启动命令：

```
# Fully async mode
bash scripts/training/text/run-qwen3-4B-8xgpu-async-npu.sh
```

- 脚本说明：

  > ⦁	环境变量：
  > ASCEND_RT_VISIBLE_DEVICES #指定卡号
  > HCCL_NPU_SOCKET_PORT_RANGE #配置HCCL在NPU侧使用的通信端口
  > HCCL_HOST_SOCKET_PORT_RANGE #配置HCCL在Host侧使用的通信端口
  > PYTORCH_NPU_ALLOC_CONF #优化内存使用，减少碎片化
  > EXP_DIR #模型、数据集地址

  > ⦁	环境入口：
  > source entrypoint：/root/Relax/scripts/entrypoint/local-npu.sh

  > ⦁	启动配置：
  > PERF_ARGS，启用 recompute（重计算） 以节省显存，同时改用动态 batch size 自适应调整。`--recompute-granularity full/--recompute-method uniform/--recompute-num-layers 1/--use-dynamic-batch-size`

  > OPTIMIZER_ARGS，内存优化，优化器状态 offload 到 CPU 并重叠通信以隐藏延迟。`--optimizer-cpu-offload/--overlap-cpu-optimizer-d2h-h2d/--use-precision-aware-optimizer`

  > SGLANG_ARGS，硬件适配：切换device为npu `--sglang-device npu`、attention backend 为 ascend（华为自研加速库）`--sglang-attention-backend ascend`、禁用 radix cache `--sglang-disable-radix-cache`;

  > SGLANG_ARGS，性能优化：开启数据并行（DP）优化 lm_head 和 attention `--sglang-enable-dp-lm-head/--sglang-enable-dp-attention`、配置 CUDA graph 预热 batch sizes `--sglang-cuda-graph-bs 4 8 16 32 64 96 128`

  > MISC_ARGS，显示启用FlashAttention实现 `--use-flash-attn`

## 下一步

- [ ] 功能特性：共卡适配
- [ ] 模型支持：Qwen3.5-9B
