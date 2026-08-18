# Hybrid 混合训练模式

## 概述

**Hybrid 模式** 是 Relax 在 [Colocate（同步）](./architecture.md) 与 [Fully Async（全异步）](./fully-async-training.md) 之间的第三种执行模式。它将以下两者结合：

- Fully Async 的 **流式数据流水线**（TransferQueue + `max-staleness` 控制 off-policy 容忍度），以及
- Colocate 的 **进程内权重共享**（TensorBackuper + `_switch_model`，使 ref / actor_fwd / advantages 全部在 actor 自身 GPU 上完成）。

具体来说，Actor 与 Rollout 仍然部署在 **独立的 GPU placement group** 上（与 Fully Async 一致），但 actor 不再把权重广播到独立的 ActorFwd / Reference / Advantages 服务，而是通过 CPU/GPU `TensorBackuper` 在 `actor`、`ref`、`old_actor`、`teacher` 等 tag 之间切换同一套权重，所有 forward 在本地完成，最后通过同步的 `UpdateWeightFromTensor` 路径把权重推给 rollout。

### 模式对比

| 维度                | Colocate（同步）                          | Fully Async（全异步）                                        | Hybrid                                                                                |
| ------------------- | ----------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| **GPU 布局**        | Actor 与 Rollout 分时复用同一组 GPU       | Actor / Rollout / ActorFwd / Reference 各自独立 GPU          | Actor 与 Rollout 独立 GPU；ref / actor_fwd / adv 复用 actor 的 GPU                    |
| **数据流水线**      | TransferQueue，批同步                     | TransferQueue + StreamingDataLoader，完全流式                | TransferQueue optimizer mini；可选 producer-chunk actor-forward 流水线               |
| **权重同步**        | 进程内 tensor 拷贝                        | 通过 DCS（Checkpoint Engine）做 NCCL broadcast               | 同步的 `UpdateWeightFromTensor` 推给 rollout；ref/actor_fwd 走 TensorBackuper          |
| **Staleness**       | `max_staleness = 0`（严格 on-policy）     | 可配置 `max_staleness`                                       | 可配置 `max_staleness`                                                                 |
| **部署的角色**      | `actor`, `critic`, `rollout`              | `actor`, `critic`, `rollout`, `advantages`, `reference`, `actor_fwd` | `actor`, `critic`, `rollout`（与 Colocate 相同；ref/actor_fwd 在 actor 内部）         |
| **`--balance-data`**| 支持                                      | 不支持                                                       | **支持**（Hybrid 存在的核心原因之一）                                                 |

### 何时选择 Hybrid

选择 **Hybrid** 的场景：

- 希望获得独立 rollout GPU 与流水线数据带来的吞吐收益，但
- 模型较大，单独部署 ref / actor_fwd 服务会浪费 GPU，或者
- 需要 `--balance-data`（DP 间均衡 micro-batch 切分），而纯 Fully Async 不支持此功能。

选择 **Fully Async**：拥有充足的 GPU 单独运行 ref / actor_fwd / advantages 服务，并希望在 step 之间做真正的并行流水线。

选择 **Colocate**：GPU 紧张，可以接受 rollout → train 的串行执行。

______________________________________________________________________

## 架构

### 角色布局

Hybrid 与 Colocate 使用相同的角色集合 —— 只部署 `actor`、`critic`（可选）、`rollout` 三个 Ray Serve 服务。判断逻辑位于 `relax/core/registry.py`：

```python
def process_role(config):
    if config.hybrid:
        # hybrid mode: actor handles ref/actor_fwd internally
        # via _switch_model, only need actor + rollout services
        return ROLES_COLOCATE
    if config.fully_async:
        ...
```

但与 Colocate 不同，actor 与 rollout 的 placement group 是 **互相独立** 的，这与 Fully Async 一致。参见 `relax/core/controller.py`：

```python
if colocate and not self.config.hybrid:
    # Sync colocate: actor and rollout share GPUs via time-sharing (offload/onload)
    actor_rollout_pgs = create_placement_group(num_gpus=num_gpus)
else:
    # fully_async (pure or hybrid): actor and rollout use separate GPUs
    actor_rollout_pgs = None
```

### 参数解析

`--hybrid` 是唯一对外暴露的开关。`relax/utils/arguments.py` 将其展开为下游已识别的两个底层参数：

```python
if args.hybrid:
    args.fully_async = True
    args.colocate = True
```

直接传入 `--fully-async --colocate` 会被拒绝，必须使用 `--hybrid`。单一开关的设计使 `args.hybrid` 成为 registry、controller 分发以及 `train_hybrid` 调用点中识别 hybrid 模式的唯一权威标志。

### 架构图

ASCII 图中保留英文以避免框线对齐错乱：

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        Controller (Orchestrator)                           │
│                     relax/core/controller.py                               │
│                                                                            │
│       ┌───────────────────────────────────┐    ┌────────────────────┐      │
│       │            Actor Service          │    │  Rollout Service   │      │
│       │  (own placement group, N GPUs)    │    │ (own PG, M GPUs)   │      │
│       │                                   │    │   SGLang engines   │      │
│       │  ┌────────────────────────────┐   │    └─────────┬──────────┘      │
│       │  │ TensorBackuper             │   │              ▲                 │
│       │  │  tags: actor / ref /       │   │              │                 │
│       │  │        old_actor / teacher │   │              │                 │
│       │  │  _switch_model(tag) swaps  │   │              │                 │
│       │  │  weights on the same GPUs  │   │              │                 │
│       │  └────────────────────────────┘   │              │                 │
│       │  train_hybrid():                  │              │                 │
│       │    ├─ ref forward   (switch:ref)  │              │                 │
│       │    ├─ actor forward (switch:actor)│                                │
│       │    ├─ advantages    (in-process)  │              │                 │
│       │    └─ train         (switch:actor)│                                │
│       └──────────────┬────────────────────┘              │                 │
│                      │ UpdateWeightFromTensor (sync) ───┘                  │
└──────────────────────┼─────────────────────────────────────────────────────┘
                       ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      TransferQueue (Data Plane)                           │
│  Rollout writes train_N partition incrementally ──► Actor consumes        │
│  in sub-batches via get_meta(batch_size, batch_index) with max-staleness  │
└───────────────────────────────────────────────────────────────────────────┘
```

______________________________________________________________________

## `train_hybrid` 训练流程

`relax/backends/megatron/actor.py:708` 实现的 Hybrid 训练步骤分为三个阶段：

1. **采集 optimizer mini 并计算 forward log-probs**

   actor 按
   `rollout_batch_size * n_samples_per_prompt / global_batch_size`
   推导 optimizer mini。对每个 mini，actor 会：

   - 从 TransferQueue 拉取数据（`_get_data_from_transfer_queue("train", rollout_id, fields, batch_size, batch_index)`）
   - 若已备份 ref 权重，执行 `_switch_model("ref")` 并计算 ref log-probs
   - 若已备份 teacher 权重（OPD 场景），执行 `_switch_model("teacher")` 并计算 teacher log-probs
   - 执行 `_switch_model("old_actor" 或 "actor")` 并计算当前 actor 的 log-probs
   - 把扩充后的 mini 追加到内存列表

   打开可选增量 actor-forward 流水线后，一个 optimizer mini 会按与 rollout
   producer 名义传输阈值对齐的固定 sample 窗口消费；物理 producer put 分组
   可以变化，该额外 actor 切分不会改变 advantage 或 optimizer 边界。

2. **合并子批次并做全局 Advantages 归一化**

   所有子批次 dict 被拼接成一个 `rollout_data`，随后 `compute_advantages_and_returns(self.args, rollout_data)` 在合并后的整批数据上执行一次。这是两阶段设计的 **核心正确性要求** —— Advantages 归一化必须看到完整的 DP-group 批次，而不是各个子批次切片。

3. **在合并批次上训练并推送权重**

   一次 `train(...)` 调用基于合并后的 batch 完成优化器步进。随后 actor 把新权重备份到 `actor` tag（如果到达 ref 更新间隔，也刷新 `ref` tag），然后调用 `self.update_weights()` 通过 `UpdateWeightFromTensor` 把最新权重同步给 rollout。

forward 阶段每次只处理一个已拉取单元，而合并后的训练步保留 Colocate 风格的
全局统计量。

______________________________________________________________________

## 配置

### 必需参数

| 参数                            | 用途                                                                                            |
| ------------------------------- | ----------------------------------------------------------------------------------------------- |
| `--hybrid`                      | 启用 Hybrid 模式（内部展开为 `fully_async=True, colocate=True`）                                |
| `--resource '{...}'`            | 分别声明 `actor` 与 `rollout` 的 placement group，例如 `{"actor":[1,4],"rollout":[1,4]}`        |
| `--num-iters-per-train-update`  | producer 传输阈值，同时作为固定的 actor fetch/forward chunk 数量                              |
| `--max-staleness`               | Off-policy 容忍度（0 = 严格 on-policy，>0 允许一定程度滞后）                                    |

### 常用可选参数

| 参数                          | 说明                                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------------- |
| `--balance-data`              | Hybrid 模式下支持（纯 fully-async 下被拒绝）。启用后做 DP 间负载均衡。                                  |
| `--num-data-storage-units`    | TransferQueue 存储 actor 的数量。                                                                       |
| `--use-streaming-dataset`     | 从磁盘流式读取 prompts，而不是全量载入内存。                                                            |
| `--ref-update-interval`       | 周期性地用最新 actor 权重刷新缓存的 ref 权重。                                                          |

### 默认值覆盖

启用 `--hybrid` 后，`relax/utils/arguments.py` 会按以下方式设置默认值（除非用户显式传入）：

- `offload_train = False` 且 `offload_rollout = False` —— actor 与 rollout GPU 独立，不需要 offload
- `compute_advantages_and_returns = True` —— actor 必须在本地计算 advantages
- `fully_async = True`、`colocate = True` —— 由 `--hybrid` 推导得出

::: warning
如果你既想做流式数据流水线，又需要 `--balance-data`，必须使用 `--hybrid`。`--fully-async --balance-data`（不带 `--hybrid`）会在参数解析阶段被拒绝。
:::

### 增量 actor-forward 流水线

Hybrid 可选地从 TransferQueue 按固定 sample count 增量请求 actor chunk，
而不是等待完整 optimizer mini：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--hybrid-pipeline-forward` | 关闭 | 足量 sample ready 后立即 fetch 固定 actor chunk 并执行 forward |
| `--hybrid-pipeline-overlap` / `--no-hybrid-pipeline-overlap` | 开启 | chunk ready 后立即 forward；或先 fetch 全部相同 chunk，作为调度匹配的性能对照 |
| `--hybrid-pipeline-trace-dir PATH` | 未设置 | 记录不含样本内容的 producer、fetch、restore、forward、advantage 和 optimizer 事件 |
| `--hybrid-pipeline-fetch-timeout-s SECONDS` | `600` | chunk 未完整到达时，带 rollout/mini/chunk 上下文终止等待 |

该开关默认关闭。参考 Qwen3.5-9B 配方中，
`global_batch_size=256`、`num_iters_per_train_update=2`、DP=1，producer
以 128 samples 为传输目标，而 actor 固定请求两个完整的 128-sample chunk。
producer 的 `async_put` 分组不是契约：`FIRST_COMPLETED` 合并、tail flush 或
backfill 都可能合法地产生 1 次、2 次或更多次 put；只要 256 samples 及
global-index fingerprint 完整守恒即可。打开开关后：

1. 每个 optimizer mini 只 restore 一次 actor，并在等待首块前执行；
2. chunk 0 ready 后立即 fetch 和 forward，此时 rollout 可继续产生 chunk 1；
3. 再 fetch 和 forward chunk 1；
4. 按 `BatchMeta.global_indexes` 对所有 per-sample 字段恢复确定顺序；
5. 记录每个 chunk 的动态均衡 microbatch 调度，将其映射为合并 batch 中的
   canonical index，并在训练 forward/backward 中回放相同的样本分组与顺序；
6. 在完整 256 samples 上只计算一次 advantage，并只执行一次 optimizer step。

该路径不修改 producer 传输策略、多模态预处理、pixel tensor 数值、GRPO
group 边界、reward normalization 或 optimizer 语义。多出的一次 actor fetch
用于暴露 rollout/actor 重叠窗口，而不是减少工作量。

microbatch 调度回放是正确性约束，而不是性能调参。多模态 packed kernel
在 BF16 下可能因 batch shape 不同产生数值舍入差异；如果 old-policy
log-prob 按 chunk 计算、训练 forward 却按另一种完整 batch 分组，就会在
权重尚未更新时产生非零 PPO ratio。回放完全相同的 chunk microbatch，可在
保持整个 optimizer mini 只更新一次的同时，对齐 old-policy 与训练 forward
的计算形状。因此该开关强制要求 `--use-dynamic-batch-size`；batch 模式不兼容
时会在 actor 启动阶段直接失败。

`--no-hybrid-pipeline-overlap` 只改变同一组 chunk 操作的执行顺序：actor
先 fetch 完所有 chunk，再开始任何 chunk forward。chunk 内动态 microbatch
调度及合并后的单次 optimizer update 均保持不变，因此它是因果性能比较的
注册 baseline。完全去掉 `--hybrid-pipeline-forward` 仍是兼容性回滚方式，
但 full-batch packing 不属于调度匹配的性能对照。

首版支持范围有意收窄；不支持的组合会 fail fast，不会静默回退：

| 维度 | 打开开关时支持的范围 |
| --- | --- |
| 模式 | Hybrid |
| 负载 | 多模态、dynamic-batch GRPO |
| Forward role | 仅 actor；不支持 ref/KL、teacher/OPD、old actor、critic、routing replay |
| 并行拓扑 | TP=2、DP=1、PP=1、VPP=1、CP=2、EP=1、ETP=1 |
| Offload | `offload_train=False`、`offload_rollout=False` |
| Dropout | attention/hidden dropout 均为 0 |
| Batch 策略 | 使用 dynamic microbatch；每次 rollout 恰好一个固定 optimizer mini（`rollout_batch_size * n_samples_per_prompt == global_batch_size`）；不支持 partial/dynamic-global batch |
| Log-prob 来源 | actor 计算；不支持 true-on-policy 或 rollout-log-prob 快捷路径 |
| TensorBackuper | 启用普通 backuper，且只有 `actor` tag |

chunk 大小必须精确重建 optimizer mini，并且是 `n_samples_per_prompt` 的
整数倍。`BatchMeta.global_indexes` 发生重复、缺失、少取或多取时，当前 step
会带可定位信息直接失败。
启动时还要求 TransferQueue `>=0.1.10.dev0`，并具备
`BatchMeta.global_indexes` 与 `async_put(custom_meta=..., is_last=...)`
契约；版本或 API 不兼容会在 Ray worker 和 rollout producer 写入数据前失败。

参考脚本通过环境变量暴露这些参数：

```bash
HYBRID_PIPELINE_FORWARD=1 \
HYBRID_PIPELINE_TRACE_DIR=/data01/LWX/relax-task21/runs/smoke/timeline \
HYBRID_PIPELINE_FETCH_TIMEOUT_S=600 \
NUM_ITERS_PER_TRAIN_UPDATE=4 \
bash scripts/training/multimodal/run-qwen35-9B-8xgpu-openr1mm-hybrid-async.sh \
  hybrid-async
```

同一脚本还提供默认行为不变的实验开关，避免为了 smoke 或配对基准临时修改
脚本正文：

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `MODEL_CONFIG_FILE` | `qwen35-9B.sh` | 选择模型并行配置脚本 |
| `MODEL_NAME` / `MODEL_RUN_NAME` | `Qwen3.5-9B` / `qwen35-9b` | 选择 checkpoint 子目录和日志前缀 |
| `MODEL_CHECKPOINT_DIR` / `REFERENCE_CHECKPOINT_DIR` | `${MODEL_DIR}/${MODEL_NAME}` | 显式固定 actor 与 reference 输入 |
| `CHECKPOINT_SAVE` | `1` | 设为 `0` 时不传 `--save*`；改用独立 rollout 与 TensorBoard 输出目录 |
| `ROLLOUT_RESULT_DIR` / `TENSORBOARD_DIR` | 保存开启时不覆盖；关闭时为 `${EXP_DIR}/rollout_result` 和 `${EXP_DIR}/tensorboard_log` | 在 no-save 运行中保留原始结果和标量；`TENSORBOARD_DIR` 会导出给 MetricsService |
| `NUM_ITERS_PER_TRAIN_UPDATE` | `2` | 设置每个 optimizer mini 中按完整 prompt group 对齐的 producer/actor 分块数 |
| `HYBRID_PIPELINE_OVERLAP` | `1` | 仅在 `HYBRID_PIPELINE_FORWARD=1` 时设为 `0`，运行调度匹配的 no-overlap 对照 |
| `ROLLOUT_MAX_RESPONSE_LEN` / `ROLLOUT_MAX_PROMPT_LEN` / `ROLLOUT_MAX_CONTEXT_LEN` | `10240` / `2048` / `12288` | 固定生成和上下文上限 |
| `ACTOR_MAX_TOKENS_PER_GPU` | `12288` | 控制 dynamic-batch actor microbatch 的 token 上限 |
| `HYBRID_ACTOR_GPUS` / `HYBRID_ROLLOUT_GPUS` | `4` / `4` | 构造 Hybrid placement resource |
| `ROLLOUT_NUM_GPUS_PER_ENGINE` | `2` | 设置每个 SGLang engine 的 TP 数 |
| `SGLANG_MEM_FRACTION_STATIC` | `0.8` | 设置 SGLang 静态显存比例 |

脚本会在提交 Ray Job 前校验正整数、上下文容量、rollout GPU 可整除关系，
以及 actor GPU 数是当前 TP=2、CP=2 拓扑的整数倍。不满足约束会直接退出，
不会在 Ray worker 内静默降级。

参考配置的 `global_batch_size=256`、`n_samples_per_prompt=8`。将
`NUM_ITERS_PER_TRAIN_UPDATE` 设为 `4` 时，每个 optimizer mini 会形成四个
64-sample stage，每个 stage 包含八组完整 prompt。严格配对中的 baseline 与
experiment 都启用 chunk forward 并使用相同分块数。baseline 设置
`HYBRID_PIPELINE_OVERLAP=0`，先 fetch 完四个 stage 再依次 forward；experiment
设置为 `1`，每个 stage ready 后立即 forward，并与后续 stage 的生成重叠。

例如，以下命令是资源受限环境中的 no-save Qwen3-VL-8B 配置，可用于明确
标注资源条件的 smoke 或配对性能实验，但结果不能与默认 8 卡 Qwen3.5 配方
合并：

```bash
MODEL_CONFIG_FILE="${MODEL_CONFIG_DIR}/qwen3-vl-8B.sh" \
MODEL_NAME=Qwen3-VL-8B-Instruct \
MODEL_RUN_NAME=qwen3-vl-8b \
CHECKPOINT_SAVE=0 \
ROLLOUT_MAX_RESPONSE_LEN=512 \
ROLLOUT_MAX_PROMPT_LEN=2048 \
ROLLOUT_MAX_CONTEXT_LEN=2560 \
ACTOR_MAX_TOKENS_PER_GPU=6144 \
NUM_ITERS_PER_TRAIN_UPDATE=4 \
HYBRID_ACTOR_GPUS=4 \
HYBRID_ROLLOUT_GPUS=1 \
ROLLOUT_NUM_GPUS_PER_ENGINE=1 \
HYBRID_PIPELINE_FORWARD=1 \
bash scripts/training/multimodal/run-qwen35-9B-8xgpu-openr1mm-hybrid-async.sh \
  hybrid-async
```

资源拓扑、模型或长度上限变化后，只能与完全相同配置的 baseline 做配对
比较；不得与默认 8 卡配方合并统计。

Trace 按 hostname、role、PID 和 global rank 分文件。内容仅包括时间戳、
样本与 token 计数、多模态 tensor 字节数、CUDA 峰值以及不可逆的 global-index
fingerprint；不会写 prompt、response、图片或样本 tensor。

校验单次运行：

```bash
python scripts/tools/analyze_hybrid_pipeline_benchmark.py \
  --run-dir /data01/LWX/relax-task21/runs/smoke \
  --validate-only
```

每个配对 seed 中，baseline 使用
`HYBRID_PIPELINE_FORWARD=1 HYBRID_PIPELINE_OVERLAP=0`，experiment 使用
`HYBRID_PIPELINE_FORWARD=1 HYBRID_PIPELINE_OVERLAP=1`。candidate commit、
模型、数据、长度上限、拓扑、seed、镜像及硬件指纹等其余 manifest 字段必须一致。

比较两次 baseline 和两次 experiment，并生成注册的 CSV、JSON 与曲线：

```bash
python scripts/tools/analyze_hybrid_pipeline_benchmark.py \
  --run-dir /data01/LWX/relax-task21/runs/A1-baseline-seed20260728 \
  --run-dir /data01/LWX/relax-task21/runs/A2-baseline-seed20260729 \
  --run-dir /data01/LWX/relax-task21/runs/B1-experiment-seed20260728 \
  --run-dir /data01/LWX/relax-task21/runs/B2-experiment-seed20260729 \
  --output-dir /data01/LWX/relax-task21/comparison \
  --enforce-targets
```

分析器会校验事件闭合、每 mini 一次 restore、每 step 一次 optimizer、样本
守恒、同机 monotonic 计时、有限数值、producer/fetch fingerprint、稳态 step
覆盖率以及预注册性能门槛。producer put 次数仅作为诊断值，actor
fetch/forward 次数仍严格固定。严格 producer 重叠定义为首个 actor forward
早于 producer 最后一次 put 的开始时刻；最后一次 put 的完成时刻仅作为传输
阶段诊断，因为其 trace 写入可能在调度上晚于 consumer。`--enforce-targets`
还会检查 baseline 在首次 forward 前已 fetch 完全部 chunk、experiment 在
最后一次 fetch 完成前已开始 forward、每次实验的严格 producer 重叠比例、
step-time p95、命令指定数量的 GPU NVML 覆盖、峰值显存、token/多模态字节工作量和
raw-reward、截断率、staleness、同权重 PPO KL 与 policy clip fraction 护栏。
同权重 `abs(train/ppo_kl)` 与 `abs(train/pg_clipfrac)` 都必须不超过 `1e-7`。
aggregate throughput 使用
`sum(step_tokens) / sum(step_time)`，不会对 per-step rate 做简单平均。
截断率护栏读取 rollout 侧的 `rollout/truncated_ratio`；训练侧
`rollout/truncated` 在分块聚合时可能被一个 step 累加多次，因此不作为截断率
统计口径。
GPU utilization 与低于 10% 的 idle ratio 只使用墙钟时间落入预注册稳态
step 区间的 500 ms NVML 样本；区间由 TensorBoard `perf/step_time` 的
step end wall time 和 duration 重建。sampled peak VRAM 仍使用 full-run
安全口径。
严格比较还要求每次 run 的 manifest 为 clean tree，candidate/image/TransferQueue
身份完全一致，并存在已校验的静态输入 hash、依赖 freeze、wheel hash、
launcher log，以及值为 0 的训练、验证和最终退出状态文件。
manifest 还会记录并交叉校验 `max_staleness`、global/rollout batch size、
每 prompt 样本数和 actor chunk 数，避免分析器 CLI 与实际工作负载静默不一致。
注册的四阶段协议要求 actor 恰好执行四次、每次 64 sample 的 fetch/forward。
producer put 可因异步完成顺序重新分组，但事件必须闭合，且聚合后的 sample 总数与
fingerprint 必须和 actor 实际消费的工作量精确一致。
在计算任何配对统计前，分析器还要求模型、模型配置、数据路径、prompt/response/
context 上限、actor token 预算、actor/rollout 资源拓扑、SGLang 确定性与显存
配置、物理卡到容器卡映射、checkpoint 模式以及 debug 捕获/回放配置完全一致；
字段缺失或取值不同都会 fail closed。
此外还要求 hostname，以及由 GPU UUID、型号、PCI 地址和驱动版本生成的
SHA-256 硬件指纹完全一致。
成对 run 的 global-index fingerprint 必须精确一致。对于稳态 campaign，异步
chunk forwarding 允许 step 间调度顺序不同，因此 analyzer 比较 warmup 之后完整
窗口的输入 fingerprint/sample 多重集合。生成的 response 长度和动态
microbatching 产生的多模态 tensor 字节数仍按 run 报告，并计入 token 吞吐率的
分子；它们是采样与动态打包的输出，不是固定 prompt/image 输入。first-step
comparison 仍保留严格的 token 与字节工作量精确匹配护栏。comparison JSON 同时
报告重复实验的均值、中位数、范围、总体标准差和变异系数。
staleness 曲线来自 trace：在 actor 首次 forward 时，用已完成 put 的最大
producer rollout ID 减去当前 actor rollout ID；若 producer 的 trace 写入稍晚，
已完成的 actor fetch 本身可证明当前 rollout 已 ready。该值不得超过 manifest
记录的 `max_staleness`（参考配方为 `2`），paired 稳态均值最多增加 `0.25`。

Task 21 中置信度更高的机制应按 PR #201 已验收的方法做独立累计消融，不要把
ProcessorPool、true-on-policy train-forward log-prob 复用与 chunk overlap 捆成
一个 experiment。以下标签有意不沿用 PR #201 的 `A1/A2/A3`：本分支没有捆绑
其 owner/ref group payload dedup，因为该机制在 #201 的配对均值中使 samples/s
回退：

```text
B   : MM_PROCESSOR_POOL_SIZE=0 HYBRID_REUSE_TRAIN_LOGPROBS=0 HYBRID_PIPELINE_FORWARD=0
P   : MM_PROCESSOR_POOL_SIZE=8 HYBRID_REUSE_TRAIN_LOGPROBS=0 HYBRID_PIPELINE_FORWARD=0
P+R : MM_PROCESSOR_POOL_SIZE=8 HYBRID_REUSE_TRAIN_LOGPROBS=1 HYBRID_PIPELINE_FORWARD=0
P+S : MM_PROCESSOR_POOL_SIZE=8 HYBRID_REUSE_TRAIN_LOGPROBS=0 HYBRID_PIPELINE_FORWARD=1
```

每个 fresh process 至少运行 40 个 optimizer step，排除 warmup；每个条件至少
两次独立运行，同时报告均值和范围。`HYBRID_REUSE_TRAIN_LOGPROBS=1` 要求前向
确定性、每个 rollout partition 恰好一个 optimizer mini，且 `max_staleness > 0`
时必须启用 TIS。参考 launcher 有意禁止它与 `HYBRID_PIPELINE_FORWARD=1` 同时
启用，以保持两项收益可归因。

回滚时去掉 `--hybrid-pipeline-forward`，或设置
`HYBRID_PIPELINE_FORWARD=0`；无需转换 checkpoint 或数据集。若要扩展
DP/PP/VPP 或 role graph，必须先新增 collective-order 与 restore-count 测试，
再重跑 frozen-input parity、多模态 smoke 和成对性能实验。

`--steady-windows 0-0` 明确统计“每次启动新进程后的第一个 optimizer step”。
它适用于固定资源窗口只能容纳单步训练的场景，但不能表述为稳态吞吐。报告中
应称为配对 first-step benchmark，至少使用两个 seed 并平衡启动顺序；资源允许
时仍应补充后续多 step 的稳态窗口。分析器会将前者记录为
`measurement_scope=fresh_process_first_step`；至少包含三个后续 step 的窗口记录为
`measurement_scope=steady_state`。

______________________________________________________________________

## 快速开始

8 GPU 多模态 Hybrid 训练的参考启动脚本位于
`scripts/training/multimodal/run-qwen35-9B-8xgpu-openr1mm-hybrid-async.sh`。

它构建的 Hybrid 调用命令为：

```bash
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 4], "rollout": [1, 4]}' \
    --max-staleness 2 \
    --num-data-storage-units 1 \
    --num-iters-per-train-update 2 \
    --balance-data \
    --hybrid \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"
```

该配置的关键点：

- 8 GPU 总量，actor 与 rollout 各占 4 张
- `max-staleness 2` —— actor 可以消费比最新权重落后最多 2 个 step 的 rollout 输出
- `num-iters-per-train-update 2` —— rollout 以半批为传输目标，可选增量路径
  固定执行两次 128-sample actor fetch；物理 producer put 次数可以变化
- `balance-data` —— 启用 DP 间负载均衡
- 算法采用 GRPO 与 `--use-tis`；该参考配方关闭 KL/ref forward

______________________________________________________________________

## 故障排除

### `train_hybrid(rollout_id=N) batch_index=K stalled for ... seconds`

该警告由 `relax/backends/megatron/actor.py` 抛出，发生在 actor 不断尝试拉取下一个子批次、partition 却始终未被标记 `all_consumed` 时。常见原因：

- Rollout 漏填了当前 partition（丢弃样本后未补齐）。
- Rollout 因健康检查失败或重启而处于暂停状态。
- Staleness 预算耗尽：rollout 必须等待新权重，因此无法继续产出数据。

在认定为代码 bug 前，请先排查 rollout 侧日志及 partition 状态。

### `--balance-data is not supported in pure fully-async mode`

你同时传入了 `--fully-async --balance-data`，但缺少 `--hybrid`。请去掉 `--balance-data`，或者改用 `--hybrid`（Hybrid 模式原生支持 DP 数据均衡）。

### Rollout 长时间看到旧权重

Hybrid 在每次 `train_hybrid` 结束时通过同步的 `UpdateWeightFromTensor` 路径推送权重。如果观察到权重更新间隔过大，请检查：

- Actor 日志中的 `update_weights()` 耗时
- Rollout 健康检查是否触发了 actor 等待（权重同步前会调用 `_check_services_health()`）

______________________________________________________________________

## 下一步

Hybrid 模式计划推进的工作：

- **接入 DCS 做权重同步** —— 用 Distributed Checkpoint Service 替换当前同步的 `UpdateWeightFromTensor` 路径，使权重向 rollout 广播能够与下一轮训练迭代重叠，消除每次 `train_hybrid` 结束时残留的同步阻塞。
- **将 `train_actor` 拆分为 `num_iters_per_train_update` 次训练迭代** —— 目前 `num_iters_per_train_update` 只对 forward 阶段做了切分，合并后的训练步仍然在整个全局 batch 上跑一次。下一步将训练步同样切成 `num_iters_per_train_update` 次，让优化器更新与 TransferQueue 数据消费形成流水线，并进一步压低训练侧峰值显存。

相关文档：

- [全异步训练流水线](./fully-async-training.md) —— Hybrid 借用的流式数据引擎
- [架构设计](./architecture.md) —— Relax 服务分层总览
- [权重更新流水线优化](./update-weights-pipeline.md) —— `UpdateWeightFromTensor` 与 DCS 的差异
