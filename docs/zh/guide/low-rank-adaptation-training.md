# LoRA RL（参数高效 RL 后训练）

Relax 支持使用 **LoRA**（Low-Rank Adaptation，低秩适配）进行 RL 后训练：不再更新全部权重，而是冻结基座模型、只训练小规模的低秩适配矩阵。这样可以大幅缩减优化器状态和每步权重同步的数据量，从而在相同 GPU 预算下容纳更大的模型。

Relax 内置了两条端到端的 LoRA rollout 路径，二者的区别仅在于**训练好的适配器如何送达 rollout 引擎**：

| 模式            | 每步同步到 rollout 的内容                   | rollout 提供的模型      | 参考启动脚本                                                          |
| --------------- | ------------------------------------------ | ----------------------- | ------------------------------------------------------------------- |
| **Merge 模式**  | 完整模型（适配器已折叠进基座）             | 一个已合并的模型        | `scripts/training/text/run-qwen3-4B-lora-merge-8xgpu.sh`             |
| **Adapter 模式**| 基座仅同步一次，之后只推送 LoRA 适配器     | 基座 + 运行时适配器     | `scripts/training/text/run-qwen3-4B-lora-adapter-8xgpu-async.sh`     |

两种模式**互斥**。如果启用了 LoRA（`--lora-rank > 0`）却没有指定模式，Relax 会强制使用 **Merge 模式**（默认、兼容性最广的路径）。

## 概述

训练侧通过 Megatron-Bridge 的 PEFT 集成来应用 LoRA（`relax/backends/megatron/model_provider.py`）：基座模型被冻结，只有注入的适配器参数会接收梯度。两种模式的差异完全体现在**权重同步路径**（`relax/backends/megatron/weight_update/`）上：

- **Merge 模式**在导出时把每个适配器折叠进对应的基座权重（`LoRAMerge`），rollout 引擎因此加载的是单个已合并的模型——推理侧无需感知 LoRA。这条路径复用标准的全量权重同步流程，所以每步带宽与全参数训练相同。
- **Adapter 模式**只同步一次冻结的基座，之后每步仅通过 SGLang 的运行时 LoRA API 推送**小规模的适配器**。rollout 请求通过 `lora_path` 选中它。这样以少量 rollout 侧的 LoRA 开销，换取每步同步带宽的大幅下降。

::: tip 该选哪种模式？
- **Merge 模式**——最简单，兼容最广的模型和两种部署布局（包括分布式 / NCCL rollout 引擎）。rollout 推理就是普通的全模型推理。代价是每步都要做一次全量权重同步。
- **Adapter 模式**——当每步权重同步带宽成为瓶颈时（基座大、适配器小）最合适。附带一些约束（`--sglang-dp-size 1`、不支持分布式 rollout 引擎、fully-async 下仅支持稠密模型）。rollout 走 SGLang 的 LoRA 内核。
:::

## 架构

两种模式共享同样的训练侧；区别只在同步箭头上传输的数据。

```
        ┌────────────────────────────────────────────────────────┐
        │                  Training side (Actor)                 │
        │  Megatron-LM + Megatron-Bridge PEFT                    │
        │  base weights FROZEN, only LoRA adapter has gradients  │
        │  (model_provider.wrap_model_provider_with_lora)        │
        └───────────────────────────┬────────────────────────────┘
                                     │
             ┌───────────────────────┴───────────────────────┐
             │                                               │
     MERGE mode: fold adapter                      ADAPTER mode: base once,
     into base, sync FULL model                    then push ADAPTER only
     every step (NCCL / IPC)                       (SGLang runtime LoRA API)
             │                                               │
             ▼                                               ▼
        ┌─────────────────────────┐              ┌─────────────────────────┐
        │  Rollout (SGLang)       │              │  Rollout (SGLang)       │
        │  one merged model       │              │  base + adapter         │
        │  plain inference        │              │  lora_path=             │
        │                         │              │   relax_policy_lora     │
        └─────────────────────────┘              └─────────────────────────┘
```

权重同步后端会根据 LoRA 模式进行分派：

- **Colocate（同置）**——`UpdateWeightFromTensor`（`weight_update/update_weight_from_tensor.py`）。Merge 模式在 HF 导出时折叠适配器；Adapter 模式走 `_update_weights_adapter_mode`（基座同步一次 + 每步通过 `load_lora_adapter_from_tensors` 在内存中推送适配器）。
- **Fully-async（全异步）**——`DeviceDirectBackend`（`distributed/checkpoint_service/backends/device_direct.py`）。Merge 模式在 NCCL 广播前折叠适配器；Adapter 模式每步把适配器写入一个共享目录，再向每个引擎扇出 HTTP `/load_lora_adapter`。

与模式无关的适配器导出/聚合逻辑（构建 HF 导出 bridge、导出本地适配器、PP 聚合、delta-skip、写 HF-PEFT 目录）都收敛在共享的 `LoraAdapterSync` 辅助类（`weight_update/lora_adapter_sync.py`）中，两个后端以组合方式复用它。

## 配置

所有 LoRA 参数都定义在 `relax/utils/arguments.py`。

| 参数                     | 类型       | 默认值                        | 说明                                                                                                 |
| ------------------------ | ---------- | ----------------------------- | ---------------------------------------------------------------------------------------------------- |
| `--lora-rank`            | int        | `0`                           | LoRA 秩。`0` 表示禁用 LoRA；任何 `> 0` 的值都会启用它。                                               |
| `--lora-alpha`           | int        | `32`                          | LoRA alpha 缩放因子。                                                                                 |
| `--lora-target-modules`  | str（列表）| `linear_qkv linear_proj`      | 要适配的 Megatron 风格模块名（见下方映射表）。以空格分隔。                                            |
| `--lora-dropout`         | float      | `0.0`                         | LoRA 层内部的 dropout 概率。                                                                          |
| `--lora-merge-mode`      | flag       | `False`                       | 同步前把适配器折叠进基座权重。与 `--lora-adapter-mode` 互斥。                                         |
| `--lora-adapter-mode`    | flag       | `False`                       | 基座只同步一次，之后每步只推送适配器。与 `--lora-merge-mode` 互斥。                                   |

### 目标模块

`--lora-target-modules` 接收 **Megatron 风格**的名字（Megatron-Bridge 的 LoRA 匹配器直接遍历的规范形式）。导出适配器时会自动扩展成 HF 风格名字（`relax/utils/megatron_peft_utils.py` 中的 `convert_megatron_to_hf_target_modules`）：

| Megatron 名字          | HF 投影                       |
| ---------------------- | ----------------------------- |
| `linear_qkv`           | `q_proj`、`k_proj`、`v_proj`  |
| `linear_proj`          | `o_proj`                      |
| `linear_fc1`           | `gate_proj`、`up_proj`        |
| `linear_fc2`           | `down_proj`                   |
| `router`               | `gate`                        |

（完整映射表，包括拆分 QKV 和 MLA 变体，见 `MEGATRON_TO_HF_MODULES`。）

### 校验规则

`lora_rank > 0` 时在 `relax/utils/arguments.py` 中强制执行：

- `--lora-merge-mode` 与 `--lora-adapter-mode` **互斥**——只能选其一。
- `--lora-adapter-mode` 要求 `--sglang-dp-size 1`（SGLang 的动态 LoRA 加载不支持 DP attention）。
- 若两个模式标志都未设置，Relax 会**强制启用 `--lora-merge-mode`** 并打印警告。

## Merge 模式配方

参考：`scripts/training/text/run-qwen3-4B-lora-merge-8xgpu.sh`（Qwen3-4B，8 卡 colocate，在 `dapo-math-17k` 上跑 GRPO）。

该脚本中的 LoRA 配置块：

```bash
LORA_ARGS=(
   --lora-rank 32
   --lora-alpha 64
   --lora-target-modules linear_qkv linear_proj
   --lora-dropout 0.0
   --lora-merge-mode
)
```

像任何 colocate 脚本一样启动：

```bash
bash scripts/training/text/run-qwen3-4B-lora-merge-8xgpu.sh
```

每步的工作方式：actor 导出自己的权重，`LoRAMerge` 在 HF 转换过程中把每个适配器折叠进配对的基座权重，然后合并后的完整模型通过正常的 IPC/NCCL 路径同步给 SGLang。rollout 引擎完全不感知 LoRA——它只是提供一个完整模型。

::: warning 快速 bridge 路径下的 Merge 模式
在快速 bridge 路径下，Merge 模式要求 `--expert-tensor-parallel-size 1`（专家 LoRA 合并会在拥有该专家的 EP rank 上本地选取一个 per-expert 切片，ETP 不匹配会导致集合通信死锁）。参考脚本已设置 `--expert-tensor-parallel-size 1`。
:::

## Adapter 模式配方

参考：`scripts/training/text/run-qwen3-4B-lora-adapter-8xgpu-async.sh`（Qwen3-4B，8 卡 fully-async，在 `dapo-math-17k` 上跑 GRPO）。

该脚本中的 LoRA 配置块：

```bash
LORA_ARGS=(
   --lora-rank 128
   --lora-alpha 64
   --lora-target-modules linear_qkv linear_proj
   --lora-dropout 0.0
   --lora-adapter-mode
)
```

注意它同时设置了 `--rollout-num-gpus-per-engine 1`（每个引擎一张卡 → `sglang_dp_size == 1`，这是 Adapter 模式的要求）。启动：

```bash
bash scripts/training/text/run-qwen3-4B-lora-adapter-8xgpu-async.sh
```

工作方式：

1. **首次同步**——把仅基座的权重推给引擎（适配器参数会从转换 bucket 中剥离，绝不合并），然后以固定名字 `relax_policy_lora` 注册 LoRA 适配器。
2. **之后每一步**——**只**刷新适配器。基座保留在引擎上不动。rollout 请求会自动带上 `lora_path=relax_policy_lora`（`relax/engine/rollout/sglang_rollout.py`），使生成走训练好的适配器。
3. **Delta-skip（增量跳过）**——如果任一 rank 上都没有适配器参数变化超过 `1e-6` 阈值，则整次推送被跳过。这是一个跨所有 rank 的集合决策（单 rank 提前返回会让聚合失去同步并挂起）。

适配器本身的传输方式因部署而异：

- **Colocate**——适配器被序列化后通过 `load_lora_adapter_from_tensors` 在内存中推送（SGLang ≥ 0.5.12，无磁盘 IO）。
- **Fully-async**——合并后的适配器**只写一次**到共享目录，然后向每个引擎扇出 HTTP `/load_lora_adapter`。该目录默认是 `<args.save>/relax_lora_live/adapter`；可用环境变量 `RELAX_LORA_LIVE_DIR` 覆盖（必须能被每个 rollout 引擎读取——在 fully-async 下**不要**指向节点本地存储）。

::: warning Adapter 模式约束
- 要求 `--sglang-dp-size 1`（参数校验时强制）。
- 在 colocate 模式下**不支持分布式（非同置的 NCCL）rollout 引擎**——适配器只会推送给同置的 IPC 引擎。请使用 colocate 或 fully-async；如需分布式 rollout，请改用 Merge 模式。
- 在 **fully-async** 路径下，**不支持** MoE（grouped-expert）LoRA（per-expert 的合并数学与稠密模型不同）。请使用稠密模型，或在 colocate 模式下运行专家 LoRA。
- Adapter 模式在 colocate 下会关闭下一步 rollout 预取（每步适配器更新约 1s，预取只会被重做）。
:::

## Checkpoint 与导出

启用 LoRA 后，保存 checkpoint 时还会在 `<checkpoint_dir>/lora_adapter/` 下写出一个**可移植的 HF-PEFT 适配器**（`relax/backends/megatron/checkpoint.py`）：

- `adapter_config.json` + `adapter_model.safetensors`——标准 HF-PEFT 布局，可用 `peft.PeftModel.from_pretrained` 加载。
- `relax_lora_meta.json`——Relax 元数据（rank、alpha、target modules、dropout、模式）。单独成文件，以免干扰标准 PEFT 加载器。

::: tip
这个 `lora_adapter/` 目录是供外部 / 推理使用的**导出产物**——它**不是**续训的来源。LoRA 参数作为普通模型参数保存在主 Megatron checkpoint 内，因此 `--load` 会像加载其他权重一样恢复它们。
:::

## 故障排除

| 现象                                                                          | 可能原因 / 解决办法                                                                                                                          |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `--lora-merge-mode and --lora-adapter-mode are mutually exclusive`             | 同时设置了两个标志。只保留其一。                                                                                                            |
| `--lora-adapter-mode requires --sglang-dp-size 1`                             | Adapter 模式遇到了 DP attention。设置 `--rollout-num-gpus-per-engine 1`（或以其他方式使 `sglang_dp_size == 1`）。                            |
| `--lora-adapter-mode does not yet support distributed ... rollout engines`     | Adapter 模式遇到非同置 rollout 引擎。改用 colocate，或用 `--lora-merge-mode` 做分布式 rollout。                                             |
| `MoE (grouped-expert) LoRA is not supported in fully-async weight sync`        | 在 fully-async 下用了专家 LoRA。改用稠密模型，或在 colocate 模式下运行专家 LoRA。                                                           |
| `[lora-merge] NO adapter tensors in backup dict`                             | `weights_getter()` 输出里缺少适配器参数——检查 `--lora-target-modules` 名字，以及 LoRA 是否真的在建模时挂上。                                |
| Adapter 模式下 rollout 质量像是基座模型                                        | 适配器推送被跳过，或 `lora_path` 未生效。确认请求带有 `lora_path=relax_policy_lora`，且适配器目录是共享可读的。                             |
