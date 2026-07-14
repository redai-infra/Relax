# Dynamic Context Parallelism

Dynamic Context Parallelism 让 Megatron 训练可以为每个 micro-batch 使用不同的 context parallel size。

## 概述

静态 Context Parallelism（CP）会为整个训练任务选择一个固定的 `context_parallel_size`。假设 `--rollout-max-context-len = a`、`--max-tokens-per-gpu = b`，长上下文训练通常需要 `cp_size = ceil(a / b)` 才能容纳最长序列。这个固定 CP size 会被所有 micro-batch 共用，包括那些由 packing 得到的短序列 micro-batch。

Dynamic Context Parallelism 解除了这个全局绑定。Relax 会针对每个 micro-batch，用其中最长序列估计所需的最小 CP size，再向上取到 2 的幂，并使用对应的 Megatron dynamic CP group。短 micro-batch 因此可以用更小的 CP size 和更多等效 data-parallel 子组，降低 CP 通信时延。

这个特性最适合 rollout 数据长度分布不均的场景，尤其是多模态任务：packing 可能把多个短样本放在同一个 micro-batch 中，但 `--rollout-max-context-len` 又必须足够大，以容纳少量很长的样本。

::: warning
当前主要验证范围是 colocate 模式下的 Megatron actor training。R3、Fully Async、Hybrid 或 SFT 生产任务使用前，需要单独验证。
:::

## 架构

```
Static CP group size = max_cp

Micro-batch 1: short sequences
┌──────────────────────────────────────────────┐
│ dynamic_cp_size = 1                          │
│ ranks become max_cp independent DP subgroups │
└──────────────────────────────────────────────┘

Micro-batch 2: medium sequences
┌──────────────────────────────────────────────┐
│ dynamic_cp_size = 2                          │
│ static CP group is split into CP subgroups   │
└──────────────────────────────────────────────┘

Micro-batch 3: longest sequences
┌──────────────────────────────────────────────┐
│ dynamic_cp_size = max_cp                     │
│ same behavior as static CP                   │
└──────────────────────────────────────────────┘
```

Dynamic CP 在 Megatron backend 中实现：

| 区域 | 实现 |
|---|---|
| 参数校验 | `relax/backends/megatron/arguments.py` 校验 `--dynamic-context-parallel`，推导最大静态 CP size，并要求启用 dynamic batching。 |
| Micro-batch CP 决策 | `relax/backends/megatron/cp_utils.py` 计算 `dynamic_cp_size = next_power_of_2(ceil(max_seq_len / max_tokens_per_gpu))`。 |
| 数据切分与合并 | `dynamic_cp_split_data` 按 token 长度在 CP 子组之间均衡样本，`dynamic_cp_merge_output` 在写回前重建完整的逐样本输出。 |
| 训练 batch 准备 | `relax/backends/megatron/data.py` 为每个 micro-batch 附加 `dynamic_cp_size`、`dynamic_cp_rank` 和 CP-aware packed sequence metadata。 |
| VLM 兼容 | `relax/backends/megatron/model.py` 为当前 micro-batch 替换 bridge CP group，并 patch GatedDeltaNet，使其读取 micro-batch 级 CP size。 |

Relax 和 verl 实现有两个重要区别。第一，Relax 支持多模态 batch：它会向 Megatron Bridge 传入未切分的 VLM 输入和 CP-aware packed sequence 参数。第二，当 micro-batch 被拆成多个子组时，Relax 按序列长度做均衡，而不是只按样本个数均分；必要时还会增大 dynamic CP size，确保每个子组都有数据。

## 功能特性

1. **Micro-batch 级 CP size**：短 packed micro-batch 可以使用 `cp_size = 1` 或 `2`，长 micro-batch 仍然使用最大 CP size。
2. **按 token 均衡的子组**：当静态 CP group 被拆成更小的 dynamic CP group 时，Relax 会按 token 长度均衡样本。
3. **避免空子组**：如果 micro-batch 样本数少于可拆出的子组数，Relax 会增大 dynamic CP size，保证 collective 调用对称。
4. **多模态路径支持**：VLM batch 保留未切分输入给 Megatron Bridge，loss 和 log-prob helper 则使用 dynamic CP metadata 做切分与 gather。
5. **Forward 输出重建**：Forward-only 输出会先在 dynamic CP group 内 gather，再跨子组 gather，最终恢复原始 micro-batch 顺序。

## 快速开始

Dynamic CP 需要和 dynamic batching 一起启用：

```bash
python3 relax/entrypoints/train.py \
  --dynamic-context-parallel \
  --use-dynamic-batch-size \
  --rollout-max-context-len 32768 \
  --max-tokens-per-gpu 8192 \
  --calculate-per-token-loss \
  # ... other training args
```

在这个例子里，Relax 会推导最大静态 CP size：

```text
ceil(32768 / 8192) = 4
next_power_of_2(4) = 4
```

运行时，如果某个 micro-batch 的最长序列只有 `6000` tokens，它可以使用 `dynamic_cp_size = 1`；接近 `32768` tokens 的 micro-batch 则使用 `dynamic_cp_size = 4`。

::: tip
如果基础脚本已经设置了 `--dynamic-context-parallel`，Relax 侧还需要补齐的 packing 开关是 `--use-dynamic-batch-size` 和 `--max-tokens-per-gpu`。
:::

## 配置

| Flag | 是否必需 | 说明 |
|---|---|---|
| `--dynamic-context-parallel` | 是 | 启用 Megatron dynamic CP group 创建，以及 Relax 的 micro-batch 级 CP 路径。 |
| `--use-dynamic-batch-size` | 是 | 校验强制要求。Dynamic CP 依赖按 token budget 打包 micro-batch。 |
| `--rollout-max-context-len` | 是 | 作为可能出现的最大上下文长度。 |
| `--max-tokens-per-gpu` | 是 | 每张 GPU 的 token budget。Dynamic CP 会把它作为 `max_seqlen_per_dp_cp_rank`。 |
| `--calculate-per-token-loss` | 是 | Megatron Bridge 在 CP 或 Dynamic CP 开启时要求设置。 |
| `--log-probs-max-tokens-per-gpu` | 可选 | Forward-only log-prob 计算可以使用不同 token budget；未设置时跟随 `--max-tokens-per-gpu`。 |

真正需要 CP 时，建议让 `--max-tokens-per-gpu` 小于 `--rollout-max-context-len`。如果 token budget 已经能覆盖最大上下文长度，Dynamic CP 通常会退化成 `cp_size = 1`，收益很小。

Relax 会在 Megatron 参数校验阶段推导并覆盖最大静态 `context_parallel_size`：

```text
max_cp = next_power_of_2(ceil(rollout_max_context_len / max_tokens_per_gpu))
```

全局 `world_size` 必须能被下面的乘积整除：

```text
tensor_model_parallel_size * pipeline_model_parallel_size * max_cp
```

## 验证与性能

在一个 5k 多模态数据集上，样本长度分布不均，平均每个样本 4.5 张图，平均图片分辨率约 `673 x 841`。内部实验观测到：

| 任务 | 训练吞吐 | 端到端吞吐 |
|---|---:|---:|
| Qwen3-VL-4B | +12% | +9% |
| Qwen3-35B-A3B | +12% | +2% |

端到端收益取决于 rollout 中有多少时间由 actor training 主导。如果瓶颈在 rollout 或数据加载，Dynamic CP 仍可能明显提升训练段，但不会等比例反映到整条 pipeline。

## 最佳实践

1. **优先用于长度方差大的数据**：当很多 micro-batch 明显短于 `--rollout-max-context-len` 时，Dynamic CP 更容易带来收益。
2. **设置现实的 token budget**：从不会 OOM 的最大 per-GPU token budget 开始，只有当长样本仍然超显存时再继续降低。
3. **观察日志**：Relax 会打印类似 `[dynamic_cp] micro-step dynamic_cp_size=...` 的日志；长度混合健康的任务通常会出现不止一种 CP size。
4. **新模式单独验证**：R3、Fully Async、Hybrid 和 SFT 都应作为独立验证目标，再用于生产。
5. **同时比较训练段和端到端指标**：训练段收益可能被 rollout、reward 或权重更新耗时稀释。

## 故障排除

### `--dynamic-context-parallel requires --use-dynamic-batch-size`

需要同时添加两个开关。Dynamic CP 依赖按 token budget 打包 micro-batch：

```bash
--dynamic-context-parallel \
--use-dynamic-batch-size \
--max-tokens-per-gpu 8192
```

### `world_size must be a multiple of tp*pp*cp`

Relax 会根据 `--rollout-max-context-len` 和 `--max-tokens-per-gpu` 推导 `cp`。可以增加总 world size、降低 TP/PP，或增大 `--max-tokens-per-gpu` 让推导出的最大 CP size 变小。

### Dynamic CP 日志始终显示同一个 CP size

检查数据集是否真的产生了短 packed micro-batch。如果每个 micro-batch 的最长序列都接近 `--rollout-max-context-len`，Dynamic CP 的行为会接近静态 CP。

### VLM 任务在 Megatron Bridge 内失败

确认当前 Megatron 版本的 `initialize_model_parallel` 支持 `dynamic_context_parallel` 参数。Relax 会在参数校验阶段检查这一点，但 Bridge 或 Megatron pin 不匹配时，仍可能表现为模型侧错误。

## 下一步

- [性能调优](./performance-tuning.md) - 调整 dynamic batching 和并行配置来提升吞吐。
- [配置说明](./configuration.md) - 查看完整训练参数参考。
- [OOM 排查](./oom-troubleshooting.md) - 为长上下文任务选择安全的 token budget。
