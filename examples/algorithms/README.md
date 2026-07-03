# Algorithm Examples

本目录包含 Relax 支持的各种策略梯度算法的启动脚本。

## 概述

Relax 框架集成了多种策略梯度算法，均通过 `--advantage-estimator` 参数选择。所有算法共享同一套训练脚本架构，用户只需替换脚本中的算法参数块（`*_ARGS`）即可快速切换算法。

## 支持的算法

| 算法      | 启用参数                      | 推荐场景                   |
| --------- | ----------------------------- | -------------------------- |
| **GRPO**  | `--advantage-estimator grpo`  | 默认、大多数场景           |
| **CISPO** | `--advantage-estimator cispo` | 保留梯度方向、需要更高精度 |
| **GSPO**  | `--advantage-estimator gspo`  | 序列级约束、稳定训练       |
| **SAPO**  | `--advantage-estimator sapo`  | 平滑优化、soft 信任域      |

## 选择建议

### GRPO（推荐首选）

- 默认算法，特性平衡，大多数场景可用
- 对超出信任域的 token 直接置零梯度
- 适合大多数强化学习场景

### CISPO（保留梯度信号）

- 对超出信任域的 token 保留梯度方向（仅限幅）
- 梯度方差较大，需配合 `--kl-loss-coef 0.001` 稳定训练
- 适合需要更精细学习信号的任务

### GSPO（序列级 KL 约束）

- 使用序列级 KL 而非 token 级 KL
- 序列内所有 token 共享统一的约束强度
- 更稳定，适合长序列任务

### SAPO（平滑信任域）

- 用 sigmoid 门控替代硬裁剪
- 梯度流更平滑，避免梯度突变
- 适合对稳定性要求高的场景

## 快速开始

### 基础操作：修改算法参数

所有脚本均采用统一的参数块设计。要切换算法，只需修改相应 `*_ARGS` 数组：

```bash
# 例如：在任意训练脚本中，将 GRPO_ARGS 替换为 CISPO_ARGS

# 原始（GRPO）：
GRPO_ARGS=(
   --advantage-estimator grpo
   --eps-clip 0.2
)

# 改为 CISPO：
CISPO_ARGS=(
   --advantage-estimator cispo
   --use-kl-loss
   --kl-loss-coef 0.001
   --eps-clip 0.2
   --eps-clip-high 10
   --use-tis
)

# 启动时使用修改后的参数
python3 -m relax.entrypoints.train \
    "${MODEL_ARGS[@]}" \
    "${CISPO_ARGS[@]}" \  # 使用 CISPO 而非 GRPO
    "${OPTIMIZER_ARGS[@]}" \
    ...
```

### 示例：运行 CISPO（多模态）

```bash
# 1) 准备模型和数据
export MODEL_DIR=/path/to/Qwen3.5-9B
export DATA_DIR=/path/to/multimodal-open-r1-8k-verified
export EXP_DIR=/path/to/experiments

# 2) 运行 CISPO 异步训练（Fully Async 模式）
cd /fengxiaoshi/Relax
bash examples/algorithms/run-qwen35-9B-8xgpu-openr1mm-cispo-async.sh async

# 或运行同步训练（Colocate 模式）
bash examples/algorithms/run-qwen35-9B-8xgpu-openr1mm-cispo-async.sh sync
```

### 示例：运行 GSPO（文本）

编辑 `scripts/training/text/run-qwen3-4B-8xgpu.sh`，将 `GRPO_ARGS` 改为：

```bash
GSPO_ARGS=(
   --advantage-estimator gspo
   --eps-clip 0.2
)
```

然后启动：

```bash
export MODEL_DIR=/path/to/Qwen3-4B
export DATA_DIR=/path/to/aime-2024
export EXP_DIR=/path/to/experiments

bash scripts/training/text/run-qwen3-4B-8xgpu.sh
```

## 关键参数说明

### 通用参数

| 参数                    | 默认值               | 说明                                                                                                    |
| ----------------------- | -------------------- | ------------------------------------------------------------------------------------------------------- |
| `--advantage-estimator` | `grpo`               | 算法类型：`grpo`, `cispo`, `gspo`, `sapo`, `ppo`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline` |
| `--eps-clip`            | `0.2`                | 下方裁剪边距（ratio 下界 = `1 - eps_clip`）                                                             |
| `--eps-clip-high`       | 与 `--eps-clip` 相同 | 上方裁剪边距（ratio 上界 = `1 + eps_clip_high`）                                                        |
| `--clip-grad`           | —                    | 梯度裁剪范数，CISPO 下推荐设为 `1.0`                                                                    |
| `--kl-coef`             | `0.0`                | KL 惩罚系数（PPO、REINFORCE++ 等用）                                                                    |

### CISPO 专用参数

| 参数             | 默认值 | 推荐值       | 说明                                |
| ---------------- | ------ | ------------ | ----------------------------------- |
| `--use-kl-loss`  | off    | on           | 启用 KL loss（CISPO 推荐必开）      |
| `--kl-loss-coef` | `0.0`  | `0.001`      | KL loss 系数，约束策略偏移          |
| `--kl-loss-type` | `k1`   | `low_var_kl` | KL 估计方式，`low_var_kl` 方差更低  |
| `--use-tis`      | off    | on           | Token Importance Sampling，推荐开启 |

### SAPO 专用参数

| 参数             | 默认值 | 说明                                             |
| ---------------- | ------ | ------------------------------------------------ |
| `--sapo-tau-pos` | `1.0`  | Positive advantage 的温度参数                    |
| `--sapo-tau-neg` | `1.05` | Negative advantage 的温度参数（更高 = 更强抑制） |

## 最佳实践

### 1. CISPO：启用 KL 约束与 TIS

```bash
CISPO_ARGS=(
   --advantage-estimator cispo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --eps-clip 0.2
   --eps-clip-high 10
   --use-tis
   --clip-grad 1.0
)
```

**为什么**：

- `--use-kl-loss --kl-loss-coef 0.001`：CISPO 保留超界梯度，易导致策略漂移；小 KL 惩罚约束偏移
- `--use-tis`：Token Importance Sampling 可增强样本有效性
- `--eps-clip-high 10`：近似取消上侧裁剪，对 positive advantage token 更宽松

### 2. GSPO：序列级约束，更稳定

```bash
GSPO_ARGS=(
   --advantage-estimator gspo
   --eps-clip 0.2
)
```

**为什么**：GSPO 使用序列级 KL，序列内 token 约束一致，减少长序列训练的振荡。

### 3. 异步 vs 同步模式选择

- **Fully Async**（`--fully-async`）：

  - 适合：GPU 充足、高吞吐要求
  - 风险：off-policy 数据，需配合 `--max-staleness` 和 `--use-health-check`

- **Colocate**（`--colocate`）：

  - 适合：GPU 有限、小规模实验
  - 优势：pure on-policy，梯度更稳定

## 文件组织

```
examples/algorithms/
├── README.md                              (本文件)
├── run-qwen35-9B-8xgpu-openr1mm-cispo-async.sh    (CISPO 多模态示例)
├── ... (其他算法脚本)
```

## 常见问题

### Q: 哪个算法性能最好？

**A**: 这取决于任务。一般规律：

- **GRPO**：all-round，快速尝试的首选
- **CISPO**：需要精细学习信号时更好，但需要 KL 约束
- **GSPO**：长序列任务，训练更稳定
- **PPO**：如果已有 Critic 资源，性能可能更好

### Q: CISPO 的梯度波动很大，正常吗？

**A**: 正常。CISPO 保留超界 token 的梯度，会导致梯度方差更大。配合 `--clip-grad 1.0` 限制实际参数更新，并配置 `--kl-loss-coef 0.001` 稳定策略。

### Q: 我应该用 async 还是 sync？

**A**:

- **Sync**（colocate）：GPU 有限（≤8）、学习率高、需要稳定梯度 → 推荐
- **Async**（fully-async）：GPU 充足（≥16）、追求高吞吐、可接受轻微 off-policy → 推荐

## 参考文献

- [Relax 算法文档](../../docs/en/examples/algorithms.md)
- [GRPO - DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [CISPO - MiniMax-M1](https://arxiv.org/abs/2506.13585)
- [PPO - Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [REINFORCE++ - Simple Efficient Alignment](https://arxiv.org/abs/2501.03262)
