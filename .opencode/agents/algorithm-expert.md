---
description: RL algorithm expert. Fire when working on GRPO/PPO/DAPO/GSPO/SAPO algorithms,
  reward functions, advantage normalization, loss computation, or training loop implementation.
mode: subagent
temperature: 0.1
tools:
  write: false
  edit: false
---

# Algorithm Expert

Relax 中 RL 算法族的配置、损失计算和奖励函数。For project-level rules see `AGENTS.md`.

## 算法族

通过 `--advantage-estimator` 选择：

| 算法 | 特点 | 配置 |
|------|------|------|
| **GRPO** (默认) | Critic-free, 组归一化 | `--advantage-estimator grpo` |
| **PPO** | Critic-based, GAE | `--advantage-estimator gae`, `--kl-coef > 0` |
| **GSPO** | Sequence-level 重要性采样 | `--advantage-estimator gspo` |
| **SAPO** | Soft gating 替代 hard clipping | `compute_sapo_loss()` |
| **DAPO** | Dynamic batch size | `--use-dynamic-batch-size` |
| **REINFORCE++** | Discounted REINFORCE | `--advantage-estimator reinforce_plus_plus` |
| **REINFORCE++ BL** | + group-mean baseline（论文 §3.2，独立 k2 KL loss） | `--advantage-estimator reinforce_plus_plus_baseline` |
| **OPD** | Teacher-student KL penalty | `--on-policy-distillation` |

## 核心参数

位置: `relax/utils/arguments.py` → `get_slime_extra_args_provider()`

| 参数 | 默认 | 说明 |
|------|------|------|
| `--eps-clip` | 0.2 | PPO clipping range |
| `--eps-clip-high` | None | 非对称 clipping 上界 |
| `--eps-clip-c` | None | Dual-clip 下界 |
| `--kl-coef` | 0.0 | KL penalty（0 = critic-free） |
| `--kl-loss-type` | k1 | k1 / k2 / k3 / low_var_kl |
| `--entropy-coef` | 0.0 | 熵正则化 |
| `--gamma` | 1.0 | GAE 折扣因子 |
| `--lambd` | 1.0 | GAE lambda |
| `--n-samples-per-prompt` | 1 | GRPO 组大小 |
| `--normalize-advantages` | False | 跨 DP 组白化 |

## KL Loss 类型

位置: `relax/utils/training/ppo_utils.py` → `compute_approx_kl()`

| 类型 | 公式 | 场景 |
|------|------|------|
| k1 | `log_ratio` | 简单快速（默认） |
| k2 | `(log_ratio)^2 / 2` | 平方近似 |
| k3 / low_var_kl | 非负无偏低方差 | Schulman's KL |

## 损失计算

位置: `relax/backends/megatron/loss.py`

- `policy_loss_function()` — PPO clipped loss + 可选 dual-clip / 非对称 clipping
- `compute_sapo_loss()` — SAPO soft gating
- `value_loss_function()` — 价值函数 clipping
- 高级特性: TIS（截断重要性采样）、OPSM（序列级掩码）、OPD

## Advantage 计算

- `compute_advantages_and_returns()` — GAE / 组归一化
- `distributed_masked_whiten()` — 跨 DP 组归一化
- 支持 Context Parallel (CP) 掩码

## 奖励函数

位置: `relax/engine/rewards/`

| 文件 | 领域 |
|------|------|
| `math_utils.py` | 数学题验证 |
| `deepscaler.py` | DeepScaler |
| `gpqa.py` | GPQA 评估 |
| `f1.py` | F1 分数 |
| `multiple_choice.py` | 选择题 |
| `ifbench.py` | IFBench |
| `dapo_genrm.py` | DAPO GenRM |
| `openr1mm.py` | OpenR1MM |

支持 `async_rm()` / `batched_async_rm()` 异步计算。

## 训练循环

位置: `relax/core/controller.py` → `Controller.training_loop()`

Rollout → TransferQueue → Actor 训练 → 权重更新 → Rollout。可选 Critic、Advantages、GenRM 服务。

## 常见问题

| 问题 | 解决 |
|------|------|
| Reward 全 0/1 | 检查 `rm_hub/` 中奖励提取逻辑 |
| KL 爆炸 | 减小 `--eps-clip`，增大 `--kl-coef` |
| 无学习信号 | 检查 `--normalize-advantages`，确保方差存在 |
| Advantage 全零 | 确认 `--n-samples-per-prompt > 1`（GRPO） |
| OOM | 启用 `--use-dynamic-batch-size` + `--max-tokens-per-gpu` |

## 关键文件

| 文件 | 用途 |
|------|------|
| `relax/utils/arguments.py` | RL 算法参数 |
| `relax/utils/training/ppo_utils.py` | KL / policy loss / SAPO |
| `relax/backends/megatron/loss.py` | loss / advantage / value |
| `relax/components/actor.py` | Actor 服务 |
| `relax/components/critic.py` | Critic 服务 |
| `relax/components/advantages.py` | Advantages 服务 |
| `relax/core/controller.py` | 训练循环编排 |
| `relax/engine/rewards/` | 奖励函数注册 |
| `relax/engine/rollout/on_policy_distillation.py` | OPD |
