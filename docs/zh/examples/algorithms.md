# 算法参考

Relax 支持多种策略梯度算法，均通过 `--advantage-estimator` 参数选择。本文档覆盖 PPO 与主要 GRPO-family 算法（OPD 在线策略蒸馏请参阅[单独文档](./on-policy-distillation.md)）。

GRPO、CISPO、GSPO 与 SAPO 使用相同的服务拓扑，可以直接在现有脚本中替换算法参数块。PPO 还需要 Critic 模型与 Advantages 服务，因此应从 [PPO 训练配置](../guide/ppo-training.md)开始，而不是只替换 `GRPO_ARGS`。

Dr.GRPO 将组内中心化 advantage 与固定尺度 token-sum policy loss 组合起来。两个修改都可以在 GRPO 训练 recipe 中显式启用。

REINFORCE++ 与 REINFORCE++-baseline 同样复用 GRPO 服务拓扑，但 return、全局归一化和 KL 契约由算法单独定义。启用任一 estimator 前，请先阅读 [REINFORCE++ 训练文档](../guide/reinforce-plus-plus.md)。

---

## GRPO

GRPO（Group Relative Policy Optimization）是 Relax 的默认算法。将组内标量奖励广播到每个 token，使用 PPO 风格的裁剪目标函数。

参考论文：[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)。

### 算法原理

GRPO 目标函数为标准 PPO-Clip：

$$J_\text{GRPO}(\theta) = \mathbb{E} \left[ \min\!\left( r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\ 1-\varepsilon,\ 1+\varepsilon)\hat{A}_t \right) \right]$$

其中 $r_t(\theta) = \pi_\theta / \pi_{\theta_\text{old}}$，$\hat{A}_t$ 是组相对 advantage（组内奖励减去组均值，按组标准差归一化）。当 ratio 超出裁剪边界时，梯度被直接置零。

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator grpo` | 默认 | 启用 GRPO |
| `--eps-clip` | `0.2` | 裁剪边距（ratio 范围 = `[1-ε, 1+ε]`） |
| `--eps-clip-high` | 与 `--eps-clip` 相同 | 上方裁剪边距，可设为不同值以构造非对称裁剪 |
| `--clip-grad` | — | 梯度裁剪范数 |

### 快速开始

GRPO 是默认算法，无需修改参数。直接使用训练脚本即可：

```bash
MODEL_DIR=/path/to/model \
DATA_DIR=/path/to/data \
EXP_DIR=/path/to/exp \
bash scripts/training/text/run-qwen3-4B-8xgpu.sh
```

---

## Dr.GRPO

Dr.GRPO 对标准 GRPO 目标做两个修改：移除 advantage 的组内标准差归一化，并使用一个固定的回复长度尺度归一化 policy-gradient token loss。在保留现有 GRPO 策略目标的同时，降低目标函数对采样回复长度的依赖。

参考论文：[Dr.GRPO](https://arxiv.org/abs/2503.20783)。

### 算法原理

对于一个奖励为 $R_1, \ldots, R_G$ 的 prompt group，Dr.GRPO 使用中心化 advantage：

$$A_i = R_i - \frac{1}{G}\sum_{j=1}^{G}R_j$$

不再将中心化后的 reward 除以组内标准差。设回复 token mask 为 $m_{i,t}$，token 级 policy loss 为 $\ell_{i,t}$，全局 response batch size 为 $B$，固定尺度为 $S$，则固定尺度 aggregation 为：

$$
\mathcal{L}_{\mathrm{Dr.GRPO}} =
\frac{1}{B}\sum_{i=1}^{B}
\frac{\sum_t m_{i,t}\ell_{i,t}}{S},
\qquad
S = \texttt{--pg-loss-scale-factor}
$$

默认的 `seq-mean-token-mean` aggregation 会让每个 response 都贡献一个等权的 token 均值。`seq-mean-token-sum-norm` 则对每个 response 求有效 token loss 之和，再除以同一个 $S$，因此 response 的相对贡献与其有效回复 token 数成比例。

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator grpo` | `grpo` | 选择 Dr.GRPO 使用的 GRPO advantage 计算方式 |
| `--disable-grpo-std-normalization` | 未设置 | 使用不进行组内标准差归一化的中心化 group reward |
| `--pg-loss-aggregation` | `seq-mean-token-mean` | 设置为 `seq-mean-token-sum-norm` 以启用固定尺度 token-sum aggregation |
| `--pg-loss-scale-factor` | `None` | 固定 token-sum 尺度；未设置时根据 `--rollout-max-response-len` 推导 |
| `--calculate-per-token-loss` | — | 启用 Dr.GRPO recipe 使用的 per-token loss 路径 |

### 快速开始

显式的 Dr.GRPO 参数块如下：

```bash
DR_GRPO_ARGS=(
   --advantage-estimator grpo
   --disable-grpo-std-normalization
   --pg-loss-aggregation seq-mean-token-sum-norm
   --calculate-per-token-loss
)
```

配对的 Qwen3.5-4B recipe 使用同一个脚本运行 Dr.GRPO 和标准 GRPO。设置 `USE_DRGRPO=1` 运行 Dr.GRPO 配置：

```bash
MODEL_DIR=/path/to/model/root \
DATA_DIR=/path/to/data/root \
USE_DRGRPO=1 \
bash examples/algorithms/dr_grpo/run-qwen35-4B-dr-grpo-2xgpu.sh
```

使用相同命令将 `USE_DRGRPO=0`，即可运行配对 recipe 中的标准 GRPO 配置。

---

## PPO

PPO（Proximal Policy Optimization）是一种 Actor-Critic 算法。Relax 会训练独立的 Critic 来预测 token 级 value，计算 GAE advantages 与 returns，对 Actor 使用 PPO-Clip，并对 Critic 使用裁剪 value loss。

参考论文：[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)。

### 算法原理

Temporal-difference residual 与 GAE 递推为：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$\hat{A}_t = \delta_t + \gamma\lambda\hat{A}_{t+1}, \qquad \hat{R}_t = \hat{A}_t + V(s_t)$$

Actor 随后使用 GRPO 一节展示的裁剪策略目标，但 advantage 来自 Critic 的 token 级估计。Critic 则最小化裁剪与未裁剪 value 平方误差中的较大值。

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator ppo` | — | 启用 PPO 与 Critic 服务图 |
| `--gamma` | `1.0` | GAE 折扣因子 |
| `--lambd` | `1.0` | GAE lambda |
| `--eps-clip` | `0.2` | Actor 裁剪边距 |
| `--value-clip` | `0.2` | Critic value 裁剪范围 |
| `--num-critic-only-steps` | `0` | 初始 Critic-only 预热步数 |
| `--critic-lr` | 与 `--lr` 相同 | Critic 学习率 |

### 快速开始

PPO 的服务图要求 `critic` 与 `advantages` 资源，因此不能只修改算法参数来启用。暂不支持 fully-async PPO；请使用专用同步 colocate 配置：

```bash
MODEL_DIR=/path/to/models \
DATA_DIR=/path/to/data \
EXP_DIR=/path/to/experiments \
bash scripts/training/text/run-qwen35-9B-8xgpu-ppo.sh
```

资源拓扑、checkpoint 规则与 KL 约束详见 [PPO 训练](../guide/ppo-training.md)。

---

## CISPO

CISPO（Clipped Importance-ratio Soft Policy Optimization）对超出信任域的 token 保留梯度信号，而非将其清零。通过 stop-gradient 系数限制梯度幅度，同时保留梯度方向。

参考论文：[MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585)。

### 算法原理

CISPO 目标函数为：

$$J_\text{CISPO}(\theta) = \mathbb{E}_{(q,a)\sim\mathcal{D},\ \{o_i\}_{i=1}^G \sim \pi_{\theta_\text{old}}(\cdot|q)} \left[ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \text{sg}\!\left(\hat{r}_{i,t}(\theta)\right) \hat{A}_{i,t} \log \pi_\theta(o_{i,t} \mid q, o_{i,<t}) \right]$$

其中 $\hat{r}_{i,t}(\theta)$ 是裁剪后的重要性采样权重：

$$\hat{r}_{i,t}(\theta) = \text{clip}\!\left(r_{i,t}(\theta),\ 1 - \varepsilon_\text{low}^\text{IS},\ 1 + \varepsilon_\text{high}^\text{IS}\right)$$

$r_{i,t}(\theta) = \pi_\theta(o_{i,t} \mid q, o_{i,<t}) / \pi_{\theta_\text{old}}(o_{i,t} \mid q, o_{i,<t})$。梯度**只**流过 $\log\pi_\theta$，$\hat{r}_{i,t}$ 和 $\hat{A}_{i,t}$ 均被 stop-gradient 处理。

### 关键参数

| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `--advantage-estimator cispo` | — | — | 启用 CISPO |
| `--eps-clip` | `0.2` | `0.2` | 下方裁剪边距（ratio 下界 = `1 - eps_clip`） |
| `--eps-clip-high` | 与 `--eps-clip` 相同 | `10` | 上方裁剪边距（ratio 上界 = `1 + eps_clip_high`）。设为 `10` 可近似取消上侧裁剪 |
| `--kl-loss-coef` | `0.0` | `0.001` | KL 损失系数。推荐设为 `0.001`，添加小幅 KL 惩罚以约束策略偏移 |
| `--use-kl-loss` | 关闭 | 开启 | 启用 KL 损失计算（`--kl-loss-coef` 生效的前提） |
| `--use-tis` | 关闭 | 开启 | Token Importance Sampling，推荐与 CISPO 同时开启 |
| `--clip-grad` | — | `1.0` | 梯度裁剪范数 |

### 快速开始

使用任意 GRPO 训练脚本，将 `GRPO_ARGS` 替换为 `CISPO_ARGS`：

```bash
CISPO_ARGS=(
   --advantage-estimator cispo
   --use-kl-loss
   --kl-loss-coef 0.001
   --eps-clip 0.2
   --eps-clip-high 10
   --use-tis
)
```

---

## GSPO

GSPO（Group-wise Sequence-level Policy Optimization）与 GRPO 的区别在于 KL 散度的计算方式：GSPO 使用**序列级** KL 而非逐 token KL。每个 token 的 KL 值等于该序列所有 token KL 的均值，这为序列内所有 token 提供统一的约束强度。

### 算法原理

GSPO 使用与 GRPO 相同的 PPO-Clip 目标函数，但 ratio 的计算基于序列级 KL：

$$\text{KL}_\text{seq} = \frac{1}{|o|} \sum_{t=1}^{|o|} \left(\log\pi_{\theta_\text{old}}(o_t) - \log\pi_\theta(o_t)\right)$$

每个 token 的 ratio 均为 $r_t = \exp(-\text{KL}_\text{seq})$，而非各自独立的 token 级 ratio。

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator gspo` | — | 启用 GSPO |
| `--eps-clip` | `0.2` | 裁剪边距 |
| `--eps-clip-high` | 与 `--eps-clip` 相同 | 上方裁剪边距 |
| `--clip-grad` | — | 梯度裁剪范数 |

### 快速开始

```bash
GSPO_ARGS=(
   --advantage-estimator gspo
   --eps-clip 0.2
)
```

---

## SAPO

SAPO（Soft Adaptive Policy Optimization）用平滑的 sigmoid 门控替代硬裁剪。通过温度参数控制门控曲线的陡峭程度，实现可微的信任域约束。

### 算法原理

SAPO 的核心是一个以 ratio=1 为中心的 sigmoid 门控函数：

$$f(r) = \frac{4}{\tau} \cdot \sigma\!\left(\tau(r - 1)\right)$$

其中 $\sigma$ 是 sigmoid 函数，$\tau$ 根据 advantage 的符号选择不同的温度：

- $A > 0$: 使用 $\tau_\text{pos}$（默认 1.0）
- $A \leq 0$: 使用 $\tau_\text{neg}$（默认 1.05，对 negative token 更强的抑制）

SAPO 目标：$J_\text{SAPO}(\theta) = f(r) \cdot A$

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator sapo` | — | 启用 SAPO |
| `--sapo-tau-pos` | `1.0` | positive advantage 的温度参数 |
| `--sapo-tau-neg` | `1.05` | negative advantage 的温度参数（更高 = 更强抑制） |
| `--clip-grad` | — | 梯度裁剪范数 |

### 快速开始

```bash
SAPO_ARGS=(
   --advantage-estimator sapo
   --sapo-tau-pos 1.0
   --sapo-tau-neg 1.05
)
```

---

## 算法对比

| 算法 | Advantage 计算 | 策略损失 | KL 约束方式 |
|------|---------------|---------|-----------|
| **PPO** | Critic value + GAE | PPO-Clip（硬裁剪） | 当前同步拓扑中禁用 |
| **GRPO** | 组相对奖励 | PPO-Clip（硬裁剪） | 可选 KL loss |
| **Dr.GRPO** | 不进行标准差归一化的中心化组奖励 | 固定尺度 token-sum aggregation | 可选 KL loss |
| **REINFORCE++** | Token KL-to-go return + 全局 token 归一化 | PPO-Clip（硬裁剪） | shaped reward 中的 k1 KL |
| **REINFORCE++-baseline** | Inclusive group mean + 全局 token 归一化 | PPO-Clip（硬裁剪） | 独立 k2 KL loss |
| **CISPO** | 组相对奖励 | Stop-gradient 系数 | 推荐 KL loss |
| **GSPO** | 组相对奖励 | PPO-Clip + 序列级 KL | 序列级 ratio |
| **SAPO** | 组相对奖励 | Sigmoid 门控 | 温度控制 |

## 下一步

- [PPO 训练](../guide/ppo-training.md)
- [REINFORCE++ 训练](../guide/reinforce-plus-plus.md)
- [快速开始](../guide/quick-start.md)
- [在线策略蒸馏](./on-policy-distillation.md)
- [生成式奖励模型](./generative-reward-model.md)
