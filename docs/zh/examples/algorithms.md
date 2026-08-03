# 算法参考

Relax 支持多种策略梯度算法，均通过 `--advantage-estimator` 参数选择。本文档覆盖 PPO 与主要 GRPO-family 算法（OPD 在线策略蒸馏请参阅[单独文档](./on-policy-distillation.md)）。

GRPO、CISPO、GSPO、SAPO 与 RLOO 使用相同的服务拓扑，可以直接在现有脚本中替换算法参数块。PPO 还需要 Critic 模型与 Advantages 服务，因此应从 [PPO 训练配置](../guide/ppo-training.md)开始，而不是只替换 `GRPO_ARGS`。

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

## RLOO

RLOO（REINFORCE Leave-One-Out）保留 GRPO 的组内采样，但改变 baseline 的构造方式：每个样本与组内**其他**样本的均值比较，而不是与包含自身的组均值比较。

参考文献：[Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740)。

### 算法原理

对同一 prompt 的 $k$ 条回复，奖励为 $R_1 \dots R_k$：

$$\hat{A}_i = R_i - \frac{1}{k-1}\sum_{j \neq i} R_j$$

由于 baseline 不含 $R_i$，它与被评估的样本统计独立，因此该估计量是无偏的。GRPO 的 baseline 包含 $R_i$，因此 baseline 与样本**正相关**（$\operatorname{Cov}(R_i, \bar{R}) = \operatorname{Var}(R)/k$）。这一相关性会收缩 advantage：$R_i - \bar{R}$ 恰好是留一法取值的 $\frac{k-1}{k}$ 倍，而下面的 $k/(k-1)$ 因子正是把它抵消回来。

两种形式在代数上等价：

$$\hat{A}_i = \frac{k}{k-1}\left(R_i - \bar{R}\right)$$

即 RLOO 等于组内中心化奖励乘以 $k/(k-1)$。由此有两点推论：

- **不做标准差缩放。** GRPO 除以组内标准差，这会让各组之间相互重新加权：奖励接近一致的组，其微小差异会被放大。RLOO 用的因子与组内离散度无关，因此各组保持原有的相对权重。这正是 Dr. GRPO 对 std 归一化提出的质疑。
- **组内 advantage 仍然和为零**，因此日志行为与 GRPO 一致。归约方式不同 —— 见下文。

#### 与 `--disable-grpo-std-normalization` 的关系

这一点值得明说，因为两者关系很近：关闭 std 归一化后，GRPO 给出的正是 $R_i - \bar{R}$，于是

$$\hat{A}^\text{RLOO}_i = \frac{k}{k-1}\,\hat{A}^{\text{GRPO,\,no-std}}_i$$

逐元素成立。Relax 要求每组样本数严格等于 `--n-samples-per-prompt`，因此 $k$ 在一次运行内恒定，该因子是一个**全局常数**。也就是说，RLOO 并非与 `--disable-grpo-std-normalization` 不同的加权方式，而是该估计量配上使 leave-one-out 基线无偏的那个特定常数。在 Adam 下，loss 上的全局常数在参数更新中基本被抵消 —— 所以选 RLOO 的理由是"要那个有名字的无偏估计量"，而不是期待得到不同的梯度方向。

真正有差别的比较对象是**默认的** GRPO：它除以每组各自的标准差，确实会改变各组之间的相对权重。

#### 对梯度裁剪的影响

RLOO 的 advantage 量级小于默认 GRPO，因为二值 reward 下组内标准差通常远小于 1。在 Qwen3-0.6B / GSM8K、$k = 8$、`--lr` 相同的实测中，`train/grad_norm` 均值 RLOO 为 0.47、GRPO 为 1.04。

Adam 对梯度的常数缩放基本不敏感，所以这**基本不构成**重调 `--lr` 的理由。但它对 `--clip-grad` 有影响 —— 裁剪会破坏这种不变性：在 Megatron 默认的 `--clip-grad 1.0` 下，同一组实验里 GRPO 有 **68% 的 step 被裁剪，RLOO 只有 2%**。切换算法时应重新审视 `--clip-grad`，而不是 `--lr`。

#### 如何读报告出来的 loss

`train/pg_loss` 有两点与裁剪类估计器不同，都是预期行为而非异常：

- **它经常是负数。** PPO-Clip 取两个取负项的最大值，天然偏正。RLOO 报的是 `-A · log π`，而 `log π < 0`，所以等于 `A · |log π|`；均值为负只是说明正 advantage 的 token 其 `|log π|` 更小，即模型对得分高的回复本来就更有信心。Qwen3-0.6B / GSM8K 实测：同一批数据下 RLOO 为 `-0.026`，GRPO 为 `+0.045`。
- **它下无界。** 对 `A < 0` 的 token，最小化 `-A · log π` 会把 `log π` 推向 `-∞`，而且没有任何东西约束它 —— 因为没有信任域。这正是裁剪被引入所要消除的性质，属于无裁剪 REINFORCE 固有，而非本实现特有。60 步的实测中表现稳定（`grad_norm` 0.45、entropy 0.49、未发散），但**判断稳定性应看 `train/entropy_loss` 与 `train/grad_norm`，而不是 `pg_loss`**；两者中任一出现漂移时应启用 `--clip-grad`。

与此相关，`rloo_no_signal_fraction` 值得从第一步就盯。同一组实验中它全程均值 **0.48，并在 60 个 rollout 内升到 0.65**，个别 step 达到 1.0：约一半的 token 落在全同分的组里，因而不贡献梯度。这就是 reward 饱和，而且远在 reward 曲线走平之前就能看到。

#### 损失是 REINFORCE，不是 PPO-Clip

RLOO 的目标函数是针对 leave-one-out baseline 的原始 REINFORCE，没有信任域：

$$L_i = -\operatorname{sg}(\hat{A}_i)\,\log \pi_\theta(y_i)$$

这是与本页其他组相对估计器（它们都包在 PPO-Clip 外）的**有意分歧**。复用裁剪目标会让它变成「带 leave-one-out baseline 的 GRPO」而非论文里的 RLOO，而裁剪本身会给一个正是因无偏才被选用的估计器引入偏差。因此 **`--eps-clip` 与 `--eps-clip-high` 对 RLOO 无效**，`train/pg_clipfrac` 恒为 0。

由于没有重要性比值校正，RLOO 假定采样策略等于训练策略。这在每轮 1 个 optimizer step 时成立；若用多个内层 epoch，估计量就变成 off-policy，且没有任何东西修正这个偏差。**这一点在启动时强制校验**，而不是交给 recipe：`rollout_batch_size * n_samples_per_prompt` 必须等于 `--global-batch-size`，且 `--num-steps-per-rollout` 必须未设置或为 `1`。

#### 归约是 completion 级的

上式中的 $\log \pi_\theta(y_i)$ 是**整条** completion 的 log-probability —— 在其 token 上求和，且**不做**长度归一化 —— 组目标是对样本求平均：

$$L = \frac{1}{k}\sum_i -\operatorname{sg}(\hat{A}_i)\sum_t \log \pi_\theta(y_{i,t})$$

这与本页其他按 token 归一化的估计器都不同。另外两种归约都会改变估计量本身，而非仅改变其尺度：

- 按样本对 token 取**均值**得到 $-\hat{A}_i / T_i \sum_t \log \pi$，等于按 response 长度对样本重新加权 —— 长 response 的每 token 梯度更小；
- 按 micro-batch 的总 token 数归一化，会让更新尺度取决于采样器这一步恰好产出了多少 token。

因此 RLOO 保留 `--calculate-per-token-loss` 给出的「每样本 token 求和」，但把梯度的分母从 token 数换成**样本数**。**该开关对 RLOO 是必需的，并在启动时强制校验**：不开它归约就变成每样本取均值，那是另一个估计量，而且这种切换本来是静默的。该计数的构造保证它在 context-parallel 组内 all-reduce 后等于真实样本数，因此任意 CP 度下目标函数完全一致。

上报指标有意保持在其他估计器共用的 per-token 尺度上，因此 `train/pg_loss`、`train/entropy_loss`、`train/ppo_kl` 仍可跨估计器对比；只有梯度归一化发生了变化。

下游其余部分 —— KL loss、TIS、DP/CP 切分 —— 均保持不变。

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator rloo` | — | 启用 RLOO |
| `--n-samples-per-prompt` | `1` | 必须 $\geq 2$；组内只有 1 个样本时没有 leave-one-out baseline，advantage 记为 0 |
| `--disable-rewards-normalization` | 关闭 | 保持关闭。该开关会完全跳过组内 baseline，直接把原始 reward 当作 advantage |
| `--disable-grpo-std-normalization` | 关闭 | 对 RLOO 无效 —— std 分支只针对 GRPO 家族，没有可禁用的对象 |
| `--eps-clip` | `0.2` | **无效** —— RLOO 不裁剪，`train/pg_clipfrac` 恒为 0 |
| `--num-steps-per-rollout`、`--global-batch-size` | — | **启动时强制每轮 rollout 恰好一次 optimizer step**：`rollout_batch_size * n_samples_per_prompt == global_batch_size`，且 `--num-steps-per-rollout` 必须未设置或为 `1`。RLOO 没有 importance-ratio 项，rollout 内的第二步会在未修正的情况下 off-policy 训练 |
| `--kl-coef` | `0.0` | **非零即拒绝。** RLOO 未实现 reward-shaped KL：advantage 来自 `get_grpo_returns`，它只使用 reference KL 的形状而不使用其数值，因此非零系数会白白付出 reference forward 的代价却不改变目标。请改用 `--use-kl-loss` |
| `--use-kl-loss` / `--kl-loss-coef` | 关闭 / `0.0` | 支持。与 GRPO 一样，KL 作为独立的 loss 项相加 |
| `--normalize-advantages` | 关闭 | **启动即拒绝。** 它会在 DP 切分*之后*跨 DP 组重新白化 advantage，把 RLOO 刻意省去的 std 归一化又加回来 |
| `--custom-reward-post-process-path`、`--agentic-custom-advantage-path`、`--custom-convert-samples-to-train-data-path` | 未设置 | **启动即拒绝。** 三者都会替换或短路构造 leave-one-out baseline 的那一步，RLOO 会静默退化为无 baseline 的 REINFORCE |
| `--fully-async` / `--hybrid` | 关闭 | **启动即拒绝。** RLOO 仅支持同步，异步从未验证 |

### 快速开始

在任意 GRPO 脚本中把 `GRPO_ARGS` 替换为 `RLOO_ARGS`：

```bash
RLOO_ARGS=(
   --advantage-estimator rloo
   --eps-clip 0.2
)
```

仓库提供了一份完整的单卡 GSM8K 配置：

```bash
MODEL_DIR=/path/to/models \
DATA_DIR=/path/to/data \
bash examples/algorithms/run-qwen3-0.6B-1xgpu-gsm8k-rloo.sh
```

::: warning
RLOO **仅支持同步，且在启动时强制**：`--fully-async` 与 `--hybrid` 会被直接拒绝而非
静默放行 —— 异步 RLOO 不在范围内且从未验证。请使用 `--colocate`。
:::

### 适用场景

与**默认** GRPO 相比，实质差异在于 RLOO 不除以每组各自的标准差：

- 组内标准差小且不稳定的奖励分布 —— 此时除以它，等于按"这组恰好有多整齐"而不是"这组有多少信息量"来重新加权
- GRPO 下 `--clip-grad` 频繁触发的场景 —— RLOO 的 advantage 量级更小，被裁剪的步数更少（见上文）
- 排查训练异常时，作为有名字的无偏参照点，判断问题是否来自 advantage 估计器

与 `--disable-grpo-std-normalization` 相比，差异只是那个常数 $k/(k-1)$，它的作用是让 leave-one-out 基线无偏。该修正在小组规模下更大（$k = 4$ 时为 $1.33$，$k = 16$ 时为 $1.07$），但它终究是常数，因此在 Adam 下不要期待由它带来行为变化。

---

## 算法对比

| 算法 | Advantage 计算 | 策略损失 | KL 约束方式 |
|------|---------------|---------|-----------|
| **PPO** | Critic value + GAE | PPO-Clip（硬裁剪） | 当前同步拓扑中禁用 |
| **GRPO** | 组相对奖励 | PPO-Clip（硬裁剪） | 可选 KL loss |
| **CISPO** | 组相对奖励 | Stop-gradient 系数 | 推荐 KL loss |
| **GSPO** | 组相对奖励 | PPO-Clip + 序列级 KL | 序列级 ratio |
| **SAPO** | 组相对奖励 | Sigmoid 门控 | 温度控制 |
| **RLOO** | Leave-one-out baseline，不做 std 缩放 | **无裁剪 REINFORCE** | 可选 KL loss |

## 下一步

- [PPO 训练](../guide/ppo-training.md)
- [快速开始](../guide/quick-start.md)
- [在线策略蒸馏](./on-policy-distillation.md)
- [生成式奖励模型](./generative-reward-model.md)
