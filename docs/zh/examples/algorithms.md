# 算法参考

Relax 支持多种策略梯度算法，均通过 `--advantage-estimator` 参数选择。本文档覆盖 PPO 与主要 GRPO-family 算法（OPD 在线策略蒸馏请参阅[单独文档](./on-policy-distillation.md)）。

GRPO、RLOO、CISPO、GSPO 与 SAPO 使用相同的 Actor/Rollout 服务拓扑；其中 RLOO 仅支持同步模式，并要求固定批量不变量。PPO 还需要 Critic 模型与 Advantages 服务，因此应从 [PPO 训练配置](../guide/ppo-training.md)开始，而不是只替换 `GRPO_ARGS`。

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

## RLOO

RLOO（REINFORCE Leave-One-Out）使用同一 prompt 的其他采样结果作为无偏基线。Relax 实现的是同步 RLOO，并采用不裁剪的 REINFORCE 策略损失，不使用 PPO ratio 或 clipping。

参考论文：[Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740)。

### 算法原理

对一个 prompt 采样 $G$ 条 response、得到标量奖励 $r_i$ 时，leave-one-out 基线与 advantage 为：

$$b_i = \frac{1}{G-1}\sum_{j\ne i}r_j, \qquad A_i = r_i-b_i = \frac{G}{G-1}(r_i-\bar r)$$

与 GRPO 不同，RLOO 不除以组内标准差。逐 token 损失为：

$$L_{i,t} = -\operatorname{stopgrad}(A_i)\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})$$

每条样本的标量 advantage 会广播到 response token。Relax 会屏蔽 padding，并按全局有效 response token 数 $N_{\mathrm{eff}}$ 归一化 token loss 之和：

$$L_{\mathrm{RLOO}} = -\frac{1}{N_{\mathrm{eff}}}\sum_i\sum_t m_{i,t}\operatorname{stopgrad}(A_i)\log\pi_{i,t}$$

这种 global-token reduction 不会为每条 response 单独附加 $1/T_i$ 权重。RLOO 不使用 clipping，因此 `train/pg_clipfrac` 始终为 `0`。

### 约束与参数

| 参数 | 要求 | 说明 |
|------|------|------|
| `--advantage-estimator rloo` | 必填 | 启用 RLOO |
| `--n-samples-per-prompt` | 至少为 `2` | 组大小 $G$；更大的组可提高 LOO 基线稳定性，但增加 rollout 成本 |
| `--rollout-batch-size × --n-samples-per-prompt` | 等于 `--global-batch-size` | 每个 rollout 恰好执行一次 optimizer update |
| `--num-steps-per-rollout` | 不设置或为 `1` | 禁止对同一 rollout 重复执行未经 ratio 校正的更新 |
| `--calculate-per-token-loss` | 开启 | 使用全局有效 token 归一化；per-response token mean 会按 $1/T_i$ 重新加权不等长 response |
| `--kl-coef` | `0` | RLOO 尚未实现 reward-side KL shaping；提供有效的 `--ref-load <checkpoint>` 后，可使用受支持的 `--use-kl-loss --kl-loss-coef <value>` 直接 KL loss |
| `--max-staleness` | `0` | 非裁剪目标没有 importance-ratio correction，因此拒绝 stale rollout |
| reward normalization | 开启 | RLOO 的组变换位于 normalized-reward 路径 |
| `--normalize-advantages` | 关闭 | DP 切分后的再次白化会改变 RLOO 语义，并使结果依赖分区 |
| `--fully-async`、`--hybrid`、`--partial-rollout`、`--use-dynamic-global-batch-size` | 关闭 | RLOO 当前要求同步、固定大小的 rollout batch |

只要保持批量等式，便可根据硬件调整批量参数。例如 `ROLLOUT_BATCH_SIZE=4`、`N_SAMPLES=8`、`GLOBAL_BATCH_SIZE=32` 仍保持每 rollout 一次更新，同时比 `16 × 8 = 128` 降低每步显存压力。

### 诊断指标

训练 rollout 日志会发布以下最终指标名：

- `rollout/rloo/baseline_mean`：LOO baseline 均值（等于组奖励均值；保留为显式 baseline 轨迹）
- `rollout/rloo/adv_abs_mean`：RLOO advantage 的平均绝对值
- `rollout/rloo/no_signal_frac`：属于零 advantage 样本的有效 loss token 比例
- `rollout/rloo/empty_response_frac`：response 字面为空的样本比例
- `rollout/rloo/zero_adv_group_frac`：所有 advantage 都为零的完整组比例
- `rollout/rloo/dropped_group_frac`：因组大小不完整而未纳入诊断的已观察组比例

这些诊断只用于训练 rollout，属于纯观测统计，不影响训练路径。Eval 使用独立的采样组大小，不会记录具有误导性的 `eval/*/rloo/*` 全零指标。当自定义 reward post-processor 或 agentic custom-advantage hook 替换标准 RLOO 信号时，也会省略这些指标，因为此时无法仅靠 raw reward 重建 optimizer 的真实输入。

### 快速开始

使用 Qwen3-0.6B GSM8K 专用 recipe；批量与 rollout 数均可通过环境变量覆盖：

```bash
MODEL_DIR=/path/to/models \
DATA_DIR=/path/to/data \
NUM_ROLLOUT=60 \
ROLLOUT_BATCH_SIZE=4 \
N_SAMPLES=8 \
GLOBAL_BATCH_SIZE=32 \
bash examples/algorithms/run-qwen3-0.6B-1xgpu-rloo.sh
```

设置 `ADVANTAGE_ESTIMATOR=grpo` 可在相同 recipe 与 seed 下运行对照臂。该 recipe 会把清洗后的 GSM8K 写入可写的 artifact cache（可用 `RLOO_DATA_CACHE_DIR` 覆盖），并在问题后追加最终答案须使用 `\boxed{...}` 的指令，以匹配 `math` reward parser 的输入契约。

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

## GDPO

GDPO（Group reward-Decoupled Normalization Policy Optimization，[arXiv 2601.05242](https://arxiv.org/abs/2601.05242)）面向**多奖励**训练。它对每个奖励分量分别做组内标准化，再合并——而不是像 GRPO 那样先把多个奖励加起来再归一化。

### 算法原理

设第 $i$ 个 prompt 采样 $G$ 条 rollout，共 $n$ 个奖励分量。

**第一步 —— 逐奖励组内标准化**：

$$A_k^{(i,j)} = \frac{r_k^{(i,j)} - \mathrm{mean}_j\{r_k^{(i,\cdot)}\}}{\mathrm{std}_j\{r_k^{(i,\cdot)}\} + \epsilon}$$

**第二步 —— 加权求和**：

$$A_\text{sum}^{(i,j)} = \sum_k w_k A_k^{(i,j)}$$

注意权重乘在**归一化后的 advantage** 上，不是乘在原始 reward 上。经过第一步各分量已在同一尺度，权重表达的是相对重要性，而不是分量的量纲。

**第三步 —— batch 级白化**：

$$\hat{A}^{(i,j)} = \frac{A_\text{sum}^{(i,j)} - \mathrm{mean}_\text{batch}}{\mathrm{std}_\text{batch} + \epsilon}$$

**相对 GRPO 的收益**：当一组 rollout 的各**分量不同、但总和恰好相同**时（例如 `(1,0)` 与 `(0,1)` 都求和为 1），GRPO 看到的组内总奖励无差异、整组 advantage 归零被丢弃；GDPO 对每个分量单独标准化，仍能保留各分量的学习信号。若某个分量在组内恒定，则只有**该分量**贡献 0、其它分量照常提供信号；若**所有**分量都恒定，GDPO 与 GRPO 一样返回零。

**关于 $\epsilon$**：GDPO 的两步都用 $\epsilon = 10^{-4}$，与参考实现（TRL `GRPOTrainer` 的 `scale_rewards` GDPO 分支）一致，而 GRPO / GSPO / SAPO / CISPO 沿用本仓库既有的 $10^{-6}$。两者的差别只在近乎塌缩的组上显现：二值 reward、组大小 8 时组内标准差约 0.4，两个取值的差异是 0.02%；但连续 reward（论文的数学实验用响应长度）可能让某组的标准差落到 $10^{-3}$ 量级，此时 $10^{-4}$ 会把该组的信号额外压低约 7%，而 $10^{-6}$ 只压低 0.08%。**完全**塌缩的组不会走到这个除法——它们由 exact 相等判定后直接置零。

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator gdpo` | — | 启用 GDPO |
| `--gdpo-reward-keys` | — | **必填**，至少两个。奖励函数返回的 dict 中要独立归一化的 key，如 `correctness format` |
| `--gdpo-reward-weights` | 全 1.0 | 各分量权重，长度须与 `--gdpo-reward-keys` 一致 |
| `--reward-key` | — | **必填**，选出用于 metrics 与 `raw_reward` 列的标量 |
| `--n-samples-per-prompt` | — | 必须 ≥ 2（组内无偏标准差在 $G=1$ 时无定义） |

奖励函数必须返回包含全部 key 的 dict。缺 key、非数值、bool、NaN/Inf 都会直接报错而不是填 0——静默填 0 会把契约违约伪装成真实的 reward collapse。

### 快速开始

```bash
GDPO_ARGS=(
   --advantage-estimator gdpo
   --gdpo-reward-keys correctness format
   --gdpo-reward-weights 1.0 1.0
   --custom-rm-path examples.gdpo.reward_gdpo.reward_func
   --reward-key score
   --n-samples-per-prompt 8
)
```

完整可运行示例见 [`examples/gdpo/`](https://github.com/redai-infra/Relax/tree/main/examples/gdpo)。

### 已知偏差

以下两点是实现与论文之间的实际差异，训练前请确认可以接受。第三步的 batch 边界曾经也在此列，现已修正——见下。

**第三步的 batch 边界（已对齐）**。论文 Eq. 6 在**一个训练批**上归一化。调用方会先把 `num_rollout_minis` 个训练批用 `concat_rollout_batches` 合并再进 advantage 阶段，所以第三步必须知道批边界。边界由 `ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY` 携带（colocate 与 hybrid 三条路径都写入），`loss.py` 作为 `mini_batch_sizes` 传给 advantage 分发器，**只有 GDPO 消费**——其余估计器由 `**_unused` 吞掉，逐位不变。每段各自跨 DP all-reduce，所以统计量既覆盖完整训练批、也覆盖全部 rank。这一点为什么重要：合并白化会把两个批都对着共同均值中心化，实测 8 个样本里有 4 个**符号翻转**——那是另一个优化目标，不是精度差异。

**`--fully-async` 仍不受支持**，参数校验阶段直接拒绝：那条路径把 advantage 计算交给单副本的 Advantages 服务，它没有数据并行通信域，也拿不到批边界，且每次只消费 `global_batch_size / num_iters_per_train_update` 的一个切片；当这个商为 1 时白化输出恒为 0，训练会安静地在零信号上跑完。

1. **单个奖励时 GDPO 不退化为 GRPO**。第三步仍然生效，结果与 GRPO 相差一个正标量（与数据相关，实测约 1.21）。要 GRPO 语义就直接用 `--advantage-estimator grpo`。
2. **$G=2$ 时幅度信息丢失**。任意两个不同值经无偏标准化后恒为 $\pm 1/\sqrt{2}$，此时分量之间的区分度只来自权重。

### 互斥项

- 不能与 `--normalize-advantages` 同用：第三步已经做过序列级白化，再叠加 token 级白化没有意义。
- 不能与 `--custom-reward-post-process-path` 同用：该钩子会整段短路奖励后处理，导致第一、二步被静默跳过，而训练日志仍然显示算法是 GDPO。
- 不能与 `--agentic-custom-advantage-path` 同用：`post_process_rewards` 里的第二个早返回点，同样赶在归一化器之前返回，后果与上一条相同。两者由 `AlgorithmSpec.allows_reward_post_process_hooks` 一起把守。
- 不能与 `--fully-async` 同用（见上）。

以上都会在参数校验阶段直接报错。配合 `--dynamic-sampling-filter-path` 时会给出警告：内置的 `check_reward_nonzero_std` 只看 `--reward-key` 那一个标量，可能丢掉只存在于其它分量的信号。

---

## 算法对比

| 算法 | Advantage 计算 | 策略损失 | KL 约束方式 |
|------|---------------|---------|-----------|
| **PPO** | Critic value + GAE | PPO-Clip（硬裁剪） | 当前同步拓扑中禁用 |
| **GRPO** | 组相对奖励 | PPO-Clip（硬裁剪） | 可选 KL loss |
| **REINFORCE++** | Token KL-to-go return + 全局 token 归一化 | PPO-Clip（硬裁剪） | shaped reward 中的 k1 KL |
| **REINFORCE++-baseline** | Inclusive group mean + 全局 token 归一化 | PPO-Clip（硬裁剪） | 独立 k2 KL loss |
| **CISPO** | 组相对奖励 | Stop-gradient 系数 | 推荐 KL loss |
| **GSPO** | 组相对奖励 | PPO-Clip + 序列级 KL | 序列级 ratio |
| **SAPO** | 组相对奖励 | Sigmoid 门控 | 温度控制 |
| **RLOO** | Leave-one-out 基线 | 非裁剪 REINFORCE | 可选 KL loss（同 GRPO） |
| **GDPO** | 逐奖励组内标准化 + 加权求和 + batch 白化 | PPO-Clip（硬裁剪） | 可选 KL loss |

## 下一步

- [PPO 训练](../guide/ppo-training.md)
- [REINFORCE++ 训练](../guide/reinforce-plus-plus.md)
- [快速开始](../guide/quick-start.md)
- [在线策略蒸馏](./on-policy-distillation.md)
- [生成式奖励模型](./generative-reward-model.md)
