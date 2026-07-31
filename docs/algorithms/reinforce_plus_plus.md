# REINFORCE++ / REINFORCE++-baseline

> 参考：[arXiv:2501.03262](https://arxiv.org/abs/2501.03262)（REINFORCE++）。
> 本文给出 Relax 中 REINFORCE++ 及其 baseline 变体的公式、归一化维度、mask 与 reduction
> 口径，并明确其与 GRPO / GSPO / SAPO 的差异。

## 1. 概述

REINFORCE++ 在经典 REINFORCE 基础上引入逐 token KL 惩罚与 advantage 归一化；其 baseline
变体额外引入 group-mean baseline。两者在 Relax 中作为 `--advantage-estimator` 的取值接入，
与 GRPO / GSPO / SAPO 共享同一套服务拓扑，仅在 advantage / loss 计算上不同。

社区中 REINFORCE 改进实现众多，命名与 baseline 口径常不统一。本文固定 Relax 的口径定义：
公式、归一化维度、mask、reduction，以及两个变体与邻近算法的差异。

## 2. 配置与命名

算法通过 `--advantage-estimator`（`relax/utils/arguments.py`）选择。两个变体在
`relax/core/registry.py:ALGOS` 中注册并复用 GRPO 服务拓扑（`_GRPO_TOPOLOGY`：rollout /
actor / advantages / reference / actor_fwd，**无 critic**），由 `--advantage-estimator` 字符串
在 advantage 组件与 `policy_loss_function` 中分发。GRPO 族算法（GRPO / GSPO / SAPO / CISPO /
REINFORCE++ / REINFORCE++-baseline）共享同一拓扑，拓扑定义集中为 `_GRPO_TOPOLOGY` 单一常量。

| 配置值 | 含义 | 必需参数 | KL 口径（论文约定） |
|---|---|---|---|
| `reinforce_plus_plus` | Monte-Carlo 折扣回报，无 group baseline | `--normalize-advantages` | `--kl-coef β` 逐 token k1-style 惩罚折入回报（§3.1） |
| `reinforce_plus_plus_baseline` | group-mean baseline 版本 | `--normalize-advantages` | `--use-kl-loss --kl-loss-type k2 --kl-loss-coef λ` 独立 k2 KL loss（§3.2）；`--kl-coef` 必须为 0 |

`arguments.py` 强制两个变体开启 `--normalize-advantages`，否则报错；并强制
`reinforce_plus_plus_baseline` 的 `--kl-coef == 0`（其 KL 正则来自独立的 k2 KL loss，折入
advantage 的惩罚项在该变体中不存在，设置 `--kl-coef` 会被静默忽略，故直接拒绝）。两个变体
命名可清晰区分；切换算法需真正修改 `--advantage-estimator`，而非复制 GRPO recipe 改名。

## 3. 公式

设第 $i$ 条样本响应长度 $R_i$、逐 token KL 为 $k_i \in \mathbb{R}^{R_i}$、响应 mask
$m_i \in \{0,1\}^{R_i}$、标量奖励 $r_i$、KL 系数 $\beta$（`kl_coef`）、折扣 $\gamma$、
独立 KL loss 系数 $\lambda$（`kl_loss_coef`）。记响应最后一个有效 token 下标为
$t^*_i = \max\{t : m_i[t]=1\}$。

### 3.1 REINFORCE++（`get_reinforce_plus_plus_returns`）

对应论文 §3.1 的 REINFORCE++（k=1）：逐 token 奖励（k1-style KL 惩罚 + 终端奖励）：

$$
\tilde r_i[t] = -\beta \cdot k_i[t] \cdot m_i[t], \quad
\tilde r_i[t^*_i] \mathrel{+}= r_i
$$

其中逐 token KL 使用 **k1-style 估计器**（论文 "k1-style penalty"：
$k_i[t] = \log \pi_{\theta_{\text{old}}}(o_t \mid \cdot) - \log \pi_{\text{ref}}(o_t \mid \cdot)$，
即 Relax `compute_approx_kl` 的 `k1`，由 `--kl-loss-type k1` 选择）。折扣回报（反向累加）：

$$
G_i[t] = \tilde r_i[t] + \gamma \cdot G_i[t+1], \quad G_i[R_i] = 0
$$

advantage = return：$A_i = G_i$。**不做 group baseline**（使用原始 $r_i$）。$\beta$ 由
`--kl-coef` 给出（recipe 默认 0.001）；论文取 $\gamma = 1$（即 $G_t = \sum_{i \geq t} \tilde r_i$），
Relax 的 `--gamma` 默认值即 1.0，可配置。

### 3.2 REINFORCE++-baseline（`get_reinforce_plus_plus_baseline_advantages`）

对应论文 §3.2 的 "REINFORCE++ /w baseline"（组采样变体，k > 1）。group baseline 在上游
`post_process_rewards`（`relax/utils/utils.py`）中预先减去：

$$
\bar r_i = r_i - \mathrm{mean}_{j \in \text{group}(i)} r_j
$$

与 GRPO / GSPO / SAPO / CISPO 不同，baseline 变体**只做 group-mean 减法，不做 std 归一化**
（`post_process_rewards` 中 std 除法分支仅对 `grpo/gspo/sapo/cispo` 生效）。随后 advantage
函数把 $\bar r_i$ 广播到每个 token——**KL 惩罚不折入 advantage**（论文 §3.2：该变体采用
独立 KL loss 项做正则）：

$$
A_i[t] = \bar r_i, \quad \text{return} = A_i
$$

KL 正则由 `policy_loss_function` 中的**独立 k2 KL loss** 提供（论文 Eq. 8：
$\mathcal{L} = \mathcal{L}_{\text{PPO}}(A^{\text{norm}}) + \lambda \cdot \mathbb{E}\left[
\frac{1}{2} \left(\log \frac{\pi_\theta}{\pi_{\text{ref}}}\right)^2 \right]$，
即 Relax `compute_approx_kl` 的 `k2`），由 `--use-kl-loss --kl-loss-type k2 --kl-loss-coef λ`
启用。`arguments.py` 强制该变体 `--kl-coef == 0`（KL 不进 advantage）。

### 3.3 优势归一化（两变体共用，`--normalize-advantages`）

在同步（colocate）路径 `relax/backends/megatron/loss.py:compute_advantages_and_returns` 中，
对拼接后的全 batch advantage 在 DP 组上做 masked whiten（`distributed_masked_whiten`，
Bessel 修正）：

$$
\hat A = (A - \mu_{\text{mask}}) / \sqrt{\sigma^2_{\text{mask}} + \epsilon}
$$

统计量在**所有有效 token（mask=1）**上计算，padding / prompt 不参与。

## 4. 归一化维度 / mask / reduction 口径

| 项 | REINFORCE++ | REINFORCE++-baseline | GRPO |
|---|---|---|---|
| return 计算 | 逐 token 折扣 MC 回报 $G_t$ | 无折扣，广播标量 advantage | 广播标量 advantage |
| group baseline | 否 | 是（group-mean，**无 std**） | 是（group-mean，**有 std**） |
| KL 计入位置 | 逐 token k1-style 计入 reward（`--kl-coef`） | **不计入 advantage**（独立 k2 KL loss，`--use-kl-loss --kl-loss-type k2`） | 不计入 advantage（走单独 KL loss） |
| advantage 归一化 | masked whiten over DP | masked whiten over DP | 可选 std 归一化（group 内） |
| mask 作用域 | response token；padding 不注入奖励、不参与统计 | 同左 | 同左 |
| loss reduction | per-token（`sum_of_sample_mean`，`--calculate-per-token-loss`） | 同左 | 同左 |

**mask 口径**：`loss_masks` 为响应级 mask（有效 response token = 1，prompt / padding = 0）。
- REINFORCE++ 中标量奖励仅注入到 $t^*_i$（最后一个有效 response token），不注入 prompt / padding；
  padding 之后的 token 回报恒为 0。
- 归一化与 loss reduction 均以 mask 加权，padding / prompt 不改变有效 token 统计。

**reduction 口径**：loss 经 `compute_policy_loss`（PPO clipped surrogate）逐 token 计算，
再由 `sum_of_sample_mean` 在有效 token 上做 sample-mean 归约（`--calculate-per-token-loss`）。

## 5. 与 GRPO / GSPO / SAPO 的差异

| 算法 | advantage / return | group baseline | std 归一化 | KL 处理 | loss |
|---|---|---|---|---|---|
| GRPO | $r - \text{mean}$ 广播 | 是 | 是 | per-token `ppo_kl`，单独 KL loss | `compute_policy_loss` |
| GSPO | 同 GRPO 优势 | 是 | 是 | **sequence-level KL** | `compute_policy_loss` + seq KL |
| SAPO | GRPO 优势 | 是 | 是 | soft sigmoid 非对称 τ | `compute_sapo_loss` |
| CISPO | GRPO 优势 | 是 | 是 | 梯度只过 `log_probs` | `compute_cispo_loss` |
| **REINFORCE++** | MC 折扣回报 $G_t$ | **否** | advantage whiten | 逐 token k1-style KL 计入 reward（`--kl-coef`） | `compute_policy_loss` |
| **REINFORCE++-baseline** | $\bar r$ 广播 | **是（无 std）** | advantage whiten | **独立 k2 KL loss**（`--use-kl-loss --kl-loss-type k2`） | `compute_policy_loss` + k2 KL loss |

核心区别：
- REINFORCE++ 用**逐 token 折扣 MC 回报**取代 GRPO 的「标量优势广播」；
- REINFORCE++ 的 KL 惩罚**逐 token 直接进入 reward**（k1-style，`--kl-coef`），
  baseline 变体的 KL 正则则是**独立 k2 KL loss**（论文 §3.2，Eq. 8），均与 GRPO 的
  `low_var_kl` 单独 KL loss 估计器不同；
- baseline 变体的 group baseline **不做 std 归一化**（与 GRPO 的关键差异）；
- loss 复用 PPO clipped surrogate（`compute_policy_loss` 的 `else` 分支）。

## 6. CP / DP 行为

### 6.1 CP（context parallel）

`get_reinforce_plus_plus_returns` 是 CP-aware：每 rank 先 `all_gather_with_cp` 重建完整响应，
在完整序列上计算 $G_t$，再 `slice_log_prob_with_cp` 切回本 rank 局部 chunk。因此 **CP 切分不改变
有效 token 上的回报统计**——各 rank 局部 chunk 拼接等于 `cp_size=1` 的完整回报。

`get_reinforce_plus_plus_baseline_advantages` 为纯逐样本计算（无 CP gather / slice），CP 切分
天然不变。

### 6.2 DP（data parallel）

`--normalize-advantages` 的 masked whiten 在 DP 组上聚合全局统计量。因统计量为全 token
求和，**DP 切分不改变每个 token 的归一化结果**。

### 6.3 已知限制：fully-async 模式不支持本族变体

advantage whiten（`distributed_masked_whiten`）只在同步路径
`loss.py:compute_advantages_and_returns` 中执行；异步路径
`relax/components/advantages.py:Advantages` 未做 whiten，且 `arguments.py` 对
fully-async 模式强制 `assert not --normalize-advantages`。由于两个变体都必须开启
`--normalize-advantages`（§2），**fully-async 下它们会被参数校验直接拒绝**（而非"归一化不生效"），
这是全算法共有的既有行为。推荐使用 sync colocate 模式。

## 7. 测试

| 测试文件 | 覆盖项 |
|---|---|
| `tests/utils/training/test_ppo_utils_reinforce.py` | 两变体 advantage / return 与独立参考实现逐元素对齐；变长、全零 reward、单样本、gamma 折扣、mask 不污染 padding、全 mask 报错；共享 `compute_policy_loss` 参考对齐 |
| `tests/backends/megatron/test_reinforce_pp_cp_parity.py` | CP 切分不变性（returns）、baseline CP 不变性、DP 分区下 masked whiten 不变性 |
| `tests/core/test_registry_reinforce.py` | 两变体已注册、复用 GRPO 拓扑、无 critic、`process_role` 返回 colocate |

测试遵循代码库惯例：fake `megatron.core.mpu` + mock 分布式原语，CPU 单进程运行（与
`test_ppo_gae_parity.py` 口径一致）。

## 8. Recipe

- `examples/algorithms/run-qwen3-0.6B-1xgpu-reinforce-pp.sh`
- `examples/algorithms/run-qwen3-0.6B-1xgpu-reinforce-pp-baseline.sh`

基于单卡 GRPO quickstart，切换 `--advantage-estimator reinforce_plus_plus[_baseline]`
并加 `--normalize-advantages`。KL 口径遵循论文（非零系数，随 recipe 生效）：

- REINFORCE++：`--kl-coef 0.001 --kl-loss-type k1`（k1-style 惩罚折入折扣回报，§3.1）；
- REINFORCE++-baseline：`--use-kl-loss --kl-loss-coef 0.001 --kl-loss-type k2`
  （独立 k2 KL loss，`--kl-coef` 保持默认 0，§3.2）。

GRPO 对比组使用 community 的 GRPO quickstart（与本地
`contributor-program/2026-cohort-1/run-qwen3-0.6B-1xgpu-grpo.sh` 逐字节一致，见
`redai-infra/community` 仓库；`--kl-loss-coef 0.00`），保证与既有 GRPO 口径一致。

## 9. 验证方法

同模型（Qwen3-0.6B）、数据（GSM8K）、预算、硬件（单卡，sync colocate）下分别运行
GRPO / REINFORCE++ / REINFORCE++-baseline，对比 reward、loss、KL、吞吐（samples/s、step time）、
GPU 利用率、峰值显存，不少于 3 个稳定窗口。稳定性护栏：无 NaN / Inf、无丢样本、有效 batch /
序列长度不下降。**KL 必须在非零系数下验证**：REINFORCE++ 用 `--kl-coef 0.001`（k1 折入回报）、
REINFORCE++-baseline 用 `--kl-loss-coef 0.001`（k2 独立 loss），recipe 内已固定，训练对比
不得以 `kl-coef=0` / `kl-loss-coef=0.00` 替代（否则退化为无 KL 的纯 MC / 纯 group-mean
口径，与本文档 §3 公式不符）。
