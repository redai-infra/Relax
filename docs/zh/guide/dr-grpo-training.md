# Dr.GRPO 训练

Relax 支持在 Megatron 后端的同步 colocate 和 hybrid 模式中，使用 Dr.GRPO（Group Relative Policy Optimization Done Right）进行 closed-window dense 模型训练。

可复现的 Qwen3.5-4B GSM8K 对照实验见 [Dr.GRPO 200-Step 训练报告](./dr-grpo-training-report.md)。

## 概述

Dr.GRPO 移除了 vanilla GRPO 中可能引入偏差的两个归一化项：按每条 response 自身长度归一化，以及按组内 reward 标准差归一化。该算法来自论文 [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)。

使用 `--advantage-estimator dr_grpo` 即可选择该算法。Dr.GRPO 与 GRPO 使用相同的 Rollout、Actor、Advantages、Reference 和 ActorFwd 服务拓扑，不需要 Critic。

## 算法

对于同一个 prompt group 中的第 `i` 条 response，Relax 默认的 reward 后处理首先计算组内中心化 reward：

$$
A_i = r_i - \frac{1}{G}\sum_{j=1}^{G} r_j
$$

其中 $G$ 是同一个 prompt 采样出的 response 数。与 GRPO 不同，Dr.GRPO 不会再用组内 reward 标准差除以 $A_i$。默认 Dr.GRPO policy objective 对整个 optimizer window 使用同一个固定分母：

$$
\mathcal{L}_{\mathrm{Dr.GRPO}} = \frac{S_{\mathrm{actor}}}{N B}
$$

其中：

- `N` 是 optimizer window 内全局 response 数；
- `B` 是 `--rollout-max-response-len`；
- 被 mask 的 token 和 padding token 不进入分子。

该固定预算避免仅仅因为 response 较短，就给其中每个 token 更大的权重。

::: warning 拒绝 advantage normalization
Dr.GRPO 会拒绝 `--normalize-advantages`。去掉 advantage 的方差归一化正是 Dr.GRPO 的核心，而该选项会重新叠加一层全局 whitening，与之矛盾。设置该选项会在启动阶段报校验错误。
:::

::: warning 支持的模型与 loss
Dr.GRPO 当前仅支持 dense 模型与 `--loss-type policy_loss`。启动阶段会拒绝 MoE 模型以及 SFT 或 custom loss，因为当前实现尚未覆盖它们的辅助 loss 与 normalization contract。
:::

## Relax 实现

Relax 保留 Megatron 对 CP 兼容的 per-token gradient normalization，并把 Dr.GRPO 实现为 optimizer-window scale。

令 $T$ 为 optimizer window 中有效 response token 数，$S$ 为组合后的 Actor algorithm loss 分子。Relax 计算：

$$
\alpha = \frac{T}{N B},
\qquad
\frac{\alpha S}{T} = \frac{S}{N B}
$$

实现复用了四个现有层次：

| 层次 | 职责 |
|---|---|
| Reward 后处理 | 在每个 prompt group 内中心化 reward，不除以组内标准差 |
| Optimizer-window preparation | 在最终 loss mask 就绪后统计全局 `(N, T)` 并计算 `alpha` |
| Data iterator metadata | 将不透明的 `__dr_grpo_window_scale__` 重放到同一 optimizer window 的每个 micro-batch |
| Megatron loss 路径 | 缩放组合后的 Actor loss，再复用 Megatron 的全局 `/T` gradient normalization |

组合 Actor loss 中的 policy-gradient、entropy 和 explicit KL 项使用同一个 scale。Micro-batch 边界只控制执行方式，不会改变 `N`、`T` 或最终分母。

### Context Parallelism

Dr.GRPO 会自动启用 `calculate_per_token_loss`。当 CP 大于 1 时，每个 CP rank 只贡献其本地 zig-zag shard 中的有效 response token。Megatron 会先汇总 CP-local token count，再执行 `/T`，因此 padding、CP degree 和 micro-batch 划分不会造成 token 重复计数。

Optimizer-window 的 `(N, T)` reduction 使用不包含 CP 的 DP group。所有 CP rank 获得相同的 `alpha`，全局 token normalizer 仍由 Megatron 负责汇总。

纯 fully-async 训练会被拒绝，因为其 streaming `__loss_scale__` 语义不同，而且 Dr.GRPO 无法从闭合的 optimizer window 准备 fixed-budget metadata。Hybrid 模式会先闭合并合并 optimizer window，因此可以与同步训练复用相同的 Dr.GRPO metadata 路径。

## 快速开始

从现有同步 GRPO 配置开始，例如 `scripts/training/text/run-qwen3-4B-8xgpu.sh`，替换其中的算法参数：

```bash
DR_GRPO_ARGS=(
   --advantage-estimator dr_grpo
   --rollout-max-response-len 8192
   --eps-clip 0.2
   --eps-clip-high 0.28
   --entropy-coef 0.0
   --kl-coef 0.0
)
```

在训练配置中显式保留 Megatron per-token 路径，使用 CP 时尤其如此：

```bash
PERF_ARGS=(
   --calculate-per-token-loss
   --context-parallel-size 1
   --qkv-format thd
   --use-dynamic-batch-size
   --max-tokens-per-gpu 10240
)
```

参数处理阶段也会由 `--advantage-estimator dr_grpo` 自动启用 `calculate_per_token_loss`，但在启动脚本中显式保留该参数可以清楚体现 CP 要求。

资源拓扑与 GRPO 相同：

```bash
--resource '{"actor": [1, 8], "rollout": [1, 8]}' \
--colocate
```

请根据实际环境调整 GPU 数、模型配置、数据路径和 token budget。

## 配置

| 参数 | 默认值 | Dr.GRPO 行为 |
|---|---|---|
| `--advantage-estimator dr_grpo` | `grpo` | 选择 Dr.GRPO 及其固定预算 reduction |
| `--loss-type policy_loss` | `policy_loss` | 必须使用；SFT 和 custom loss 会被拒绝 |
| `--num-experts` | `None` | 必须保持未设置；当前不支持 MoE 模型 |
| `--rollout-max-response-len` | `None` | 定义 `B`；应设置为生成时的 response token budget |
| `--n-samples-per-prompt` | `1` | 每个 reward centering group 中的 response 数 |
| `--global-batch-size` | `None` | 常规固定 batch schedule 下，每个 optimizer step 的 response 数 |
| `--calculate-per-token-loss` | 关闭 | Dr.GRPO 会自动启用；Megatron 在 CP 场景下也强制要求 |
| `--normalize-advantages` | 关闭 | 会被拒绝，因为 Dr.GRPO 在设计上去掉了 advantage 方差归一化 |
| `--disable-rewards-normalization` | 关闭 | 会被拒绝，因为 Dr.GRPO 必须执行组内中心化 |
| `--kl-coef` | `0.0` | 必须保持为零；Dr.GRPO 不添加 reward-side KL |
| `--use-kl-loss` | 关闭 | 添加 explicit KL loss 项 |
| `--kl-loss-coef` | `0.0` | Explicit KL loss 的系数 |
| `--entropy-coef` | `0.0` | Entropy bonus 的系数 |

需要显式 KL penalty 时，请组合使用 `--use-kl-loss` 和正数 `--kl-loss-coef`。Reward-side `--kl-coef` 会被拒绝，确保 GRPO 与 Dr.GRPO 对照只包含 Dr.GRPO 的两处 normalization 改动。

## 最佳实践

1. 显式设置 `--rollout-max-response-len`，并保证它与 Rollout 使用的 response budget 一致。它是 Dr.GRPO objective 的组成部分，不只是显存限制。
2. 对比 GRPO 与 Dr.GRPO 时，应观察 reward、evaluation accuracy 和 response length。两种算法的 Actor loss 与 gradient norm 使用不同分母，不能直接比较绝对值。
3. 不要设置 `--normalize-advantages`。Dr.GRPO 会拒绝该选项；如果需要 whitened advantage，请改用标准 GRPO estimator。
4. 自定义 reward 后处理会覆盖默认的组内中心化逻辑，因此必须自行提供符合预期的 Dr.GRPO advantage。
5. 使用 `--custom-pg-loss-reducer-function-path` 时，确保自定义 reducer 保留最终 fixed-budget scale 所期望的 token-sum 分子。

## 故障排除

### CP 训练要求 per-token loss

保持启用 `--calculate-per-token-loss`。Dr.GRPO 会自动设置该参数；当 `--context-parallel-size` 大于 1 时，Megatron 也会校验同一要求。

### Fixed denominator 不正确

检查 `--rollout-max-response-len` 是否与预期 response budget 一致，并确认自定义 rollout 后处理在 Actor 训练前已经生成最终 `loss_masks`。`T` 根据这些最终 mask 统计。

### 纯 fully-async 训练被拒绝

请使用同步 colocate 或 hybrid 模式。纯 fully-async streaming 尚未准备 fixed denominator 所需的 closed-window `(N, T)` metadata。

### MoE 或非 policy loss 被拒绝

请使用 dense 模型与 `--loss-type policy_loss`。MoE auxiliary loss 和 router expert-bias update 需要当前 Dr.GRPO 尚未实现的 empty-window contract，而 SFT 和 custom loss 不符合其 policy objective contract。

## 下一步

- [Dr.GRPO 200-Step 训练报告](./dr-grpo-training-report.md) — 复现 GRPO/Dr.GRPO 配对实验
- [算法参考](../examples/algorithms.md) — 对比 Dr.GRPO 与其他 policy-gradient 算法
- [配置说明](./configuration.md) — 查看 rollout、batch 和并行参数
- [PPO 训练](./ppo-training.md) — 使用 Actor-Critic 训练路径
