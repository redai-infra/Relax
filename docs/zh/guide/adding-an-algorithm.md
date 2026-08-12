# 接入一个新算法

Relax 的算法通过 `relax/algorithms/` 下的注册表接入。一个算法名不再散落在各处的 `if/elif` 里——它由一条 `AlgorithmSpec` 描述，各个阶段按需查表。

## 注册表的结构

```
relax/algorithms/
├── spec.py        AlgorithmSpec 定义 + ALGORITHM_SPECS 注册表
├── rewards.py     reward 归一化策略 + REWARD_NORMALIZERS
├── advantages.py  advantage 估计器 + ADVANTAGE_FNS
└── policy.py      policy loss 适配器 + POLICY_LOSS_FNS
```

三条硬约束：

1. **`relax/algorithms/` 下禁止顶层 import 重依赖**——不能有 `megatron`、`ray`、`transfer_queue`、`tensordict`、`relax.components`、`relax.backends`。注册表会被参数解析和两个 worker 进程 import；一个重依赖会把整个训练栈拖进 `--help` 和只有 CPU 的 CI。确实需要时在函数内 import。
2. **spec 的字段存字符串标识符，不存函数引用**。advantage 计算跑在 Ray Serve 的 `Advantages` 进程，policy loss 跑在 Megatron worker 进程，两者 import 的模块子集不同。跨进程只传算法名，各进程本地查表。
3. **`ALGOS` 角色表不用手改**。它从注册表自动派生，新算法自动获得标准 RL 角色集合。

## 接入一个新算法要改多少

先说实话：**不是「加一条 dict entry」就完事**。

| 情况 | 要改的文件 |
|---|---|
| 复用现成的 reward 归一化 / advantage / policy loss，只是组合方式不同 | 1 个（`spec.py`） |
| 需要一种新的数学（如新的 advantage 公式） | 2–3 个（`spec.py` + 对应的实现模块） |
| 还需要新的命令行参数（如 GDPO 的 `--gdpo-reward-keys`） | 4–6 个（上述 + `arguments.py` 的参数声明与校验 + 示例 + 文档） |

注册表消除的是「同一个算法名散落在 6 处 if/elif」，不是「新增算法零成本」。GDPO 走的是最后一档。

`ALGOS` 角色表是唯一真正做到零改动的部分——它从注册表自动派生。

## 步骤

### 1. 加一条 spec

编辑 `relax/algorithms/spec.py` 的 `ALGORITHM_SPECS`：

```python
"my_algo": AlgorithmSpec(
    name="my_algo",
    reward_normalizer="group_mean_std",   # 复用现成的，或见第 2 步
    advantage_fn="grpo_broadcast",
    policy_loss_fn="ppo_clip",
),
```

如果新算法在某个阶段与已有算法完全一致，直接复用那个标识符即可——例如 GRPO / GSPO / SAPO / CISPO 在 advantage 层完全等价，四者共享 `"grpo_broadcast"`。

可用的能力字段：

| 字段 | 作用 |
|------|------|
| `kl_level` | `"token"` 或 `"sequence"`（GSPO 用序列级） |
| `needs_full_log_probs` | loss 是否需要 CP all-gather 后的完整 log probs |
| `advantage_normalization` | `--normalize-advantages` 的归一化方式：`"whiten"`（掩码白化）或 `"token_global"`（REINFORCE++ 的全局 token 级归一化，同时切换掩码安全的 loss reducer） |
| `needs_critic` | 是否需要 critic 服务，驱动 `args.use_critic` |
| `requires_normalize_advantages` | 强制要求 `--normalize-advantages` |
| `forbids_normalize_advantages` | 禁止 `--normalize-advantages`（算法刻意保留了 advantage 的尺度时） |
| `requires_rewards_normalization` | 禁止 `--disable-rewards-normalization` |
| `min_group_size` | `--n-samples-per-prompt` 的下限 |
| `forbids_reward_side_kl` | 要求 `--kl-coef 0`（reward 侧 KL 项无处可放；`--use-kl-loss` 不受影响） |
| `requires_global_token_loss` | 强制要求 `--calculate-per-token-loss`（否则按样本取 token 均值，会按 `1 / response_length` 重新加权） |
| `requires_on_policy_updates` | 一次性拒绝五项：`--fully-async` / `--hybrid`、`--max-staleness != 0`、`--num-steps-per-rollout != 1`、`rollout_batch_size * n_samples != global_batch_size`、`--partial-rollout` / `--use-dynamic-global-batch-size`。适用于没有重要性比值修正的目标函数 |
| `supports_fully_async` | 设为 `False` 可拒绝 `--fully-async`（该模式下 advantage 由单副本服务按切片计算，无 DP 通信域） |
| `allows_reward_post_process_hooks` | 设为 `False` 可同时拦住 `--custom-reward-post-process-path` 与 `--agentic-custom-advantage-path`——这两个钩子都会在归一化器之前从 `post_process_rewards` 返回，静默跳过本算法的奖励阶段 |
| `uses_reward_components` | 算法消费多个具名奖励分量而非单个标量，驱动 `--gdpo-reward-keys` 校验 |

表里除 `kl_level`、`needs_full_log_probs` 和 `advantage_normalization` 之外的字段，都由 `relax/utils/arguments.py` 的四个 `validate_*` 函数统一消费，**声明即生效**，不需要再去 `arguments.py` 加 `if`。（拆成四个是因为参数校验本身有推导顺序——例如 `--kl-coef` 必须在「检查 `--ref-load` 是否存在」之前判掉，one-update 等式必须在 `global_batch_size` 定稿之后判——与算法特殊性无关。）那三个字段是在 `relax/backends/megatron/loss.py` 里读的：新增一个前所未有的取值需要在那里加分支，复用已有取值则不用。

### 2. 需要新公式时，写纯函数并登记

只有当新算法在某个阶段的数学与现有算法都不同时才需要这一步。

**Reward 归一化**（`relax/algorithms/rewards.py`），签名固定为 `fn(args, samples, raw_rewards) -> list[float]`：

```python
def normalize_my_strategy(args, samples, raw_rewards):
    positions_by_group = group_positions(samples, args.n_samples_per_prompt)
    ...
    return normalized  # 每个 sample 一个标量

REWARD_NORMALIZERS["my_strategy"] = normalize_my_strategy
```

产出必须是**每个 sample 一个标量**。这条约束让 TransferQueue 的 schema 保持不变——多奖励算法（如 GDPO）也是在这一层把各分量收敛成一个标量的。

**Advantage 估计器**（`relax/algorithms/advantages.py`），签名 `fn(args, *, rewards, kl, loss_masks, response_lengths, total_lengths, values) -> (advantages, returns)`，两者都是 `list[Tensor]`：

```python
def advantage_my_algo(args, *, rewards, kl, **_unused):
    ...
    return advantages, returns

ADVANTAGE_FNS["my_algo"] = advantage_my_algo
```

**Policy loss**（`relax/algorithms/policy.py`），签名 `fn(args, *, log_probs, ppo_kl, advantages) -> (pg_loss, pg_clipfrac)`。底层算子签名不一致，适配器负责统一。

### 3. 写单测

`tests/algorithms/` 下的测试不依赖 megatron / ray / transfer_queue，只要 torch 就能跑：

```bash
pytest tests/algorithms/ -v
```

至少覆盖：

- 注册与分发：算法名在 `ALGORITHM_SPECS` 里；能力字段与预期一致；未注册名报错。
- 数值：手算一个小例子做对照，别用全零或全相同的 reward——那种输入下任何公式都输出 0，测不出东西。
- 退化场景：组内 reward 全相同、`n_samples_per_prompt` 取边界值、缺字段、非数值输入。
- **改动已有算法时**：把旧实现冻结进测试文件当参照，逐位对拍（`view(torch.int32).equal`），不要用 `allclose`——它的默认容差足以吞掉无偏/有偏标准差的差异。`tests/algorithms/test_reward_normalizers.py` 是现成范例。

### 4. 加示例与文档

- `examples/<algo>/`：启动脚本，必要时附自定义 reward 函数。
- `docs/{zh,en}/examples/algorithms.md`：算法原理、关键参数表、快速开始，以及**已知偏差**——实现与论文不一致的地方要写出来，不要留给使用者去发现。

## 参数

新增算法专用参数时改 `relax/utils/arguments.py` 的 `add_algo_arguments`。`--advantage-estimator` 的 `choices` 由 `list_algorithm_names()` 生成，注册即可用，不需要手动维护名单。

跨参数的校验写进 `validate_algorithm_args`，并优先用 spec 字段表达而不是比较算法名——后者正是这套注册表要消除的东西。

## 参考

- [算法参考](../examples/algorithms.md)
- GDPO 是最近一个走完整个流程的例子，可以对照 `relax/algorithms/` 与 `examples/gdpo/` 阅读。
