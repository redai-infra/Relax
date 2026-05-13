# Relax 接入 Dynamic Context Parallel 改造方案

> **状态**：Draft / 设计阶段
> **背景**：verl 在 PR [#5057](https://github.com/volcengine/verl/pull/5057)（commit `7e9a07c4`，2026-03-31）落地了 dynamic CP，能让每个 micro-batch 按当前最大 seq_len 自适应选 cp_size，对 RL 训练的长短样本混合场景明显友好。本文档给出 Relax 端的改造方案。
> **参考**：verl 实现详见 `/root/repos/verl/verl_cp.md` §6。

---

## 0. 前置事实（决定方案形态）

| 维度 | verl 现状 | Relax 现状 | 影响 |
|---|---|---|---|
| 引擎封装 | `MegatronEngine` 类 | 无类，直接 `mpu.*`（散布在 `cp_utils.py`/`ppo_utils.py`/`loss.py`/`data.py`/`actor.py`/`model.py`） | verl 的 "DP=1 伪装" 做不到方法重写，需另想 |
| Batch 类型 | TensorDict + `non_tensor_data` | `dict[str, list[Tensor]]`（`utils/types.py:194`） | `local_cp_size` 必须显式作为字段流转 |
| Megatron-LM | 上游 PR #3405 后的 dev 分支 | Bridge pin `2faedbf6...` 拉的是 **main 分支** `f4a071039`，**不含** `dynamic_context_parallel` 形参与 `get_dynamic_data_context_parallel_groups` API（已实测）；PR #3405 的 commit `cde56a469` 只在 dev 分支 | **必须先解决依赖**：要么 bump bridge pin → dev，要么打补丁把 #3405 + #2000 cherry-pick 进当前 main |
| 数据切分 | `preprocess_thd_engine` 一处 | `get_batch`（`data.py:106-335`）+ `cp_utils.py` 一组帮助函数 | 改面更广，但都在两个文件里 |
| Token budget | `max_token_len * sp_size` | `max_tokens_per_gpu * cp_size`（`data.py:533`，写死） | 必须改成 per-microbatch 的 effective cp_size |
| 损失尺度 | `loss * num_micro_batch`（一处） | `loss.py:1080-1087` 多处含 `mpu.get_*_world_size(with_context_parallel=True)` 因子 | 需引入 per-microbatch normalizer |
| VLM | thd + bridge VL 透传，align `tp*cp*2` | 同（`data.py:177-212`、`cp_utils.py:9-29`），同样 hardcode `tp*cp*2` | 同样需要 per-microbatch align |

---

## 1. 设计原则

**支点**：`local_cp_size` 通过 batch dict 一路下传到 `get_batch`、`forward_step`、loss、postprocess。所有 `mpu.get_context_parallel_*()` 调用改为接受可选的 `cp_group`/`cp_size` 参数，当外部传入则用之，否则退回全局 mpu（保持向后兼容）。

**两段式落地**：
- **Phase 1 MVP**：cp_size 在 DP 内**统一**（取 max），按 micro-batch 自适应。无 sub-DP 路由、不动数据 partition。一个开关上线即可享受"短样本不付 CP 通信代价"的收益。
- **Phase 2 verl 等价**：引入 sub-DP 路由 + 异构 cp_size。需要 `mpu.get_dynamic_data_context_parallel_groups`、`dynamic_cp_split_batch`、`dynamic_cp_merge_output`。

> 强烈建议 **Phase 1 先上线、跑稳、再做 Phase 2**。Phase 1 的总工作量 ≈ Phase 2 的 1/3，且 Phase 2 多出来的 sub-DP 路由对 RL 训练里的 advantage/log-prob 全局聚合有破坏性影响（verl 自己也留了 TODO 没完成）。

---

## 2. Phase 0：依赖准备

### 2.1 决定 Megatron-LM 来源（二选一）

| 选项 | 做法 | 收益 | 代价 |
|---|---|---|---|
| A. Bump Bridge pin | 把 `docker/Dockerfile:78` 的 `MEGATRON_BRIDGE_COMMIT` 升到含 mcore-dev 的版本，并把 `switch_mcore.sh` 切到 `dev` | 自带 dynamic CP + 持续维护 | dev 分支稳定性差，需要回归全套现有训练任务 |
| B. 现有 main + 补丁 | 在 `docker/patch/megatron/` 下加一个补丁，cherry-pick `cde56a469` (#3405) 和 `2d6e946ba` (#2000 Dynamic CP part 2) | 影响面小，可控 | 维护补丁 conflict 风险 |

**推荐 B**：cherry-pick 两个 commit 落到 `docker/patch/latest/megatron-dynamic-cp.patch`。改造期间任何升级 megatron 的人都看到 patch，风险显式化。

### 2.2 验证脚本

在 `relax/backends/megatron/initialize.py` 加运行时探测：

```python
import inspect
HAS_DYNAMIC_CP = "dynamic_context_parallel" in inspect.signature(mpu.initialize_model_parallel).parameters
```

供后续条件性启用。

---

## 3. Phase 1（MVP）改动清单

### 3.1 配置（args + validate）

```python
# relax/utils/arguments.py（新增 group "Dynamic CP"）
parser.add_argument("--dynamic-context-parallel", action="store_true",
                    help="Enable per-microbatch adaptive CP size (requires Megatron-LM PR #3405).")
parser.add_argument("--max-seqlen-per-dp-cp-rank", type=int, default=None,
                    help="Upper bound of tokens per DP×CP rank. Required when --dynamic-context-parallel.")
```

```python
# relax/backends/megatron/arguments.py: validate_args
if args.dynamic_context_parallel:
    assert HAS_DYNAMIC_CP, "Megatron-LM lacks PR #3405; bump pin or apply patch."
    assert args.max_seqlen_per_dp_cp_rank is not None
    assert args.context_parallel_size >= 1
    dp_size = compute_dp_size(...)
    assert dp_size * args.max_seqlen_per_dp_cp_rank >= args.max_response_len + args.max_prompt_len
```

### 3.2 mpu 初始化（`initialize.py:39-55`）

```python
extra = {}
if args.dynamic_context_parallel and HAS_DYNAMIC_CP:
    extra["dynamic_context_parallel"] = True
mpu.initialize_model_parallel(..., **extra)
```

### 3.3 Bridge config 反向欺骗（`model_provider.py:175-215`，**仅 bridge 模式**）

```python
if args.dynamic_context_parallel:
    overrides["max_seqlen_per_dp_cp_rank"] = args.max_seqlen_per_dp_cp_rank
    overrides["dynamic_context_parallel"] = False                # 同 verl 注释里的 "bad coupling" 绕道
    overrides["context_parallel_size"] = mpu.get_data_parallel_world_size()  # 让 bridge 误以为 cp=DP
```

raw 模式下不做这件事（直接用 `core_transformer_config_from_args(args)` 即可，但要在那里同样防止 `args.context_parallel_size` 被 transformer config 误读 —— 或者干脆 raw 模式 Phase 1 不支持，给出清晰报错）。

### 3.4 Per-microbatch CP 决策

新增 `relax/backends/megatron/dynamic_cp.py`：

```python
def decide_local_cp_size(samples_in_microbatch, max_seqlen_per_dp_cp_rank, dp_size):
    """同 verl: ceil(max_seqlen / cap), 向上取 2 的幂, clamp ≤ dp_size."""
    max_seq = max(len(s["input_ids"]) for s in samples_in_microbatch)
    n = math.ceil(max_seq / max_seqlen_per_dp_cp_rank)
    n = max(1, 1 << (n - 1).bit_length())
    return min(n, dp_size)


def gather_max_local_cp(local_cp: int, dp_group) -> int:
    """Phase 1 MVP: DP 间统一取 MAX。"""
    t = torch.tensor([local_cp], device="cuda")
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=dp_group)
    return int(t.item())
```

### 3.5 数据流（`data.py`）

#### a) 注入决策点（`get_data_iterator`，`data.py:449-573`）

在 `get_seqlen_balanced_partitions` 之后、构造 `DataIterator` 之前：

```python
if args.dynamic_context_parallel:
    per_mb_cp_size = []
    for mb_samples in partitioned_microbatches:
        local = decide_local_cp_size(mb_samples, args.max_seqlen_per_dp_cp_rank,
                                      mpu.get_data_parallel_world_size(with_context_parallel=False))
        per_mb_cp_size.append(gather_max_local_cp(local, mpu.get_data_parallel_group()))
else:
    per_mb_cp_size = [args.context_parallel_size] * len(partitioned_microbatches)

# 把 per_mb_cp_size 挂到 DataIterator 上
```

#### b) Token budget（`data.py:533`）

```python
# 改前: max_tokens_per_gpu * cp_size
# 改后:
budget_cp = args.context_parallel_size if not args.dynamic_context_parallel \
            else 1   # MVP: 不预知，就按最坏的 cp=1 打包，让 dynamic_cp 决策时再涨 cp
get_minimum_num_micro_batch_size(samples[start:end],
                                 args.max_tokens_per_gpu * budget_cp)
```

> 这里有个微妙取舍：dynamic CP 的 "动态" 是在 batch 已经分好后的 second pass。打包阶段用 `cp=1` 的 budget 意味着每 micro-batch 都按 "塞满单卡" 打 → 长样本最多触发 cp=8，能跑通；但短样本浪费空间。Phase 2 可以重排打包。

#### c) `get_batch`（`data.py:106-335`）

签名加可选 `local_cp_size: Optional[int] = None`：

```python
def get_batch(args, samples, ..., local_cp_size: Optional[int] = None):
    if local_cp_size is not None:
        cp_size = local_cp_size
        cp_group = mpu.get_dynamic_data_context_parallel_groups(group_size=local_cp_size)
        cp_rank = torch.distributed.get_rank(cp_group)
    else:
        cp_size = mpu.get_context_parallel_world_size()
        cp_group = mpu.get_context_parallel_group()
        cp_rank = mpu.get_context_parallel_rank()
    ...
    # 所有用 cp_size/cp_rank 的地方改用上面的本地变量
    # PackedSeqParams 上挂 cp_group, local_cp_size
    packed_seq_params = PackedSeqParams(
        ..., cp_group=cp_group, local_cp_size=local_cp_size,
    )
```

`slice_with_cp`（`cp_utils.py:210-251`）同样加 `cp_size, cp_rank` 参数。

#### d) `DataIterator.get_next`

返回的 dict 多带一个键 `"local_cp_size"`，由 `get_batch` 透传到 `forward_step`。

### 3.6 Forward & loss

#### a) `forward_step`（`model.py:222-303`、`399-489`）

```python
local_cp_size = batch.get("local_cp_size")  # None 表示静态 CP
batch_processed = get_batch(..., local_cp_size=local_cp_size)
output_tensor = model(...)
return output_tensor, partial(postprocess_fn, ..., local_cp_size=local_cp_size,
                              cp_group=batch_processed["cp_group"])
```

#### b) `cp_utils.py` 全部公共 helper

涉及函数：`get_logits_and_tokens_offset_with_cp`、`all_gather_with_cp`、`get_sum_of_sample_mean`、`slice_log_prob_with_cp`、`maybe_padded_total_lengths`。

签名加 `cp_size: Optional[int] = None, cp_group: Optional[ProcessGroup] = None`。内部 `mpu.get_context_parallel_*()` 改为：

```python
cp_size = cp_size or mpu.get_context_parallel_world_size()
cp_group = cp_group or mpu.get_context_parallel_group()
cp_rank = torch.distributed.get_rank(cp_group) if cp_group else mpu.get_context_parallel_rank()
```

调用方（`loss.py`、`ppo_utils.py:298-336/402-423/463-515`、`actor.py:1047`、`advantages.py:159`）补传两个参数。

#### c) `loss.py:1080-1087` 损失尺度

```python
# 改前:
# loss = loss * num_microbatches / global_batch_size * mpu.get_data_parallel_world_size(with_context_parallel=True)
# loss = loss * mpu.get_context_parallel_world_size()

# 改后:
effective_cp = local_cp_size if local_cp_size is not None else mpu.get_context_parallel_world_size()
# DP 维度 Phase 1 MVP 仍按全局 DP（cp 同 DP 内统一），不变
dp_x_cp = mpu.get_data_parallel_world_size(with_context_parallel=False) * effective_cp
loss = loss * num_microbatches / global_batch_size * dp_x_cp        # sample-mean
loss = loss * effective_cp                                          # per-token
```

> ⚠️ **Phase 1 关键约束**：MVP 同一 micro-batch 内所有 DP rank 用相同 cp，所以 `mpu.get_data_parallel_world_size(with_context_parallel=True)` 在 Phase 1 ≡ `dp * effective_cp`，等价。Phase 2 引入异构后才会破。

### 3.7 VLM（`data.py:177-212`，`cp_utils.py:9-29`）

把硬编码的 `tp * cp * 2` 替换为 `tp * effective_cp * 2`，其中 `effective_cp = local_cp_size or args.context_parallel_size`。`vlm_packed_seq_params` 同样要带 `cp_group`，让 Bridge 内部走对的 ring-CP 通信。

> 风险：Bridge 自己内部如何调度 dynamic CP 取决于 PR #3405 + Bridge 自身。建议 **Phase 1 先在纯文本上线，VLM + dynamic CP 单独作为 Phase 1.5 验证**。

### 3.8 不动的部分

- `actor.py` 里的 `data_system_client` 数据接收：rollout 阶段不感知 CP，CP 只是 trainer 内部事。
- `RolloutBatch` 类型定义：保持 `dict[str, list[Tensor]]`，新键加在 micro-batch 那一层。
- 所有 raw 模式相关代码：Phase 1 报错 "raw mode 暂不支持 dynamic CP"。

---

## 4. Phase 2（verl 完整等价）追加改动

待 Phase 1 稳定后再做。

### 4.1 引入 sub-DP 路由

新增 `dynamic_cp_split_batch` —— 在 `data.py` 的 `get_data_iterator` 末尾对每个 micro-batch：

```python
if local_cp_size < dp_size:
    local_dp_rank = dp_rank // local_cp_size
    local_dp_size = dp_size // local_cp_size
    # 把 partitioned_microbatches[i] 进一步切成 local_dp_size 份
    # 每个 sub-DP 拿自己那份
```

需要的 mpu 新 API：`get_dynamic_data_context_parallel_groups(group_size=local_cp_size)`（PR #3405 已提供）。

### 4.2 引入 `dynamic_cp_merge_output`

postprocess 阶段对需要跨 sub-DP all_gather 的输出（log_probs、entropy、advantages 等），同 verl 用 `all_gather_object` 在 `dp_group` 内按 stride 重组。

### 4.3 DP "伪装"

由于 Relax 没有 `engine.get_data_parallel_size()` 方法，建议反过来：在 `compute_dp_size` 旁加一个 `get_logical_dp_size(args)`，dynamic CP 时返回 1。所有 dynamic-batching/loss 尺度计算改用 `get_logical_dp_size`，物理 DP 通信仍用 `mpu`。

### 4.4 损失跨 sub-DP 聚合

verl 在这里留了 TODO，Relax 不能照抄。建议至少在 `loss.py` 增加一次跨 sub-DP 的 weighted average（按 sub-group 的样本数权重），否则 advantage normalize / KL 估计会 biased。

### 4.5 数据 scheduler 长度感知

verl post-merge TODO 里要做的事 —— 在 `get_seqlen_balanced_partitions` 之前按长度排序并按桶分箱，同桶 micro-batch 用相同 cp。能让 DP 内 `MAX(local_cp)` 趋近 `MEAN(local_cp)`。

---

## 5. 测试方案

| 层级 | 测试 | 通过标准 |
|---|---|---|
| 单元 | `decide_local_cp_size` / `gather_max_local_cp` | 输入构造的 batch → 期望 cp_size |
| 单元 | `get_batch(local_cp_size=2)` vs `get_batch()` 在 cp=2 静态时 | bit-exact |
| 集成 | 8GPU 单机 SFT，TP=4 PP=1 CP=1，开/关 dynamic | loss 曲线在数值容差内对齐 |
| 集成 | 8GPU GRPO，长样本 + 短样本混合，dynamic vs 静态 cp=4 | reward 曲线一致，throughput dynamic 更高 |
| 回归 | 现有所有 e2e 训练脚本（`scripts/training/`），关闭 dynamic CP | 必须 bit-exact |
| VLM（Phase 1.5） | Qwen3-VL 8GPU | loss 对齐 |

---

## 6. 风险与决策点

| # | 风险 | 缓解 |
|---|---|---|
| 1 | Megatron pin 升级或 patch 维护负担 | 倾向 patch 路线，写在 `docker/patch/latest/`，CI 自动 apply |
| 2 | Phase 1 损失尺度公式在 cp 同 DP 内统一时等价，但需要严谨证明 | 在 PR 描述里写出代数推导 + 单元测试覆盖两种 mode |
| 3 | VLM 的 Bridge 内部 CP 是否随 `local_cp_size` 自动适应未知 | Phase 1.5 单独验证；最坏情况 VLM 不支持 dynamic CP（同 verl bshd 的处理） |
| 4 | raw 模式 Phase 1 不支持 | 显式报错引导用户切到 bridge 模式 |
| 5 | `cp_utils.py` 全部 helper 改签名是 breaking 修改 | 给所有参数加 `Optional` 默认值，保证既有调用方一行不动 |
| 6 | Phase 2 的 sub-DP 路由对 RL 全局统计（advantage normalize）是破坏性的 | Phase 2 启动前，先盘清 `relax/components/advantages.py` 和 `relax/utils/training/ppo_utils.py` 里所有跨 DP 聚合点，逐一决定是否需要补 cross-sub-DP 聚合 |

---

## 7. 工作量预估（人日）

| 阶段 | 模块 | 估算 |
|---|---|---|
| Phase 0 | Megatron patch + 探测 | 1-2 |
| Phase 1 | 配置 + mpu/bridge + dynamic_cp.py | 2 |
| Phase 1 | data.py / cp_utils.py / forward_step / loss.py 改造 | 4-5 |
| Phase 1 | 单元 + 集成测试 + 回归 | 3-4 |
| **Phase 1 小计** | | **~10-13 人日** |
| Phase 1.5 | VLM 验证 + 修复 | 3-5 |
| Phase 2 | sub-DP 路由 + merge + DP 伪装 + 损失聚合 | 8-12 |
| Phase 2 | 测试 + 回归 | 5 |
| **总计** | | **~25-35 人日** |

---

## 8. 落地起始的最小补丁清单（P1 第一周）

1. `docker/patch/latest/megatron-dynamic-cp.patch` —— cherry-pick #3405 + #2000
2. `relax/utils/arguments.py` —— 加 2 个 flag
3. `relax/backends/megatron/arguments.py` —— validate
4. `relax/backends/megatron/initialize.py` —— mpu init + `HAS_DYNAMIC_CP`
5. `relax/backends/megatron/model_provider.py` —— bridge 反向欺骗
6. `relax/backends/megatron/dynamic_cp.py`（新文件） —— `decide_local_cp_size` / `gather_max_local_cp`

走通这 6 处 → 已经可以 `dynamic_context_parallel=True` 跑起来（虽然 cp_size 还没真在 batch 里变化），后续再分别接通 data → forward → loss 三段。

---

## 9. 参考

- verl PR #5057：<https://github.com/volcengine/verl/pull/5057>
- verl 实现详解：`/root/repos/verl/verl_cp.md` §6（来源 / mpu+bridge 双重欺骗 / split+merge / 限制汇总）
- Megatron-LM PR #3405：<https://github.com/NVIDIA/Megatron-LM/pull/3405>（dev 分支已合，main 未合）
- Megatron-LM PR #2000：Dynamic CP part 2（提供 `get_dynamic_data_context_parallel_groups`）
