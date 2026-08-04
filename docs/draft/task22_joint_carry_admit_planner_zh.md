# Task 22 Joint Carry-Admit Planner 设计

## 目标

Joint Carry-Admit Planner 在同一个决策点处理两类工作：

- 已经开始生成、可能跨权重版本继续运行的 carried work；
- 为 previous debt 和 current partition 新准入的 fresh work。

Planner 的目标不是降低某一个 gate，也不是最大化 KV 保留量，而是最小化一个 Actor
权重发布边界的总成本：

```text
boundary_cost
= publication_gate_wait
+ next_partition_fill_wait
+ abort_replay_cost
+ carried_tail_delay
+ strict_refresh_cost
```

第一版不修改 SGLang 内核，不降低全局 permit capacity，不改变每 step 发布一次权重，
不按 step 编号、奇偶、prompt、reward 或离线快慢模板分支。

## 实验结论约束

### Gate 降低不代表流水线加速

Clean Sync-Intent V2 canary 显示，named gate wait 可以下降，但等待会迁移到 train wait、
partition fill 或下一轮 debt。去掉初始化异常 step 后，`step 3–12` 的稳态
token-normalized throughput 约为 `-0.84%`。该结论来自
`../../../task22_probe_artifacts/clean_sync_intent_v2_20260803/`，不能与更早的
work-conserving Admission A/B 合并成同一实验。

因此 Planner 的主指标必须是完整 paired-cycle wall 和 token-normalized throughput，
不能只优化 publication gate。

### 静态水位会在重 debt 时过量投放

Work-conserving old-debt-first Admission A/B 使用固定 candidate window 和 minimum
inflight。old debt 已有 7–8 groups 时仍补 4–5 fresh，physical envelope 被推到
15–16 groups，最终 wall 和 samples/s 退化约 3%。该结论来自
`../../../task22_probe_artifacts/admission_matched_bb03d9a420e6_20260801_165830/`。

因此 fresh budget 必须扣除 carried/debt 的剩余工作，不能只按 group 数和固定水位计算。

### 低容量 lane 会损伤 decode aggregate throughput

长上下文 profile 中，双 engine 满载窗口的主要瓶颈是 paged attention，约占 GPU kernel
时间的 77.8%。该结论来自
`../../../task22_probe_artifacts/deep_profile_longctx_20260801_205418/011_real_long_context_deep_profile_decision_zh.md`。
历史 current-floor 与 hard-lane 模拟也显示，主动压低全局 active request 水位会损伤
decode aggregate throughput。

因此 Planner 只改变工作组合，不建立低容量 lane，也不降低已有全局并发上限。

### Dynamic debt quorum 是可保留的正向原语

离线 request-level trace 显示，previous debt 不必永久绑定到 adoption 顺序中的前 D 个
group。让全部 eligible carry 竞争 debt quorum，按 useful completion 顺序将前 D 个归入
previous partition，已有离线模拟给出的 pooled 关键路径回收上界约 `+2.97%`；加入
soft floor 和 2 秒额外等待上限后约 `+2.65%`。这两个数字是待 J0b 使用原始 ledger 和
固定脚本复现的离线上界估计，不能直接作为线上收益声明。现有计算记录在
`../../../task22_probe_artifacts/current_floor_lane_simulation_20260803/FEASIBILITY_ANALYSIS_ZH.md`。

因此 Planner 使用动态 quorum，而不是固定前 D 个 carried group 为 old debt。

### A3 的收益空间有限

已有估计表明 replay-only 可攻击空间约占完整 wall 的 2%–3%。A3 若保留错误长尾、
冻结 engine placement 或制造 strict cliff，很容易抵消全部收益。

因此 carry 数量本身不是成功指标。Planner 必须报告 carry completion、剩余工作、
strict fallback 和 partition fill 的联合结果。

## 能力边界

当前 SGLang publication 支持全局 `in_place` 或全局 `abort`，没有经过验证的逐 group
selective abort 接口。原 asyncio task 还绑定原 engine placement。

所以分三个阶段：

| 阶段 | 能力 | 不做的事 |
|---|---|---|
| J0a Primitive | 纯 planner、输入验证和性质测试 | 不读取 ledger，不改变 runtime |
| J0b Shadow replay | 读取两条既有 ledger 并输出影子计划 | 不改变 admission 或 carry |
| J1 Shared budget | 联合计算 work budget、fresh admit、dynamic debt quorum | 不逐 group 淘汰 carry，不迁移 engine |
| J2 Selective carry | 按 carry value 选择保留/中止任务 | 需要逐请求 abort 和可验证的 task cleanup |

J1 是当前可安全落地的最小版本。J2 不能通过取消本地 asyncio task 冒充实现，否则可能
留下后端 orphan request。

## 状态模型

### Group 描述

每个在途 group 形成一个只包含运行状态的描述：

```python
@dataclass(frozen=True)
class JointWorkGroup:
    group_id: str
    origin: Literal[
        "carry_debt_eligible",
        "carry_current",
        "fresh_current",
        "strict_retry",
    ]
    estimated_remaining_tokens: int
    response_prefix_tokens: int
    protected: bool
    version_gap: int
    engine_id: str | None
```

`group_id` 只能是稳定的内部 ID，不记录 prompt 或 response 文本。

### Planner 输入

```python
@dataclass(frozen=True)
class JointCarryAdmitInput:
    phase: str | None
    debt_target: int
    debt_committed: int
    current_target: int
    current_committed: int
    lifecycle_progress: int
    resident_group_ids: tuple[str, ...]
    carry_groups: tuple[JointWorkGroup, ...]
    debt_eligible_group_ids: tuple[str, ...]
    carry_current_group_ids: tuple[str, ...]
    fresh_current_group_ids: tuple[str, ...]
    rollout_batch_size: int
    max_response_length: int
    strict_retry_pending: bool
    final_backfill: bool
```

### Planner 输出

```python
@dataclass(frozen=True)
class JointCarryAdmitPlan:
    debt_remaining: int
    current_deficit: int
    carry_work_equivalents: float
    resident_cap: int
    current_reserve: int
    fresh_admit_groups: int
    work_overcommit_equivalents: float
    reason: str
```

Planner 从稳定 group ID 集合派生计数，并验证 debt-eligible 与 carry-current 都是 carry
子集、二者互斥、fresh-current 与 carry 互斥，且所有在途 group 都属于 resident 集合。
J1 不输出可执行的
`carry_budget_groups`，因为当前没有安全的逐 group abort；J2 才增加 selective-carry
预算。

## 剩余工作估计

继续使用在线条件生存估计，不读取离线 run 的 step 模板：

```text
residual_samples
= completed_length - current_length
  for completed_length > current_length

estimated_remaining
= P75(residual_samples)
```

历史条件样本不足 4 个时：

```text
estimated_remaining
= max_response_length - current_length
```

GRPO group 等待最慢 sibling：

```text
group_remaining
= max(unfinished sibling estimated_remaining)
```

`COMPLETED` 和 `TRUNCATED` sibling 的剩余量为 0。

换算为满长 group 等价量：

```text
work_equivalent(group)
= clamp(group_remaining / max_response_length, 0, 1)
```

它不是精确运行时间预测，只用于避免把 8 个长尾 carry 和 8 个近完成 carry视为相同容量。

## 联合预算

### Resident 上限

保留当前有界上限：

```text
resident_cap
= rollout_batch_size + max(1, rollout_batch_size // 4)
```

batch 为 8 时，resident cap 为 10。

### Soft work cap

正常 work cap 保持一个 rollout batch：

```text
normal_work_cap = rollout_batch_size

carry_work
= sum(work_equivalent(group) for group in carry_groups)

fresh_inflight_work
= fresh_current_inflight

fresh_by_work
= floor(max(
    normal_work_cap
    - carry_work
    - fresh_inflight_work,
    0,
))
```

### Current progress reserve

为了避免未归属 current 的 carried groups 暂时占满逻辑窗口后 current partition 完全没有
useful inflight，保留最多 2 个 current reserve：

```text
reserve_cap
= max(1, rollout_batch_size // 4)

current_reserve
= min(
    reserve_cap,
    current_deficit,
    free_resident_slots,
)
```

Reserve 只在以下条件同时成立时启用：

```text
carry_current_inflight + fresh_current_inflight == 0
phase permits fresh
not strict_retry_pending
not final_backfill
```

最终准入：

```text
fresh_admit
= min(
    current_deficit_after_inflight,
    free_resident_slots,
    max(fresh_by_work, current_reserve),
)
```

Reserve 是显式、可观测的 soft-cap overcommit，而不是 hard work-cap 内的免费容量：

```text
planned_work <= normal_work_cap + current_reserve
work_overcommit_equivalents
= max(planned_work - normal_work_cap, 0)
```

它每次 completion 后重算，不能突破 resident cap；当前 batch 8 时最多显式 overcommit
2 个新的满长等价 group。`work_overcommit_equivalents` 报告总 overcommit 真值，因此
若 Planner 介入前 carried work 已超过 soft cap，它可以大于本次 `current_reserve`。
实验必须单独报告该值，不能声称所有计划都不超过 work cap。

## Dynamic debt quorum

### Adoption

所有从上一 physical rollout 接管、且有 useful completion 潜力的 carry 先标记为：

```text
carry_debt_eligible
```

不再按 adoption 列表顺序将前 D 个永久标为 old debt。

### Completion 归属

每个 group 完成后，在同一协程临界段原子决定归属：

```python
if not useful_completion:
    assign_lifecycle_only()
elif debt_committed < debt_target:
    assign_previous_partition()
    debt_committed += 1
elif current_committed < current_target:
    assign_current_partition()
    current_committed += 1
else:
    assign_surplus()
```

`useful_completion` 要求整个 group 都为 `COMPLETED/TRUNCATED`。`ABORTED` 只增加
physical-rollout lifecycle progress，不能满足 debt quorum，也不能减少 current deficit。

这样 previous partition 的关闭时间由 eligible carry 中第 D 个 useful completion 决定，
而不是固定前 D 个 group 中最慢者决定。

### Strict fallback

Mixed group 在 strict retry 完成前不算 useful completion。一次 strict fallback 出现后，
当前 physical rollout 的 fresh admission 保持 fail-closed 为 0，避免 strict rebuild 与
fresh reserve 同时争抢 KV 和 scheduler。

## Planner 状态机

```text
OVERLAP
  debt quorum 优先
  joint planner 控制 fresh
      |
      | sync intent -> quiesce
      v
QUIESCE
  fresh_admit = 0
  等待已有任务到安全点
      |
      | publication in_place
      v
CARRY
  持久化合法 task
  下一 physical rollout adopt 后重算 planner
      |
      | strict publication
      v
STRICT_REBUILD
  fresh_admit = 0
  mixed group retry
```

Planner 在以下事件后重算：

- carry adoption；
- 任一 group completion；
- dynamic filter drop；
- strict fallback；
- sync phase 变化；
- fresh fetch 返回；
- resident 数变化。

## 不变量

### 所有权

1. 一个 task 只能由一个 physical rollout 持有。
2. 一个 useful group 只能归 previous、current 或 surplus 之一。
3. train data 与 buffer Sample identity 必须不相交。
4. marker 只保存跨轮事实，不能保存 physical-rollout 局部计数。

### 计数

1. 所有 group ID 非空且在各自集合内唯一。
2. debt-eligible 与 carry-current 是 carry 的互斥子集。
3. carry 与 fresh-current 互斥，且都属于 resident。
4. `resident_groups <= resident_cap`。
5. `ABORTED` 不增加 debt/current committed。
6. 每次 completion 的归属和 counter 更新不可分割。
7. `debt_eligible_inflight + carry_current_inflight <= carry_groups`。
8. `carry_groups + fresh_current_inflight <= resident_groups`。

### 分区

1. previous partition 仅在 `debt_committed == debt_target` 时发送 `is_last=True`。
2. current partition 仅在 useful accepted 达到 current target 时关闭。
3. lifecycle progress 只允许 physical rollout 收尾，不能作为 partition complete 的证据。
4. final backfill 只补 debt，不启用 current reserve。

### 降级

| 条件 | 行为 |
|---|---|
| 无 carry | 按 current deficit 正常准入 |
| 剩余量历史不足 | 使用安全上界，倾向少准入 |
| `quiesce` / `weight_sync` | fresh 为 0 |
| strict retry pending | fresh 为 0 |
| final backfill | 只处理 debt |
| snapshot stale/mismatch | 禁止新 fresh，并由控制面 fail-closed |
| 重复 identity、双重所有权、counter underflow | 立即使 run invalid |
| Planner 关闭 | 保留当前 Admission+A3 路径 |

Planner 异常时不能静默回退到 eager 16-group window，因为该行为已被 Admission V2 证明
会在重 debt 阶段过量投放。

## J1 决策伪代码

```python
def plan_joint_carry_admission(state):
    validate_counters_and_identity(state)

    if state.final_backfill:
        return no_fresh("final_backfill")
    if state.strict_retry_pending:
        return no_fresh("strict_retry")
    if state.phase in {"quiesce", "weight_sync"}:
        return no_fresh("sync_phase")

    debt_remaining = max(state.debt_target - state.debt_committed, 0)
    current_deficit = max(
        state.current_target
        - state.current_committed
        - state.carry_current_inflight
        - state.fresh_current_inflight,
        0,
    )
    free_slots = max(resident_cap - state.resident_groups, 0)

    carry_work = sum(work_equivalent(group) for group in state.carry_groups)
    fresh_by_work = floor(max(
        state.rollout_batch_size
        - carry_work
        - state.fresh_current_inflight,
        0,
    ))

    reserve = 0
    if (
        state.carry_current_inflight + state.fresh_current_inflight == 0
        and current_deficit > 0
    ):
        reserve = min(reserve_cap, current_deficit, free_slots)

    fresh_admit = min(
        current_deficit,
        free_slots,
        max(fresh_by_work, reserve),
    )
    return plan(fresh_admit, carry_work, debt_remaining)
```

Debt fetch 与 current fresh fetch 分开执行。若 debt quorum 尚未满足且没有足够
debt-eligible inflight，先补 debt；current fresh 只能使用剩余 resident/work budget。

## 指标

每次 planner 重算记录一条无文本事件：

```text
rollout/joint/planner_recomputes
rollout/joint/debt_target
rollout/joint/debt_committed
rollout/joint/debt_eligible_inflight
rollout/joint/current_target
rollout/joint/current_committed
rollout/joint/carry_current_inflight
rollout/joint/fresh_current_inflight
rollout/joint/carry_groups
rollout/joint/carry_work_equivalents
rollout/joint/free_resident_slots
rollout/joint/current_reserve
rollout/joint/fresh_admit_groups
rollout/joint/work_overcommit_equivalents
rollout/joint/strict_blocked_decisions
rollout/joint/fallback_reason
```

Boundary 级别必须继续报告：

```text
publication_gate_wait
next_partition_fill_wait
actor_train_wait
strict_fallback_groups
strict_fallback_prefix_tokens
carry_completion_ratio
paired_cycle_wall
token_normalized_throughput
```

如果 gate 下降但 train wait、partition fill 或 paired-cycle wall 上升，判定为成本搬移，
不能宣称优化成立。

## 实施顺序

### J0a Planner primitive

新增纯函数 planner、输入集合约束、预算性质测试和 fail-closed 校验。它没有 production
caller，不改变 runtime。

### J0b Shadow replay

新增纯函数 planner 和 replay 工具，读取现有 request-level ledger，输出 shadow plan。
不改变线上 admission/carry。

必须验证：

- old debt 7–8 时不再固定补 4–5 fresh；
- resident 不超过 10；
- 不产生 15–16 group 重 envelope；
- dynamic quorum 的 previous close 不慢于固定前 D；
- 每个 group 归属唯一；
- 两条已有 trace 上都不降低 useful current progress。

### J1 Shared budget

使用单独环境开关：

```text
RELAX_JOINT_CARRY_ADMIT_PLANNER=1
```

唯一行为变化：

- dynamic debt quorum；
- joint work/resident budget；
- current reserve；
- 每个 completion 后重新规划。

关闭旧的独立 progress hedge 决策，避免两个控制器同时写 fresh budget。

### J2 Selective carry

只有具备以下能力后启用：

- 后端逐 request/group abort；
- abort 后确认 task、session 和 engine request 都已清理；
- per-engine remaining-work snapshot；
- version-aware placement；
- orphan request 检测为 0。

J2 才使用 carry value：

```text
carry_value
= replay_tokens_saved
+ partition_criticality
- estimated_remaining_tokens
- kv_pressure
- engine_imbalance
- strict_refresh_risk
```

J1 不使用这个评分淘汰 carry。

## 测试

### Planner 纯函数

1. 无 carry、current deficit 8 时 admit 8。
2. 8 个满工作量 carry 时，普通 refill 为 0，current reserve 最多 2。
3. 8 个近完成 carry 时仍受 resident cap 限制。
4. 7–8 个 debt-eligible carry 时不固定补 4–5 fresh。
5. completion 释放 slot 后增量 refill。
6. `quiesce`、`weight_sync`、strict retry、final backfill 返回 0 fresh。
7. 非有限或越界 remaining estimate fail-closed。
8. 输出不超过 current deficit 和 resident cap。
9. 本次新增 work overcommit 不超过 current reserve；总 overcommit 报告实际真值。
10. carry work 与 fresh inflight 不重复计费。

### Dynamic quorum

1. debt target 3、eligible carry 5，前三个 useful completion 归 previous。
2. 第四、第五个归 current，不依赖 adoption 顺序。
3. ABORTED 不增加 debt/current committed。
4. mixed group 在 strict retry 完成前不计 useful completion。
5. previous `is_last` 只在第三个 useful completion 后发送。
6. 重复 completion 和重复归属立即失败。

### Rollout 集成

1. carry/adopt 保持原 Task 与 Sample identity。
2. 每个 completion 后只补释放出的 slots。
3. strict fallback 后 fresh admission 单调关闭。
4. accepted data 与 aborted buffer identity 不相交。
5. marker 在 accepted、drop、surplus、abort-to-buffer 路径全部闭合。
6. Planner OFF 保持当前路径。

## 实验判定

J1 首轮使用完整双步 cycle，不挑有利前缀。

机制 GO：

- boundary identity mismatch/stale 为 0；
- resident cap violation 为 0；
- debt quorum 重复/漏记为 0；
- aborted useful completion 为 0；
- duplicate identity、orphan、counter underflow 为 0；
- strict fallback 后 fresh admission 为 0。

性能 GO：

- token-normalized throughput 至少 `+2%`；
- paired-cycle wall 下降；
- boundary control wait 不增加；
- Actor weighted train token/s 在 `±2%`；
- current commit 不再稳定出现 `1/8 -> 8/8` 强振荡；
- gate 下降时 train wait 和 partition fill 不反向增加。

任一成本搬移、15–16 group envelope 复现、hard-cap wait、partition 未闭合或 fatal 都是
NO-GO。

## J0b 回放结果

J0b 使用以下两条 pre-A3 request-level trace：

```text
phase_feedback_fix_on_20260802_212004
phase_feedback_on20_20260802_230341
```

回放工具：

```text
scripts/task22/replay_joint_carry_admit.py
```

修正了旧模拟中 driver 秒级日志偏移与 permit Unix epoch 的不同原点问题。Driver 日志
显式按 `UTC+08:00` 解析，分析 wall 从旧 `simulation_results.json` 读取。

19 个 headline-window boundary 的结果：

```text
analysis-window coverage: PASS
identity / assignment checks: PASS
zero-resident buffer-refill cap diagnostic: PASS
observed dynamic quorum upper bound: PASS
joint carry budget replay: BLOCKED
closed-loop A3 trace: MISSING
J1 decision: HOLD
```

保守单边上界：

```text
dynamic debt quorum:
  78.1614 s
  +2.7566%

dynamic debt quorum + soft floor 2 + cap 2 s:
  66.6384 s
  +2.3407%
```

零 resident buffer-refill 初始计划最大 admit 为 9，resident cap violation 为 0。该结果
不能和历史 physical rollout 的累计 16 unique groups 直接比较，也不能证明 A3 carry
下的 shared budget 已闭合。

J1 保持 HOLD，因为两条 run 均未启用 A3，且没有 `joint_planner_ledger`。需要一条
`A3 enabled + joint ledger non-empty + no fatal` 的 shadow trace，才能继续 runtime 集成。
