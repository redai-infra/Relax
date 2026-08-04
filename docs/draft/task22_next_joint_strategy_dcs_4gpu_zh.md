# Task 22 下一轮联合策略与 4GPU DCS 子模块

## 目标

下一轮只运行一次 4GPU 联合 ON，资源拓扑固定为：

```text
Actor: 2 GPU, TP=2
Rollout: 2 个独立 engine, 每个 1 GPU
DCS rollout weight-update group: Actor PP source + 2 receivers, world_size=3
```

联合策略由以下部分组成：

1. per-request KV version age 与 targeted retirement；
2. 删除 phase-based blanket fresh freeze，保留 old-debt priority；
3. bounded dynamic candidate window；
4. per-engine estimated work accounting；
5. synchronous Hybrid DCS weight sync；
6. GPU-resident Actor snapshot。

当前分支已实现 per-request publication 协议；只有本地与正式环境预检全部通过后才启动付费实验。

## PR #184 复核

PR 中 GPU snapshot 的底层代码已经存在：

- 正式参数：`--hybrid-weights-backuper-on-gpu`
- 提交者回复中的 `--hybrid-weights-backup-on-gpu` 少了 `er`
- 两个拼写在本分支中都映射到同一 destination

PR 的 2GPU Hybrid 启动脚本没有传该参数，因此原测试仍使用 CPU pinned snapshot，经 H2D 后进入 DCS，不能代表 GPU-resident snapshot 的性能。

PR 树中的 `run-qwen3-vl-4B-2xgpu.sh` 是 colocate 脚本，不是 Hybrid DCS 脚本。真正的 DCS 验证脚本是 `run-qwen3-vl-4B-2xgpu-openr1mm-hybrid-async.sh`，拓扑为 Actor 1 GPU + Rollout 1 GPU。

Task 22 不复用多模态 workload，使用纯文本 Qwen3-4B clean runner。4GPU 入口为：

```text
scripts/task22/run_qwen3_4b_4gpu_dcs_joint_on.sh
```

## GPU Snapshot

候选 arm 强制：

```text
--hybrid-dcs-weight-sync
--hybrid-weights-backuper-on-gpu
--checkpoint-engine-backend nccl
```

仅 `actor` snapshot 常驻 GPU；ref/teacher/old_actor 仍在 host。

Qwen3-4B BF16 TP2 的理论额外参数副本约 4GB/GPU，保守预算 5GB/GPU。历史 Actor 峰值约 75GB，GPU 总显存约 95GB，预计仍保留约 15GB 余量。真实 run 前仍需检查初始化后 GPU 0/1 峰值。

## 单点埋点

每个 logical step 输出：

```text
TASK22_WEIGHT_SNAPSHOT
TASK22_DCS_WEIGHT_SYNC
```

统计：

- snapshot D2D 时间、tensor 数、local bytes；
- topology fetch；
- cold group setup / steady group reuse；
- source materialize 与 H2D bytes；
- TP all-gather；
- HF conversion；
- lock wait；
- NCCL broadcast；
- receiver finalize；
- pause/flush/continue；
- broadcast bytes 与 fanout bytes；
- backend/client total；
- snapshot + client total。

严格门槛：

- init marker 恰好 1 条；
- train marker 覆盖 step 0..10；
- headline 2..9 全部 `group_reused=true`；
- `world_size=3`；
- `rollout_receivers=2`；
- GPU snapshot `on_device=true`；
- DCS `source_h2d_bytes=0`；
- `fanout_bytes=2*broadcast_bytes`。
- train steps 至少观察到 1 个 safe RID continuation；
- train steps 至少观察到 1 个 expired RID targeted retirement；
- 每个 publication 恰好一对 `prepare -> commit`，不得出现 `fail`。

Analyzer：

```text
scripts/task22/analyze_dcs_weight_sync.py
```

## Publication 协议

DCS 原有 publication 是：

```text
pause all -> flush all cache -> broadcast -> continue
```

它会破坏 A3 KV continuation。当前联合 arm 已将其替换为 request-version ledger
协调的 targeted retirement + in-place publication，DCS-only arm 不再作为付费实验入口。

当前联合 publication 为：

```text
freeze admission/retry
-> targeted abort only expired RID
-> wait retired RID terminal
-> in-place DCS update
-> continue safe RID
-> strict retry retired RID
```

SGLang 已支持按 RID 调用 `/abort_request`，主要工程量在 RID→engine→KV epoch ledger、publication fence、retry 时序和异常恢复。
