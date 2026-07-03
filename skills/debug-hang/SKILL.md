---
name: debug-hang
description: 自动排查 Ray 调度的分布式训练任务 hang 问题。使用当训练任务无响应、资源利用率异常、任务长时间无进度时。自动收集集群状态、任务调用栈、Actor 状态，分析阻塞链条并定位根因。
allowed-tools:
  - bash
  - read
  - grep
  - glob
---

# Ray 分布式训练 Hang 问题自动排查

## 排查流程

### Phase 1: 集群状态概览

**目标**: 确认集群健康状态和资源使用情况

```bash
ray status --address <address>
ray job list --address="<address>" | grep RUNNING
```

**关注指标**: 节点存活、CPU/GPU 使用率（异常低 → hang）、Pending resource demands、Object store 内存。

### Phase 2: 定位阻塞 Tasks

```bash
ray list tasks --address="<address>" --filter "JOB_ID=<job_id>" --filter "state=RUNNING" --format yaml
```

**关键字段**: `name`（业务逻辑）、`actor_id`、`worker_pid`（调用栈用）、`node_id`（py-spy 必须在正确节点执行）。

> **⚠️ 返回 `No resource in the cluster` / 空列表是常态，不是错误**。Actor 内部 `await` / `time.sleep` / `dist.barrier` 等阻塞**不会**显示为 RUNNING task — actor 主线程一直停在 `worker.main_loop`，业务逻辑跑在后台线程或 asyncio coroutine 里。**不要继续试 `state=SUBMITTED_TO_WORKER` / `PENDING_NODE_ASSIGNMENT` 等其他 state**，直接跳到 Phase 4 拉 actor 列表 + Phase 3 对 actor PID 跑 py-spy。

### Phase 3: 收集调用栈

> **重要**: `py-spy dump --pid <pid>` 必须在目标进程所在的节点上执行。

```bash
# 列出所有节点
ray job submit --working-dir "./" --address="<address>" -- \
  python scripts/tools/run_on_each_ray_node.py --list

# 在指定节点执行 py-spy（推荐）
ray job submit --working-dir "./" --address="<address>" -- \
  python scripts/tools/run_on_each_ray_node.py -n <node_id_or_ip> "py-spy dump --pid <pid>"

# 在所有 GPU 节点执行（单节点集群适用）
ray job submit --working-dir "./" --address="<address>" -- \
  python scripts/tools/run_on_each_ray_node.py "py-spy dump --pid <pid>"
```

**重点关注**: 主线程阻塞点、后台线程状态、`[Has the GIL]` 标记。

#### ⚠️ 反模式（实际踩过的坑）

| 反模式 | 现象 | 正确做法 |
|--------|------|----------|
| 本地 `py-spy dump --pid <remote_pid>` | `Error: No such file or directory (os error 2)` | PID 来自远端 actor 的 `worker_pid`，必须通过 `ray job submit` + `run_on_each_ray_node.py -n <node_id>` 在对应节点执行 |
| `RAY_ADDRESS=... python scripts/tools/run_on_each_ray_node.py --list` | 启了**新的本地 Ray 实例**，看不到目标集群 | `run_on_each_ray_node.py` 内部 `ray.init()` 不带 address，env var 不生效；必须 `ray job submit --address="<addr>" -- python scripts/tools/run_on_each_ray_node.py --list` |
| `ray job submit -- bash -c 'for pid in 1 2 3; do py-spy --pid $pid; done'` | py-spy 收到空的 `--pid`，`$pid` 被 ray 的引号嵌套吃掉 | 把循环写到 `pyspy_dump.sh` 文件，再 `ray job submit --working-dir "./" -- bash pyspy_dump.sh`（脚本随 working-dir 一起上传） |
| 一个 PID 一个 `ray job submit` | 每次 ~30-60s 启动开销 × N 个 PID | 写一个脚本文件循环 dump 所有 PID，**单次** `ray job submit` 跑完 |

### Phase 4: Actor 依赖链分析

```bash
ray list actors --address="<address>" --filter "JOB_ID=<job_id>" --filter "STATE=ALIVE" --format yaml
```

**分析维度**: 数据流方向（生产者→消费者）、调用关系（parent_task_id→task_id）、资源竞争。

### Phase 5: 阻塞模式匹配

| 模式 | 调用栈特征 | 排查方向 |
|------|-----------|----------|
| 数据等待 | `time.sleep` 在迭代器/队列中 | 上游数据生产者是否工作 |
| 分布式同步 | `dist.broadcast`, `dist.all_reduce`, `dist.barrier` | 所有 rank 是否到达同步点 |
| 条件等待 | `while True: if condition: break; sleep` | 条件是否有机会满足 |
| 资源竞争 | 锁/信号量等待 | 是否存在死锁 |
| 远程调用阻塞 | `ray.get` 等待 | 被调用方是否响应 |
| 网络 I/O | socket read/write | 对端是否存活 |

### Phase 6: 根因定位

1. **追溯阻塞链**: 从阻塞点向上游追溯（数据等待→生产者状态，分布式同步→其他 rank 状态，条件等待→条件设置逻辑）
2. **状态机验证**: 初始状态是否正确、状态转换条件是否可达、是否存在状态机死锁
3. **配置一致性**: 条件分支依赖的配置项、默认值与预期值的差异

## 自动化诊断脚本

```bash
#!/bin/bash
# scripts/tools/diagnose_ray_hang.sh
set -e

RAY_ADDRESS="${1:-$RAY_ADDRESS}"
OUTPUT_DIR="${2:-/tmp/ray_hang_diag_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR"

echo "=== Phase 1: Cluster Status ===" | tee "$OUTPUT_DIR/01_cluster.txt"
ray status --address "$RAY_ADDRESS" 2>&1 | tee -a "$OUTPUT_DIR/01_cluster.txt"

echo -e "\n=== Phase 2: Running Jobs ===" | tee "$OUTPUT_DIR/02_jobs.txt"
ray job list --address="$RAY_ADDRESS" 2>&1 | tee "$OUTPUT_DIR/02_jobs.txt"

JOB_ID=$(grep -oP "job_id='\K[^']+" "$OUTPUT_DIR/02_jobs.txt" | head -1)
[ -z "$JOB_ID" ] && echo "No running job found" && exit 1
echo "Target Job ID: $JOB_ID" | tee -a "$OUTPUT_DIR/02_jobs.txt"

echo -e "\n=== Phase 3: Running Tasks ===" | tee "$OUTPUT_DIR/03_tasks.txt"
ray list tasks --address="$RAY_ADDRESS" --filter "JOB_ID=$JOB_ID" --filter "state=RUNNING" --format yaml 2>&1 | tee "$OUTPUT_DIR/03_tasks.txt"

echo -e "\n=== Phase 4: Active Actors ===" | tee "$OUTPUT_DIR/04_actors.txt"
ray list actors --address="$RAY_ADDRESS" --filter "JOB_ID=$JOB_ID" --filter "STATE=ALIVE" --format yaml 2>&1 | tee "$OUTPUT_DIR/04_actors.txt"

echo -e "\n=== Phase 5: Stack Traces ===" | tee "$OUTPUT_DIR/05_stacks.txt"
awk '
  /node_id:/ { node=$2 }
  /worker_pid:/ { pid=$2; print pid, node }
' "$OUTPUT_DIR/03_tasks.txt" | while read PID NODE_ID; do
    echo -e "\n--- PID $PID (node: $NODE_ID) ---" | tee -a "$OUTPUT_DIR/05_stacks.txt"
    ray job submit --working-dir "./" --address="$RAY_ADDRESS" -- \
        python scripts/tools/run_on_each_ray_node.py -n "$NODE_ID" "py-spy dump --pid $PID" 2>&1 | tee -a "$OUTPUT_DIR/05_stacks.txt"
done

echo -e "\n=== Diagnosis Complete ==="
echo "Output saved to: $OUTPUT_DIR"
```

## 输出报告模板

```markdown
## 集群状态摘要
- 活跃节点: X / Y
- GPU 使用率: Z%
- 异常信号: ...

## 阻塞 Task 分析
| Task Name | PID | 阻塞位置 | 模式分类 |
|-----------|-----|----------|----------|

## 阻塞链条
Actor A (阻塞于条件 X)
  ↑ 等待
Actor B (阻塞于数据 Y)
  ↑ 等待
Actor C (空闲，未生产数据 Y) ← 根因

## 根因诊断
- 主要原因 / 触发条件 / 影响范围

## 修复建议
```

## 关键排查原则

1. **从现象到本质**: GPU 低利用率 → 阻塞 Task → 阻塞代码行 → 阻塞原因
2. **追溯数据流**: 消费者阻塞 → 检查生产者 → 检查上游
3. **验证假设**: 提出假设 → 检查相关状态 → 确认或否定
4. **最小化改动**: 定位根因后，用最小改动修复
5. **Ray address 默认端口**: 用户给出的 `RAY_ADDRESS` 若未显式指定端口，按 `6379` 处理（如 `x.x.x.x` → `x.x.x.x:6379`）

## 常见踩坑速查（实战教训）

下表汇总了实际 debug 中浪费时间的反模式，**遇到对应现象立即跳到「正确做法」**，不要重复试错。

| 阶段 | 错误做法 | 现象 | 正确做法 |
|------|---------|------|---------|
| Phase 2 | `state=RUNNING` 返回空 → 继续试 `SUBMITTED_TO_WORKER` / `PENDING_NODE_ASSIGNMENT` | 一直查不到 task | actor-internal hang（await/sleep/barrier）**不显示为 RUNNING task**，直接跳 Phase 4 拉 actors，对 actor PID 跑 py-spy |
| Phase 3 | 本地 `py-spy dump --pid <remote_pid>` | `Error: No such file or directory` | PID 是远端的，必须 `ray job submit` + `run_on_each_ray_node.py -n <node_id>` |
| Phase 3 | `RAY_ADDRESS=... python scripts/tools/run_on_each_ray_node.py --list` | 启了新本地 Ray，看不到目标集群 | 该脚本内部 `ray.init()` 不带 address，env var 不生效；必须 `ray job submit --address="<addr>" -- python ...` |
| Phase 3 | `ray job submit -- bash -c 'for pid in 1 2 3; do py-spy --pid $pid; done'` | py-spy 收到空 `--pid`，输出乱 | `$pid` 被 ray 的引号嵌套吞了；写脚本文件 `pyspy_dump.sh` 然后 `ray job submit --working-dir "./" -- bash pyspy_dump.sh` |
| Phase 3 | 一个 PID 一次 `ray job submit` × N | 每次 30-60s 启动开销 | 写循环到 .sh 脚本里，**单次** submit 跑完所有 PID |

---

## 参考案例

| 案例 | 描述 |
|------|------|
| [case-rollout-eval-onload-hang.md](references/case-rollout-eval-onload-hang.md) | Rollout eval 等待 onload 状态导致 hang（配置与逻辑不匹配） |
