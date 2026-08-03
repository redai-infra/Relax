# DeepEyes V2 Agentic — Pitfalls

适配 DeepEyes V2 在 Relax agentic 栈上的踩坑清单。按"先看这个"优先级排。

## 0. 调试期不要开 `--use-fault-tolerance` / `--use-health-check`

**症状**：rollout step 1 一直循环跑，前面 session 挂了自动被 kill 再拉起，后面新 session 又不断进来，看起来 rollout 永远不结束、pool 一直有活儿。

**根因**：`--use-fault-tolerance` 会启动 `RolloutHealthMonitor`（`relax/distributed/ray/rollout.py:830`）对每个 engine group 做健康检查，失败静默 restart。这条链路把真正的错误（OOM / IMA / session livelock / adapter 崩）**藏起来了**，只在训练日志里看到不断重启和新一轮 rollout。

**做法**：适配初期 **一律关掉** `--use-fault-tolerance` 和 `--use-health-check`。让第一个错误直接抛出来，看清是 SGLang crash 还是 agent session 挂了还是 reward 出错。稳定之后再考虑开。

不要用 fault tolerance 去"绕过"没定位的错。

______________________________________________________________________

## 1. colocate 必须 `--max-staleness 0`

`--max-staleness > 0` 是 `--fully-async` / `--hybrid` 的特性。纯 colocate 下必须 0，否则调度器会让 rollout 提前 dispatch，SGLang wake_up 时踩 memory_saver region，**第一次 prefill** 就 IMA，堆栈指向 `HybridReqToTokenPool.alloc` 类的 mamba 池 op，容易被误导去查 SGLang sizing bug。

代码 docstring 在 `relax/utils/arguments.py:67-73` 明说。

______________________________________________________________________

## 2. Agentic prepare-gate livelock（rollout 卡 20h+）

**症状**：`emitted_materialized_session_count_total` 平台化，进度条在 248/256 附近震荡，日志刷 `POST /agentic_api/chat/completions CANCELLED 900099ms`，`prepare_gate_blocked_ir_count` 一直涨到几百，SGLang `#running-req` 却很低甚至 idle。

**根因**：`_execute_managed_session_input` 对超时 session 发 `SIGTERM forget=True`，但 Python OpenAI SDK 在 httpx 请求里把 SIGTERM 吞掉，session 继续发请求最多 900s × retries。zombie IR 堆在 `pending_chat_waiters` 把 prepare gate 堵死，新 group 永远等不到 waiter。

**修复三件套（缺一不可）**：

1. `relax/agentic/pipeline/runtime.py:322` — `except asyncio.CancelledError` 升级为 SIGKILL
2. `examples/deepeyes_v2_agentic/app/agent.py` — `AsyncOpenAI(timeout=1200s, max_retries=0)`（快失败，SIGKILL 是唯一强制路径）
3. 启动脚本 `--agent-timeout 600`（兜底）

发生后必须**重启**，运行中的 instance 无法自愈。

**跟 straggler 区分**：straggler 的 SGLang `#running-req` 是从高慢慢降到 1，chat 请求没有 CANCELLED；livelock 是 chat 请求刷 CANCELLED + gate 计数涨 + SGLang idle。

______________________________________________________________________

## 3. 可恢复 tool error 不要终止 session

`code_extract_failed` / `tool_call_extract_failed` / sandbox unavailable / sandbox exec failed / `search_failed` 应返回 `done=False` + natural language feedback，让模型自纠。直接 terminate 会浪费 rollout 且贡献 zero-reward。

见 `examples/deepeyes_v2_agentic/app/env_deepeyes_v2.py` 里 `done=False` 分支。

______________________________________________________________________

## 4. apptainer session tmpdir 泄露

**症状**：单节点上一堆 `relax-apptainer-*` 目录，跨多次 run 累积；严重时打满容器盘触发 eviction。

**做法**：

- session tmpdir 加 pid 前缀 + atexit sweeper（`app/sandboxes/backends/apptainer_jupyter_backend.py`）
- 启动脚本先 `find /tmp -name 'relax-apptainer-*' -mmin +240 -exec rm -rf {} +` 扫掉 4h 以上的旧目录

单节点跑久了也要看 `apptainer instance list` 有没有 zombie，`losetup -a` 有没有孤儿 loop device。

______________________________________________________________________

## 5. jsonl `agent_turns` 骗人，看 TB `num_turn/mean`

`rollout_result/train/*.jsonl` 里 `agent_turns` 字段用 `metadata["rollout_turns"]`，agentic 新栈只写 `metadata["agentic_trace"]["turn_count"]`——共享 dump 只识别老 key，导致 agentic 的 jsonl `agent_turns` 永远是默认值 1，误以为模型不 tool-call。

**优先信 TB 的 `rollout/num_turn/mean`**。已 mitigation：`relax/agentic/session/state.py:848` 导出 Sample 时把 `len(turns)` 镜像回老 key，但要确认代码是否是最新。

______________________________________________________________________

## 6. Reward 必须给 tool bonus，否则 tool use 崩

上游 `deepeyesv2.py` 的 `tool_reward` 是**死代码**：算了但从不加进 final_score。final_score 只算 `0.8*acc + 0.2*format`。论文靠 cold-start SFT 打入工具调用习惯，在没 SFT 的 base 上直接跑必收敛到 1-turn 直答。

我们的 port `reward_deepeyes_v2.py` 在 perception/reason 加了 `0.2 * tool_bonus`（binary，多次调 = 1 次调），且 gate 在 `acc_reward >= 0.5` 上防 reward hacking（模型乱包 `<tool_call>` 骗 bonus）。search split 不动避免和 search_penalty 打架。

上游 search_penalty 触发是 raw `<tool_call>` tag count（任何 name 都算），我们限制到 `name ∈ {search, image_search}` 排除 `python_exec`。

______________________________________________________________________

## 7. 多模必须 `--no-rope-fusion`

Relax 多模训练（image / video / omni）都要带 `--no-rope-fusion`。RoPE fusion 与多模 position id 处理不兼容。不要在 perf-doctor review 时把它当遗留 flag 建议删。

______________________________________________________________________

## 8. `--agent-env` 多次出现互相覆盖

`--agent-env` 在 `relax/utils/arguments.py:1201-1209` 用 `nargs="+"`（不是 `action="append"`），同 dest 多次出现后值覆盖前值。所有 KV 必须写在**同一个** `--agent-env` flag 后：

```bash
--agent-env
    "NEMO_GYM_ADAPTER=..."
    "AGENT_DEBUG_LOG_DIR=..."
    "OPENAI_BASE_URL=..."
```

拆成多行等于只有最后一行生效。DeepEyes v2 目前不用 nemo_gym adapter，但走 agent-env 传 debug log dir 等的话要注意。

______________________________________________________________________

## 排查动作清单

1. **别一上来 py-spy** — 先 `git log --oneline -15 -- <相关路径>` + `git show <最近相关 commit>` 看有没有可疑改动
2. **确认执行模式** — `--colocate / --fully-async / --hybrid`、`--max-staleness`、`--use-fault-tolerance / --use-health-check` 状态
3. **首日 sanity check `rollout_result/train/0.jsonl`** — reward 是不是机制性 0、agent_turns 是不是全 1、response 里模型有没有输出 `<tool_call>`
4. **`Prepare-owned managed agent session completed before producing a chat IR`** — 第一动作看 `${AGENT_DEBUG_LOG_DIR}/agentic_session_*.log` 里的真实 traceback，不要纠结上层 stack
5. **rollout 卡但不像 hang** — 看 `#running-req` 曲线：慢慢降到 1 = straggler；一直 idle 但 chat 刷 CANCELLED = livelock
