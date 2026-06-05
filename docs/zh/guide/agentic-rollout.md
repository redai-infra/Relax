# Agentic Rollout

Agentic rollout 是 Relax 用于训练外部 agent app 的 rollout 路线。Relax 将每个 agent app 作为普通进程运行，向它提供 OpenAI-compatible chat endpoint，并把完成的 agent session 转换为训练样本。Relax 负责调度、请求锁定与恢复、reward 编排、dump、metrics，以及向训练侧的 TransferQueue 写入数据。

## 核心能力

1. **用已有 agent app 做 Agentic RL**
   通过 OpenAI-compatible chat interface 接入已有 agent app，并连接到 Relax 训练。

   ![Agent app integration](/agentic/agent_app.svg)

2. **Agent process warmup**
   在 rollout 执行前提前启动 agent process，隐藏进程启动、tool setup 和环境初始化耗时。

   ![Agent process warmup](/agentic/warmup.svg)

3. **Request-level partial rollout**
   在 Relax 内部中断和恢复 active model request，同时让 agent app 继续使用普通 chat-completion 流程。

   ![Request-level partial rollout](/agentic/partial_rollout.svg)

## 端到端生命周期

![Agentic rollout lifecycle](/agentic/lifecycle.svg)

Agent process 看到的是 task input、chat API endpoint 和可选 output path。Relax 在这个进程外管理 resident pipeline：

- Dataset row 被准备成 agent session input。
- Prepare 启动 agent process，并等待 session 到达 rollout admission 可消费状态。
- Session 中段是重复的 model-turn chat loop：agent 向 Relax 发送 chat request，Relax 把它调度为 `InflightRequest`，SGLang 生成 response，agent 可以执行工具后再请求下一轮。
- Runtime 将完成的 session finalize 成 sample。
- Reward 对 finalized sample 或 group 打分。
- Transfer 将 rewarded group 写入 TransferQueue。

## 架构

![Agentic rollout architecture](/agentic/architecture.svg)

Agentic rollout 分为四个 resident domain：

- **Prepare domain** 在 rollout step 消费前准备 agent session 并启动 agent process。
- **Runtime domain** 启动 agent process，暴露 session input/output 文件，跟踪 chat request，并 finalize 已完成的 session。
- **Reward domain** 对 completed sample 或 group 打分。
- **Transfer domain** 将 rewarded group 写入 TransferQueue。

这些 domain 会跨 rollout step 常驻。每个 step 会释放已经 export 或 transfer 的 batch；在 partial rollout 或 fully async rollout 启用时，prepared group 和 partially completed resident tail 可以继续进入后续 step。

## 配置 Agent 启动

接入 agent app 需要准备三项内容：

1. Agent code。
2. Launch command。
3. Working directory，通常是 agent code repository root。

启用 agentic rollout，并把 working directory 和 launch command 传给 Relax：

```bash
--use-agentic-rollout
--agent-cwd /path/to/agent_repo
--agent-command ". ./run_agent_app.sh"
```

可以通过 `--agent-env "FOO=bar" "BAZ=qux"` 向 agent process 传入额外环境变量。`RELAX_` 前缀由训练框架保留。

`--agent-timeout` 控制每个 managed-command agent session 的 active runtime budget，单位是秒。Session 已经 admitted 且 managed process 正在运行时计入预算；session gated 时暂停计时。超时后，Relax 向 managed agent process group 发送 `SIGTERM`，记录 managed-command timeout，并丢弃对应 runtime group。

Relax 在 `--agent-cwd` 下运行 command，并注入这些环境变量：

| 变量 | 含义 |
| --- | --- |
| `RELAX_INPUT_JSON` | `RELAX_SESSION_IO_DIR` 内的 input JSON file path。 |
| `RELAX_OUTPUT_JSON` | `RELAX_SESSION_IO_DIR` 内可选的 output JSON file path。 |
| `RELAX_SESSION_IO_DIR` | 每个 session 独立的临时 IO directory。 |
| `RELAX_BASE_URL` | Relax OpenAI-compatible chat API 的 base URL。 |
| `RELAX_SESSION_ID` | Session id，通常作为 OpenAI API key 使用。 |
| `RELAX_ROLLOUT_MODE` | `train` 或 `eval`。 |
| `RELAX_GROUP_ID` | 这个 session 所属的 runtime group id。 |

启动脚本通常把这些变量适配成 agent 已有的运行方式：

```bash
#!/usr/bin/env bash

export OPENAI_BASE_URL="${RELAX_BASE_URL}"
export OPENAI_API_KEY="${RELAX_SESSION_ID}"

python -m my_agent --arg1 val1 --arg2 val2
```

如果 agent 需要显式 session file path：

```bash
python -m my_agent --task "${RELAX_INPUT_JSON}" --result "${RELAX_OUTPUT_JSON}"
```

## Session Input 与 Multimodal Data

`RELAX_INPUT_JSON` 包含一个 JSON object，字段为 `messages` 和可选 `metadata`：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [{"type": "text", "text": "Solve the task."}]
    }
  ],
  "metadata": {
    "question_id": "sample-0001"
  }
}
```

`messages` 使用 OpenAI chat messages format，可以直接传给 `chat.completions.create(...)`。`metadata` 来自 `--metadata-key` 选中的 dataset row 字段，可用于 task context、environment initialization 或 agent strategy selection。

Dataset 到 input 的转换关系：

| 参数 | 默认值 | 转换路径 |
| --- | --- | --- |
| `--input-key` | `input` | `row[input_key]` -> `Sample.prompt` -> `RELAX_INPUT_JSON["messages"]` |
| `--metadata-key` | `metadata` | `row[metadata_key]` -> `Sample.metadata` -> `RELAX_INPUT_JSON["metadata"]` |
| `--multimodal-keys` | - | 启用 multimodal input 的 placeholder replacement。 |

对于 image input，`--multimodal-keys` 声明哪些 dataset field 存放 image data：

```bash
--multimodal-keys '{"image":"images"}'
```

在这个 mapping 中，`image` 是 modality name，`images` 是 dataset field。`row[input_key]` 里的 `<image>` placeholder 会按顺序匹配 `row["images"]` 里的 image。Relax 加载每张 image，将其编码为 PNG data URL，并写入 `messages`：

```json
{
  "type": "image_url",
  "image_url": {"url": "data:image/png;base64,..."}
}
```

如果 dataset prompt 已经包含 `{"type": "image", "image": ...}`，Relax 也会把它转换为 `image_url`。如果需要直接控制 multimodal 格式，可以把所需数据或引用放入 dataset metadata，再由 agent 读取 `RELAX_INPUT_JSON`。

## 调用 Relax Chat API

Relax 暴露 `/v1/chat/completions`。Agent 可以使用 OpenAI Python SDK、LiteLLM 或任意 HTTP client。

将 assistant response 追加回 history 时，保持 assistant `content` 为 string：

```python
messages.append({"role": "assistant", "content": choice.message.content})
```

支持的 request fields：

| 字段 | 含义 |
| --- | --- |
| `messages` | 必填的 OpenAI chat messages。 |
| `tools` | 可选的 OpenAI tool definitions。 |
| `chat_template_kwargs` | 对 `--apply-chat-template-kwargs` 的 per-request overrides。 |
| `temperature` | Per-turn sampling temperature。 |
| `top_p` | Per-turn nucleus sampling value。 |
| `logprobs` | 为 `true` 时，在返回 choice 中包含 token logprobs。 |
| `max_completion_tokens` | Per-turn generation length override。 |
| `stop` | Stop string 或 stop string list。 |
| `seed` | Per-turn sampling seed。 |

Relax 会先将 `chat_template_kwargs` 与 `--apply-chat-template-kwargs` 合并，再进行 state hashing 和 prompt compilation。`add_generation_prompt`、`tokenize` 和 `tools` 由 Relax 保留。

当某一轮需要不同于默认值的 generation length 时，使用 `max_completion_tokens`。如果省略，Relax 使用 `--rollout-max-response-len`。

当前限制：

| 字段 | 接受值 | 其他值 |
| --- | --- | --- |
| `stream` | omitted, `null`, or `false` | 拒绝 |
| `n` | omitted or `1` | 拒绝 |
| `top_logprobs` | omitted or `null` | 拒绝 |
| `functions` | omitted, `null`, or `[]` | 拒绝 |
| `function_call` | omitted, `null`, or `"none"` | 拒绝 |
| `tool_choice` | omitted, `null`, or `"none"` | 拒绝 |
| `parallel_tool_calls` | omitted, `null`, or `false` | 拒绝 |

在当前 agentic chat protocol 中，agent 需要从 assistant response 文本中解析 tool call，执行 tool，并把 tool result 追加到 `messages`。**后续会接入 SGLang function calling support**。

对于 `context_length_exceeded` 等上下文错误，agent 可以将 session 标记为 `finish_length` 并正常退出。对于其他 API error，推荐直接 raise exception，让 Relax 按 session lifecycle 执行清理。

## InflightRequest 作为调度单元

![InflightRequest scheduling unit](/agentic/inflight_request.svg)

每个 agent chat call 都会成为 Relax 内部的一个 `InflightRequest`。Agent 看到的是普通 OpenAI-compatible request/response exchange。Relax 看到的是可调度单元，包含 request state、rollout id、backend-started state、abort count、token deltas、loss masks、logprobs 和 export metadata。

关键属性是 lockability。Relax 可以把 `InflightRequest` 保留在 session shard 中而不发送给 SGLang，并在 rollout-step admission point 解锁。相同的 request 抽象同时支持 agent process warmup 和 partial rollout。

Agent 退出后，Relax 会 finalize 由这些 `InflightRequest` 构成的 session：

- exportable terminal response 会成为 training sample；
- non-finalizable 或 discarded session 会由 runtime 清理；
- 可选的 agent output metadata 或 reward 会合并到 finalized sample。

## Agent Process Warmup

![Agent process warmup flow](/agentic/warmup_flow.svg)

Agentic rollout 会在 rollout step 消费前启动 prepare group。一个 prepared agent 可能已经发出第一条 chat request；Relax 会把它记录为 `InflightRequest`，放在 prepare gate 后面，并在 rollout step admit 该 group 时解锁。

使用 agent 原本的 startup path，并尽量在进程开始阶段启动耗时较长的准备工作。Container startup、environment creation、tool initialization 和 client setup 不必在第一轮 model turn 前全部完成，除非这一轮依赖这些结果。已经启动的工作可以在 warmup 和 model generation 期间继续推进，Relax 因此可以隐藏一部分或全部等待时间。

## Request-Level Partial Rollout

![Request-level partial rollout sequence](/agentic/partial_rollout_sequence.svg)

Partial rollout 会在足够多 trainable group 到达目标后关闭当前 rollout step。Active agent chat call 已经表示为 `InflightRequest`，因此 Relax 可以在 request level 关闭 step，而不是在 agent process level 关闭。

当前 rollout step 关闭时，调度决策由 Relax 负责：

- 尚未到达 SGLang 的 request 可以继续锁在 Relax 内部。
- 已经到达 SGLang 的 request 可以被中断、记录，并由 Relax 标记为可恢复。

下一个 rollout step 再次 admit 该 session 时，Relax 解锁对应的 `InflightRequest` 并继续执行。Agent 仍然使用同一次 chat request；partial interruption 和 resumption 留在 Relax 内部完成。

## Session Output、Reward 与 Metadata

写入 `RELAX_OUTPUT_JSON` 是可选的。如果 agent 退出时没有写这个文件，Relax 使用空 output metadata 和无 agent-provided reward。

当 agent 需要返回 reward 或 metadata 时，可以写 output JSON：

```python
{
  "metadata": {
    "num_turn": 3,
    "stop_reason": "env_done",
    "env_infos": []
  },
  "reward": None
}
```

Top-level fields：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `metadata` | object | 合并到 `Sample.metadata` 的 agent-produced metadata。 |
| `reward` | `None`、number 或 object | Agent 计算出的 reward。 |

如果 agent 写入 `reward`，Relax 会将它直接挂到生成的 sample 上。如果 agent 没有写入 `reward`，可以配置 `--custom-rm-path` 等 reward function 在 session finalized 后计算 reward。

Finalize 后，output `metadata` 会合并进 `Sample.metadata`：

- Reward function 可以在计算 reward 时读取 `Sample.metadata`。
- Agentic dump 会包含 `Sample.metadata`，用于离线检查。
- Rollout metric reporting，包括启用 ClearML 时，可以聚合 numeric metadata。

Metric aggregation 使用 `Sample.metadata` 顶层 numeric field。Key 会在满足这些条件时被上报：非空 string、不以 `_` 开头、不是框架内部 key，且 value 是 `int` 或 `float`。对每个 accepted key，Relax 上报 `key/mean`、`key/median`、`key/max` 和 `key/min`。Nested object、list、string 和非 numeric value 会保留在 `Sample.metadata` 和 dump 中，但不会被聚合成 metrics。

## 下一步

- 阅读 [Fully Async Training](./fully-async-training.md) 了解 service-level parallel training。
- 阅读 [Dataset Design](./dataset-design.md) 了解 prompt 和 metadata dataset conventions。
- 阅读 [Configuration](./configuration.md) 了解常用训练配置。
