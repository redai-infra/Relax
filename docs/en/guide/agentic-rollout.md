# Agentic Rollout

Agentic rollout is Relax's rollout path for training external agent applications. Relax runs each agent app as a normal process, gives it an OpenAI-compatible chat endpoint, and converts completed agent sessions into training samples. Relax owns scheduling, request locking and resumption, reward orchestration, dumping, metrics, and transfer into training.

## Core Capabilities

1. **Agentic RL with existing agent apps**
   Run existing agent apps through an OpenAI-compatible chat interface and connect them to Relax training.

   ![Agent app integration](/agentic/agent_app.svg)

2. **Agent process warmup**
   Launch agent processes ahead of rollout execution, hiding process startup, tool setup, and environment initialization overhead.

   ![Agent process warmup](/agentic/warmup.svg)

3. **Request-level partial rollout**
   Interrupt and resume active model requests inside Relax while the agent app keeps using a normal chat-completion flow.

   ![Request-level partial rollout](/agentic/partial_rollout.svg)

## End-to-End Lifecycle

![Agentic rollout lifecycle](/agentic/lifecycle.svg)

The agent process sees a task input, a chat API endpoint, and an optional output path. Relax manages the resident pipeline around that process:

- Dataset rows are prepared as agent session inputs.
- Prepare launches agent processes and waits until their sessions are ready for rollout admission.
- The middle of a session is a repeated model-turn chat loop: the agent sends a chat request to Relax, Relax schedules it as an `InflightRequest`, SGLang generates the response, and the agent may execute tools before asking for the next turn.
- Runtime finalizes completed sessions into samples.
- Reward scores finalized samples or groups.
- Transfer writes rewarded groups to TransferQueue.

## Architecture

![Agentic rollout architecture](/agentic/architecture.svg)

Agentic rollout is split into four resident domains:

- **Prepare domain** prepares agent sessions and launches agent processes before the rollout step consumes them.
- **Runtime domain** launches agent processes, exposes session input and output files, tracks chat requests, and finalizes completed sessions.
- **Reward domain** scores completed samples or groups.
- **Transfer domain** writes rewarded groups to TransferQueue.

These domains stay resident across rollout steps. Each step releases the batches it has already exported or transferred, while prepared groups and partially completed resident tail can continue into later steps when partial rollout or fully async rollout is enabled.

## Configure Agent Startup

Prepare three things to connect an agent app to training:

1. Agent code.
2. A launch command.
3. A working directory, usually the agent code repository root.

Enable agentic rollout and pass the working directory and launch command to Relax:

```bash
--use-agentic-rollout
--agent-cwd /path/to/agent_repo
--agent-command ". ./run_agent_app.sh"
```

Optional extra environment variables for the agent process can be passed with `--agent-env "FOO=bar" "BAZ=qux"`. The `RELAX_` prefix is reserved by the training framework.

`--agent-timeout` controls each managed-command agent session's active runtime budget in seconds. The budget is charged while an admitted managed process is running and pauses while the session is gated. On timeout, Relax sends `SIGTERM` to the managed agent process group, records a managed-command timeout, and drops the corresponding runtime group.

Relax runs the command under `--agent-cwd` and injects these environment variables:

| Variable | Meaning |
| --- | --- |
| `RELAX_INPUT_JSON` | Input JSON file path inside `RELAX_SESSION_IO_DIR`. |
| `RELAX_OUTPUT_JSON` | Optional output JSON file path inside `RELAX_SESSION_IO_DIR`. |
| `RELAX_SESSION_IO_DIR` | Per-session temporary IO directory. |
| `RELAX_BASE_URL` | Base URL of the Relax OpenAI-compatible chat API. |
| `RELAX_SESSION_ID` | Session id; commonly used as the OpenAI API key. |
| `RELAX_ROLLOUT_MODE` | `train` or `eval`. |
| `RELAX_GROUP_ID` | Runtime group id for this session. |

A launch script usually adapts these variables to whatever the agent already expects:

```bash
#!/usr/bin/env bash

export OPENAI_BASE_URL="${RELAX_BASE_URL}"
export OPENAI_API_KEY="${RELAX_SESSION_ID}"

python -m my_agent --arg1 val1 --arg2 val2
```

For an agent that expects explicit session file paths:

```bash
python -m my_agent --task "${RELAX_INPUT_JSON}" --result "${RELAX_OUTPUT_JSON}"
```

## Session Input and Multimodal Data

`RELAX_INPUT_JSON` contains a JSON object with `messages` and optional `metadata`:

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

`messages` uses the OpenAI chat messages format and can be passed to `chat.completions.create(...)`. `metadata` comes from the dataset row selected by `--metadata-key`; use it for task context, environment initialization, or agent strategy selection.

Dataset-to-input conversion:

| Argument | Default | Conversion path |
| --- | --- | --- |
| `--input-key` | `input` | `row[input_key]` -> `Sample.prompt` -> `RELAX_INPUT_JSON["messages"]` |
| `--metadata-key` | `metadata` | `row[metadata_key]` -> `Sample.metadata` -> `RELAX_INPUT_JSON["metadata"]` |
| `--multimodal-keys` | - | Enables placeholder replacement for multimodal inputs. |

For image inputs, `--multimodal-keys` declares which dataset fields contain image data:

```bash
--multimodal-keys '{"image":"images"}'
```

In this mapping, `image` is the modality name and `images` is the dataset field. `<image>` placeholders in `row[input_key]` are matched with images from `row["images"]` in order. Relax loads each image, encodes it as a PNG data URL, and writes it into `messages` as:

```json
{
  "type": "image_url",
  "image_url": {"url": "data:image/png;base64,..."}
}
```

If the dataset prompt already contains `{"type": "image", "image": ...}`, Relax also converts it to `image_url`. To control multimodal formatting directly, put the needed data or references in dataset metadata and let the agent read `RELAX_INPUT_JSON`.

## Calling the Relax Chat API

Relax exposes `/v1/chat/completions`. The agent can use the OpenAI Python SDK, LiteLLM, or any HTTP client.

When appending the assistant response back into history, keep assistant `content` as a string:

```python
messages.append({"role": "assistant", "content": choice.message.content})
```

Supported request fields:

| Field | Meaning |
| --- | --- |
| `messages` | Required OpenAI chat messages. |
| `tools` | Optional OpenAI tool definitions. |
| `chat_template_kwargs` | Per-request overrides for `--apply-chat-template-kwargs`. |
| `temperature` | Per-turn sampling temperature. |
| `top_p` | Per-turn nucleus sampling value. |
| `logprobs` | When `true`, include token logprobs in the returned choice. |
| `max_completion_tokens` | Per-turn generation length override. |
| `stop` | Stop string or list of stop strings. |
| `seed` | Per-turn sampling seed. |

Relax merges `chat_template_kwargs` with `--apply-chat-template-kwargs` before state hashing and prompt compilation. The keys `add_generation_prompt`, `tokenize`, and `tools` are reserved by Relax.

Use `max_completion_tokens` when a turn needs a generation length different from the default. If omitted, Relax uses `--rollout-max-response-len`.

Current limitations:

| Field | Accepted value | Other values |
| --- | --- | --- |
| `stream` | omitted, `null`, or `false` | rejected |
| `n` | omitted or `1` | rejected |
| `top_logprobs` | omitted or `null` | rejected |
| `functions` | omitted, `null`, or `[]` | rejected |
| `function_call` | omitted, `null`, or `"none"` | rejected |
| `tool_choice` | omitted, `null`, or `"none"` | rejected |
| `parallel_tool_calls` | omitted, `null`, or `false` | rejected |

In the current agentic chat protocol, the agent needs to parse tool-call text from the assistant response, execute the tool, and append the tool result to `messages`. **SGLang function calling support will be integrated in the future**.

For context errors such as `context_length_exceeded`, the agent may mark the session as `finish_length` and exit normally. For other API errors, raising the exception is recommended so Relax can clean up according to the session lifecycle.

## InflightRequest as the Scheduling Unit

![InflightRequest scheduling unit](/agentic/inflight_request.svg)

Each agent chat call becomes an `InflightRequest` inside Relax. The agent sees a normal OpenAI-compatible request and response exchange. Relax sees a schedulable unit that carries request state, rollout id, backend-started state, abort count, token deltas, loss masks, logprobs, and export metadata.

The important property is lockability. Relax can keep an `InflightRequest` in the session shard without sending it to SGLang, then unlock it at the rollout-step admission point. The same request abstraction supports both agent process warmup and partial rollout.

When the agent exits, Relax finalizes the session built from its `InflightRequest`s:

- exportable terminal responses become training samples;
- non-finalizable or discarded sessions are cleaned up by the runtime;
- optional agent output metadata or reward is merged into the finalized sample.

## Agent Process Warmup

![Agent process warmup flow](/agentic/warmup_flow.svg)

Agentic rollout launches prepare groups before the rollout step consumes them. A prepared agent may already issue its first chat request; Relax records it as an `InflightRequest`, holds it behind the prepare gate, and unlocks it when the rollout step admits the group.

Use the agent's normal startup path, and start expensive setup near the beginning of the process when possible. Container startup, environment creation, tool initialization, and client setup do not have to finish before the first model turn unless that turn depends on them. Work that has already been started can continue during warmup and model generation, allowing Relax to hide part or all of its latency before the agent needs the result.

## Request-Level Partial Rollout

![Request-level partial rollout sequence](/agentic/partial_rollout_sequence.svg)

Partial rollout closes the current rollout step once enough trainable groups have reached the target. Active agent chat calls are already represented as `InflightRequest`s, so Relax can close the step at the request level instead of the agent process level.

When the current rollout step is closed, Relax owns the scheduling decision:

- A request that has not reached SGLang can stay locked inside Relax.
- A request that has already reached SGLang can be interrupted, recorded, and made resumable by Relax.

When the next rollout step admits the session again, Relax unlocks the corresponding `InflightRequest` and continues it. The agent keeps using the same chat request; partial interruption and resumption stay inside Relax.

## Session Output, Reward, and Metadata

Writing the file at `RELAX_OUTPUT_JSON` is optional. If the agent exits without writing it, Relax uses empty output metadata and no agent-provided reward.

Write output JSON when the agent needs to return reward or metadata:

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

Top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `metadata` | object | Agent-produced metadata to merge into `Sample.metadata`. |
| `reward` | `None`, number, or object | Reward computed by the agent. |

If the agent writes `reward`, Relax attaches it to the generated sample directly. If the agent does not write `reward`, configure a reward function such as `--custom-rm-path` to compute reward after the session is finalized.

After finalization, output `metadata` is merged into `Sample.metadata`:

- Reward functions can read `Sample.metadata` while computing reward.
- Agentic dumps include `Sample.metadata` for offline inspection.
- Rollout metric reporting, including ClearML when enabled, can aggregate numeric metadata.

Metric aggregation uses top-level numeric fields in `Sample.metadata`. A key is reported when it is a non-empty string, does not start with `_`, is not an internal framework key, and its value is an `int` or `float`. For every accepted key, Relax reports `key/mean`, `key/median`, `key/max`, and `key/min`. Nested objects, lists, strings, and non-numeric values remain in `Sample.metadata` and dumps but are not aggregated as metrics.

## Next Steps

- Read [Fully Async Training](./fully-async-training.md) for service-level parallel training.
- Read [Dataset Design](./dataset-design.md) for prompt and metadata dataset conventions.
- Read [Configuration](./configuration.md) for common training options.
