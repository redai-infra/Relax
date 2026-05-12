# 生成式奖励模型 (GenRM) 示例

使用 **生成式奖励模型**（GenRM）——一种 LLM-as-judge 方法——对 rollout 生成的响应进行评分，替代传统的训练式奖励模型。

## 概述

GenRM（Generative Reward Model，生成式奖励模型）利用预训练的大语言模型（如 Qwen3-VL-30B-A3B-Instruct）来评估模型响应是否与标准答案一致。与训练专用奖励模型不同，GenRM 通过部署为独立 Ray Serve 服务的 [SGLang](https://github.com/sgl-project/sglang) 引擎在推理时进行评估。

核心优势：

- **零训练成本** — 直接使用已有的预训练 LLM，无需额外训练奖励模型
- **泛化能力强** — 利用 LLM 的推理能力，对未见过的任务也能有效评估
- **灵活可控** — 可通过 prompt 模板调整评估标准

本示例中的两个脚本均使用 **GRPO** 算法在 `dapo-math-17k` 数据集上训练 **Qwen3-4B**，通过 GenRM（`--rm-type dapo-genrm`）进行奖励评分，并使用 AIME-2024 进行评估。

## 架构

在推荐的 **colocate 模式**下，全部 GPU 归 Actor（训练）所有。在推理阶段，Actor 卸载权重，GPU 由 Rollout 和 GenRM 分时复用。推理完成后，GenRM 和 Rollout 卸载权重，全部 GPU 重新归 Actor 训练使用。这意味着 GenRM 的 GPU 不会被浪费——它们在不进行评估时直接加速训练。

根据 GPU 分配，框架会自动识别两种 colocate 子模式：

- **Split 模式**（`rollout_num_gpus + genrm_num_gpus == actor_total_gpus`）：Rollout 和 GenRM 占用**不重叠**的 GPU bundle。适合两个引擎都不大、各自独占一组 GPU 比共享更高效的场景。
- **Shared 模式**（`rollout_num_gpus == genrm_num_gpus == actor_total_gpus`）：Rollout 和 GenRM 占用**相同**的 GPU bundle，通过 SGLang 的 `mem_fraction_static` 切分每张 GPU 的显存。适合 GenRM 较大（如 30B）、需要全集群 TP，但又不想牺牲 Rollout 吞吐的场景。**无需新增 CLI flag，自动启用**。

```
                 8-GPU Colocate (Split)

 ┌──────────── Placement Group (8 GPU) ────────────┐
 │                                                 │
 │  Inference phase:                               │
 │  ┌───────────────────┐   ┌───────────────────┐  │
 │  │  Rollout  (4 GPU) │──►│  GenRM   (4 GPU)  │  │
 │  │  bundles 0..3     │◄──│  bundles 4..7     │  │
 │  └───────────────────┘   └───────────────────┘  │
 │                                                 │
 │  Training phase (offload inference weights):    │
 │  ┌─────────────────────────────────────────┐    │
 │  │         Actor  (8 GPU)                  │    │
 │  │         Megatron Training               │    │
 │  └─────────────────────────────────────────┘    │
 └─────────────────────────────────────────────────┘


              8-GPU Colocate (Shared)

 ┌──────────── Placement Group (8 GPU) ────────────┐
 │                                                 │
 │  Inference phase (same bundles 0..7):           │
 │  ┌─────────────────────────────────────────┐    │
 │  │  Rollout: mem_fraction_static = 0.6     │    │
 │  │  GenRM  : mem_fraction_static = 0.3     │    │
 │  │  ~0.1 reserved for cuda / activations   │    │
 │  └─────────────────────────────────────────┘    │
 │                                                 │
 │  Training phase (offload inference weights):    │
 │  ┌─────────────────────────────────────────┐    │
 │  │         Actor  (8 GPU)                  │    │
 │  │         Megatron Training               │    │
 │  └─────────────────────────────────────────┘    │
 └─────────────────────────────────────────────────┘
```

所有组件共处同一个 placement group。Rollout 生成候选响应，然后将响应与标准答案一起通过 HTTP 发送给 GenRM；GenRM 返回二值评分（1 = 一致，0 = 不一致）。奖励计算完成后，推理权重被卸载，全部 GPU 归还给 Actor 用于训练。

## 脚本

| 脚本                               | 模式                  | 描述                                                    |
| :--------------------------------- | :-------------------- | :------------------------------------------------------ |
| `run-qwen3-4B-8xgpu-colocated.sh` | Colocate（推荐）       | 全部 8 GPU 用于训练；rollout 与 GenRM 通过 offload 分时复用 |
| `run-qwen3-4B-8xgpu-async.sh`     | Fully Async           | 每个角色使用独立 GPU 池；rollout 与训练完全并行            |

### 资源分配

**Colocate / Split**（`--colocate`，推荐用于小 GenRM）：

```
Actor（训练）:  8 GPU（全部 GPU 参与训练）
Rollout:       4 GPU（bundles 0..3，与 actor 分时复用）
GenRM:         4 GPU（bundles 4..7，与 actor 分时复用）
```

**Colocate / Shared**（`--colocate`，推荐用于大 GenRM）：

```
Actor（训练）:  8 GPU（全部 GPU 参与训练）
Rollout:       8 GPU（bundles 0..7，mem_fraction_static = 0.6）
GenRM:         8 GPU（bundles 0..7，mem_fraction_static = 0.3）
```

两种 colocate 子模式下，GenRM 的 GPU 在训练阶段都会卸载回 Actor 做梯度计算，训练始终能使用全部 8 GPU 并行能力。Shared 模式额外允许 GenRM 使用更大的 TP（例如 30B 模型用 TP=8），同时不牺牲 Rollout 吞吐。

**Async 模式**（`--fully-async`）：

```
Actor（训练）:  2 GPU（专用）
Rollout:       3 GPU（专用）
Reference:     1 GPU
Actor Forward: 1 GPU
GenRM:         1 GPU（专用）
```

## 快速开始

### 前置条件

1. **模型权重** — 下载 Qwen3-4B（策略模型）和 Qwen3-VL-30B-A3B-Instruct（GenRM 评估模型）：

   ```bash
   # 放置在 exps/ 目录下（或设置 EXP_DIR / MODEL_DIR）
   exps/Qwen3-4B/
   exps/Qwen3-VL-30B-A3B-Instruct/
   ```

2. **数据集** — 准备 `dapo-math-17k` 用于训练，`aime-2024` 用于评估：

   ```bash
   exps/dapo-math-17k/dapo-math-17k.jsonl
   exps/aime-2024/aime-2024.jsonl
   ```
3. **Ray 集群** — 一个可访问的 Ray 集群，地址为 `http://127.0.0.1:8265`。

### 启动训练

```bash
# Colocate 模式（推荐，至少 8 GPU）
bash examples/generate_reward_model/run-qwen3-4B-8xgpu-colocated.sh

# Fully async 模式（至少 8 GPU）
bash examples/generate_reward_model/run-qwen3-4B-8xgpu-async.sh
```

### 验证服务健康状态

训练任务启动后，检查 GenRM 服务是否正常运行：

```bash
curl http://localhost:8000/genrm/health
```

预期响应：

```json
{
  "status": "healthy",
  "service": "genrm"
}
```

## 配置

### GenRM 专用命令行参数

| 参数                          | 类型   | 默认值 | 描述                                                               |
| :---------------------------- | :----- | :----- | :----------------------------------------------------------------- |
| `--genrm-model-path`          | `str`  | `None` | GenRM 模型路径，设置后启用 GenRM                                    |
| `--genrm-num-gpus`            | `int`  | `1`    | GenRM 使用的 GPU 总数                                               |
| `--genrm-num-gpus-per-engine` | `int`  | `1`    | 每个 GenRM 引擎使用的 GPU 数量                                      |
| `--genrm-engine-config`       | `JSON` | `None` | 引擎初始化 JSON 配置（如 `max_context_len`、`dp_size`、`pp_size`）  |
| `--genrm-sampling-config`     | `JSON` | `None` | 采样参数 JSON 配置                                                  |

### 引擎配置键

| 键                    | 类型    | 默认值          | 描述                                                                  |
| :-------------------- | :------ | :-------------- | :-------------------------------------------------------------------- |
| `max_context_len`     | `int`   | `8192`          | 最大上下文长度                                                         |
| `dp_size`             | `int`   | `1`             | 数据并行大小                                                           |
| `pp_size`             | `int`   | `1`             | 流水线并行大小                                                         |
| `ep_size`             | `int`   | `1`             | 专家并行大小                                                           |
| `mem_fraction_static` | `float` | SGLang 默认值   | 单引擎 SGLang 静态显存比例。**Shared 模式下必须设置**（见下文配置示例）。 |

### 采样配置键

| 键                 | 类型    | 默认值 | 描述                      |
| :----------------- | :------ | :----- | :------------------------ |
| `temperature`      | `float` | `0.1`  | 采样温度                   |
| `top_p`            | `float` | `1.0`  | 核采样概率                 |
| `top_k`            | `int`   | `-1`   | Top-k 采样（-1 表示禁用）  |
| `max_response_len` | `int`   | `4096` | 最大响应长度               |

### 资源分配

GenRM 在 `--resource` JSON 中作为 `"genrm"` 角色配置，格式为 `[num_groups, num_gpus_per_group]`。

**Colocated / Split**（小 GenRM，默认）：

```bash
python3 relax/entrypoints/train.py \
    --genrm-model-path /path/to/genrm/model \
    --genrm-num-gpus-per-engine 4 \
    --genrm-engine-config '{"max_context_len": 10240}' \
    --genrm-sampling-config '{"temperature": 0.1, "top_p": 1.0, "top_k": -1, "max_response_len": 1024}' \
    --resource '{"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 4]}' \
    --colocate \
    --rm-type dapo-genrm
```

**Colocated / Shared**（大 GenRM，新增）：把 rollout 和 genrm 都设为 actor 的全部 GPU；框架自动识别为 shared 模式，让两个引擎通过 `mem_fraction_static` 切分每张 GPU 的显存：

```bash
python3 relax/entrypoints/train.py \
    --genrm-model-path /path/to/genrm/model \
    --genrm-num-gpus-per-engine 8 \
    --genrm-engine-config '{"max_context_len": 10240, "mem_fraction_static": 0.3}' \
    --genrm-sampling-config '{"temperature": 0.1, "top_p": 1.0, "top_k": -1, "max_response_len": 1024}' \
    --rollout-num-gpus-per-engine 1 \
    --sglang-mem-fraction-static 0.6 \
    --resource '{"actor": [1, 8], "rollout": [1, 8], "genrm": [1, 8]}' \
    --colocate \
    --rm-type dapo-genrm
```

::: tip 自动识别 colocate 子模式
启用 `--colocate` 且配置了 GenRM 时，GPU 分配决定子模式：

| 分配                                                  | 子模式                                  |
| :---------------------------------------------------- | :-------------------------------------- |
| `rollout_num_gpus + genrm_num_gpus == actor_total`    | **Split**（rollout 和 genrm 在不同 bundle） |
| `rollout_num_gpus == genrm_num_gpus == actor_total`   | **Shared**（rollout 和 genrm 在同一组 bundle） |
| 其他                                                  | 启动时报错拒绝                            |
:::

::: warning Shared 模式必须设置 `mem_fraction_static`
Shared 模式下两个 SGLang 引擎共占同一组 GPU，必须设置各自的 `mem_fraction_static`，使**单卡之和 < 1.0**（建议 ≤ 0.9，剩余给 cuda graph + activations）。Rollout 通过 `--sglang-mem-fraction-static`（或 `--sglang-config` YAML overrides）配置；GenRM 通过 `--genrm-engine-config` 中的 `mem_fraction_static` 配置。
:::

**Fully-Async 模式**：

```bash
python3 relax/entrypoints/train.py \
    --genrm-model-path /path/to/genrm/model \
    --genrm-num-gpus-per-engine 1 \
    --genrm-engine-config '{"max_context_len": 10240}' \
    --genrm-sampling-config '{"temperature": 0.1, "top_p": 1.0, "top_k": -1, "max_response_len": 1024}' \
    --resource '{"actor": [1, 2], "rollout": [1, 3], "reference": [1, 1], "actor_fwd": [1, 1], "advantages": [1, 0], "genrm": [1, 1]}' \
    --fully-async \
    --rm-type dapo-genrm
```

## 脚本详解

两个脚本共享相同的结构，以下是关键配置组的详细说明：

### 奖励配置

启用 GenRM 的关键设置是 `--rm-type dapo-genrm`，它将奖励计算路由到 `relax/engine/rewards/dapo_genrm.py` 中的 `async_compute_score_genrm()` 函数。核心实现如下：

```python
DAPO_GENRM_PROMPT_TEMPLATE = """Below are two answers to a question. ...
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""

def _format_messages(question, ground_truth, predict_str):
    # 提取 "Answer:" 之后的部分，若无则截取末尾 300 字符
    if "Answer:" in predict_str:
        predict_str = predict_str.split("Answer:")[-1]
    else:
        predict_str = predict_str[-300:]
    prompt = DAPO_GENRM_PROMPT_TEMPLATE.format(
        question=question, ground_truth=ground_truth, predict_str=predict_str,
    )
    return [{"role": "user", "content": prompt}]

async def async_compute_score_genrm(args, sample) -> dict:
    genrm_client = get_genrm_client()          # 单例 HTTP 客户端
    question = sample.metadata.get("question", "")
    ground_truth = sample.metadata.get("label", "")
    messages = _format_messages(question, ground_truth, sample.response)

    response = await genrm_client.generate(messages)  # 调用 GenRM 服务
    prediction = response.strip()

    # 严格相等：只有精确的 "1" 才产生正分
    score = 1.0 if prediction == "1" else 0.0
    return {"score": score, "acc": int(score), "pred": prediction}
```

```bash
ROLLOUT_ARGS=(
   --rm-type dapo-genrm        # 使用 GenRM 进行奖励评分
   --reward-key score           # 输出字典中的奖励键
   --n-samples-per-prompt 8     # 每个 prompt 生成 8 个响应
   --rollout-max-response-len 8192
   --rollout-temperature 1
)
```

### 训练配置

两个脚本均使用 GRPO 算法，超参数如下：

```bash
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis                    # 截断重要性采样
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
)
```

### GenRM 服务配置

GenRM 模型和引擎在 `ray job submit` 级别进行配置：

```bash
--genrm-model-path ${MODEL_DIR}/Qwen3-VL-30B-A3B-Instruct/ \
--genrm-num-gpus-per-engine 1 \
--genrm-engine-config '{"max_context_len": 10240}' \
--genrm-sampling-config '{"temperature": 0.1, "top_p": 1.0, "top_k": -1, "max_response_len": 1024}'
```

::: tip 提示
建议为 GenRM 使用较低的温度（如 0.1），以产生确定性的评估结果。较高的温度会引入评估方差。
:::

## 使用示例

### 直接调用 GenRM API

```bash
curl -X POST http://localhost:8000/genrm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Evaluate the answer consistency..."}
    ]
  }'
```

响应：

```json
{
  "response": "1"
}
```

### 在 Python 中使用 GenRMClient

```python
from relax.utils.genrm_client import get_genrm_client

# 获取单例客户端（避免每次请求创建新客户端的开销）
client = get_genrm_client()

# 异步生成
response = await client.generate(
    messages=[{"role": "user", "content": "Evaluate..."}],
    sampling_params={"temperature": 0.2},
)
print(response)  # "1" 或 "0"
```

## 最佳实践

1. **优先使用 colocate 模式**：在 colocate 模式下，GenRM 的 GPU 在不进行评估时会卸载回训练，全部 GPU 都参与梯度计算。比 async 模式的 GPU 利用率更高（async 模式下 GenRM 的 GPU 在训练阶段处于闲置）。
2. **选对 colocate 子模式**：
   - GenRM 较小、能放在部分 GPU 上时（如 4B reward model 用 4 GPU），用 **Split**。
   - GenRM 较大、需要全集群 TP 时（如 30B MoE 用 TP=8），用 **Shared**。Shared 模式还能避免「GenRM 太大装不下 4 GPU、Rollout 又被挤压」的两难。
3. **Shared 模式下谨慎设置 `mem_fraction_static`**：单卡上各引擎之和 ≤ 0.9。常用起点：rollout 0.6、genrm 0.3。
4. **设置合适的上下文长度**：引擎配置中的 `max_context_len` 应能容纳最长的 prompt + 响应组合。
5. **使用低采样温度**：温度 0.1 可产生确定性的评估结果；仅在需要评估多样性时提高。
6. **监控健康状态**：定期检查 `/health` 端点，确保 GenRM 引擎正常运行。
7. **按模型大小分配 GPU**：大型 GenRM 模型（如 30B）建议 shared 模式 + `--genrm-num-gpus-per-engine` 设为整个集群规模。

## 故障排除

### GenRM 未启用

确保设置了 `--genrm-model-path` 参数。只有当该参数不为 `None` 时，GenRM 才会被激活。

### Colocated 模式下资源分配错误

在启用 GenRM 的 colocated 模式下，GPU 分配必须**恰好**满足以下两种之一：

- **Split**：`rollout_num_gpus + genrm_num_gpus == actor_total_gpus`
- **Shared**：`rollout_num_gpus == genrm_num_gpus == actor_total_gpus`

其它组合（例如 `rollout + genrm < actor_total`，或 `rollout < actor_total < rollout + genrm`）会在启动阶段被拒绝。请把 `--resource` 中的 `rollout` / `genrm` 调整到这两种合法布局之一。

### Shared 模式 OOM 或引擎初始化失败

如果 shared 模式启动时 OOM 或 cuda graph capture 失败，降低一个或两个引擎的 `mem_fraction_static`，让单卡之和 ≤ 0.9。对大 MoE GenRM，可能还需要禁用 cuda graph 或减小 `max_context_len`。

### 引擎初始化超时

如果 GenRM 引擎初始化失败：

1. 检查模型路径是否在所有节点上都可以访问
2. 确认有足够的 GPU 显存可用
3. 查看 Ray 日志中 SGLang 引擎的启动错误信息

### GenRM 始终返回 0

DAPO-GenRM 奖励函数使用严格相等来解析响应 — 只有精确的 `"1"` 字符串才会产生正分。如果 GenRM 模型输出了其他内容（如 `"1."`、`"Yes"` 或多行文本），分数将为 0。请验证 GenRM 模型和 prompt 模板能产生干净的 `"1"` / `"0"` 输出。

## 文件结构

```
examples/generate_reward_model/
├── README.md                            # 示例概述
├── run-qwen3-4B-8xgpu-colocated.sh     # Colocate 模式启动脚本
└── run-qwen3-4B-8xgpu-async.sh         # Fully async 模式启动脚本
```

## 延伸阅读

- [架构设计](/zh/guide/architecture) — 了解 Relax 的整体架构
- [全异步训练流水线](/zh/guide/fully-async-training) — Relax 中异步模式的工作原理
- [配置说明](/zh/guide/configuration) — 完整的配置参考
- [GenRM API](/zh/api/genrm) — GenRM 服务的 HTTP API 参考
