# Relax-SDPO 示例

本目录提供 Relax-SDPO 的文本训练示例。SDPO（Self-Distillation from Preference
Optimization）是一种让模型学会「照做成功回答」的训练方法：先让同一个问题生成多个回答，
再用 reward 找出其中的成功回答，把成功回答作为示范注入 teacher prompt，最后让学生模型
通过 on-policy distillation loss 学习修正自己的回答。

这里的所有 launcher 都帮你把整套流程跑通：rollout → reward → feedback → managed
SGLang teacher → `student_topk + JSD` 蒸馏训练，你只需要准备数据和修改环境配置。

> **运行前请先确认当前 pod 上没有其他 Ray 任务。** `env.sh` 会执行 `ray stop`，因此
> source 它会停止当前 pod 上的 Ray 进程。训练机上如果有其他任务正在运行，不要直接执行
> 这些 launcher。

## 这个示例包含什么

- 六个开箱即用的四卡 colocate 训练脚本（SciKnowEval × 4、ToolUse、ToolAlpaca）
- 数据转换工具 `prepare_data.py`：把参考数据转成 Relax 的 `prompt`/`label`/`metadata`
  JSONL schema
- 规则 reward 与 SDPO feedback 实现（`reward.py` 与 `relax.utils.opd.sdpo.feedback`）
- 由 Relax 自动管理的 SGLang teacher，无需手动部署

## 前置条件

开始之前请确认以下内容都已就绪：

- 一台可用 **4×GPU** 的机器 / pod，且当前没有其他 Ray 任务在运行
- Relax worktree，以及配套的 Relax-SDPO Python 环境（见下方 `env.sh` 的 `RELAX_VENV`）
- Qwen3-8B checkpoint（student 与 teacher 默认共用同一个）
- SDPO 参考数据（SciKnowEval / ToolUse / ToolAlpaca，或你自己的文本数据）
- SGLang 已应用 per-position token-id patch（launcher 会设置
  `RELAX_OPD_PER_POS_TOKEN_IDS=1`），详见
  [通用 OPD 文档中的 SGLang Patch 说明](../README.md#sglang-patch)

## 快速开始

```bash
# 1. 进入项目根目录，确认环境可用（无残留 Ray 任务）
cd <relax-worktree>
ray status          # 或检查当前 pod 的 tmux，确保没有其他任务
nvidia-smi          # 确认四张 GPU 空闲

# 2. 准备数据：以 SciKnowEval Chemistry 为例
python3 -m examples.on_policy_distillation.sdpo.prepare_data \
  --dataset sciknoweval \
  --input <sdpo-source-root>/datasets/sciknoweval/chemistry/train.json \
  --domain chemistry \
  --source-split train \
  --output <data-root>/SDPO/sciknoweval/chemistry/train.jsonl

# 3. 修改 examples/on_policy_distillation/sdpo/env.sh：
#    更新 RELAX_VENV / MEGATRON / STUDENT_MODEL_PATH / TEACHER_MODEL_PATH / SDPO_DATA_ROOT

# 4. 启动训练
bash examples/on_policy_distillation/sdpo/run-sciknoweval-chemistry-4xgpu-colocate.sh

# 5. 观察日志与指标（TensorBoard 曲线、训练 loss / reward / teacher 状态等）
```

首次验证建议先跑小规模 smoke：`prepare_data.py` 加 `--max-rows 2` 只转两条数据，并临时调小
launcher 的 `--num-rollout` / `--rollout-batch-size` 后启动。

## 训练流程

每个 launcher 的一次训练迭代大致经过以下阶段：

```text
prompt-data
    │
    ├── 学生 rollout：同一问题生成多个 response，组成一个 group
    │
    ├── custom reward：计算 score，并记录 feedback
    │
    ├── SDPO feedback：构造每个样本的 teacher prompt
    │       ├── SciKnowEval：共享同 group/UID 内的成功回答，并加入当前样本反馈
    │       └── ToolUse/Code：同样共享同 group/UID 内的成功回答，并加入当前样本反馈
    │
    ├── managed SGLang teacher：对动态 teacher prompt 计算 top-K log-probability
    │
    └── student_topk + JSD loss：更新学生模型
```

当前脚本使用以下关键配置：

| 配置                                         | 当前值         | 说明                                                                           |
| -------------------------------------------- | -------------- | ------------------------------------------------------------------------------ |
| `--use-opd`                                  | 开启           | 启用 on-policy distillation                                                    |
| `--opd-type`                                 | `sglang`       | teacher 由 Relax 管理的 SGLang 服务提供 log-probability                        |
| `--opd-token-selection`                      | `student_topk` | 在学生 rollout 的 top-K token 集合上计算 SDPO 信号                             |
| `--opd-log-prob-top-k`                       | `16`           | 每个位置收集 16 个 token 的 log-probability                                    |
| `--opd-kl-type`                              | `jsd`          | 使用 JSD 形式的 token-level distillation criterion                             |
| `--opd-norm-mode`                            | `tail`         | 保留 top-K 之外的 tail probability mass                                        |
| `--opd-loss-coef`                            | `1.0`          | 将 distillation signal 作为 loss 注入训练                                      |
| `--opd-kl-coef`                              | `0.0`          | 不使用 advantage 形式的 OPD KL                                                 |
| `--opd-disable-rl-reward`                    | 开启           | 不把基础 RL outcome reward 注入 actor 优化；custom reward 仍用于 SDPO feedback |
| `--group-rm`                                 | 开启           | 让同一个 prompt 的多个 rollout 进入同一 reward group                           |
| `--use-rollout-logprobs`                     | 开启           | 复用学生 rollout 阶段的 log-probability 数据                                   |
| `--colocate`                                 | 开启           | 在 rollout、teacher 和 actor 之间切换共享 GPU 资源                             |
| `--teacher-sglang-enable-weights-cpu-backup` | 开启           | colocate sleep/wake 时把 teacher 权重备份到 CPU，避免唤醒后权重丢失            |

`student_topk` 模式需要 SGLang 支持按位置返回 token ID。launcher 会设置
`RELAX_OPD_PER_POS_TOKEN_IDS=1`；运行环境还必须安装对应的 SGLang source patch。详见
[通用 OPD 文档中的 SGLang Patch 说明](../README.md#sglang-patch)。

## Launcher 与资源布局

所有当前 launcher 都是单机四卡 colocate 配置：

```text
四张 GPU 的 colocate resource pool
├── actor   ：4 GPU（TP=4），训练阶段使用整个 pool
├── rollout ：3 GPU，rollout 阶段使用
└── teacher ：1 GPU，rollout 阶段使用 managed SGLang teacher
```

脚本中的资源配置为：

```json
{"actor": [1, 4], "rollout": [1, 3], "teacher": [1, 1]}
```

六个 launcher 的训练参数完全一致：`--num-rollout 5000`、`--rollout-batch-size 32`、
`--n-samples-per-prompt 8`、`--global-batch-size 256`、`--eval-interval 5`、
`--n-samples-per-eval-prompt 16`。

| 脚本                                                                                         | 数据入口                            | eval 入口                          | Feedback 类                 |
| -------------------------------------------------------------------------------------------- | ----------------------------------- | ---------------------------------- | --------------------------- |
| [`run-sciknoweval-biology-4xgpu-colocate.sh`](run-sciknoweval-biology-4xgpu-colocate.sh)     | `sciknoweval/biology/train.jsonl`   | `sciknoweval/biology/eval.jsonl`   | `GoldenAnswerSDPOFeedback` |
| [`run-sciknoweval-chemistry-4xgpu-colocate.sh`](run-sciknoweval-chemistry-4xgpu-colocate.sh) | `sciknoweval/chemistry/train.jsonl` | `sciknoweval/chemistry/eval.jsonl` | `GoldenAnswerSDPOFeedback` |
| [`run-sciknoweval-physics-4xgpu-colocate.sh`](run-sciknoweval-physics-4xgpu-colocate.sh)     | `sciknoweval/physics/train.jsonl`   | `sciknoweval/physics/eval.jsonl`   | `GoldenAnswerSDPOFeedback` |
| [`run-sciknoweval-material-4xgpu-colocate.sh`](run-sciknoweval-material-4xgpu-colocate.sh)   | `sciknoweval/material/train.jsonl`  | `sciknoweval/material/eval.jsonl`  | `GoldenAnswerSDPOFeedback` |
| [`run-tooluse-4xgpu-colocate.sh`](run-tooluse-4xgpu-colocate.sh)                             | `tooluse/train.jsonl`               | `tooluse/eval.jsonl`               | `GoldenAnswerSDPOFeedback` |
| [`run-toolalpaca-4xgpu-colocate.sh`](run-toolalpaca-4xgpu-colocate.sh)                       | `toolalpaca/train.jsonl`            | `toolalpaca/eval.jsonl`            | `GoldenAnswerSDPOFeedback` |

当前脚本没有独立的公共 SDPO launcher，每个数据入口都显式指定了自己的 feedback 类。
student rollout 统一为 `--rollout-max-response-len 8192`；teacher 请求超时由各 launcher 的
`--opd-teacher-timeout-s` 控制（launcher 显式设置为 600 s），需要时可自行调整。

## 文件结构

| 文件                                 | 用途                                                                            |
| ------------------------------------ | ------------------------------------------------------------------------------- |
| [`env.sh`](env.sh)                   | 设置项目根目录、Python/Megatron 环境、模型路径和数据根目录，并停止当前 Ray 进程 |
| [`prepare_data.py`](prepare_data.py) | 将参考数据转换为 Relax 的 `prompt`/`label`/`metadata` JSONL schema              |
| [`reward.py`](reward.py)             | 提供 SciKnowEval、ToolUse 和 ToolAlpaca 的 rule-based reward                    |
| `run-*-4xgpu-colocate.sh`            | 按数据集启动四卡 colocate SDPO 训练                                             |

## 环境准备

### 模型

当前脚本通过 `scripts/models/qwen3-8B.sh` 加载 Qwen3-8B 的学生模型配置，
并默认使用同一 Qwen3-8B checkpoint 作为 student 和 teacher：

```text
<model-root>/Qwen3-8B  # student
<model-root>/Qwen3-8B  # teacher
```

也可以使用不同的文本 teacher checkpoint，但必须确保 checkpoint 能被当前 managed
SGLang teacher 和 Qwen3-8B 训练配置正确加载。当前 SDPO prompt-routing 路径只支持文本
输入，不支持多模态字段。

### `env.sh`

launcher 会自行回到项目根目录，并在内部 source `examples/on_policy_distillation/sdpo/env.sh`。
请先根据当前机器修改其中的环境路径和 checkpoint 路径：

| 变量                 | 用途                                                         |
| -------------------- | ------------------------------------------------------------ |
| `RELAX_VENV`         | Relax-SDPO Python 虚拟环境                                   |
| `RELAX_PYTHON`       | 训练入口使用的 Python                                        |
| `MEGATRON`           | 与当前 Relax worktree 配套的 Megatron 和 Python package 路径 |
| `PYTHONPATH`         | 由项目根目录和 `MEGATRON` 路径组成                           |
| `STUDENT_MODEL_PATH` | 学生模型 HF checkpoint                                       |
| `TEACHER_MODEL_PATH` | managed SGLang teacher 的 HF checkpoint                      |
| `SDPO_DATA_ROOT`     | 准备好的 SDPO JSONL 数据根目录                               |

`env.sh` 当前对上述变量使用固定默认值，而不是 `${VAR:-default}` 形式。因此，在命令行
预先 export `STUDENT_MODEL_PATH` 或 `TEACHER_MODEL_PATH` 会被 `env.sh` 中的赋值覆盖；如果
需要更换模型或运行环境，应直接修改 `env.sh`，或维护一份本地 launcher/environment 副本。

## 数据准备

从 Relax 项目根目录执行 `python3 -m examples.on_policy_distillation.sdpo.prepare_data`。
输入可以是 JSON、JSONL 或 Parquet，输出统一为 JSONL。

### 输出 schema

每一行至少包含 `prompt`、`label` 和 `metadata`：

```json
{
  "prompt": "question and optional choices",
  "label": "gold answer",
  "metadata": {
    "data_source": "sciknoweval",
    "source_split": "train",
    "domain": "Chemistry",
    "task_type": "mcq"
  }
}
```

`metadata.data_source` 是 reward 路由键；`metadata` 中的 `answer_key`、`golden_answer` 等
字段由 reward 使用。不要在 launcher 中把 `--metadata-key metadata` 改成其他字段，除非
同时修改数据输出 schema 和 reward 逻辑。

### SciKnowEval

数据通常按 domain 保存。以下命令以 Chemistry 为例；Physics、
Biology 和 Materials 只需要替换 domain、输入路径和输出路径：

```bash
python3 -m examples.on_policy_distillation.sdpo.prepare_data \
  --dataset sciknoweval \
  --input <sdpo-source-root>/datasets/sciknoweval/chemistry/train.json \
  --domain chemistry \
  --source-split train \
  --output <data-root>/SDPO/sciknoweval/chemistry/train.jsonl
```

也可以处理已经整理成扁平 schema 的参考数据。对于原始 SciKnowEval 格式，转换器会只保留
L3 样本，并将 `material` 规范化为 `Materials`：

```bash
python3 -m examples.on_policy_distillation.sdpo.prepare_data \
  --dataset sciknoweval \
  --input <sciknoweval-root>/chemistry/train.json \
  --source-split train \
  --output <data-root>/SDPO/sciknoweval/chemistry/train.jsonl
```

测试 split 可以用同样的命令生成，只需将输入和 `--source-split` 改为 `test`。当前训练
launcher 默认每 5 个训练迭代使用 `<dataset>/eval.jsonl` 做一次周期性评测
（`--eval-prompt-data`）。

如果只有 train 源数据、没有独立的 test 集，可以在生成时按比例留出 eval 集：
`--eval-ratio` 会把该比例的规范化行写到 `<output>.parent/eval.jsonl`，其余写
到 `--output`（train）。launcher 通过 `--eval-prompt-data` 指向这个 `eval.jsonl`
即可周期性评测：

```bash
python3 -m examples.on_policy_distillation.sdpo.prepare_data \
  --dataset sciknoweval \
  --input <sdpo-source-root>/datasets/sciknoweval/biology/train.json \
  --domain biology \
  --source-split train \
  --eval-ratio 0.1 \
  --seed 42 \
  --output <data-root>/SDPO/sciknoweval/biology/train.jsonl
# 生成 <data-root>/SDPO/sciknoweval/biology/train.jsonl + eval.jsonl
```

### ToolUse

如果使用参考 SDPO 仓库中的工具调用数据：

```bash
python3 -m examples.on_policy_distillation.sdpo.prepare_data \
  --dataset tooluse \
  --input <sdpo-source-root>/datasets/tooluse/train.json \
  --source-split train \
  --output <data-root>/SDPO/tooluse/train.jsonl
```

### ToolAlpaca

ToolAlpaca 输入通常是 Parquet：

```bash
python3 -m examples.on_policy_distillation.sdpo.prepare_data \
  --dataset toolalpaca \
  --input <toolalpaca-root>/data/train-00000-of-00001.parquet \
  --source-split train \
  --output <data-root>/SDPO/toolalpaca/train.jsonl
```

读取 Parquet 需要当前 Python 环境安装 `pyarrow`。

## 启动训练

### 使用 `SDPO_DATA_ROOT` 默认路径

如果 `env.sh` 中的 `SDPO_DATA_ROOT` 已经包含以下目录之一，可以直接启动：

```text
$SDPO_DATA_ROOT/
├── sciknoweval/
│   ├── biology/train.jsonl
│   ├── chemistry/train.jsonl
│   ├── material/train.jsonl
│   └── physics/train.jsonl
├── toolalpaca/train.jsonl
└── tooluse/train.jsonl
```

例如：

```bash
bash examples/on_policy_distillation/sdpo/run-sciknoweval-chemistry-4xgpu-colocate.sh
```

### 使用自定义数据路径

launcher 会优先读取 `DATA_PATH` 环境变量，未设置时才回退到 `SDPO_DATA_ROOT` 下的默认目录，
因此也可以直接指定你自己的数据文件：

```bash
DATA_PATH=/path/to/my/train.jsonl \
bash examples/on_policy_distillation/sdpo/run-sciknoweval-chemistry-4xgpu-colocate.sh
```

ToolAlpaca 和 ToolUse 的启动方式相同，只需替换 launcher：

```bash
DATA_PATH=<data-root>/SDPO/toolalpaca/train.jsonl \
bash examples/on_policy_distillation/sdpo/run-toolalpaca-4xgpu-colocate.sh

DATA_PATH=<data-root>/SDPO/tooluse/train.jsonl \
bash examples/on_policy_distillation/sdpo/run-tooluse-4xgpu-colocate.sh
```

训练开始前，Relax 会启动 managed SGLang teacher，并在 actor、rollout 和 teacher 之间按
colocate 配置切换 GPU。launcher 默认每 5 个训练迭代在 `<dataset>/eval.jsonl` 上评测一次，
并在第一个训练迭代前先跑一次 eval 作为基线。

## Reward 与 Feedback

### SciKnowEval

`reward.py` 要求回答包含 `<answer>...</answer>` 标签并从中提取选项字母；缺失 `<answer>`
标签即视为格式错误（score=0，即使回答里出现了正确选项）。普通选择题会和
`metadata.answer_key` 比较，true/false 任务会进行归一化比较。

三类数据（SciKnowEval、ToolUse/ToolAlpaca、Code）按同一套决策矩阵决定是否进蒸馏、
注入什么（按优先级）；前两者共用 `GoldenAnswerSDPOFeedback`（静态 golden-answer 文本
任务），Code 预留 `CodeSDPOFeedback` 占位（reward 未接入，调用即报错）：

1. 自身成功（`score >= success_reward_threshold`，默认 1.0，可经 `--opd-feedback-kwargs` 调整）→ 注入同 `group_index`/`metadata.uid` 内成功 peer 的正确解，
   无 peer 时用自己的解，进蒸馏；
2. 失败但有成功 peer → 只注入 peer 的正确解（丢弃当前样本的 feedback），进蒸馏；
3. 失败、无 peer、且是格式/截断错误 → 注入格式/截断反馈文字，进蒸馏；
4. 失败、无 peer、普通算错 → 不注入、不进蒸馏。

没有 `group_index` 时使用 `metadata.uid` 隔离；不同 group/UID 之间不会共享回答。

失败样本的反馈文字只有两种（截断优先于格式错误），且都不泄露正确答案：

```text
Your response was truncated because it exceeded the maximum length.
Your answer had the wrong format. The solution must be given in the format: <answer>X</answer>.
```

普通算错不产生任何反馈文字，因此这类样本只能靠同题的 peer 正确解学习。

### ToolUse 与 ToolAlpaca

模型回答需要包含：

```text
Action: <tool name>
Action Input: <JSON object>
```

reward 分别检查 tool action 和 JSON 参数：缺失 `Action/Action Input` 格式 → 格式反馈；
回答被截断 → 截断反馈（优先于格式）；action/input 不匹配视为普通算错 → 无反馈（不泄露
gold）。进蒸馏的决策与 SciKnowEval 相同：成功/有成功 peer 时注入 peer 正确解；失败且
无 peer 时只有格式/截断错误才注入反馈文字。

## 常见问题

### 提示 `Set STUDENT_MODEL_PATH` 或 `Set SDPO_DATA_ROOT`

检查 `env.sh` 中的模型、Python/Megatron 和 `SDPO_DATA_ROOT` 配置。若使用自定义数据，
可以直接设置 `DATA_PATH`（见上方「使用自定义数据路径」），这样 launcher 不需要依赖
`SDPO_DATA_ROOT` 对应的默认目录。

### 提示 `No rows matched`

检查 `--dataset` 是否与输入格式匹配。SciKnowEval 原始格式还需要有效的 L3 domain；
ToolAlpaca 输入必须包含 `golden_answer`；ToolUse 输入必须包含参考格式中的 `prompt` 和
`answer`。

### teacher 请求超时或显存不足

检查 teacher/rollout 是否确实各分配一张 GPU，并确认 `--colocate`
和 `--teacher-sglang-enable-weights-cpu-backup` 没有被删除。
所有 launcher 的默认 rollout 规模较大（5000 prompts / global-batch 256）；首次验证建议
先生成小规模 smoke 数据再训练：

```bash
python3 -m examples.on_policy_distillation.sdpo.prepare_data \
  --dataset sciknoweval \
  --input <sdpo-source-root>/datasets/sciknoweval/chemistry/train.json \
  --domain chemistry \
  --source-split train \
  --max-rows 2 \
  --output <data-root>/SDPO/sciknoweval/chemistry/train-smoke.jsonl
```

必要时应在对应 launcher 中调整 rollout 数量、response 长度或 batch 配置。

### Top-K log-probability 不可用

确认 SGLang 已应用 [通用 OPD 文档中的 per-position token-id patch](../README.md#sglang-patch)，
并保留 `RELAX_OPD_PER_POS_TOKEN_IDS=1`。当前 SDPO 路径不能退回到
`student_sampled`，因为 SDPO prompt routing 只支持 `student_topk`。

## 参考

- [On-Policy Distillation 通用说明](../README.md)
- [通用 OPD 的 token selection、loss 和 SGLang 配置](../README.md#token-selection-modes)
- [lasgroup/SDPO](https://github.com/lasgroup/SDPO)
- [SciKnowEval](https://github.com/HICAI-ZJU/SciKnowEval)
- [ToolAlpaca](https://huggingface.co/datasets/Ahren09/ToolAlpaca)
