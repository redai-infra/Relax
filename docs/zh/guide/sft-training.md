# SFT 训练

本指南展示 Relax 中监督微调（SFT）的完整流程，并以当前 [`scripts/training/sft/`](../../../scripts/training/sft/) 下的启动脚本为准，覆盖 math 和 Pokemon 数据准备、模型与路径配置、启动命令以及常用调参方法。

开始之前，请先完成[安装](./installation.md)。

## 概述

SFT 通过 `--loss-type sft` 启用。该模式下，Relax 会启动一个 SFT producer，从 `--prompt-data` 读取数据，用模型的 chat template 渲染样本，将 packed 样本写入 TransferQueue，然后由 Megatron Actor 训练。如果设置了 `--eval-interval`，还会在评估集上运行 PPL 评估。如果设置了 `--sft-predict-interval`，Relax 会额外使用 Rollout 角色和 SGLang 周期性做生成式 predict。

当前启动脚本使用 `ray job submit`，并且在没有外部 entrypoint 预先准备 Ray 环境时自动 source [`scripts/entrypoint/local.sh`](../../../scripts/entrypoint/local.sh)。

## 脚本

| 脚本 | 数据集 | 模型 | 默认资源 | 说明 |
| --- | --- | --- | --- | --- |
| [`run-qwen3.5-9B-math-8xgpu.sh`](../../../scripts/training/sft/run-qwen3.5-9B-math-8xgpu.sh) | `OpenMathReasoning-mini` | `Qwen3.5-9B` | 8 GPU Actor，加 SFT producer 和 Rollout | 文本 SFT，使用 `problem` -> `generated_solution`，开启 PPL eval 和 predict。 |
| [`run-qwen3-vl-4B-math-8xgpu.sh`](../../../scripts/training/sft/run-qwen3-vl-4B-math-8xgpu.sh) | `OpenMathReasoning-mini` | `Qwen3-VL-4B-Instruct` | 8 GPU Actor，加 SFT producer 和 Rollout | 使用 VL checkpoint 训练纯文本 math 数据。 |
| [`run-qwen3-vl-4B-pokemon-8xgpu.sh`](../../../scripts/training/sft/run-qwen3-vl-4B-pokemon-8xgpu.sh) | `pokemon-gpt4o-captions` | `Qwen3-VL-4B-Instruct` | 8 GPU Actor，加 SFT producer 和 Rollout | 图像多模态 SFT，使用两个 parquet 文件，并开启预取。 |
| [`run-qwen3-vl-4B-pokemon-1xgpu.sh`](../../../scripts/training/sft/run-qwen3-vl-4B-pokemon-1xgpu.sh) | `pokemon-gpt4o-captions` | `Qwen3-VL-4B-Instruct` | 1 GPU Actor，加 SFT producer | 低资源 Pokemon SFT，开启 CPU optimizer offload。 |
| [`run-qwen3.5-35B-A3B-mtp-sft-16xgpu.sh`](../../../scripts/training/sft/run-qwen3.5-35B-A3B-mtp-sft-16xgpu.sh) | `OpenMathReasoning-mini` | `Qwen3.5-35B-A3B` | 16 GPU Actor，加 SFT producer | 进阶 MTP SFT。多个参数已暴露为环境变量。 |

## 数据准备

SFT 脚本默认使用 `DATA_DIR=${SCRIPT_DIR}/data`，除非显式覆盖 `DATA_DIR`。对于可复用训练任务，建议把数据放在持久化存储中，并显式设置 `DATA_DIR`。

```bash
cd /root/Relax
export DATA_DIR=/root
mkdir -p "${DATA_DIR}/sft/data"
```

### Math: OpenMathReasoning-mini

math 脚本期望以下文件：

```text
${DATA_DIR}/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet
```

从 [unsloth/OpenMathReasoning-mini](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini) 下载：

```bash
hf download --repo-type dataset unsloth/OpenMathReasoning-mini \
  data/cot-00000-of-00001.parquet \
  --local-dir "${DATA_DIR}/sft/data/OpenMathReasoning-mini"
```

当前 math 脚本配置为：

```bash
--prompt-data "${DATA_DIR}/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet"
--input-key problem
--label-key generated_solution
```

设置 `--label-key` 后，每行数据必须在 `problem` 字段中提供 prompt 字符串，并在 `generated_solution` 字段中提供目标回复。Relax 会用 `problem` 构造 user message，并把 `generated_solution` 追加为 assistant message。

最小行结构：

```json
{
  "problem": "Find the value of x.",
  "generated_solution": "We solve the equation step by step..."
}
```

### Pokemon: pokemon-gpt4o-captions

Pokemon 脚本期望以下文件：

```text
${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_en.parquet
${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_zh.parquet
```

从 [llamafactory/pokemon-gpt4o-captions](https://huggingface.co/datasets/llamafactory/pokemon-gpt4o-captions) 下载：

```bash
hf download --repo-type dataset llamafactory/pokemon-gpt4o-captions \
  pokemon_gpt4o_en.parquet pokemon_gpt4o_zh.parquet \
  --local-dir "${DATA_DIR}/sft/data/pokemon-gpt4o-captions"
```

当前 Pokemon 脚本会从两个文件构造 list 形式的 `PROMPT_DATA`：

```bash
TRAIN_FILES=(
    "'${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_en.parquet'"
    "'${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_zh.parquet'"
)
PROMPT_DATA="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"
```

然后配置：

```bash
--prompt-data "${PROMPT_DATA}"
--input-key conversations
--multimodal-keys '{"image":"images"}'
--conversation-key-map '{"from":"role","value":"content","human":"user","gpt":"assistant"}'
```

由于没有设置 `--label-key`，`conversations` 必须是一列完整的消息列表。`--conversation-key-map` 会把 ShareGPT 风格的消息字段和值改写为 SFT 期望的 OpenAI 风格 `{role, content}`。`--multimodal-keys` 告诉 Relax 从每行的 `images` 字段加载图片路径。文本中每个图片条目都应该有对应的 `<image>` 占位符。

最小行结构：

```json
{
  "conversations": [
    {"from": "human", "value": "Identify the object of this image.<image>"},
    {"from": "gpt", "value": "A round, pink Pokemon with a gentle expression."}
  ],
  "images": ["/path/to/pokemon.png"]
}
```

### 验证数据文件

启动前运行下面的检查，提前发现缺文件或列名不匹配：

```bash
python - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq

checks = {
    "math": (
        Path("/root/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet"),
        {"problem", "generated_solution"},
    ),
    "pokemon_en": (
        Path("/root/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_en.parquet"),
        {"conversations", "images"},
    ),
    "pokemon_zh": (
        Path("/root/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_zh.parquet"),
        {"conversations", "images"},
    ),
}

for name, (path, required) in checks.items():
    if not path.exists():
        print(f"{name}: missing {path}")
        continue
    names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = required - names
    print(f"{name}: rows={pq.ParquetFile(path).metadata.num_rows}, missing={sorted(missing)}")
PY
```

如果你使用的不是 `/root` 作为 `DATA_DIR`，请同步修改检查脚本中的路径。

## 模型准备

将模型权重下载到持久化存储。math 和 Pokemon 脚本从 `MODEL_DIR` 读取模型路径，而不是从 `EXP_DIR` 读取，因此这几份脚本建议显式设置 `MODEL_DIR`。

```bash
# Math 脚本：run-qwen3.5-9B-math-8xgpu.sh
hf download Qwen/Qwen3.5-9B --local-dir /root/Qwen3.5-9B

# Pokemon 脚本：run-qwen3-vl-4B-pokemon-*.sh
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /root/Qwen3-VL-4B-Instruct
```

对于 MTP 脚本，默认约定是 `EXP_DIR=/root`，然后 `MODEL_DIR=${EXP_DIR}`、`DATA_DIR=${EXP_DIR}`，除非你单独覆盖它们。

## 配置详解

启动脚本按参数块组织。调参时可以直接修改选中的脚本；如果脚本已经暴露了环境变量，也可以通过环境变量覆盖。

### 路径与 checkpoint

对于 math 和 Pokemon 脚本：

```bash
export MODEL_DIR=/root
export DATA_DIR=/root
```

`CKPT_ARGS` 参数块设置：

| 参数 | 作用 |
| --- | --- |
| `--hf-checkpoint` | HF checkpoint，用于 tokenizer、config，以及开启 predict 时初始化 SGLang。 |
| `--ref-load` | 初始参考 checkpoint 路径。 |
| `--load` | 训练加载路径。若这是已有 Megatron checkpoint，则从它恢复训练；否则 bridge 模式从 HF 权重开始。 |
| `--megatron-to-hf-mode bridge` | 使用 Megatron Bridge 完成 HF <-> Megatron 权重转换。 |
| `--save` | Megatron checkpoint 输出目录。 |
| `--save-interval` | 每 N 个训练 step 保存一次。 |
| `--num-epoch` | 数据集 epoch 数。Relax 会根据数据集大小和 global batch size 推导实际训练步数。 |

新训练时，可以保持当前脚本里 `--load` 和 `--save` 指向同一个实验目录。断点续训时，保持同一个 `--save`，并确保目录中存在有效 checkpoint 和 `latest_checkpointed_iteration.txt`。

### MTP 参数

MTP SFT 脚本开启 `--mtp-num-layers ${MTP_NUM_LAYERS:-1}`、`--enable-mtp-training` 和 `--mtp-loss-scaling-factor ${MTP_LOSS_SCALING_FACTOR:-0.2}`。只有模型和 checkpoint 确实包含对应 MTP 层时才调大 `MTP_NUM_LAYERS`；`MTP_LOSS_SCALING_FACTOR` 是辅助 loss 权重，建议从 `0.2` 起调。

### SFT 数据参数

SFT 必需参数：

```bash
--loss-type sft
--prompt-data "${PROMPT_DATA}"
--use-dynamic-batch-size
--max-tokens-per-gpu <tokens>
```

SFT 必须使用动态 batching。缺少 `--use-dynamic-batch-size` 时，参数校验会失败。如果没有设置 `--balance-data`，SFT 校验会自动开启它，以保持 SFT producer 和 Megatron 数据路径一致。

数据行格式二选一：

| 格式 | 参数 | 适用场景 |
| --- | --- | --- |
| Prompt 加 label | `--input-key problem --label-key generated_solution` | prompt 和监督答案分别在两列的文本数据。 |
| 完整 messages | `--input-key conversations`，且不设置 `--label-key` | 已包含 user 和 assistant 轮次的对话或多模态数据。ShareGPT 风格字段需加 `--conversation-key-map`。 |

SFT 不需要额外添加 `--apply-chat-template`。SFT 数据集会在内部使用 tokenizer chat template 渲染样本，并从渲染结果构造 assistant loss mask。

### 评估与 predict

PPL 评估由以下参数控制：

```bash
--eval-size 0.01
--eval-interval 10
```

`--eval-size` 会从 `--prompt-data` 末尾切出 eval 集，并从训练池移除。小于 1 的值表示比例；10 或更大的值可视为绝对样本数。也可以使用 `--eval-prompt-data name path` 提供独立评估集，但 SFT 模式下不要使用 `--eval-config`。

生成式 predict 由以下参数控制：

```bash
--sft-predict-interval 10
--eval-temperature 0.0
--eval-max-response-len 10240
```

设置 `--sft-predict-interval` 后，Relax 会自动拉起 Rollout 角色，并把预测结果写到：

```text
<save>/predict/predictions_step_<rollout_id>.jsonl
```

math 推理任务通常需要更长的 `--eval-max-response-len`；caption 或图像描述任务可以设短一些。

### 并行与资源

脚本中的 `PERF_ARGS` 和 `--resource` 需要与集群 GPU 数一致。

Math 8 GPU 脚本：

```bash
--tensor-model-parallel-size 4
--pipeline-model-parallel-size 2
--context-parallel-size 1
--resource '{"sft": [1, 0], "actor": [1, 8], "rollout": [1, 8]}'
```

Pokemon 8 GPU 脚本：

```bash
--tensor-model-parallel-size 2
--pipeline-model-parallel-size 1
--context-parallel-size 1
--per-rank-fetch
--num-data-storage-units 8
--resource '{"sft": [1, 0], "actor": [1, 8], "rollout": [1, 8]}'
```

Pokemon 1 GPU 脚本：

```bash
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
--resource '{"sft": [1, 0], "actor": [1, 1], "rollout": [1, 1]}'
```

`"sft": [1, 0]` 表示 SFT producer 是 CPU-only。Actor 使用训练 GPU。开启周期性 predict 时，Rollout 也需要 GPU 资源。

## 启动

### 单机

直接运行脚本即可。脚本会 source `local.sh`、启动本地 Ray head node，并提交 Ray job：

```bash
cd /root/Relax
export MODEL_DIR=/root
export DATA_DIR=/root

bash scripts/training/sft/run-qwen3.5-9B-math-8xgpu.sh
bash scripts/training/sft/run-qwen3-vl-4B-pokemon-8xgpu.sh
```

运行 1 GPU Pokemon 脚本：

```bash
cd /root/Relax
export MODEL_DIR=/root
export DATA_DIR=/root
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1

bash scripts/training/sft/run-qwen3-vl-4B-pokemon-1xgpu.sh
```

### 已有 Ray 集群

如果 Ray 集群已经启动，使用 [`scripts/entrypoint/ray-job.sh`](../../../scripts/entrypoint/ray-job.sh)。它会准备运行环境，不会停止 Ray，然后委托给训练脚本：

```bash
cd /root/Relax
export MODEL_DIR=/root
export DATA_DIR=/root
export RAY_ADDRESS=http://127.0.0.1:8265

bash scripts/entrypoint/ray-job.sh scripts/training/sft/run-qwen3-vl-4B-pokemon-8xgpu.sh
```

### 多机

使用 [`scripts/entrypoint/spmd-multinode.sh`](../../../scripts/entrypoint/spmd-multinode.sh)。必需环境变量为 `MASTER_ADDR`、`POD_NAME`、`HOST_IP`、`WORLD_SIZE`；`NUM_GPUS` 默认每节点 8 张。

```bash
cd /root/Relax
export MODEL_DIR=/root
export DATA_DIR=/root
export WORLD_SIZE=2
export NUM_GPUS=8

bash scripts/entrypoint/spmd-multinode.sh \
  scripts/training/sft/run-qwen3.5-35B-A3B-mtp-sft-16xgpu.sh
```

## 调参流程

建议按下面顺序调参，让每次改动都有明确目标。

### 1. 先让显存放得下

如果训练阶段 OOM：

| 参数 | 调整方向 | 作用 |
| --- | --- | --- |
| `--max-tokens-per-gpu` | 调小 | 降低动态 micro-batch 的 token 容量。这是 SFT 首选显存参数。 |
| `--global-batch-size` | 调小 | 降低每次 optimizer update 的样本数，但会改变优化动态。 |
| `--recompute-num-layers` | 调大 | 用更多计算换 activation 显存。 |
| `--optimizer-cpu-offload` | 开启 | 节省 GPU 显存，适合 1 GPU 或显存紧张的 VL 训练。 |
| `--sft-oversize-strategy skip` | 谨慎开启 | 丢弃超过 `max_tokens_per_gpu * context_parallel_size` 的样本。 |

使用 context parallelism 时，容量为 `--max-tokens-per-gpu * --context-parallel-size`。只有模型和 Megatron 配置支持时才建议增加 CP。

### 2. 提升吞吐

如果 GPU 在等 SFT 数据：

| 参数 | 调整方向 | 作用 |
| --- | --- | --- |
| `--sft-prefetch-buffer-size` | 从 256 往上调 | 缓存更多已渲染样本。 |
| `--sft-prefetch-num-workers` | 调大 | 提高图片解码和多模态 I/O 并行度。 |
| `--sft-prefetch-chunk-size` | 调大 | 一次派发更多预取样本，但会增加内存压力。 |
| `--per-rank-fetch` | 多 GPU 时开启 | 让 TP/PP rank 直接从 TransferQueue 拉数据，需配足 `--num-data-storage-units`。 |
| `--max-staleness` | I/O 重时调大 | 允许 producer 提前生产。Pokemon 8 GPU 脚本使用 `--max-staleness 4`。 |

纯文本 math 任务通常更受序列长度和模型并行影响；Pokemon 任务更容易被图片读取和 processor 工作拖慢。

### 3. 保持效果稳定

先使用脚本默认值，再一次只改一个优化参数：

| 参数 | Math 默认值 | Pokemon 8 GPU 默认值 | 说明 |
| --- | --- | --- | --- |
| `--lr` | `1e-5` | `1e-5` | eval loss 快速上升或从强 checkpoint 恢复时可调低。 |
| `--lr-decay-style` | `cosine` | `cosine` | 1 GPU Pokemon 脚本使用 `constant` 和 `3e-5`，更激进。 |
| `--weight-decay` | `0.1` | `0.1` | 没有系统 sweep 时建议保持不变。 |
| `--clip-grad` | `1.0` | `1.0` | 只有梯度尖峰明显时再调低。 |
| `--num-epoch` | `10` | `10` | smoke test 可调小；想调大必须配合 eval 监控。 |

### 4. 控制评估成本

| 参数 | 调整方向 | 作用 |
| --- | --- | --- |
| `--eval-size` | 调小 | 减少 holdout 数据和 PPL eval 成本。 |
| `--eval-interval` | 调大 | 减少评估轮次。 |
| `--sft-predict-interval` | 调大或关闭 | 减少 SGLang predict 开销。 |
| `--eval-max-response-len` | 调小 | 限制 predict 生成成本。 |

快速 bring-up 时，可以先注释掉 `PREDICT_ARGS`，并使用较小 `--eval-size`。训练循环稳定后再恢复 predict。

## 故障排除

### `--loss-type sft requires --use-dynamic-batch-size`

SFT 会拒绝静态 micro-batch。添加 `--use-dynamic-batch-size`，并设置 `--max-tokens-per-gpu`。

### `Under --loss-type sft with --eval-interval set...`

`--eval-interval` 需要且只能有一个 eval 来源：`--eval-size` 或 `--eval-prompt-data name path`。不要同时设置两者。

### `SFT row missing prompt key`

`--input-key` 和数据列名不匹配。math 使用 `problem`；Pokemon 使用 `conversations`。

### `--label-key is not set ... expects ... messages list`

你传入的是字符串 prompt，但没有设置 `--label-key`。要么为 prompt-plus-label 数据添加 `--label-key`，要么把行数据转换成完整 message list。

### 多模态占位符不匹配

图像 SFT 中，`--multimodal-keys '{"image":"images"}'` 引用的每张图片都必须在 conversation content 中有对应 `<image>` 占位符。图片数量多于或少于占位符都会导致数据处理错误。

### Predict 没有写文件

检查是否设置了 `--sft-predict-interval`、`--save`，并且至少存在一个 eval 来源。预测文件会写到 `<save>/predict/`。

## 下一步

- 阅读[配置参考手册](./configuration.md)，查看完整 SFT 参数表。
- 阅读[性能调优](./performance-tuning.md)，了解更完整的吞吐优化方法。
- 如果任务在模型加载或训练中 OOM，阅读[OOM 排查](./oom-troubleshooting.md)。
