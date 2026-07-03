# SFT Training

This guide shows the end-to-end supervised fine-tuning (SFT) workflow in Relax, using the current scripts under [`scripts/training/sft/`](../../../scripts/training/sft/). It covers data preparation for the math and Pokemon datasets, model and path configuration, launch commands, and practical tuning.

Make sure you have completed [Installation](./installation.md) before running the commands below.

## Overview

SFT is enabled with `--loss-type sft`. In this mode, Relax starts an SFT producer that reads `--prompt-data`, renders samples through the model chat template, writes packed samples into TransferQueue, and trains the Megatron actor. If `--eval-interval` is set, it also runs PPL evaluation on an eval split. If `--sft-predict-interval` is set, Relax additionally uses the Rollout role and SGLang for periodic generative prediction.

The current launch scripts use `ray job submit` and auto-source [`scripts/entrypoint/local.sh`](../../../scripts/entrypoint/local.sh) when no external entrypoint has already prepared the Ray environment.

## Scripts

| Script | Dataset | Model | Default resources | Notes |
| --- | --- | --- | --- | --- |
| [`run-qwen3.5-9B-math-8xgpu.sh`](../../../scripts/training/sft/run-qwen3.5-9B-math-8xgpu.sh) | `OpenMathReasoning-mini` | `Qwen3.5-9B` | 8 GPU actor plus SFT producer and Rollout | Text SFT with `problem` -> `generated_solution`, PPL eval, and predict. |
| [`run-qwen3-vl-4B-math-8xgpu.sh`](../../../scripts/training/sft/run-qwen3-vl-4B-math-8xgpu.sh) | `OpenMathReasoning-mini` | `Qwen3-VL-4B-Instruct` | 8 GPU actor plus SFT producer and Rollout | Text-only math SFT using a VL checkpoint. |
| [`run-qwen3-vl-4B-pokemon-8xgpu.sh`](../../../scripts/training/sft/run-qwen3-vl-4B-pokemon-8xgpu.sh) | `pokemon-gpt4o-captions` | `Qwen3-VL-4B-Instruct` | 8 GPU actor plus SFT producer and Rollout | Multimodal image SFT with two parquet files and prefetch enabled. |
| [`run-qwen3-vl-4B-pokemon-1xgpu.sh`](../../../scripts/training/sft/run-qwen3-vl-4B-pokemon-1xgpu.sh) | `pokemon-gpt4o-captions` | `Qwen3-VL-4B-Instruct` | 1 GPU actor plus SFT producer | Low-resource Pokemon SFT with CPU optimizer offload. |
| [`run-qwen3.5-35B-A3B-mtp-sft-16xgpu.sh`](../../../scripts/training/sft/run-qwen3.5-35B-A3B-mtp-sft-16xgpu.sh) | `OpenMathReasoning-mini` | `Qwen3.5-35B-A3B` | 16 GPU actor plus SFT producer | Advanced MTP SFT. Many knobs are exposed as environment variables. |

## Data Preparation

The SFT scripts default to `DATA_DIR=${SCRIPT_DIR}/data` unless you override `DATA_DIR`. For reusable jobs, put datasets on persistent storage and export `DATA_DIR` explicitly.

```bash
cd /root/Relax
export DATA_DIR=/root
mkdir -p "${DATA_DIR}/sft/data"
```

### Math: OpenMathReasoning-mini

The math scripts expect this exact file:

```text
${DATA_DIR}/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet
```

Download it from [unsloth/OpenMathReasoning-mini](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini):

```bash
hf download --repo-type dataset unsloth/OpenMathReasoning-mini \
  data/cot-00000-of-00001.parquet \
  --local-dir "${DATA_DIR}/sft/data/OpenMathReasoning-mini"
```

The current math scripts configure:

```bash
--prompt-data "${DATA_DIR}/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet"
--input-key problem
--label-key generated_solution
```

With `--label-key` set, each row must contain a prompt string under `problem` and a target response under `generated_solution`. Relax builds a user message from `problem` and appends an assistant message from `generated_solution`.

Minimal row shape:

```json
{
  "problem": "Find the value of x.",
  "generated_solution": "We solve the equation step by step..."
}
```

### Pokemon: pokemon-gpt4o-captions

The Pokemon scripts expect these files:

```text
${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_en.parquet
${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_zh.parquet
```

Download them from [llamafactory/pokemon-gpt4o-captions](https://huggingface.co/datasets/llamafactory/pokemon-gpt4o-captions):

```bash
hf download --repo-type dataset llamafactory/pokemon-gpt4o-captions \
  pokemon_gpt4o_en.parquet pokemon_gpt4o_zh.parquet \
  --local-dir "${DATA_DIR}/sft/data/pokemon-gpt4o-captions"
```

The current Pokemon scripts build a list-valued `PROMPT_DATA` from both files:

```bash
TRAIN_FILES=(
    "'${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_en.parquet'"
    "'${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_zh.parquet'"
)
PROMPT_DATA="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"
```

They then configure:

```bash
--prompt-data "${PROMPT_DATA}"
--input-key conversations
--multimodal-keys '{"image":"images"}'
--conversation-key-map '{"from":"role","value":"content","human":"user","gpt":"assistant"}'
```

Because `--label-key` is not set, `conversations` must be a complete message list. The `--conversation-key-map` rewrites ShareGPT-style message fields and role values into the OpenAI-style `{role, content}` format expected by SFT. The `--multimodal-keys` mapping tells Relax to load image paths from the row's `images` column. The text should contain an `<image>` placeholder for each image item.

Minimal row shape:

```json
{
  "conversations": [
    {"from": "human", "value": "Identify the object of this image.<image>"},
    {"from": "gpt", "value": "A round, pink Pokemon with a gentle expression."}
  ],
  "images": ["/path/to/pokemon.png"]
}
```

### Verify Data Files

Run this before launching to catch missing files or wrong columns:

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

Change `/root` in the check if you used a different `DATA_DIR`.

## Model Preparation

Download model weights to persistent storage. The math and Pokemon scripts read model paths from `MODEL_DIR` rather than from `EXP_DIR`, so set `MODEL_DIR` explicitly for these scripts.

```bash
# Math script: run-qwen3.5-9B-math-8xgpu.sh
hf download Qwen/Qwen3.5-9B --local-dir /root/Qwen3.5-9B

# Pokemon scripts: run-qwen3-vl-4B-pokemon-*.sh
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /root/Qwen3-VL-4B-Instruct
```

For the MTP script, the default convention is `EXP_DIR=/root`, then `MODEL_DIR=${EXP_DIR}` and `DATA_DIR=${EXP_DIR}` unless you override them.

## Configuration Walkthrough

The scripts are organized into argument blocks. Tune by editing the selected script, or by using environment variables where the script already exposes them.

### Paths and Checkpoints

For math and Pokemon scripts:

```bash
export MODEL_DIR=/root
export DATA_DIR=/root
```

The `CKPT_ARGS` block sets:

| Flag | Purpose |
| --- | --- |
| `--hf-checkpoint` | HF checkpoint used for tokenizer, config, and SGLang initialization when predict is enabled. |
| `--ref-load` | Initial reference checkpoint path. |
| `--load` | Training load path. If this is an existing Megatron checkpoint, training resumes from it. Otherwise bridge mode starts from HF weights. |
| `--megatron-to-hf-mode bridge` | Uses Megatron Bridge for HF <-> Megatron conversion. |
| `--save` | Megatron checkpoint output directory. |
| `--save-interval` | Save every N training steps. |
| `--num-epoch` | Number of dataset epochs. Relax resolves the actual training steps from dataset size and global batch size. |

For a fresh run, keep `--load` and `--save` pointing at the same experiment directory as the current scripts do. For resume, keep the same `--save` and make sure it contains a valid checkpoint with `latest_checkpointed_iteration.txt`.

### MTP Arguments

The MTP SFT script enables `--mtp-num-layers ${MTP_NUM_LAYERS:-1}`, `--enable-mtp-training`, and `--mtp-loss-scaling-factor ${MTP_LOSS_SCALING_FACTOR:-0.2}`. Increase `MTP_NUM_LAYERS` only for models/checkpoints with matching MTP layers; tune `MTP_LOSS_SCALING_FACTOR` as an auxiliary-loss weight, starting from `0.2`.

### SFT Data Arguments

Required SFT flags:

```bash
--loss-type sft
--prompt-data "${PROMPT_DATA}"
--use-dynamic-batch-size
--max-tokens-per-gpu <tokens>
```

SFT requires dynamic batching. If `--use-dynamic-batch-size` is missing, argument validation fails. If `--balance-data` is not set, SFT validation auto-enables it so the SFT producer and Megatron data path stay consistent.

Use one of two row formats:

| Format | Flags | When to use |
| --- | --- | --- |
| Prompt plus label | `--input-key problem --label-key generated_solution` | Text rows where prompt and supervised answer are separate columns. |
| Full messages | `--input-key conversations` and no `--label-key` | Chat or multimodal rows that already contain user and assistant turns. Add `--conversation-key-map` for ShareGPT-style fields. |

Do not add `--apply-chat-template` for SFT. The SFT dataset renders samples with the tokenizer chat template internally and builds the assistant loss mask from that render.

### Evaluation and Predict

PPL evaluation is controlled by:

```bash
--eval-size 0.01
--eval-interval 10
```

`--eval-size` reserves the tail of `--prompt-data` for eval and removes it from the train pool. A value below 1 is a fraction; a value of 10 or higher is an absolute sample count. You may use `--eval-prompt-data name path` instead, but in SFT mode do not use `--eval-config`.

Generative prediction is controlled by:

```bash
--sft-predict-interval 10
--eval-temperature 0.0
--eval-max-response-len 10240
```

When `--sft-predict-interval` is set, Relax spins up the Rollout role automatically, and predictions are written under:

```text
<save>/predict/predictions_step_<rollout_id>.jsonl
```

Use a longer `--eval-max-response-len` for math reasoning and a shorter value for captioning or image description tasks.

### Parallelism and Resources

The script-level `PERF_ARGS` and `--resource` must agree with the cluster size.

Math 8 GPU script:

```bash
--tensor-model-parallel-size 4
--pipeline-model-parallel-size 2
--context-parallel-size 1
--resource '{"sft": [1, 0], "actor": [1, 8], "rollout": [1, 8]}'
```

Pokemon 8 GPU script:

```bash
--tensor-model-parallel-size 2
--pipeline-model-parallel-size 1
--context-parallel-size 1
--per-rank-fetch
--num-data-storage-units 8
--resource '{"sft": [1, 0], "actor": [1, 8], "rollout": [1, 8]}'
```

Pokemon 1 GPU script:

```bash
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
--resource '{"sft": [1, 0], "actor": [1, 1], "rollout": [1, 1]}'
```

`"sft": [1, 0]` means the SFT producer is CPU-only. The Actor owns training GPUs. Rollout GPUs are needed when periodic predict is enabled.

## Launch

### Single Node

Use the script directly. It sources `local.sh`, starts a local Ray head node, and submits the job:

```bash
cd /root/Relax
export MODEL_DIR=/root
export DATA_DIR=/root

bash scripts/training/sft/run-qwen3.5-9B-math-8xgpu.sh
bash scripts/training/sft/run-qwen3-vl-4B-pokemon-8xgpu.sh
```

For the 1 GPU Pokemon script:

```bash
cd /root/Relax
export MODEL_DIR=/root
export DATA_DIR=/root
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1

bash scripts/training/sft/run-qwen3-vl-4B-pokemon-1xgpu.sh
```

### Existing Ray Cluster

If a Ray cluster is already running, use [`scripts/entrypoint/ray-job.sh`](../../../scripts/entrypoint/ray-job.sh). It prepares the runtime environment, avoids stopping Ray, and then delegates to the run script:

```bash
cd /root/Relax
export MODEL_DIR=/root
export DATA_DIR=/root
export RAY_ADDRESS=http://127.0.0.1:8265

bash scripts/entrypoint/ray-job.sh scripts/training/sft/run-qwen3-vl-4B-pokemon-8xgpu.sh
```

### Multi-Node

Use [`scripts/entrypoint/spmd-multinode.sh`](../../../scripts/entrypoint/spmd-multinode.sh). Required environment variables are `MASTER_ADDR`, `POD_NAME`, `HOST_IP`, and `WORLD_SIZE`; `NUM_GPUS` defaults to 8 per node.

```bash
cd /root/Relax
export MODEL_DIR=/root
export DATA_DIR=/root
export WORLD_SIZE=2
export NUM_GPUS=8

bash scripts/entrypoint/spmd-multinode.sh \
  scripts/training/sft/run-qwen3.5-35B-A3B-mtp-sft-16xgpu.sh
```

## Tuning Workflow

Tune in this order so each change has a clear purpose.

### 1. Fit in Memory

If the job OOMs during training:

| Knob | Direction | Effect |
| --- | --- | --- |
| `--max-tokens-per-gpu` | Decrease | Lowers dynamic micro-batch token capacity. This is the first SFT memory knob. |
| `--global-batch-size` | Decrease | Reduces samples per optimizer update, but changes optimization dynamics. |
| `--recompute-num-layers` | Increase | Saves activation memory at the cost of compute. |
| `--optimizer-cpu-offload` | Enable | Saves GPU memory, useful for 1 GPU or tight VL runs. |
| `--sft-oversize-strategy skip` | Enable carefully | Drops samples longer than `max_tokens_per_gpu * context_parallel_size`. |

For context parallelism, capacity is `--max-tokens-per-gpu * --context-parallel-size`. Increase CP only when the model and Megatron configuration support it.

### 2. Improve Throughput

If GPUs wait on SFT data:

| Knob | Direction | Effect |
| --- | --- | --- |
| `--sft-prefetch-buffer-size` | Increase from 256 | Keeps more rendered samples ready. |
| `--sft-prefetch-num-workers` | Increase | Improves image decode and multimodal I/O parallelism. |
| `--sft-prefetch-chunk-size` | Increase | Dispatches larger prefetch chunks, with higher memory pressure. |
| `--per-rank-fetch` | Enable for multi-GPU | Lets TP/PP ranks pull from TransferQueue directly. Pair with enough `--num-data-storage-units`. |
| `--max-staleness` | Increase for I/O-heavy SFT | Lets the producer run ahead. The Pokemon 8 GPU script uses `--max-staleness 4`. |

For text-only math, prefetch usually matters less than sequence length and model parallelism. For Pokemon, image loading and processor work are common bottlenecks.

### 3. Preserve Quality

Start with the script defaults, then change one optimization knob at a time:

| Knob | Math default | Pokemon 8 GPU default | Notes |
| --- | --- | --- | --- |
| `--lr` | `1e-5` | `1e-5` | Lower it if eval loss rises quickly or resume from a strong checkpoint. |
| `--lr-decay-style` | `cosine` | `cosine` | The 1 GPU Pokemon script uses `constant` with `3e-5`, which is more aggressive. |
| `--weight-decay` | `0.1` | `0.1` | Keep stable unless you are doing a controlled sweep. |
| `--clip-grad` | `1.0` | `1.0` | Lower only if gradients spike. |
| `--num-epoch` | `10` | `10` | Reduce for smoke tests or increase only with eval monitoring. |

### 4. Control Evaluation Cost

| Knob | Direction | Effect |
| --- | --- | --- |
| `--eval-size` | Lower | Less held-out data and cheaper PPL eval. |
| `--eval-interval` | Increase | Fewer eval rounds. |
| `--sft-predict-interval` | Increase or disable | Reduces SGLang predict overhead. |
| `--eval-max-response-len` | Lower | Caps generation cost for predict. |

For fast bring-up, disable predict by commenting out `PREDICT_ARGS` and use a small `--eval-size`. Re-enable predict once the training loop is stable.

## Troubleshooting

### `--loss-type sft requires --use-dynamic-batch-size`

SFT intentionally rejects static micro-batching. Add `--use-dynamic-batch-size` and set `--max-tokens-per-gpu`.

### `Under --loss-type sft with --eval-interval set...`

`--eval-interval` requires exactly one eval source: either `--eval-size` or `--eval-prompt-data name path`. Do not set both.

### `SFT row missing prompt key`

The value of `--input-key` does not match the dataset column. Math uses `problem`; Pokemon uses `conversations`.

### `--label-key is not set ... expects ... messages list`

You passed a string prompt without `--label-key`. Either add `--label-key` for prompt-plus-label rows, or convert the row to a full message list.

### Multimodal Placeholder Mismatch

For image SFT, each image referenced by `--multimodal-keys '{"image":"images"}'` must have a matching `<image>` placeholder in the conversation content. Extra images or missing placeholders cause data processing errors.

### Predict Does Not Write Files

Check that `--sft-predict-interval` is set, `--save` is set, and an eval source exists. Prediction files are written under `<save>/predict/`.

## Next Steps

- Read [Configuration Reference](./configuration.md) for the full SFT parameter table.
- Read [Performance Tuning](./performance-tuning.md) for broader throughput tuning.
- Read [OOM Troubleshooting](./oom-troubleshooting.md) if the job fails during model load or training.
