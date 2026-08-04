# MemAgent on ReLax

This example trains Qwen3-4B to update a bounded textual memory while reading a long document chunk by chunk. Every memory-update turn and the final-answer turn is saved as an independent training row. Only the final boxed answer receives a rule-based reward; GRPO normalization happens before the trajectory is expanded.

The training log reports the trajectory-level 0/1 outcome as `rollout/mem_agent_raw_reward/mean` on every rollout step. This diagnostic mirrors the primary `score` exactly but is not consumed by GRPO. `run-pipeline.sh` then runs `summarize_reward.py` and writes `training-reward.summary.json`, containing every raw point, the first/last-window means, their delta, and the peak. It rejects a run whose rollout ids are incomplete instead of producing a partial trend.

The reproducibility contract is frozen to:

- model: `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`;
- dataset: `BytedTsinghua-SIA/hotpotqa@27275ff4fee67ac0acb6478e405e7ac07efbdc1a`;
- chunk/memory/final limits: 2048/1024/256 tokens, at most 64 chunks;
- GRPO group size 8, split credit, LR `1e-6`, KL coefficient `0.001`;
- 100 rollout steps with checkpoints every 50 steps.

## Prepare model and data

Download the exact model revision to a local directory with your preferred Hugging Face client. Then prepare all frozen train/eval files and their SHA-256 manifest:

```bash
DATA_DIR=/data/mem-agent bash examples/mem_agent/prepare-data.sh
```

`prepare-data.sh` downloads `hotpotqa_train_32k.parquet`, `hotpotqa_dev.parquet`, and `eval_50/200/800.json` at the pinned dataset revision. It writes converted JSONL files plus `artifact_manifest.json`.

## Train

```bash
MODEL_PATH=/data/models/Qwen3-4B \
DATA_DIR=/data/mem-agent \
SAVE_DIR=/data/checkpoints/mem-agent-relax \
bash examples/mem_agent/run-qwen3-4B-train.sh
```

For the required two-step correctness smoke, add `NUM_ROLLOUT=2` and use the same command. A smoke run validates the pipeline but is not an effects result.

The train-side SGLang context envelope is 8192 tokens because each request is an independent chunk turn (2K chunk + 1K memory + response), not the concatenated trajectory. This retains the frozen 9216-token per-GPU packing budget and sample-mean loss used for split credit.

## Convert and evaluate

```bash
MODEL_PATH=/data/models/Qwen3-4B \
CHECKPOINT_DIR=/data/checkpoints/mem-agent-relax \
CHECKPOINT_TAG=iter_0000099 \
bash examples/mem_agent/convert-to-hf.sh

MODEL_PATH=/data/checkpoints/mem-agent-relax-HF/iter_0000099 \
TOKENIZER_PATH=/data/models/Qwen3-4B \
DATA_DIR=/data/mem-agent \
RESULTS_DIR=/data/results/mem-agent-relax \
bash examples/mem_agent/run-eval.sh
```

The evaluator writes raw per-sample JSONL and a summary JSON for HotpotQA dev and RULER-HQA 50/200/800. Its 64-chunk limit is the effective value of fixed VIME's official `run-eval.sh`: that script sources `_common.sh`, which exports `MEM_MAX_CHUNKS=64`, even though the Python evaluator alone has a 512 fallback. Failed requests keep their ground truth in the raw file and remain in the denominator with score zero. Formal comparison additionally rejects empty runs and any run with request errors. Each summary records the normalized input file SHA-256 and evaluator schema version, so equal paths with different bytes or incompatible evaluator revisions cannot be compared. `boxed_em_pct` is the HotpotQA reward-compatible accuracy and `sub_em_pct` is the primary VIME-compatible RULER-HQA metric. Set `MODE=base` to run the optional single-context diagnostic; its context truncation always preserves the question and answer instruction.

`TOKENIZER_PATH` should point to the frozen base snapshot. `run-pipeline.sh` preserves it automatically before switching `MODEL_PATH` to the converted checkpoint. When `NUM_ROLLOUT=2` is used, the pipeline also selects `iter_0000001` automatically instead of the 100-step default `iter_0000099`.

For the VIME reproduction tolerance, evaluate an official VIME checkpoint when one is available; otherwise use a checkpoint produced once from the fixed VIME recipe. The acceptance runner holds the tokenizer, data, prompts, sampling parameters, recurrent inference mode, and evaluator constant across frozen base, VIME, and ReLax. It evaluates RULER-HQA 50/200/800 by default; `LENGTHS` can freeze a smaller pre-agreed subset before any result is observed. It exits non-zero unless every selected VIME/ReLax `sub_em_pct` gap is at most 3 percentage points, ReLax beats frozen base on every selected RULER-HQA `sub_em_pct`, and ReLax beats frozen base on HotpotQA `boxed_em_pct`. Raw per-sample files are retained for review:

```bash
BASE_MODEL_PATH=/data/models/Qwen3-4B \
VIME_MODEL_PATH=/data/checkpoints/vime-hf \
RELAX_MODEL_PATH=/data/checkpoints/mem-agent-relax-hf \
TOKENIZER_PATH=/data/models/Qwen3-4B \
DATA_DIR=/data/mem-agent \
RESULTS_DIR=/data/results/vime-vs-relax \
LENGTHS="50 200 800" \
bash examples/mem_agent/run-paired-eval.sh
```

## One-command chain

With the environment paths set, `run-pipeline.sh` executes data preparation, training, checkpoint conversion, and evaluation in order:

```bash
MODEL_PATH=/data/models/Qwen3-4B \
DATA_DIR=/data/mem-agent \
SAVE_DIR=/data/checkpoints/mem-agent-relax \
RESULTS_DIR=/data/results/mem-agent-relax \
bash examples/mem_agent/run-pipeline.sh
```

GPU execution is intentionally not started by the CPU test suite. The caller remains responsible for starting the ReLax/Ray environment described by the repository deployment guide.

## Qwen3-0.6B single-4090 pilot

The 0.6B recipe is a low-cost pipeline and learnability diagnostic; it does not replace the frozen Qwen3-4B VIME/ReLax acceptance run. It fixes `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`, disables Qwen3 thinking, uses 512-token chunks, a 128-token memory, a 64-token final answer, and at most four chunks.

Prepare 24 immutable 2--4-chunk candidates before allocating a GPU:

```bash
TOKENIZER_PATH=/data/models/Qwen3-0.6B \
DATA_DIR=/data/task36-pilot \
bash examples/mem_agent/prepare-pilot-candidates.sh
```

The baseline samples every candidate eight times. Selection fails unless at least 12 prompts have both a success and a failure; 2--6 successes out of 8 are preferred. Eight prompts become the training split and four disjoint prompts become the held-out pilot split. This Pass@N-screened set deliberately supplies GRPO reward variance and must not be reported as an unbiased HotpotQA metric.

```bash
MODEL_PATH=/data/models/Qwen3-0.6B \
DATA_DIR=/data/task36-pilot \
RESULTS_DIR=/data/task36-runs/baseline \
bash examples/mem_agent/run-qwen3-0.6B-baseline.sh

MODEL_PATH=/data/models/Qwen3-0.6B \
DATA_DIR=/data/task36-pilot \
RUN_ROOT=/data/task36-runs/train \
NUM_ROLLOUT=2 \
bash examples/mem_agent/run-qwen3-0.6B-train.sh
```

The train script is TP=1 and produces the complete Ray job log, TensorBoard events, `training-reward.summary.json`, exact `training-reward.csv` points, and `training-reward.svg`. Run the two-step smoke first. Only after its generated/transferred/consumed row counts agree should a longer run start. Converted checkpoints are evaluated against the frozen pilot split with the same seed, sampling parameters, prompt path, and tokenizer via `run-qwen3-0.6B-eval.sh`.
