# MemAgent on ReLax

This example trains Qwen3-4B to update a bounded textual memory while reading a long document chunk by chunk. Every memory-update turn and the final-answer turn is saved as an independent training row. Only the final boxed answer receives a rule-based reward; GRPO normalization happens before the trajectory is expanded.

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

The evaluator writes raw per-sample JSONL and a summary JSON for HotpotQA dev and RULER-HQA 50/200/800. Failed requests keep their ground truth in the raw file and remain in the denominator with score zero. `boxed_em_pct` is the HotpotQA reward-compatible accuracy and `sub_em_pct` is the primary VIME-compatible RULER-HQA metric. Set `MODE=base` to run the optional single-context diagnostic; its context truncation always preserves the question and answer instruction.

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
