# Dr.GRPO 200-Step Training Report

This report records the equal-budget Qwen3.5-4B GSM8K comparison used to validate Relax's Dr.GRPO implementation. It is an implementation and stability study with one run per algorithm, not a statistical claim that Dr.GRPO is generally superior to GRPO.

## Scope and Evidence Boundary

- Pull request: [Relax #239](https://github.com/redai-infra/Relax/pull/239)
- Algorithms: GRPO and Dr.GRPO
- Formal budget: exactly 200 optimizer steps for each run
- Responses: 3,200 per run (`200 * 16`)
- Hardware: 4 x NVIDIA H20
- Execution source base: `62022e01ce4cbe48b4720bcf20335af2415e41ec`
- Execution patch SHA256: `e0f560236024420b51e03124b03b7041c9c0936f5d431de81c4fc770191b1f03`
- Frozen execution worktree SHA256: `30ab4f7524e378bba6350d0b67641090485c5b904fc375eb8e51ba726fb876e7`
- Container image: `ghcr.io/redai-infra/relaxrl@sha256:d6ee9f015f1a92987931ed5f7229e6cfb6f089bba9fa8e6bc1ea289c2503a7af`

::: warning Evidence release candidate
The sanitized evidence archives are built and SHA256-verified locally. Their public GitHub Release URL and the final clean reproduction commit must be added before the PR evidence update is considered complete. The experiment source is currently identified by the base commit plus the archived patch and worktree hashes above.
:::

The prepared path-free main archive SHA256 is `7510d747a3f3371e52a84f19315fde5f7cc54c9438c5d30970fab4dec465c96c`. The separate 1,600-dump loss-mask source archive SHA256 is `521c554bb11b2c5891e72e103eb669286dbf7b1531903f7b258773deeaae28f3`.

The repository publishes the lightweight, path-free results used below:

- [all 400 optimizer-step records](../../public/dr-grpo/training_metrics_long.csv);
- [per-run summary](../../public/dr-grpo/training_summary.json);
- [machine-readable experiment manifest](../../public/dr-grpo/experiment_manifest.json);
- [result table](../../public/dr-grpo/tables.md);
- [training curves](../../public/dr-grpo/training_curves.svg).

Checkpoints, model weights, the complete GSM8K source dataset, raw logs, rollout JSONL, and per-rank training dumps are not stored in Git.

## Frozen Environment and Workload

| Item | Value |
|---|---|
| Model | `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Precision / update | BF16 / full parameter update |
| Python / PyTorch | 3.12.3 / 2.11.0+cu129 |
| CUDA / NCCL | 12.9 / 2.28.9 |
| Ray / SGLang | 2.57.0 / 0.5.15.post1 |
| Transformer Engine | 2.14.1 |
| Parallelism | TP2, PP1, CP1, DP2 |
| Batch geometry | 4 prompts x 4 responses; global batch 16 |
| Response budget | 4,096 tokens |
| Optimizer steps | 200 per algorithm; no post-hoc truncation |
| Optimizer | Adam, `lr=1e-6`, betas `(0.9, 0.98)`, weight decay `0.1`, constant LR |
| PPO clipping | low `0.2`, high `0.28` |
| KL / entropy coefficients | reward-side KL `0`; explicit KL `0`; entropy `0` |
| Sampling | temperature `0.7`, top-p `0.8`, top-k `20` |
| Seeds | train `1234`, rollout `42`, dataset shuffle `42` |

Both jobs were configured with `--num-rollout 200`, produced rollout files `0.jsonl` through `199.jsonl`, produced 800 per-rank dumps, and reached the iteration-199 checkpoint. No step outside `0..199` exists in either accepted run.

## Model and Dataset Preparation

Download the pinned model revision:

```bash
hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir /path/to/Qwen3.5-4B
```

Download the pinned GSM8K train parquet and build the exact 256-row training subset:

```bash
hf download openai/gsm8k main/train-00000-of-00001.parquet \
  --repo-type dataset \
  --revision 740312add88f781978c0658806c59bc2815b9866 \
  --local-dir /path/to/gsm8k-pinned

python scripts/testing/convert_gsm8k_for_dr_grpo_e2e.py \
  --input /path/to/gsm8k-pinned/main/train-00000-of-00001.parquet \
  --output /path/to/gsm8k-train-shuffle42-first256.jsonl

sha256sum /path/to/gsm8k-train-shuffle42-first256.jsonl
```

The converter rejects an input whose SHA256 is not `ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d`. It shuffles all 7,473 train rows with `random.Random(42)`, keeps the first 256, converts the final `####` answer to a scalar label, and requires this output SHA256:

```text
8f8580875e50e5da2828ad586f97ee20e55f3ac1dfd7a6f019103ddad1a0f9d1
```

The experiment does not use `gsm8k-test.jsonl` or the GSM8K test split.

## Reproduction Commands

Pull the immutable image and start an isolated four-GPU container with the repository, model, dataset, and output directory mounted at the paths expected by the recipe:

```bash
docker pull ghcr.io/redai-infra/relaxrl@sha256:d6ee9f015f1a92987931ed5f7229e6cfb6f089bba9fa8e6bc1ea289c2503a7af
```

Run the GRPO control arm:

```bash
MODEL_PATH=/path/to/Qwen3.5-4B \
PROMPT_SET=/path/to/gsm8k-train-shuffle42-first256.jsonl \
OUTPUT_DIR=/path/to/output/grpo \
TRAIN_DATA_DIR=/path/to/output/grpo/train_data \
TENSORBOARD_DIR=/path/to/output/grpo/tensorboard \
NUM_ROLLOUT=200 \
ADVANTAGE_ESTIMATOR=grpo \
RUN_ID=qwen35-4b-grpo-cp1-final \
SAVE_CHECKPOINT=1 \
SAVE_INTERVAL=50 \
NCCL_NVLS_ENABLE=0 \
bash scripts/training/text/run-qwen35-4B-4xgpu-dr-grpo.sh
```

After the control arm has terminated and its Ray Serve state has been shut down, run Dr.GRPO from the same initial model:

```bash
MODEL_PATH=/path/to/Qwen3.5-4B \
PROMPT_SET=/path/to/gsm8k-train-shuffle42-first256.jsonl \
OUTPUT_DIR=/path/to/output/dr_grpo \
TRAIN_DATA_DIR=/path/to/output/dr_grpo/train_data \
TENSORBOARD_DIR=/path/to/output/dr_grpo/tensorboard \
NUM_ROLLOUT=200 \
ADVANTAGE_ESTIMATOR=dr_grpo \
RUN_ID=qwen35-4b-dr-grpo-cp1-final \
SAVE_CHECKPOINT=1 \
SAVE_INTERVAL=50 \
NCCL_NVLS_ENABLE=0 \
bash scripts/training/text/run-qwen35-4B-4xgpu-dr-grpo.sh
```

The recipe fixes every other algorithm, optimizer, sampling, batch, and parallelism argument. The sanitized evidence bundle contains the two fully expanded `ray job submit -- python3 -m relax.entrypoints.train ...` commands emitted by `bash -x`.

## Metric Generation

The report is generated from the two raw training logs, 400 rollout JSONL files, and 1,600 unmodified per-rank `--save-debug-train-data` dumps:

```bash
python scripts/testing/summarize_dr_grpo_qwen35_gsm8k.py \
  --experiment-root /path/to/paired-experiment \
  --output-dir /path/to/report \
  --num-steps 200 \
  --response-budget 4096 \
  --global-batch-size 16 \
  --world-size 4 \
  --model-parallel-size 2
```

One invocation validates all expected files and fields and generates `training_metrics_long.csv`, `training_summary.json`, `tables.md`, `training_curves.svg`, and `incorrect_length_10step_pooled.svg`. There is no manual CSV-to-table or CSV-to-SVG step.

### Exact Loss-Mask Token Count

`train/loss_mask_tokens` is the exact integer sum of the final training `loss_masks`, not `sum(response_length)`. Each accepted step has four Actor-rank dumps. TP replicas are verified to contain identical sample-mask signatures and removed once; the remaining DP shards are summed to recover the global `T`. The analysis fails if a step/rank is absent, replica masks disagree, a mask is non-binary, or the reconstructed global sample count is not 16.

The separate evidence archive retains all 1,600 source dumps and their individual SHA256 values. The main evidence archive retains the per-step exact counts and the complete generated CSV.

`train/reference_kl` is then reconstructed as strict `sum(KL) / T`. For Dr.GRPO, the logged fixed-budget component is multiplied by `N * B` and divided by exact `T`. For GRPO, the historical CP1 per-response floor is first undone; no fully masked response occurred in either accepted run. A zero-token step would be reported as unavailable rather than divided by zero.

## Results

| Algorithm | Reward all / last 20 | Accuracy | Correct length all / last 20 | Incorrect length all / last 20 | Exact T | Grad norm mean / p95 / max |
|---|---:|---:|---:|---:|---:|---:|
| GRPO | 0.83125 / 0.86250 | 91.5625% | 1067.9 / 905.3 | 2969.2 / 2833.7 | 3,930,695 | 0.4466 / 2.4393 / 8.7898 |
| Dr.GRPO | 0.85750 / 0.91250 | 92.8750% | 929.7 / 948.5 | 2248.4 / 3139.9 | 3,275,766 | 0.1032 / 0.5420 / 2.0179 |

The accuracy is pooled over all 3,200 responses in each run. Length means are pooled by response count, rather than averaging per-step means with equal weight. The last-20 window is optimizer steps 180 through 199.

![GRPO and Dr.GRPO training curves](../../public/dr-grpo/training_curves.svg)

The exact token-pooled policy-reference KL was `0.31151` for GRPO and `0.16972` for Dr.GRPO over all 200 steps. The inference/training probability mismatch remained small: mean `train/train_rollout_prob_abs_diff` was `0.003474` for GRPO and `0.003470` for Dr.GRPO.

Both jobs completed successfully without OOM, NCCL failure, NaN/Inf metrics, or a training traceback. Each run retained checkpoints at iterations 49, 99, 149, and 199. Checkpoints are operational recovery artifacts and are excluded from the public evidence bundle.

## Limitations

1. There is one stochastic run per algorithm. Differences in reward, accuracy, length, KL, or gradient norm are descriptive for this pair and do not establish statistical superiority.
2. Training uses a fixed 256-prompt subset of GSM8K train, not the test split or a complete benchmark evaluation.
3. `kl_loss_coef=0`, so reference KL is a diagnostic and does not contribute to the optimized objective.
4. The run covers dense Qwen3.5-4B with CP1. Dr.GRPO currently rejects MoE models, non-policy losses, pure fully-async training, and `--normalize-advantages`.
5. There were no fully masked responses or zero-token optimizer windows in these two formal runs; those boundaries are covered by focused regression tests instead.

Within those limits, both algorithms completed the identical declared 200-step budget, the report uses exact training loss-mask counts, and the observed Dr.GRPO run had higher pooled accuracy and lower gradient norms than its GRPO control.

## Next Steps

- [Dr.GRPO Training](./dr-grpo-training.md) — Algorithm contract and supported configurations
- [Algorithm Reference](../examples/algorithms.md) — Compare Relax policy-gradient estimators
- [Configuration](./configuration.md) — Review training and rollout arguments
