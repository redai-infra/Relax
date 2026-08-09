# Mixture-of-LoRA RL Training

Mixture-of-LoRA keeps the base model frozen and trains multiple LoRA experts at each target projection. A learned token-level router selects `K` experts and combines their outputs with normalized Top-K weights.

Use this path when one LoRA adapter does not provide enough capacity. Existing single LoRA behavior remains active when `--lora-num-experts 1` is used or omitted.

## Configuration

The Mixture path is enabled when both `--lora-rank` is positive and `--lora-num-experts` is greater than one.

```bash
--lora-rank 16
--lora-alpha 32
--lora-target-modules linear_qkv linear_proj
--lora-dropout 0.0
--lora-num-experts 4
--lora-router-top-k 2
--lora-router-temperature 1.0
--lora-router-aux-loss-coef 0.01
```

| Option | Meaning |
| --- | --- |
| `--lora-num-experts` | Number of LoRA experts at each routed projection. Values greater than one enable Mixture-of-LoRA. |
| `--lora-router-top-k` | Number of experts selected for each token. It must satisfy `1 <= K <= N`. |
| `--lora-router-temperature` | Temperature applied before the router softmax. |
| `--lora-router-aux-loss-coef` | Coefficient for the per-site router balance loss. Use `0` to disable its gradient while retaining routing metrics. |

The three router options must be provided explicitly for `N > 1`. Mixture mode does not use `--lora-merge-mode` or `--lora-adapter-mode`.

## Supported Setup

The first implementation supports:

- dense Qwen3 models;
- `linear_qkv` and `linear_proj` attention targets;
- Megatron data, tensor, sequence, pipeline, and static context parallelism;
- synchronous colocate training;
- multiple independent SGLang engines where each engine uses TP=1 and DP=1;
- native Megatron distributed checkpoints.

Fully async training, dynamic context parallelism, VLMs, MoE base models, MLP targets, and SGLang engines with internal TP or DP greater than one are rejected during startup.

## Routing and Balance Loss

For each token, the router computes FP32 probabilities over `N` experts, selects the configured Top-K entries, and renormalizes the selected probabilities to sum to one. Training and SGLang rollout call the same routing function and use equivalent dense expert equations; the Megatron executor additionally handles its TP/SP collectives.

The balance objective is calculated independently for each routed site and averaged over all sites:

```text
L_balance = N * sum_e(F_e * P_e)
F_e = selection_count_e / (valid_response_tokens * K)
P_e = mean pre-Top-K router probability for expert e
```

Prompt, padding, and dummy tokens do not enter the balance loss or routing metrics. Under static context parallelism, the counts and probability sums are combined over the CP group before the objective is formed.

## Routing Metrics

Metrics are emitted under `molora/<site_id>/...` and `molora/global/...`:

- `expert_<id>_pre_topk_mean_prob`;
- `expert_<id>_post_topk_mean_weight`;
- `expert_<id>_selection_share`;
- `expert_<id>_top1_fraction`;
- `pre_topk_normalized_entropy` and `post_topk_normalized_entropy`;
- `balance_loss` per site and `molora/aux_loss` globally.

Use the per-site post-Top-K weights, selection shares, and entropy to detect collapse. Global metrics are useful summaries but can hide a collapsed layer.

## Rollout Weight Updates

Relax starts the Qwen3 SGLang external model automatically when Mixture mode is enabled. The first colocate update sends the frozen base plus all expert and router tensors. Later updates send all current expert and router tensors without resending the base. A weight version is published only after the final routed tensor is accepted.

If an update fails, Relax resumes generation, keeps the previous weight version, and reports the failed update instead of silently serving a partial policy.

## Checkpoints

Expert and router tensors are ordinary model parameters in the native Megatron distributed checkpoint. The same checkpoint also restores optimizer, scheduler, iteration, and RNG state. Mixture mode does not create a separate HF PEFT adapter export.

Resume by launching the same recipe with `--load` and `--save` pointing to the existing output directory. The saved Mixture metadata is checked against the current expert count, rank, Top-K, temperature, coefficient, alpha, target modules, dtype, and site dimensions before tensors are loaded.

## Qwen3-4B DAPO Recipe

The reference script runs Qwen3-4B GRPO for 200 rollouts on eight colocated GPUs:

```bash
MODEL_PATH=/path/to/Qwen3-4B \
PROMPT_DATA=/path/to/dapo-math-17k.jsonl \
OUTPUT_DIR=/path/to/qwen3-4b-mixture-lora \
bash scripts/training/text/run-qwen3-4B-mixture-lora-8xgpu.sh
```

The actor uses TP=2 with sequence parallelism. The rollout allocation creates eight independent one-GPU SGLang engines. Environment variables such as `NUM_ROLLOUT`, `LORA_NUM_EXPERTS`, `LORA_RANK`, and `LORA_ROUTER_TOP_K` can override the recipe values. Additional Relax arguments can be appended to the command.
