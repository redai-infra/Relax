# Dr.GRPO Training

Relax supports Dr.GRPO (Group Relative Policy Optimization Done Right) for closed-window dense-model training in synchronous colocate and hybrid modes with the Megatron backend.

For the reproducible Qwen3.5-4B GSM8K comparison, see the [Dr.GRPO 200-Step Training Report](./dr-grpo-training-report.md).

## Overview

Dr.GRPO removes two normalization terms that can bias vanilla GRPO: division of each response by its own length and division of group-relative rewards by the group standard deviation. The algorithm was introduced in [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783).

Select it with `--advantage-estimator dr_grpo`. Dr.GRPO uses the same Rollout, Actor, Advantages, Reference, and ActorFwd service topology as GRPO; it does not require a Critic.

## Algorithm

For response `i` in a prompt group, Relax's default reward post-processing first computes the group-centered reward:

$$
A_i = r_i - \frac{1}{G}\sum_{j=1}^{G} r_j
$$

where $G$ is the number of responses sampled for the same prompt. Unlike GRPO, Dr.GRPO does not divide $A_i$ by the group reward standard deviation. The default Dr.GRPO policy objective uses one fixed denominator for the whole optimizer window:

$$
\mathcal{L}_{\mathrm{Dr.GRPO}} = \frac{S_{\mathrm{actor}}}{N B}
$$

where:

- `N` is the global number of responses in the optimizer window;
- `B` is `--rollout-max-response-len`;
- masked or padding tokens do not contribute to the numerator.

This fixed budget prevents short responses from receiving a larger per-token weight only because they contain fewer response tokens.

::: warning Advantage normalization is rejected
`--normalize-advantages` is rejected for Dr.GRPO. Removing the advantage variance normalization is the core of Dr.GRPO, and the flag re-applies a global whitening step that contradicts it. Startup fails with a validation error if the flag is set.
:::

::: warning Supported model and loss
Dr.GRPO currently supports dense models with `--loss-type policy_loss` only. Startup rejects MoE models and SFT or custom losses because their auxiliary losses and normalization contracts are not covered by the current implementation.
:::

## Relax Implementation

Relax keeps Megatron's CP-compatible per-token gradient normalization and applies Dr.GRPO as an optimizer-window scale.

Let $T$ be the number of valid response tokens in the optimizer window and let $S$ be the combined Actor algorithm loss numerator. Relax computes:

$$
\alpha = \frac{T}{N B},
\qquad
\frac{\alpha S}{T} = \frac{S}{N B}
$$

The implementation is split across four existing layers:

| Layer | Responsibility |
|---|---|
| Reward post-processing | Center rewards within each prompt group without group-std division |
| Optimizer-window preparation | Count global `(N, T)` after the final loss masks are available and compute `alpha` |
| Data iterator metadata | Replay the opaque `__dr_grpo_window_scale__` value on every micro-batch in the same optimizer window |
| Megatron loss path | Scale the combined Actor loss, then reuse Megatron's global `/T` gradient normalization |

The same scale is applied to the policy-gradient, entropy, and explicit KL terms in the combined Actor loss. Micro-batch boundaries only control execution and do not change `N`, `T`, or the final denominator.

### Context Parallelism

Dr.GRPO automatically enables `calculate_per_token_loss`. With CP greater than one, each CP rank contributes only the valid response tokens in its local zig-zag shard. Megatron sums the CP-local token counts before applying `/T`, so padding, CP degree, and micro-batch partitioning do not duplicate token counts.

The optimizer-window `(N, T)` reduction uses the DP group without CP. CP ranks receive the same `alpha`, while Megatron remains responsible for assembling the global token normalizer.

Pure fully-async training is rejected because its streaming `__loss_scale__` has different semantics and the Dr.GRPO fixed-budget metadata is not prepared from a closed optimizer window. Hybrid mode first closes and merges the optimizer window, so it can use the same Dr.GRPO metadata path as synchronous training.

## Quick Start

Start from an existing synchronous GRPO recipe, such as `scripts/training/text/run-qwen3-4B-8xgpu.sh`, and replace its algorithm block:

```bash
DR_GRPO_ARGS=(
   --advantage-estimator dr_grpo
   --rollout-max-response-len 8192
   --eps-clip 0.2
   --eps-clip-high 0.28
   --entropy-coef 0.0
   --kl-coef 0.0
)
```

Keep the Megatron per-token path enabled explicitly in the recipe, especially when using CP:

```bash
PERF_ARGS=(
   --calculate-per-token-loss
   --context-parallel-size 1
   --qkv-format thd
   --use-dynamic-batch-size
   --max-tokens-per-gpu 10240
)
```

`--advantage-estimator dr_grpo` also enables `calculate_per_token_loss` during argument processing, but keeping the flag in a launch script makes its CP requirement visible.

The resource topology is unchanged from GRPO:

```bash
--resource '{"actor": [1, 8], "rollout": [1, 8]}' \
--colocate
```

Adjust the GPU counts, model configuration, data paths, and token budget for your environment.

## Configuration

| Parameter | Default | Dr.GRPO behavior |
|---|---|---|
| `--advantage-estimator dr_grpo` | `grpo` | Select Dr.GRPO and its fixed-budget reduction |
| `--loss-type policy_loss` | `policy_loss` | Required; SFT and custom losses are rejected |
| `--num-experts` | `None` | Must remain unset; MoE models are not currently supported |
| `--rollout-max-response-len` | `None` | Defines `B`; set it to the generation response-token budget |
| `--n-samples-per-prompt` | `1` | Number of responses in each reward-centering group |
| `--global-batch-size` | `None` | Number of responses in one optimizer step for the usual fixed-size schedule |
| `--calculate-per-token-loss` | off | Automatically enabled for Dr.GRPO; required by Megatron when CP is enabled |
| `--normalize-advantages` | off | Rejected because Dr.GRPO removes advantage variance normalization by design |
| `--disable-rewards-normalization` | off | Rejected because group centering is mandatory for Dr.GRPO |
| `--kl-coef` | `0.0` | Must remain zero; Dr.GRPO does not add reward-side KL |
| `--use-kl-loss` | off | Add an explicit KL loss term |
| `--kl-loss-coef` | `0.0` | Coefficient of the explicit KL loss |
| `--entropy-coef` | `0.0` | Coefficient of the entropy bonus |

Use `--use-kl-loss` with a positive `--kl-loss-coef` when an explicit KL penalty is required. Reward-side `--kl-coef` is rejected so GRPO and Dr.GRPO comparisons differ only in the two Dr.GRPO normalization changes.

## Best Practices

1. Set `--rollout-max-response-len` explicitly and keep it equal to the response budget used by Rollout. It is part of the Dr.GRPO objective, not only a memory limit.
2. Compare reward, evaluation accuracy, and response length when comparing GRPO and Dr.GRPO. Their Actor loss and gradient norm magnitudes use different denominators and are not directly comparable.
3. Keep `--normalize-advantages` unset. Dr.GRPO rejects it, so use the standard GRPO estimator if you want whitened advantages.
4. A custom reward post-processor overrides the default group centering. It must provide the intended Dr.GRPO advantages.
5. When using `--custom-pg-loss-reducer-function-path`, ensure the custom reducer preserves the token-sum numerator expected by the final fixed-budget scale.

## Troubleshooting

### CP training requires per-token loss

Keep `--calculate-per-token-loss` enabled. Dr.GRPO sets it automatically, and Megatron validates the same requirement when `--context-parallel-size` is greater than one.

### The fixed denominator is incorrect

Check that `--rollout-max-response-len` matches the intended response budget and that custom rollout post-processing produces the final `loss_masks` before Actor training. `T` is counted from those final masks.

### Pure fully-async training is rejected

Use synchronous colocate or hybrid mode. Pure fully-async streaming does not yet prepare the closed-window `(N, T)` metadata required by the fixed denominator.

### MoE or a non-policy loss is rejected

Use a dense model with `--loss-type policy_loss`. MoE auxiliary losses and router expert-bias updates require an empty-window contract that is not implemented for Dr.GRPO, while SFT and custom losses do not satisfy its policy-objective contract.

## Next Steps

- [Dr.GRPO 200-Step Training Report](./dr-grpo-training-report.md) — Reproduce the paired GRPO/Dr.GRPO experiment
- [Algorithm Reference](../examples/algorithms.md) — Compare Dr.GRPO with other policy-gradient algorithms
- [Configuration](./configuration.md) — Review rollout, batch, and parallelism parameters
- [PPO Training](./ppo-training.md) — Use the Actor-Critic training path
