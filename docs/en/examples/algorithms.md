# Algorithm Reference

Relax supports multiple policy gradient algorithms, all selected via the `--advantage-estimator` flag. This document covers PPO and the primary GRPO-family algorithms (for On-Policy Distillation, see the [dedicated page](./on-policy-distillation.md)).

GRPO, RLOO, CISPO, GSPO, and SAPO share the same actor/rollout service topology, although RLOO is synchronous-only and enforces fixed batch invariants. PPO additionally requires a Critic model and an Advantages service; start from the [PPO training recipe](../guide/ppo-training.md) instead of only replacing `GRPO_ARGS`.

REINFORCE++ and REINFORCE++-baseline also reuse the GRPO service topology, but
their return, global normalization and KL contracts are algorithm-specific.
See [REINFORCE++ Training](../guide/reinforce-plus-plus.md) before enabling
either estimator.

---

## GRPO

GRPO (Group Relative Policy Optimization) is the default algorithm in Relax. It broadcasts the group-relative scalar reward to every token and uses a standard PPO-Clip objective.

Reference: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300).

### How It Works

The GRPO objective is the standard PPO-Clip:

$$J_\text{GRPO}(\theta) = \mathbb{E} \left[ \min\!\left( r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\ 1-\varepsilon,\ 1+\varepsilon)\hat{A}_t \right) \right]$$

where $r_t(\theta) = \pi_\theta / \pi_{\theta_\text{old}}$, and $\hat{A}_t$ is the group-relative advantage (reward minus group mean, normalized by group standard deviation). Gradients are zeroed out when the ratio exceeds the clipping bounds.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--advantage-estimator grpo` | default | Enable GRPO |
| `--eps-clip` | `0.2` | Clipping margin (ratio range = `[1-ε, 1+ε]`) |
| `--eps-clip-high` | same as `--eps-clip` | Upper clipping margin; can be set differently for asymmetric clipping |
| `--clip-grad` | — | Gradient clipping norm |

### Quick Start

GRPO is the default algorithm — no parameter changes needed. Just run the training script directly:

```bash
MODEL_DIR=/path/to/model \
DATA_DIR=/path/to/data \
EXP_DIR=/path/to/exp \
bash scripts/training/text/run-qwen3-4B-8xgpu.sh
```

---

## RLOO

RLOO (REINFORCE Leave-One-Out) uses the other samples for the same prompt as an unbiased baseline. Relax implements synchronous RLOO with an unclipped REINFORCE policy loss; it does not use PPO ratios or clipping.

Reference: [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740).

### How It Works

For a prompt with $G$ sampled responses and scalar rewards $r_i$, the leave-one-out baseline and advantage are:

$$b_i = \frac{1}{G-1}\sum_{j\ne i}r_j, \qquad A_i = r_i-b_i = \frac{G}{G-1}(r_i-\bar r)$$

Unlike GRPO, RLOO does not divide by the group standard deviation. The token loss is:

$$L_{i,t} = -\operatorname{stopgrad}(A_i)\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})$$

Each sample's scalar advantage is broadcast to its response tokens. Relax masks padding and normalizes the summed loss by the global number of valid response tokens $N_{\mathrm{eff}}$:

$$L_{\mathrm{RLOO}} = -\frac{1}{N_{\mathrm{eff}}}\sum_i\sum_t m_{i,t}\operatorname{stopgrad}(A_i)\log\pi_{i,t}$$

This global-token reduction does not apply a separate $1/T_i$ weight to each response. `train/pg_clipfrac` is always `0` because RLOO uses no clipping.

### Requirements and Parameters

| Parameter | Requirement | Description |
|-----------|-------------|-------------|
| `--advantage-estimator rloo` | required | Enable RLOO |
| `--n-samples-per-prompt` | at least `2` | Group size $G$; larger groups provide a more stable LOO baseline at higher rollout cost |
| `--rollout-batch-size × --n-samples-per-prompt` | equals `--global-batch-size` | Exactly one optimizer update per rollout |
| `--num-steps-per-rollout` | unset or `1` | Reusing the same rollout for multiple unclipped updates is rejected |
| `--calculate-per-token-loss` | enabled | Use global valid-token normalization; per-response token means would reweight unequal-length responses by $1/T_i$ |
| `--kl-coef` | `0` | Reward-side KL shaping is not implemented for RLOO; with a valid `--ref-load <checkpoint>`, use `--use-kl-loss --kl-loss-coef <value>` for the supported direct KL penalty |
| `--max-staleness` | `0` | Stale rollouts are rejected because the unclipped objective has no importance-ratio correction |
| reward normalization | enabled | RLOO's group transformation runs in the normalized-reward path |
| `--normalize-advantages` | disabled | Post-DP whitening would change RLOO semantics and make results partition-dependent |
| `--fully-async`, `--hybrid`, `--partial-rollout`, `--use-dynamic-global-batch-size` | disabled | RLOO currently requires synchronous, fixed-size rollout batches |

The batch sizes are hardware-tunable as long as their equality is preserved. For example, `ROLLOUT_BATCH_SIZE=4`, `N_SAMPLES=8`, and `GLOBAL_BATCH_SIZE=32` retain one update per rollout while reducing per-step memory relative to `16 × 8 = 128`.

### Diagnostics

Training rollout logs publish the following final metric names:

- `rollout/rloo/baseline_mean`: mean LOO baseline (equal to the mean group reward; retained as an explicit baseline trace)
- `rollout/rloo/adv_abs_mean`: mean absolute RLOO advantage
- `rollout/rloo/no_signal_frac`: fraction of effective loss tokens attached to zero-advantage samples
- `rollout/rloo/empty_response_frac`: fraction of samples with a literally empty response
- `rollout/rloo/zero_adv_group_frac`: fraction of complete groups with zero advantages throughout
- `rollout/rloo/dropped_group_frac`: fraction of observed groups omitted from diagnostics because their size is incomplete

These diagnostics are training-only, purely observational rollout statistics; they do not affect the training path. Evaluation uses its own sampling group size and does not emit misleading `eval/*/rloo/*` values. They are also omitted when a custom reward post-processor or agentic custom-advantage hook replaces the standard RLOO signal, because raw rewards cannot reconstruct the optimizer input in those modes.

### Quick Start

Use the dedicated Qwen3-0.6B GSM8K recipe. Its batch and rollout settings can be overridden with environment variables:

```bash
MODEL_DIR=/path/to/models \
DATA_DIR=/path/to/data \
NUM_ROLLOUT=60 \
ROLLOUT_BATCH_SIZE=4 \
N_SAMPLES=8 \
GLOBAL_BATCH_SIZE=32 \
bash examples/algorithms/run-qwen3-0.6B-1xgpu-rloo.sh
```

Set `ADVANTAGE_ESTIMATOR=grpo` to run a control arm with the same recipe and seeds. The recipe writes normalized GSM8K data to a writable artifact cache (override with `RLOO_DATA_CACHE_DIR`) and appends an instruction to emit the final answer as `\boxed{...}`, matching the `math` reward parser contract.

---

## PPO

PPO (Proximal Policy Optimization) is an actor-critic algorithm. Relax trains a separate Critic to predict token-level values, computes GAE advantages and returns, applies PPO-Clip to the Actor, and applies clipped value loss to the Critic.

Reference: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).

### How It Works

The temporal-difference residual and GAE recursion are:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$\hat{A}_t = \delta_t + \gamma\lambda\hat{A}_{t+1}, \qquad \hat{R}_t = \hat{A}_t + V(s_t)$$

The Actor then uses the same clipped policy objective shown for GRPO, but with Critic-derived token-level advantages. The Critic minimizes the maximum of clipped and unclipped squared value errors.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--advantage-estimator ppo` | — | Enable PPO and the Critic service graph |
| `--gamma` | `1.0` | GAE discount factor |
| `--lambd` | `1.0` | GAE lambda |
| `--eps-clip` | `0.2` | Actor clipping margin |
| `--value-clip` | `0.2` | Critic value clipping range |
| `--num-critic-only-steps` | `0` | Initial Critic-only warmup steps |
| `--critic-lr` | same as `--lr` | Critic learning rate |

### Quick Start

PPO cannot be enabled by changing only the algorithm argument because its service graph requires `critic` and `advantages` resources. Fully-async PPO is not currently supported; use the dedicated synchronous colocate recipe:

```bash
MODEL_DIR=/path/to/models \
DATA_DIR=/path/to/data \
EXP_DIR=/path/to/experiments \
bash scripts/training/text/run-qwen35-9B-8xgpu-ppo.sh
```

See [PPO Training](../guide/ppo-training.md) for the resource topology, checkpoint rules, and KL constraints.

---

## CISPO

CISPO (Clipped Importance-ratio Soft Policy Optimization) preserves gradient signal for out-of-trust-region tokens instead of zeroing it out. It caps gradient magnitude via a stop-gradient'd coefficient while keeping the gradient direction alive.

Reference: [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585).

### How It Works

The CISPO objective is:

$$J_\text{CISPO}(\theta) = \mathbb{E}_{(q,a)\sim\mathcal{D},\ \{o_i\}_{i=1}^G \sim \pi_{\theta_\text{old}}(\cdot|q)} \left[ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \text{sg}\!\left(\hat{r}_{i,t}(\theta)\right) \hat{A}_{i,t} \log \pi_\theta(o_{i,t} \mid q, o_{i,<t}) \right]$$

where $\hat{r}_{i,t}(\theta)$ is the clipped importance-sampling weight:

$$\hat{r}_{i,t}(\theta) = \text{clip}\!\left(r_{i,t}(\theta),\ 1 - \varepsilon_\text{low}^\text{IS},\ 1 + \varepsilon_\text{high}^\text{IS}\right)$$

and $r_{i,t}(\theta) = \pi_\theta(o_{i,t} \mid q, o_{i,<t}) / \pi_{\theta_\text{old}}(o_{i,t} \mid q, o_{i,<t})$. Gradients flow **only** through $\log\pi_\theta$; both $\hat{r}_{i,t}$ and $\hat{A}_{i,t}$ are stop-gradient'd.

### Key Parameters

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `--advantage-estimator cispo` | — | — | Enable CISPO |
| `--eps-clip` | `0.2` | `0.2` | Lower clipping margin (ratio lower bound = `1 - eps_clip`) |
| `--eps-clip-high` | same as `--eps-clip` | `10` | Upper clipping margin (ratio upper bound = `1 + eps_clip_high`). Set to `10` to effectively unclamp the upper side |
| `--kl-loss-coef` | `0.0` | `0.001` | KL loss coefficient. Recommended: `0.001` to add a small KL penalty that constrains policy drift |
| `--use-kl-loss` | off | on | Enable KL loss computation (required for `--kl-loss-coef` to take effect) |
| `--use-tis` | off | on | Token Importance Sampling — recommended to enable with CISPO |
| `--clip-grad` | — | `1.0` | Gradient clipping norm |

### Quick Start

Use any existing GRPO training script and replace `GRPO_ARGS` with `CISPO_ARGS`:

```bash
CISPO_ARGS=(
   --advantage-estimator cispo
   --use-kl-loss
   --kl-loss-coef 0.001
   --eps-clip 0.2
   --eps-clip-high 10
   --use-tis
)
```

---

## GSPO

GSPO (Group-wise Sequence-level Policy Optimization) differs from GRPO in how KL divergence is computed: GSPO uses **sequence-level** KL instead of per-token KL. Every token in a sequence shares the same KL value (the mean over all tokens in that sequence), providing uniform constraint strength within a sequence.

### How It Works

GSPO uses the same PPO-Clip objective as GRPO, but the ratio is computed from sequence-level KL:

$$\text{KL}_\text{seq} = \frac{1}{|o|} \sum_{t=1}^{|o|} \left(\log\pi_{\theta_\text{old}}(o_t) - \log\pi_\theta(o_t)\right)$$

Every token's ratio is $r_t = \exp(-\text{KL}_\text{seq})$, rather than an independent per-token ratio.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--advantage-estimator gspo` | — | Enable GSPO |
| `--eps-clip` | `0.2` | Clipping margin |
| `--eps-clip-high` | same as `--eps-clip` | Upper clipping margin |
| `--clip-grad` | — | Gradient clipping norm |

### Quick Start

```bash
GSPO_ARGS=(
   --advantage-estimator gspo
   --eps-clip 0.2
)
```

---

## SAPO

SAPO (Soft Adaptive Policy Optimization) replaces hard clipping with a smooth sigmoid gate. The gate's steepness is controlled by a temperature parameter, implementing a differentiable trust region constraint.

### How It Works

SAPO's core is a sigmoid gate centered at ratio=1:

$$f(r) = \frac{4}{\tau} \cdot \sigma\!\left(\tau(r - 1)\right)$$

where $\sigma$ is the sigmoid function, and $\tau$ is selected based on the advantage sign:

- $A > 0$: use $\tau_\text{pos}$ (default 1.0)
- $A \leq 0$: use $\tau_\text{neg}$ (default 1.05, stronger suppression for negative tokens)

SAPO objective: $J_\text{SAPO}(\theta) = f(r) \cdot A$

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--advantage-estimator sapo` | — | Enable SAPO |
| `--sapo-tau-pos` | `1.0` | Temperature for positive advantages |
| `--sapo-tau-neg` | `1.05` | Temperature for negative advantages (higher = stronger suppression) |
| `--clip-grad` | — | Gradient clipping norm |

### Quick Start

```bash
SAPO_ARGS=(
   --advantage-estimator sapo
   --sapo-tau-pos 1.0
   --sapo-tau-neg 1.05
)
```

---

## GDPO

GDPO (Group reward-Decoupled Normalization Policy Optimization, [arXiv 2601.05242](https://arxiv.org/abs/2601.05242)) targets **multi-reward** training. It standardizes each reward component within its prompt group and only then combines them, instead of summing the rewards first and normalizing once as GRPO does.

### How It Works

For prompt $i$ with $G$ rollouts and $n$ reward components:

**Step 1 — per-reward group standardization:**

$$A_k^{(i,j)} = \frac{r_k^{(i,j)} - \mathrm{mean}_j\{r_k^{(i,\cdot)}\}}{\mathrm{std}_j\{r_k^{(i,\cdot)}\} + \epsilon}$$

**Step 2 — weighted sum:**

$$A_\text{sum}^{(i,j)} = \sum_k w_k A_k^{(i,j)}$$

The weights multiply the **normalized** advantages, not the raw rewards. After step 1 every component is on the same scale, so a weight expresses relative importance rather than the component's units.

**Step 3 — batch-wise whitening:**

$$\hat{A}^{(i,j)} = \frac{A_\text{sum}^{(i,j)} - \mathrm{mean}_\text{batch}}{\mathrm{std}_\text{batch} + \epsilon}$$

**Why this beats GRPO:** when one component is constant across a group (reward collapse), GRPO's summed reward collapses too, the whole group's advantages go to zero, and the samples are wasted. Under GDPO only *that component* contributes zero while the others still carry signal.

**On $\epsilon$:** GDPO uses $\epsilon = 10^{-4}$ at both steps, matching the reference implementation (the `scale_rewards` GDPO branch of TRL's `GRPOTrainer`), whereas GRPO / GSPO / SAPO / CISPO keep this repository's existing $10^{-6}$. The two only diverge on near-degenerate groups: with binary rewards and a group of 8 the within-group standard deviation is around 0.4 and the constants differ by 0.02%, but a continuous reward (the paper's maths setup scores response length) can leave a group at a standard deviation of ~$10^{-3}$, where $10^{-4}$ damps that group's signal by about 7% against 0.08% for $10^{-6}$. Groups that collapse *exactly* never reach this division; they are detected by exact equality and zeroed.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--advantage-estimator gdpo` | — | Enable GDPO |
| `--gdpo-reward-keys` | — | **Required**, at least two. Keys in the reward dict to standardize independently, e.g. `correctness format` |
| `--gdpo-reward-weights` | all 1.0 | Per-component weights; length must match `--gdpo-reward-keys` |
| `--reward-key` | — | **Required**; selects the scalar used for metrics and the `raw_reward` column |
| `--n-samples-per-prompt` | — | Must be >= 2 (the unbiased group std is undefined at $G=1$) |

The reward function must return a dict containing every key. A missing key, a non-numeric value, a bool, or NaN/Inf raises rather than defaulting to 0.0 — a silently zeroed component is indistinguishable from a genuinely collapsed one.

### Quick Start

```bash
GDPO_ARGS=(
   --advantage-estimator gdpo
   --gdpo-reward-keys correctness format
   --gdpo-reward-weights 1.0 1.0
   --custom-rm-path examples.gdpo.reward_gdpo.reward_func
   --reward-key score
   --n-samples-per-prompt 8
)
```

A complete runnable example lives in [`examples/gdpo/`](https://github.com/redai-infra/Relax/tree/main/examples/gdpo).

### Known Deviations

Two differences between this implementation and the paper. Confirm they are acceptable before training. Step 3's batch boundary used to be a third; it has since been corrected — see below.

**Step 3's batch boundary (now aligned).** Eq. 6 normalises over one training batch. The caller merges `num_rollout_minis` of them with `concat_rollout_batches` before the advantage stage, so step 3 has to be told where the boundaries are. They travel in `ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY`, which all three actor paths (colocate and hybrid) set; `loss.py` forwards them to the advantage dispatcher as `mini_batch_sizes`, and **only GDPO reads it** — every other estimator absorbs it in `**_unused` and is bit-identical either way. Each segment all-reduces across the data-parallel group, so the statistics cover both a whole training batch and every rank. Why it matters: whitening merged batches centres them all on a pooled mean, and on a measured example four of eight samples **change sign** — a different objective, not a precision difference.

**`--fully-async` remains unsupported** and is rejected during argument validation: it hands advantage computation to the single-replica Advantages deployment, which has no data-parallel group, never sees the batch boundaries, and consumes one `global_batch_size / num_iters_per_train_update` slice at a time — when that quotient is 1 the whitened output is identically zero and the run trains on no signal at all, quietly.

1. **A single reward does not reduce to GRPO.** Step 3 still applies, leaving a positive scalar difference from GRPO (data-dependent, measured around 1.21). Use `--advantage-estimator grpo` if you want GRPO semantics.
2. **$G=2$ discards magnitude.** Any two distinct values standardize to exactly $\pm 1/\sqrt{2}$, so with a group of two the only thing distinguishing components is their weights.

### Mutually Exclusive Options

- `--normalize-advantages`: step 3 already whitens per sequence; adding the token-level pass on top is not meaningful.
- `--custom-reward-post-process-path`: that hook short-circuits reward post-processing entirely, silently skipping steps 1 and 2 while the run still reports itself as GDPO.
- `--agentic-custom-advantage-path`: the second early return in `post_process_rewards`, which likewise returns ahead of the normalizer, with the same consequence. One flag, `AlgorithmSpec.allows_reward_post_process_hooks`, guards both.
- `--fully-async`: see above.

All of these fail during argument validation. Combining it with `--dynamic-sampling-filter-path` logs a warning instead: the built-in `check_reward_nonzero_std` judges a group by the single `--reward-key` scalar and may drop groups whose signal lives in the other components.

---

## Algorithm Comparison

| Algorithm | Advantage Computation | Policy Loss | KL Constraint |
|-----------|----------------------|-------------|---------------|
| **PPO** | Critic values + GAE | PPO-Clip (hard clip) | Disabled in the current synchronous topology |
| **GRPO** | Group-relative reward | PPO-Clip (hard clip) | Optional KL loss |
| **REINFORCE++** | Token KL-to-go return + global token normalization | PPO-Clip (hard clip) | k1 KL in shaped reward |
| **REINFORCE++-baseline** | Inclusive group mean + global token normalization | PPO-Clip (hard clip) | Separate k2 KL loss |
| **CISPO** | Group-relative reward | Stop-gradient coefficient | Recommended KL loss |
| **GSPO** | Group-relative reward | PPO-Clip + sequence-level KL | Sequence-level ratio |
| **SAPO** | Group-relative reward | Sigmoid gate | Temperature-controlled |
| **RLOO** | Leave-one-out baseline | Unclipped REINFORCE | Optional KL loss (same as GRPO) |
| **GDPO** | Per-reward group standardization + weighted sum + batch whitening | PPO-Clip (hard clip) | Optional KL loss |

## Next Steps

- [PPO Training](../guide/ppo-training.md)
- [REINFORCE++ Training](../guide/reinforce-plus-plus.md)
- [Quick Start](../guide/quick-start.md)
- [On-Policy Distillation](./on-policy-distillation.md)
- [Generative Reward Model](./generative-reward-model.md)
