# Algorithm Reference

Relax supports multiple policy gradient algorithms, all selected via the `--advantage-estimator` flag. This document covers PPO and the primary GRPO-family algorithms (for On-Policy Distillation, see the [dedicated page](./on-policy-distillation.md)).

GRPO, CISPO, GSPO, SAPO, and RLOO share the same service topology, so their argument blocks can be swapped in existing scripts. PPO additionally requires a Critic model and an Advantages service; start from the [PPO training recipe](../guide/ppo-training.md) instead of only replacing `GRPO_ARGS`.

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

## RLOO

RLOO (REINFORCE Leave-One-Out) keeps GRPO's group sampling but changes the baseline: each sample is scored against the mean of the **other** samples in its group rather than against the group mean including itself.

Reference: [Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740).

### How It Works

For a group of $k$ responses to the same prompt with rewards $R_1 \dots R_k$:

$$\hat{A}_i = R_i - \frac{1}{k-1}\sum_{j \neq i} R_j$$

Because the baseline excludes $R_i$, it is statistically independent of the sample it evaluates, which makes the estimator unbiased. GRPO's baseline includes $R_i$, so baseline and sample are **positively** correlated ($\operatorname{Cov}(R_i, \bar{R}) = \operatorname{Var}(R)/k$). That correlation shrinks the advantage: $R_i - \bar{R}$ is exactly $\frac{k-1}{k}$ times the leave-one-out value, which is what the $k/(k-1)$ factor below undoes.

The two forms are algebraically related:

$$\hat{A}_i = \frac{k}{k-1}\left(R_i - \bar{R}\right)$$

so RLOO is the group-centered reward rescaled by $k/(k-1)$. Two consequences follow:

- **No standard-deviation scaling.** GRPO divides by the group standard deviation, which reweights groups against each other: a group whose rewards are nearly identical gets its small differences amplified. RLOO applies a single factor that does not depend on the group's spread, so groups keep their relative weight. This is the same concern Dr. GRPO raises about std normalization.
- **Advantages still sum to zero** within each group, so the reduction and logging behaviour is identical to GRPO.

#### Relationship to `--disable-grpo-std-normalization`

Worth being explicit, because the two are closely related: with std normalization disabled, GRPO produces exactly $R_i - \bar{R}$, so

$$\hat{A}^\text{RLOO}_i = \frac{k}{k-1}\,\hat{A}^{\text{GRPO,\,no-std}}_i$$

element-wise. Relax requires every group to have exactly `--n-samples-per-prompt` samples, so $k$ is constant within a run and that factor is a **global constant**. RLOO is therefore not a different reweighting from `--disable-grpo-std-normalization`; it is that estimator with the specific constant that makes the leave-one-out baseline unbiased. Under Adam a global constant on the loss largely cancels in the update, so choose RLOO when you want the named, unbiased estimator — not in the expectation of a different gradient direction.

The comparison that does differ is against **default** GRPO, which divides by the per-group std and so does reweight groups relative to each other.

#### Effect on gradient clipping

RLOO's advantages are smaller in magnitude than default GRPO's, because the group std for binary rewards is typically well below 1. Measured on Qwen3-0.6B / GSM8K at $k = 8$ and identical `--lr`, `train/grad_norm` averaged 0.47 for RLOO versus 1.04 for GRPO.

Adam is largely invariant to a constant gradient rescale, so this is mostly *not* a reason to retune `--lr`. It does matter for `--clip-grad`, which breaks that invariance: at Megatron's default `--clip-grad 1.0`, the same run clipped **68% of steps under GRPO but only 2% under RLOO**. Revisit `--clip-grad` rather than `--lr` when switching.

#### Reading the reported loss

Two things about `train/pg_loss` differ from the clipped estimators and are expected rather than symptoms:

- **It is routinely negative.** PPO-Clip takes a maximum of two negated terms, which biases it positive. RLOO reports `-A · log π`, and since `log π < 0` this equals `A · |log π|`; a negative mean simply says that positive-advantage tokens carry smaller `|log π|`, i.e. the model is already more confident on the responses that scored well. Measured on Qwen3-0.6B / GSM8K: `-0.026` for RLOO against `+0.045` for GRPO on the same data.
- **It is unbounded below.** For a token with `A < 0`, minimising `-A · log π` pushes `log π` toward `-∞`; nothing caps it, because there is no trust region. This is the property clipping was introduced to remove, and it is intrinsic to unclipped REINFORCE rather than specific to this implementation. In a 60-step run it stayed well behaved (`grad_norm` 0.45, entropy 0.49, no divergence), but **watch `train/entropy_loss` and `train/grad_norm` rather than `pg_loss` for stability**, and reach for `--clip-grad` if either drifts.

Relatedly, `rloo_no_signal_fraction` is worth watching from the first step. On the same run it averaged **0.48 and rose to 0.65** over 60 rollouts, reaching 1.0 on individual steps: roughly half the tokens sat in groups that scored identically and therefore contributed no gradient. That is reward saturation, visible long before the reward curve flattens.

#### The loss is REINFORCE, not PPO-Clip

RLOO's objective is plain REINFORCE against the leave-one-out baseline, with no trust region:

$$L_i = -\operatorname{sg}(\hat{A}_i)\,\log \pi_\theta(y_i)$$

This is a deliberate departure from the other group-relative estimators here, which all wrap PPO-Clip. Reusing the clipped objective would make this "GRPO with a leave-one-out baseline" rather than RLOO as published, and the clipping would bias an estimator chosen precisely for being unbiased. Consequently **`--eps-clip` and `--eps-clip-high` have no effect under RLOO**, and `train/pg_clipfrac` is always 0.

Because there is no importance-ratio correction, RLOO assumes the sampling policy equals the training policy. That holds with one optimizer step per rollout; with several inner epochs the estimator becomes off-policy and the paper's guarantees no longer apply. The provided recipe keeps one step per rollout.

Everything else downstream — KL loss, TIS, DP/CP splitting, the reduction — is unchanged.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--advantage-estimator rloo` | — | Enable RLOO |
| `--n-samples-per-prompt` | `1` | Must be $\geq 2$; a group of one has no leave-one-out baseline and gets a zero advantage |
| `--disable-rewards-normalization` | off | Leave off. This flag skips the group baseline entirely, leaving the raw reward as the advantage |
| `--disable-grpo-std-normalization` | off | No effect under RLOO — the std branch is GRPO-family only, so there is nothing to disable |
| `--eps-clip` | `0.2` | **No effect** — RLOO does not clip; `train/pg_clipfrac` stays 0 |
| `--fully-async` / `--hybrid` | off | **Rejected at startup.** RLOO is synchronous-only and asynchronous RLOO was never validated |

### Quick Start

Replace `GRPO_ARGS` with `RLOO_ARGS` in any GRPO script:

```bash
RLOO_ARGS=(
   --advantage-estimator rloo
   --eps-clip 0.2
)
```

A complete single-GPU recipe on GSM8K is provided:

```bash
MODEL_DIR=/path/to/models \
DATA_DIR=/path/to/data \
bash examples/algorithms/run-qwen3-0.6B-1xgpu-gsm8k-rloo.sh
```

::: warning
RLOO is **synchronous-only and enforced as such**: `--fully-async` and `--hybrid` are
rejected at startup rather than silently allowed, because asynchronous RLOO is out of
scope and was never validated. Use `--colocate`.
:::

### When to Use

Against **default** GRPO, the substantive difference is that RLOO does not divide by the per-group standard deviation:

- Reward distributions where the group std is small and unstable, so dividing by it reweights groups by how uniform they happen to be rather than by how informative they are
- Workloads where `--clip-grad` binds often under GRPO — RLOO's smaller advantage magnitudes leave more steps unclipped (see above)
- As a named, unbiased reference point when diagnosing whether the advantage estimator is implicated in a training pathology

Against `--disable-grpo-std-normalization`, the difference is only the constant $k/(k-1)$, which is what makes the leave-one-out baseline unbiased. That correction is larger for small groups ($1.33$ at $k = 4$, $1.07$ at $k = 16$), but it is still a constant, so do not expect a behavioural change from it under Adam.

---

## Algorithm Comparison

| Algorithm | Advantage Computation | Policy Loss | KL Constraint |
|-----------|----------------------|-------------|---------------|
| **PPO** | Critic values + GAE | PPO-Clip (hard clip) | Disabled in the current synchronous topology |
| **GRPO** | Group-relative reward | PPO-Clip (hard clip) | Optional KL loss |
| **CISPO** | Group-relative reward | Stop-gradient coefficient | Recommended KL loss |
| **GSPO** | Group-relative reward | PPO-Clip + sequence-level KL | Sequence-level ratio |
| **SAPO** | Group-relative reward | Sigmoid gate | Temperature-controlled |
| **RLOO** | Leave-one-out baseline, no std scaling | **Unclipped REINFORCE** | Optional KL loss |

## Next Steps

- [PPO Training](../guide/ppo-training.md)
- [Quick Start](../guide/quick-start.md)
- [On-Policy Distillation](./on-policy-distillation.md)
- [Generative Reward Model](./generative-reward-model.md)
