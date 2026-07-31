# REINFORCE++ and REINFORCE++-baseline

Relax exposes two separate estimator names because the baseline variant is not
just a different recipe:

- `reinforce_plus_plus`
- `reinforce_plus_plus_baseline`

This page fixes their return, advantage, normalization, mask, KL and reduction
semantics. The implementation follows the main equations of
[REINFORCE++ arXiv v9](https://arxiv.org/abs/2501.03262) and the executable
normalization convention in OpenRLHF commit
[`bc71bb1`](https://github.com/OpenRLHF/OpenRLHF/tree/bc71bb19464aca306b33080b2d2bb45d154e2f49).

## Version and naming note

The paper changed between v1 and v9. In addition, v9's main method and Appendix
B.2 do not describe exactly the same token placement:

- the main method defines a token KL-to-go return and global normalization;
- Appendix B.2 describes zero advantage before the final token and sample-level
  reward normalization.

Relax selects the v9 main-equation interpretation. The implementation is
therefore described as **v9 main-equation and OpenRLHF aligned**, rather than as
an implementation of every sentence in the appendix.

## Notation and mask contract

For response (i) and response position (t):

- (R_i) is the scalar terminal reward;
- (m_{i,t}\in\{0,1\}) is the response loss mask;
- (T_i) is the final valid response position;
- (L_i=\sum_t m_{i,t}) is the valid response length.

Prompt tokens, padding and response tokens with `mask=0` do not contribute to
reward shaping, return, normalization or loss. Production return and advantage
tensors are explicitly zero outside the mask.

## REINFORCE++

Let the signed k1 estimator be

$$
d_{i,t}=\log\pi_{old}(a_{i,t})-\log\pi_{ref}(a_{i,t}).
$$

The shaped token reward is

$$
r_{i,t}=m_{i,t}\left[-\beta d_{i,t}+\mathbf{1}(t=T_i)R_i\right].
$$

The terminal reward is added only to the final valid response token. Returns
are accumulated backwards:

$$
G_{i,t}=m_{i,t}\sum_{u=t}^{T_i}\gamma^{u-t}r_{i,u}.
$$

The formal recipe fixes `gamma=1`. The raw advantage is (G), followed by the
global masked normalization below. This variant uses token KL reward shaping
and does not add a second KL loss.

```text
--advantage-estimator reinforce_plus_plus
--normalize-advantages
--gamma 1.0
--kl-coef 0.01
--kl-loss-type k1
```

## REINFORCE++-baseline

For a prompt group (g) with (K) sampled responses,

$$
b_g=\frac{1}{K}\sum_{j\in g}R_j,\qquad C_i=R_i-b_g.
$$

The group mean includes the current response. It is not a leave-one-out
baseline. Relax does not divide (C_i) by a group standard deviation.

The raw token advantage is

$$
A^{raw}_{i,t}=m_{i,t}C_i.
$$

Token KL is not subtracted from this advantage. Reference regularization is a
separate k2 loss:

$$
D^{k2}_{i,t}=\frac{1}{2}
\left(\log\pi_\theta(a_{i,t})-\log\pi_{ref}(a_{i,t})\right)^2.
$$

```text
--advantage-estimator reinforce_plus_plus_baseline
--normalize-advantages
--n-samples-per-prompt 8
--kl-coef 0
--use-kl-loss
--kl-loss-type k2
--kl-loss-coef 0.01
```

The baseline variant requires more than one sample per prompt. The group mean
includes each sample itself, so `n_samples_per_prompt=1` would collapse every
raw advantage to zero and is rejected.

## Global masked normalization

The statistical population consists of every valid response token in the
closed synchronous global batch across all data-parallel ranks:

$$
S=\{(r,i,t)\mid m_{r,i,t}=1\},\qquad N=|S|.
$$

Relax uses population variance (`ddof=0`):

$$
\mu=\frac{1}{N}\sum_{S}A,
\qquad
\sigma^2=\frac{1}{N}\sum_{S}(A-\mu)^2.
$$

The normalized output is

$$
\hat A=m(A-\mu)\left[\max(\sigma^2,10^{-8})\right]^{-1/2}.
$$

The epsilon convention is a **variance floor**, matching the pinned OpenRLHF
implementation. It is different from both `sqrt(var) + epsilon` and Relax's
legacy `sqrt(unbiased_var + epsilon)` helper. The REINFORCE++ variants use a
dedicated helper; existing algorithms keep their current normalization.

Expected edge behavior:

- zero variance and a single valid token produce finite zero advantages;
- an all-zero baseline reward group produces finite zeros;
- an all-zero reward with nonzero KL can produce finite KL-shaped
  REINFORCE++ returns;
- a globally empty mask raises `ValueError` on every participating rank.

Because the baseline scalar is broadcast to each valid token, longer responses
have greater weight in these token-level global moments. This is intentional.

## PPO and KL reduction

Both variants use the ordinary token PPO clipped surrogate. Their formal scalar
objective is a response mean:

$$
L_{PG}=\frac{1}{B}\sum_i\frac{1}{L_i}
\sum_t m_{i,t}\max\left(
-\rho_{i,t}\hat A_{i,t},
-\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_{i,t}
\right).
$$

The baseline k2 loss uses the same response-mean reduction. The initial
implementation rejects `--calculate-per-token-loss` for these variants because
it changes the objective to a global token mean.

## Comparison

| Algorithm | Advantage | Ratio/loss | Reference regularization |
|---|---|---|---|
| REINFORCE++ | token KL-to-go return, then global valid-token normalization | token PPO clip | k1 inside token reward |
| REINFORCE++-baseline | inclusive group-mean baseline, then global valid-token normalization | token PPO clip | separate k2 loss |
| GRPO | group mean and group standard deviation | token PPO clip | existing configurable Relax KL |
| GSPO | GRPO group advantage | response-level ratio expanded to tokens | existing configurable Relax KL |
| SAPO | GRPO group advantage | sigmoid-shaped token ratio | existing configurable Relax KL |

This feature does not change GRPO, GSPO or SAPO defaults.

## Supported modes

The first implementation supports:

- synchronous colocate training;
- data-parallel normalization;
- `context_parallel_size=1`;
- response-mean loss reduction.

It rejects fully-async, hybrid, context parallelism greater than one, and
per-token global loss reduction. Fully-async does not currently define a closed
global batch over which these moments can be calculated, while CP greater than
one requires a separately verified unique-token ownership contract.

## Monitoring

The rollout metrics include:

- raw global advantage mean and standard deviation;
- normalized advantage mean and standard deviation;
- valid-token count;
- zero-variance indicator;
- ordinary reward, return and advantage summaries.

Training metrics continue to report policy loss, PPO KL, clip fraction and the
independent k2 loss when enabled.

## Testing

The numerical tests use an independent float64 reference that does not invoke
the production return, advantage, normalization or loss functions. Coverage
includes variable response lengths, padding, internal mask holes, large values
outside the mask, zero rewards, zero variance, a single valid token, PPO
clipping and response-reduced policy/k2 losses.

Distributed normalization is tested with two real Gloo processes and a real
`all_reduce`, including a case where one rank has no valid token. Its output is
compared with the independently concatenated global population.

See
`examples/algorithms/run-qwen3-0.6B-1xgpu-reinforce-plus-plus.sh` for the
parameterized Qwen3-0.6B recipe.

The equal-budget Qwen3-0.6B stability experiment, numerical evidence, curves,
and comparison with GRPO are documented in the
[training and numerical validation report](./reinforce-plus-plus-training-report.md).
