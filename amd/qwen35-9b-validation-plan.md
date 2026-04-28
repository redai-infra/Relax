# Qwen3.5-9B DAPO-Math AMD Validation Plan

## Goal

Validate Qwen3.5-9B DAPO-Math on AMD ROCm with a staged path:

1. first prove the Relax service/data/training loop closes,
2. then verify numerical health,
3. then scale toward a baseline-like run and evaluate quality.

## Current Questions

- TensorBoard is acceptable as the first metrics UI.
- AIME is useful as a held-out evaluation benchmark, but it is not the first signal for “training is learning.”
- For early runs, use train reward, passrate, KL/logprob/entropy, response length, and absence of NaN/OOM as health signals.
- Use AIME after the smoke and small learning run are stable, because AIME is small and high variance.

## Checklist

- [x] Enable TensorBoard logging for Qwen3.5 runner.
- [x] Start TensorBoard server on port `6006`.
- [x] Select free GPUs dynamically before starting Ray; exit cleanly if fewer than the requested GPUs are available.
- [ ] Tiny smoke: complete one full step through rollout, reference logprob, actor_fwd logprob, advantages, actor train, and weight sync.
- [ ] Small learning run: run `NUM_ROLLOUT=16` or `32` with short response length and verify reward/passrate/KL/logprob health.
- [ ] Baseline-like run: raise `n_samples_per_prompt` toward `8`.
- [ ] Baseline-like run: raise `rollout_max_response_len` toward `4096` and then `8192`.
- [ ] Enable AIME eval after the small learning run is stable.
- [ ] Compare curves: train reward/passrate, AIME pass@k, KL, entropy, response length, and step time.

## Why Not Start With AIME?

AIME is a good downstream quality check, but it is a poor first debugging signal.

The first AMD validation question is whether every Relax role works together:

- `rollout` can generate samples with SGLang.
- `reference` can compute reference logprob.
- `actor_fwd` can compute current actor logprob.
- `advantages` can compute advantage and return.
- `actor` can consume the batch and update weights.
- DCS can synchronize weights back to rollout and actor_fwd.

Only after that loop closes should AIME be used to judge whether quality improves.

## TensorBoard

The runner writes TensorBoard logs under:

```text
/data/models/minimax/Relax/amd/tensorboard/qwen35-9b
```

Open with:

```bash
tensorboard --logdir /data/models/minimax/Relax/amd/tensorboard --host 0.0.0.0 --port 6006
```

## 2026-04-28 Dynamic GPU Run Notes

- Dynamic GPU selection worked: with GPUs `0,1,2,3,4,7` above the free-VRAM threshold, the runner selected a 6-GPU profile.
- The generated resource plan was `actor=2`, `rollout=2`, `reference=1`, `actor_fwd=1`, `advantages=0`.
- TensorBoard metrics service initialized successfully and wrote to `/data/models/minimax/Relax/amd/tensorboard/qwen35-9b`.
- The run reached service startup, SGLang readiness, rollout generation, and step-0 actor/reference/actor_fwd execution.
- New blocker: DCS/NCCL timeout in `update_actor_pp_0` during actor weight synchronization, with `WorkNCCL(... OpType=BROADCAST ...)` timeout.
- Current suspicion: the 6-GPU topology with `actor` TP=2 and rollout/ref/actor_fwd split across the remaining GPUs needs extra DCS topology validation before using it as the default profile.
- Decision: skip the 6-GPU profile for now. The adaptive runner now supports only `4` and `8` GPU profiles by default.

## 2026-04-28 4-GPU Smoke Notes

- The runner correctly fell back from 6 eligible GPUs to the supported 4-GPU profile.
- Resource plan: `actor=1`, `rollout=1`, `reference=1`, `actor_fwd=1`, `advantages=0`.
- TensorBoard adapter initialized successfully.
- All 5 services registered successfully.
- Actor, rollout, reference, actor_fwd, and advantages entered step 0.
- Rollout 0 completed and wrote a 2-sample batch to TransferQueue.
- New blocker: DCS/NCCL timeout still occurred in `update_actor_pp_0` during actor weight synchronization (`BROADCAST` timeout).
- Conclusion: the next debugging target is DCS actor-to-rollout/actor_fwd weight synchronization, not GPU selection or rollout memory.
