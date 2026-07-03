# MTP RL Training

This guide shows how to jointly train a model's Multi-Token Prediction (MTP) head with the policy during RL post-training in Relax. The MTP head is trained as an auxiliary objective alongside the main GRPO loss — the same pattern slime uses.

Make sure you have completed [Installation](./installation.md) and have run at least one baseline GRPO job (see [Quick Start](./quick-start.md)) before enabling MTP.

## When to enable MTP RL training

Enable MTP joint training when you want the MTP head's weights to stay calibrated with the evolving policy during RL, so that the same checkpoint can later serve speculative decoding (EAGLE/NEXTN in SGLang or vLLM) without a separate distillation pass.

If you only need the base policy and have no downstream speculative-decoding plan, leave MTP off — the auxiliary loss adds a small amount of compute and memory per step.

## Prerequisites

- The HF checkpoint must already contain MTP weights, i.e. `config.json` has `num_nextn_predict_layers >= 1`. Today this covers **Qwen3.5**, **Qwen3-next**, **MiMo-7B-RL**, **DeepSeek-V3 / V3.1**, and **GLM-4.7-MoE**.
- The Megatron backend (this is the only training backend in Relax). MTP requires the Megatron MTP patch shipped at [`docker/patch/megatron/20260506-85bced0ae.patch`](../../../docker/patch/megatron/20260506-85bced0ae.patch); the official Relax image applies it automatically.
- Combined 1F1B pipeline schedule must be off — it is incompatible with MTP and is asserted out at [`relax/backends/megatron/model.py:493`](../../../relax/backends/megatron/model.py).

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--enable-mtp-training` | off | Switches on MTP forward injection and MTP auxiliary loss. |
| `--mtp-num-layers` | required when enabled | Number of MTP layers in the model. Must match the HF checkpoint. |
| `--mtp-loss-scaling-factor` | `0.1` (recommended for RL) | Scalar multiplied onto the MTP loss before it is added to the main loss. |

The first two are validated jointly at [`relax/utils/arguments.py:2765-2766`](../../../relax/utils/arguments.py). All three are inherited from the Megatron parser, so they are accepted by every Relax launch script without code changes.

## Scripts

| Script | Model | Resources | Notes |
| --- | --- | --- | --- |
| [`run-qwen35-9B-mtp-8xgpu.sh`](../../../scripts/training/text/run-qwen35-9B-mtp-8xgpu.sh) | `Qwen3.5-9B` | 8 GPU colocate | Dense model, cheap smoke test before scaling up. |
| [`run-qwen35-35B-A3B-mtp-16xgpu.sh`](../../../scripts/training/text/run-qwen35-35B-A3B-mtp-16xgpu.sh) | `Qwen3.5-35B-A3B` | 16 GPU (2-node) colocate | Production target. Mirrors the baseline GRPO script plus `MTP_ARGS`. |

Both scripts read the same env-var overrides:

```bash
MTP_NUM_LAYERS=1 MTP_LOSS_SCALING_FACTOR=0.1 \
  bash scripts/training/text/run-qwen35-9B-mtp-8xgpu.sh
```

## Tuning the scaling factor

The default `0.1` is conservative for RL because the main GRPO loss carries gradient noise from advantage estimation. If you observe:

- `train/mtp_loss` plateaus very early while `train/loss` looks normal → try `0.2`–`0.3`.
- `train/loss` becomes unstable after enabling MTP → drop to `0.05`.
- MTP grads dominate (look at `train/grad_norm` relative to the no-MTP baseline) → drop scaling.

For comparison, the SFT MTP script uses `0.2` because SFT gradients are cleaner. The slime defaults are also `0.2`; lowering for RL is a Relax-specific recommendation.

## What to watch in logs

A healthy run emits, per train step:

```text
train/loss              # main GRPO loss, comparable to baseline
train/grad_norm         # comparable to baseline (≤2×)
train/mtp_loss          # bounded, gradually decreasing
```

If `train/mtp_loss` never appears, the MTP block was not built — usually a checkpoint mismatch (the HF ckpt has no MTP weights). Confirm with:

```bash
python -c "from transformers import AutoConfig; \
  print(AutoConfig.from_pretrained('/path/to/ckpt').num_nextn_predict_layers)"
```


## See also

- [SFT Training](./sft-training.md) — the MTP head can also be trained in SFT, see `run-qwen3.5-35B-A3B-mtp-sft-16xgpu.sh`.
- [Update Weights Pipeline](./update-weights-pipeline.md) — how Megatron→SGLang weight sync handles MTP layers.
