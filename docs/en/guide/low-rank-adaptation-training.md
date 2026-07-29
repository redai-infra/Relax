# LoRA RL (Parameter-Efficient RL Post-Training)

Relax supports **LoRA** (Low-Rank Adaptation) for RL post-training: instead of updating every weight, only small low-rank adapter matrices are trained while the base model stays frozen. This shrinks the optimizer state and the per-step weight-sync payload, so much larger models fit on the same GPU budget.

Two end-to-end LoRA rollout paths are wired up, differing only in **how the trained adapter reaches the rollout engine**:

| Mode              | What is synced to rollout each step          | Rollout serves         | Reference launch script                                                |
| ----------------- | -------------------------------------------- | ---------------------- | --------------------------------------------------------------------- |
| **Merge mode**    | Full model (adapter folded into base)        | one merged model       | `scripts/training/text/run-qwen3-4B-lora-merge-8xgpu.sh`               |
| **Adapter mode**  | Base once, then only the LoRA adapter        | base + runtime adapter | `scripts/training/text/run-qwen3-4B-lora-adapter-8xgpu-async.sh`       |

The two modes are **mutually exclusive**. If you enable LoRA (`--lora-rank > 0`) without picking a mode, Relax forces **merge mode** (the default, broadest-support path).

## Overview

LoRA is applied on the training side through Megatron-Bridge's PEFT integration (`relax/backends/megatron/model_provider.py`): the base model is frozen and only the injected adapter parameters receive gradients. The difference between the two modes is entirely on the **weight-sync path** (`relax/backends/megatron/weight_update/`):

- **Merge mode** folds each adapter into its base weight at export time (`LoRAMerge`), so the rollout engine loads a single merged model — no LoRA awareness required on the inference side. This reuses the standard full-weight-sync path, so bandwidth per step is the same as a full-parameter run.
- **Adapter mode** syncs the frozen base **once**, then every step pushes **only the small adapter** to the rollout engine via SGLang's runtime LoRA API. Rollout requests select it with `lora_path`. This trades a bit of rollout-side LoRA overhead for a large drop in per-step sync bandwidth.

::: tip Which mode should I use?
- **Merge mode** — simplest, works with the widest set of models and both deployment layouts (including distributed/NCCL rollout engines). Rollout inference is plain full-model inference. Costs a full weight sync every step.
- **Adapter mode** — best when the per-step weight-sync bandwidth dominates (large base model, small adapter). Adds constraints (`--sglang-dp-size 1`, no distributed rollout engines, dense models only in fully-async). Rollout runs through SGLang's LoRA kernels.
:::

## Architecture

Both modes share the same training side; only the payload on the sync arrow differs.

```
        ┌────────────────────────────────────────────────────────┐
        │                  Training side (Actor)                 │
        │  Megatron-LM + Megatron-Bridge PEFT                    │
        │  base weights FROZEN, only LoRA adapter has gradients  │
        │  (model_provider.wrap_model_provider_with_lora)        │
        └───────────────────────────┬────────────────────────────┘
                                     │
             ┌───────────────────────┴───────────────────────┐
             │                                               │
     MERGE mode: fold adapter                      ADAPTER mode: base once,
     into base, sync FULL model                    then push ADAPTER only
     every step (NCCL / IPC)                       (SGLang runtime LoRA API)
             │                                               │
             ▼                                               ▼
        ┌─────────────────────────┐              ┌─────────────────────────┐
        │  Rollout (SGLang)       │              │  Rollout (SGLang)       │
        │  one merged model       │              │  base + adapter         │
        │  plain inference        │              │  lora_path=             │
        │                         │              │   relax_policy_lora     │
        └─────────────────────────┘              └─────────────────────────┘
```

The weight-sync backend dispatches on the LoRA mode:

- **Colocate** — `UpdateWeightFromTensor` (`weight_update/update_weight_from_tensor.py`). Merge mode folds adapters during HF export; adapter mode uses `_update_weights_adapter_mode` (base-once + per-step in-memory adapter push via `load_lora_adapter_from_tensors`).
- **Fully-async** — `DeviceDirectBackend` (`distributed/checkpoint_service/backends/device_direct.py`). Merge mode folds adapters before the NCCL broadcast; adapter mode writes the adapter to a shared directory once per step and fans out an HTTP `/load_lora_adapter` to every engine.

The mode-independent adapter export/gather logic (build the HF export bridge, export the local adapter, PP-gather, delta-skip, write the HF-PEFT dir) lives in the shared `LoraAdapterSync` helper (`weight_update/lora_adapter_sync.py`), which both backends compose.

## Configuration

All LoRA flags are defined in `relax/utils/arguments.py`.

| Flag                     | Type       | Default                       | Description                                                                                          |
| ------------------------ | ---------- | ----------------------------- | ---------------------------------------------------------------------------------------------------- |
| `--lora-rank`            | int        | `0`                           | LoRA rank. `0` disables LoRA; any value `> 0` enables it.                                             |
| `--lora-alpha`           | int        | `32`                          | LoRA alpha scaling factor.                                                                            |
| `--lora-target-modules`  | str (list) | `linear_qkv linear_proj`      | Megatron-style module names to adapt (see mapping below). Space-separated.                            |
| `--lora-dropout`         | float      | `0.0`                         | Dropout probability applied inside the LoRA layers.                                                   |
| `--lora-merge-mode`      | flag       | `False`                       | Fold adapters into base weights before sync. Mutually exclusive with `--lora-adapter-mode`.           |
| `--lora-adapter-mode`    | flag       | `False`                       | Sync base once, then push only the adapter each step. Mutually exclusive with `--lora-merge-mode`.    |

### Target modules

`--lora-target-modules` takes **Megatron-style** names (the canonical form Megatron-Bridge's LoRA matcher walks). They are expanded to HF-style names automatically when exporting the adapter (`convert_megatron_to_hf_target_modules` in `relax/utils/megatron_peft_utils.py`):

| Megatron name          | HF projection(s)              |
| ---------------------- | ----------------------------- |
| `linear_qkv`           | `q_proj`, `k_proj`, `v_proj`  |
| `linear_proj`          | `o_proj`                      |
| `linear_fc1`           | `gate_proj`, `up_proj`        |
| `linear_fc2`           | `down_proj`                   |
| `router`               | `gate`                        |

(The full map, including split-QKV and MLA variants, is in `MEGATRON_TO_HF_MODULES`.)

### Validation rules

Enforced in `relax/utils/arguments.py` when `lora_rank > 0`:

- `--lora-merge-mode` and `--lora-adapter-mode` are **mutually exclusive** — pick one.
- `--lora-adapter-mode` requires `--sglang-dp-size 1` (SGLang dynamic LoRA loading does not support DP attention).
- If neither mode flag is set, Relax **forces `--lora-merge-mode`** and logs a warning.

## Merge Mode Recipe

Reference: `scripts/training/text/run-qwen3-4B-lora-merge-8xgpu.sh` (Qwen3-4B, 8-GPU colocate, GRPO on `dapo-math-17k`).

The LoRA block in that script:

```bash
LORA_ARGS=(
   --lora-rank 32
   --lora-alpha 64
   --lora-target-modules linear_qkv linear_proj
   --lora-dropout 0.0
   --lora-merge-mode
)
```

Launch it like any other colocate script:

```bash
bash scripts/training/text/run-qwen3-4B-lora-merge-8xgpu.sh
```

How it works each step: the actor exports its weights, `LoRAMerge` folds every adapter into its paired base weight during HF conversion, and the merged full model is synced to SGLang over the normal IPC/NCCL path. The rollout engine is completely LoRA-agnostic — it just serves a full model.

::: warning Merge mode in the fast bridge path
The fast bridge path requires `--expert-tensor-parallel-size 1` for merge mode (expert LoRA merge selects a per-expert slice locally on the owning EP rank; a mismatched ETP would deadlock the collective). The reference script already sets `--expert-tensor-parallel-size 1`.
:::

## Adapter Mode Recipe

Reference: `scripts/training/text/run-qwen3-4B-lora-adapter-8xgpu-async.sh` (Qwen3-4B, 8-GPU fully-async, GRPO on `dapo-math-17k`).

The LoRA block in that script:

```bash
LORA_ARGS=(
   --lora-rank 128
   --lora-alpha 64
   --lora-target-modules linear_qkv linear_proj
   --lora-dropout 0.0
   --lora-adapter-mode
)
```

Note it also sets `--rollout-num-gpus-per-engine 1` (one GPU per engine → `sglang_dp_size == 1`, required by adapter mode). Launch it:

```bash
bash scripts/training/text/run-qwen3-4B-lora-adapter-8xgpu-async.sh
```

How it works:

1. **First sync** — push base-only weights to the engine (adapter params are pulled out of the conversion buckets, never merged), then register the LoRA adapter under the fixed name `relax_policy_lora`.
2. **Every subsequent step** — refresh **only** the adapter. The base stays put on the engine. Rollout requests automatically pass `lora_path=relax_policy_lora` (`relax/engine/rollout/sglang_rollout.py`) so generation runs through the trained adapter.
3. **Delta-skip** — if no adapter parameter changed beyond a `1e-6` threshold on any rank, the whole push is skipped. This is a collective decision across all ranks (a per-rank early return would desync the gather and hang).

The transport of the adapter itself differs by deployment:

- **Colocate** — the adapter is serialized and pushed in-memory via `load_lora_adapter_from_tensors` (SGLang ≥ 0.5.12, no disk IO).
- **Fully-async** — the merged adapter is written **once** to a shared directory, then an HTTP `/load_lora_adapter` is fanned out to every engine. The directory defaults to `<args.save>/relax_lora_live/adapter`; override with the `RELAX_LORA_LIVE_DIR` environment variable (must be readable by every rollout engine — do **not** point it at node-local storage in fully-async).

::: warning Adapter mode constraints
- `--sglang-dp-size 1` is required (enforced at arg-validation time).
- **Distributed (non-colocated NCCL) rollout engines are not supported** in colocate mode — the adapter is only pushed to colocated IPC engines. Use colocate or fully-async; for distributed rollout use merge mode instead.
- In the **fully-async** path, MoE (grouped-expert) LoRA is **not supported** (the per-expert merge math differs from dense). Use a dense model, or colocate mode for expert LoRA.
- Adapter mode disables next-step rollout prefetch in colocate (the per-step adapter update is ~1s, so prefetch would just be re-done).
:::

## Checkpointing and Export

When LoRA is enabled, checkpoint save also writes a **portable HF-PEFT adapter** under `<checkpoint_dir>/lora_adapter/` (`relax/backends/megatron/checkpoint.py`):

- `adapter_config.json` + `adapter_model.safetensors` — standard HF-PEFT layout, loadable with `peft.PeftModel.from_pretrained`.
- `relax_lora_meta.json` — Relax metadata (rank, alpha, target modules, dropout, mode). Kept as a separate file so it never confuses a standard PEFT loader.

::: tip
This `lora_adapter/` directory is an **export artifact** for external / inference use — it is **not** the resume source. LoRA parameters are ordinary model parameters saved inside the main Megatron checkpoint, so `--load` resumes them like any other weight.
:::

## Troubleshooting

| Symptom                                                                        | Likely cause / fix                                                                                                                            |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `--lora-merge-mode and --lora-adapter-mode are mutually exclusive`             | Both flags set. Pick exactly one.                                                                                                            |
| `--lora-adapter-mode requires --sglang-dp-size 1`                             | Adapter mode with DP attention. Set `--rollout-num-gpus-per-engine 1` (or otherwise force `sglang_dp_size == 1`).                            |
| `--lora-adapter-mode does not yet support distributed ... rollout engines`     | Adapter mode with non-colocated rollout engines. Switch to colocate, or use `--lora-merge-mode` for distributed rollout.                    |
| `MoE (grouped-expert) LoRA is not supported in fully-async weight sync`        | Expert LoRA in fully-async. Use a dense model, or run expert LoRA in colocate mode.                                                          |
| `[lora-merge] NO adapter tensors in backup dict`                              | Adapter params missing from `weights_getter()` output — check `--lora-target-modules` names and that LoRA actually attached at model build.  |
| Rollout quality looks like the base model in adapter mode                      | The adapter push was skipped or `lora_path` not applied. Confirm requests carry `lora_path=relax_policy_lora` and the adapter dir is shared. |
