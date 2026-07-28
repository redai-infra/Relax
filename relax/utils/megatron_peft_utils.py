# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Utilities for PEFT (Parameter-Efficient Fine-Tuning) of Megatron in
Relax."""

from typing import Tuple

import torch


# Fixed name under which the trained policy LoRA adapter is registered on the rollout
# engines in adapter mode. Generation requests must pass ``lora_path=LORA_ADAPTER_NAME``
# for the adapter to take effect (see sglang_rollout.generate and _push_lora_adapter).
LORA_ADAPTER_NAME = "relax_policy_lora"


def count_adapter_parameters(model) -> Tuple[int, int, float]:
    """Count the number of trainable adapter parameters.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (adapter_params, total_params, percentage)
    """
    from megatron.core.utils import unwrap_model

    unwrapped = unwrap_model(model)
    if not isinstance(unwrapped, list):
        unwrapped = [unwrapped]

    adapter_params = 0
    total_params = 0

    # Sum across every virtual-pipeline chunk; with VPP `model` is a list of chunks.
    for chunk in unwrapped:
        for name, param in chunk.named_parameters():
            total_params += param.numel()
            if is_lora_adapter_param(name) and param.requires_grad:
                adapter_params += param.numel()

    percentage = 100 * adapter_params / total_params if total_params > 0 else 0

    return adapter_params, total_params, percentage


# Fused Megatron module name -> the HF-style projections it expands to. Megatron is the
# canonical form used everywhere internally (CLI, injection, weight sync); the mapping is
# one-to-many because a single fused Megatron linear covers several HF projections
# (linear_qkv -> q/k/v_proj, linear_fc1 -> gate/up_proj). Expansion is therefore lossless,
# unlike the reverse direction which could not be expressed uniquely.
MEGATRON_TO_HF_MODULES = {
    "linear_qkv": ["q_proj", "k_proj", "v_proj"],
    "linear_proj": ["o_proj"],
    "linear_fc1": ["gate_proj", "up_proj"],
    "linear_fc2": ["down_proj"],
    "router": ["gate"],
    # Split-QKV / split-gate-up variants
    "linear_q": ["q_proj"],
    "linear_k": ["k_proj"],
    "linear_v": ["v_proj"],
    "linear_fc1_gate": ["gate_proj"],
    "linear_fc1_up": ["up_proj"],
    # MLA projections
    "linear_kv_down_proj": ["kv_a_proj_with_mqa"],
    "linear_kv_up_proj": ["kv_b_proj"],
    "linear_q_down_proj": ["q_a_proj"],
    "linear_q_up_proj": ["q_b_proj"],
    "linear_q_proj": ["q_proj"],
}


def convert_megatron_to_hf_target_modules(megatron_modules: list[str]) -> list[str]:
    """Expand Megatron-style LoRA target module names to HF-style names.

    Relax uses Megatron-style names (``linear_qkv``/``linear_proj``/...) as the
    canonical form: the CLI accepts them and Megatron-Bridge's LoRA matcher walks
    the Megatron module tree directly. HF-style names are only needed when writing a
    standard HF-PEFT ``adapter_config.json`` (or building an inference PEFT config),
    so translation happens at export time via this one-to-many expansion.

    Args:
        megatron_modules: List of Megatron-style module names
            (e.g. ``["linear_qkv", "linear_proj"]``).

    Returns:
        List of HF-style module names with duplicates removed. Unknown names are
        passed through unchanged (already HF-style or custom).
    """
    hf_target_modules = []
    for module in megatron_modules:
        hf_target_modules.extend(MEGATRON_TO_HF_MODULES.get(module, [module]))
    # Remove duplicates while preserving order
    return list(dict.fromkeys(hf_target_modules))


def is_lora_adapter_param(name: str) -> bool:
    """Return True if ``name`` is a LoRA adapter parameter.

    Matches both naming conventions, anchored on the dotted module segment so a
    base weight that merely *contains* the substring (e.g. a module named
    ``lora_gate``) is not misclassified — a false positive here would silently
    drop a base weight from weight-sync buckets:
    - Megatron-Bridge: ``...adapter.linear_in.weight`` / ``...adapter.linear_out.weight``
    - Standard PEFT:   ``...lora_A.weight`` / ``...lora_B.weight``
    """
    return ".lora_A." in name or ".lora_B." in name or ".adapter.linear_in." in name or ".adapter.linear_out." in name


def build_hf_peft_config_dict(
    *,
    lora_rank: int,
    lora_alpha: int,
    target_modules,
    lora_dropout: float,
) -> dict:
    """Build the HF-PEFT adapter config dict (the content of
    ``adapter_config.json``).

    Single source of truth for both transports: the disk path serializes this
    to ``adapter_config.json`` and the from-tensors path passes it straight to
    SGLang's ``LoRAConfig.from_dict``. JSON-safe (no enums). ``target_modules``
    must already be HF-style names (see
    ``convert_megatron_to_hf_target_modules``).
    """
    return {
        "r": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": list(target_modules),
        "lora_dropout": lora_dropout,
        "bias": "none",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
    }


def write_hf_peft_adapter(
    merged: dict[str, torch.Tensor],
    adapter_dir,
    *,
    lora_rank: int,
    lora_alpha: int,
    target_modules,
    lora_dropout: float,
) -> str:
    """Write a merged LoRA adapter as a standard HF-PEFT directory.

    Produces ``adapter_config.json`` + ``adapter_model.safetensors`` — the on-disk
    format used by the checkpoint save, which Megatron-Bridge's ``load_peft_adapter``
    can read back.

    Args:
        merged: Full (TP-gathered, PP-merged) adapter tensors keyed by HF name.
        adapter_dir: Target directory (created if missing).
        lora_rank/lora_alpha/target_modules/lora_dropout: PEFT config written to
            ``adapter_config.json`` (HF-style target module names).

    Returns:
        The adapter directory path as a string.
    """
    import json
    from pathlib import Path

    from safetensors.torch import save_file

    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # HF PEFT adapter config (JSON-safe: no enums).
    config_dict = build_hf_peft_config_dict(
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
    )
    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump(config_dict, f)

    # safetensors requires contiguous CPU tensors.
    state = {name: t.contiguous() for name, t in merged.items()}
    save_file(state, str(adapter_dir / "adapter_model.safetensors"))
    return str(adapter_dir)


def is_lora_enabled(args) -> bool:
    """Check if LoRA is enabled in training arguments."""
    return hasattr(args, "lora_rank") and args.lora_rank > 0


def is_lora_merge_mode(args) -> bool:
    """Check if LoRA merge mode is enabled.

    In merge mode, LoRA adapters are merged into base weights before sync.

    Args:
        args: Training arguments

    Returns:
        True if merge mode, False otherwise
    """
    return hasattr(args, "lora_merge_mode") and args.lora_merge_mode


def is_lora_adapter_mode(args) -> bool:
    """Check if LoRA adapter mode is enabled.

    In adapter mode the base model is synced once and each step only the trained LoRA
    adapter is pushed to the rollout engine via SGLang's runtime LoRA API. Mutually
    exclusive with merge mode (enforced at arg-validation time).

    Args:
        args: Training arguments

    Returns:
        True if adapter mode, False otherwise
    """
    return hasattr(args, "lora_adapter_mode") and args.lora_adapter_mode


def extract_lora_delta(
    new_lora_state: dict[str, torch.Tensor],
    old_lora_state: dict[str, torch.Tensor] | None = None,
    threshold: float = 1e-6,
) -> Tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Compute the change (delta) of LoRA parameters between two syncs.

    Used by adapter mode's incremental sync: the first sync returns the full state,
    subsequent syncs return only parameters that changed by more than ``threshold``.
    The delta is used ONLY to decide whether a sync is needed (delta-skip); the tensors
    actually pushed to the engine are the absolute adapter values, not this delta.

    Args:
        new_lora_state: Current LoRA parameter state.
        old_lora_state: Previous state (None means first sync -> full state returned).
        threshold: Ignore changes below this value (float noise).

    Returns:
        ``(delta_dict, new_state_dict)``:
        - delta_dict: parameters that changed and would need syncing.
        - new_state_dict: a clone of the current state, saved for the next comparison.
    """
    if old_lora_state is None:
        # First sync: everything is "new"; clone twice so caller's saved state is
        # decoupled from the returned delta.
        return (
            {k: v.clone() for k, v in new_lora_state.items()},
            {k: v.clone() for k, v in new_lora_state.items()},
        )

    delta: dict[str, torch.Tensor] = {}
    for name, new_tensor in new_lora_state.items():
        if name not in old_lora_state:
            # Brand-new parameter: include its full value.
            delta[name] = new_tensor.clone()
            continue
        param_delta = new_tensor - old_lora_state[name]
        # .abs().max().item() forces a GPU->CPU sync; acceptable here because this runs
        # on the (infrequent) weight-update path, not a training/rollout hot path.
        if param_delta.abs().max().item() > threshold:
            delta[name] = param_delta

    new_state = {k: v.clone() for k, v in new_lora_state.items()}
    return delta, new_state


def build_lora_peft(args):
    """Build a Megatron-Bridge PEFT (LoRA) object from training arguments.

    Single source of truth for the LoRA config: the model provider applies this to
    the model, and adapter (re)loading reuses it so the adapter-key filter matches
    exactly what was applied. CLI exposes Megatron-style module names (``linear_qkv``,
    ...) which is what Megatron-Bridge's LoRA matcher walks, so they pass through
    unchanged.

    Args:
        args: Training arguments carrying ``lora_rank`` / ``lora_alpha`` /
            ``lora_target_modules`` / ``lora_dropout``.

    Returns:
        The Megatron-Bridge PEFT object (not yet applied to a model).
    """
    try:
        from megatron.bridge.peft.utils import create_peft
    except ImportError as e:
        raise RuntimeError(
            "LoRA training requires a newer training image with LoRA-enabled Megatron-Bridge. "
            "Please upgrade the training image."
        ) from e

    peft_config = {
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "target_modules": list(args.lora_target_modules),
        "dropout": args.lora_dropout,
        "bias": "none",
    }
    return create_peft(peft_config)


__all__ = [
    "LORA_ADAPTER_NAME",
    "count_adapter_parameters",
    "convert_megatron_to_hf_target_modules",
    "MEGATRON_TO_HF_MODULES",
    "write_hf_peft_adapter",
    "extract_lora_delta",
    "is_lora_enabled",
    "is_lora_merge_mode",
    "is_lora_adapter_mode",
    "build_lora_peft",
]
