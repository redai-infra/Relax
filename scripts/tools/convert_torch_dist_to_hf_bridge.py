# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import os

import megatron.bridge.training.model_load_save as _model_load_save_module
import safetensors.torch as _safetensors_torch
from megatron.bridge import AutoBridge


# Some megatron_to_hf mappings in Megatron Bridge yield non-contiguous tensors (e.g. after
# transpose/narrow/chunk without a trailing .contiguous()). The shared-tensor dedup check in
# safetensors.save_file runs `tensor.view(-1)[-1]`, which raises
# "view size is not compatible with input tensor's size and stride" on such tensors.
# Force .contiguous() at the save boundary as a safety net.
_original_save_file = _safetensors_torch.save_file


def _save_file_ensure_contiguous(tensors, filename, metadata=None):
    fixed = {}
    for k, v in tensors.items():
        if hasattr(v, "is_contiguous") and not v.is_contiguous():
            print(
                f"[convert] forcing .contiguous() on non-contig tensor: {k} shape={tuple(v.shape)} stride={v.stride()}"
            )
            v = v.contiguous()
        fixed[k] = v
    return _original_save_file(fixed, filename, metadata=metadata)


_safetensors_torch.save_file = _save_file_ensure_contiguous


# Here we need to patch Megatron Bridge's `load_model_config`, since the checkpoint is saved
# by Megatron and lack of provider information.
_provider_override = {}
_original_load_model_config = _model_load_save_module.load_model_config


def _patched_load_model_config(checkpoint_path):
    model_cfg, mlm_args = _original_load_model_config(checkpoint_path)
    provider = _provider_override.get("provider")
    if provider is not None:
        from megatron.bridge.models.model_provider import ModelProviderMixin

        if not isinstance(model_cfg, ModelProviderMixin):
            print(f"[convert] Overriding MLM TransformerConfig with Bridge provider: {type(provider).__name__}")
            return provider, mlm_args
    return model_cfg, mlm_args


_model_load_save_module.load_model_config = _patched_load_model_config


def _checkpoint_has_mtp(input_dir):
    """Return True if the torch-dist checkpoint actually stores MTP weights.

    `input_dir` may be a Megatron checkpoint root (containing
    `latest_checkpointed_iteration.txt` and `iter_*` subdirs) or a single
    checkpoint directory. Detection reads the torch DCP `.metadata` and looks
    for any `mtp` key.
    """
    from torch.distributed.checkpoint import FileSystemReader

    ckpt_dir = input_dir
    latest_file = os.path.join(input_dir, "latest_checkpointed_iteration.txt")
    if os.path.exists(latest_file):
        with open(latest_file) as f:
            tag = f.read().strip()
        iter_dir = os.path.join(input_dir, f"iter_{int(tag):07d}")
        if os.path.isdir(iter_dir):
            ckpt_dir = iter_dir

    metadata = FileSystemReader(ckpt_dir).read_metadata()
    return any("mtp" in k.lower() for k in metadata.state_dict_metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert torch distributed checkpoint to HuggingFace format using Megatron Bridge"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Path to the torch distributed checkpoint directory"
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save the HuggingFace checkpoint")
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        required=True,
        help="Path to the original HuggingFace model directory (for config)",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the output directory if it exists."
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    print(f"Loading config from {args.origin_hf_dir}")
    bridge = AutoBridge.from_hf_pretrained(args.origin_hf_dir, trust_remote_code=True)

    # Use Bridge's provider so the correct model class is created (e.g., Qwen3VLModel
    # instead of GPTModel). This is needed because MLM checkpoints lack run_config.yaml.
    provider = bridge.to_megatron_provider(load_weights=False)

    # Some HF configurations enable MTP layers, but RL-trained Megatron checkpoints lack MTP weights, which causes a loading error.
    if getattr(provider, "mtp_num_layers", 0) and not _checkpoint_has_mtp(args.input_dir):
        print(f"[convert] Checkpoint has no MTP weights; disabling MTP (was mtp_num_layers={provider.mtp_num_layers})")
        provider.mtp_num_layers = 0

    _provider_override["provider"] = provider
    print(f"[convert] Using Bridge provider: {type(provider).__name__}")

    print(f"Exporting checkpoint from {args.input_dir} to {args.output_dir}")
    bridge.export_ckpt(args.input_dir, args.output_dir)

    print("Done!")
