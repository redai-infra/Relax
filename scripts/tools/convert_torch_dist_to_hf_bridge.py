# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import os
import sys
from pathlib import Path

import megatron.bridge.training.model_load_save as _model_load_save_module
import safetensors.torch as _safetensors_torch
import torch
from megatron.bridge import AutoBridge


_RELAX_ROOT = str(Path(__file__).resolve().parents[2])
if _RELAX_ROOT in sys.path:
    sys.path.remove(_RELAX_ROOT)
sys.path.insert(0, _RELAX_ROOT)

_pythonpath_entries = [
    entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry and entry != _RELAX_ROOT
]
os.environ["PYTHONPATH"] = os.pathsep.join([_RELAX_ROOT, *_pythonpath_entries])


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
            # A list-valued MoE pattern makes TransformerBlock use layer-indexed
            # distributed-checkpoint keys. Preserve that representation from the
            # checkpoint even when it is semantically equivalent to provider's
            # integer value (for example, [1] * num_layers versus 1).
            checkpoint_moe_layer_freq = getattr(model_cfg, "moe_layer_freq", None)
            if isinstance(checkpoint_moe_layer_freq, list):
                provider.moe_layer_freq = checkpoint_moe_layer_freq
                print(
                    "[convert] Preserving checkpoint moe_layer_freq list "
                    f"({len(checkpoint_moe_layer_freq)} layers) for sharded key compatibility"
                )
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
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Quantize each exported HF tensor to FP8 before it is buffered for safetensors output.",
    )
    parser.add_argument(
        "--fp8-strategy",
        choices=["block", "channel", "tensor"],
        default="block",
        help="FP8 quantization strategy (default: block).",
    )
    parser.add_argument(
        "--fp8-block-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("ROWS", "COLS"),
        help="Block shape for block FP8 (default: 128 128).",
    )
    parser.add_argument(
        "--fp8-device",
        type=str,
        default="cuda",
        help="Device used for one-tensor-at-a-time FP8 quantization (default: cuda).",
    )
    parser.add_argument(
        "--fp8-max-shard-size-mb",
        type=int,
        default=4096,
        help="Target FP8 safetensors shard size in MiB; one converted tensor group may exceed it (default: 4096).",
    )
    args = parser.parse_args()

    if args.fp8:
        try:
            output_is_origin = os.path.samefile(args.output_dir, args.origin_hf_dir)
        except FileNotFoundError:
            output_is_origin = os.path.realpath(args.output_dir) == os.path.realpath(args.origin_hf_dir)
        if output_is_origin:
            raise ValueError("--output-dir must differ from --origin-hf-dir for online FP8 conversion")
    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    fp8_block_size = args.fp8_block_size
    if args.fp8 and args.fp8_strategy == "block" and fp8_block_size is None:
        fp8_block_size = [128, 128]
    if args.fp8:
        from relax.utils.quant_cast.fp8 import build_quantization_config, validate_fp8_options
        from relax.utils.quant_cast.fp8_checkpoint import StreamingFP8Writer

        validate_fp8_options(args.fp8_strategy, fp8_block_size)
        if args.fp8_max_shard_size_mb <= 0:
            raise ValueError("--fp8-max-shard-size-mb must be positive")
        fp8_device = torch.device(args.fp8_device)
        if fp8_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("--fp8-device points to CUDA, but torch.cuda.is_available() is false")

    print(f"Loading config from {args.origin_hf_dir}")
    bridge = AutoBridge.from_hf_pretrained(args.origin_hf_dir, trust_remote_code=True)

    # Use Bridge's provider so the correct model class is created (e.g., Qwen3VLModel
    # instead of GPTModel). This is needed because MLM checkpoints lack run_config.yaml.
    provider = bridge.to_megatron_provider(load_weights=False)

    # Some HF configurations enable MTP layers, but RL-trained Megatron checkpoints lack MTP weights, which causes a loading error.
    allow_missing_mtp_keys = False
    if getattr(provider, "mtp_num_layers", 0) and not _checkpoint_has_mtp(args.input_dir):
        print(f"[convert] Checkpoint has no MTP weights; disabling MTP (was mtp_num_layers={provider.mtp_num_layers})")
        provider.mtp_num_layers = 0
        allow_missing_mtp_keys = True

    _provider_override["provider"] = provider
    print(f"[convert] Using Bridge provider: {type(provider).__name__}")

    fp8_writer = None
    source = None
    original_save_generator = None
    if args.fp8:
        state = getattr(bridge.hf_pretrained, "state", None)
        source = getattr(state, "source", None)
        if source is None or not hasattr(source, "key_to_filename_map"):
            raise ValueError("Online FP8 conversion requires --origin-hf-dir to contain safetensors weights")
        fp8_writer = StreamingFP8Writer(
            source.key_to_filename_map,
            args.fp8_strategy,
            fp8_block_size,
            args.fp8_device,
            args.fp8_max_shard_size_mb * 1024**2,
        )
        original_save_generator = source.save_generator
        source.save_generator = fp8_writer.save_generator
        print(
            f"[convert] Enabled streaming FP8: strategy={args.fp8_strategy}, "
            f"block_size={fp8_block_size}, device={args.fp8_device}, "
            f"max_shard_size_mb={args.fp8_max_shard_size_mb}"
        )

    print(f"Exporting checkpoint from {args.input_dir} to {args.output_dir}")
    try:
        bridge.export_ckpt(args.input_dir, args.output_dir, strict=args.fp8 and not allow_missing_mtp_keys)
    finally:
        if source is not None and original_save_generator is not None:
            source.save_generator = original_save_generator

    # Work around a Megatron-Bridge bug: with strict=False, export_ckpt writes each
    # incomplete shard without the tensors the checkpoint lacked (e.g. MTP), yet still
    # lists those keys in model.safetensors.index.json, producing "ghost" entries that
    # point at shards which do not contain them. Reconcile the index against the shards
    # and supplement missing MTP weights from the reference model so the export stays
    # loadable/deployable (e.g. for EAGLE speculative decoding). The FP8 path uses a
    # separate streaming writer with its own index and is left untouched here.
    if not args.fp8:
        from relax.utils.hf_export import reconcile_hf_export_index

        reconcile_summary = reconcile_hf_export_index(
            args.output_dir,
            reference_hf_dir=args.origin_hf_dir,
            supplement_mtp=allow_missing_mtp_keys,
        )
        if reconcile_summary["ghosts"]:
            print(
                f"[convert] reconciled index: {len(reconcile_summary['ghosts'])} ghost key(s), "
                f"{len(reconcile_summary['supplemented'])} MTP supplemented, "
                f"{len(reconcile_summary['dropped'])} dropped"
            )

    # Make the output dir consumable by older transformers 4.x releases.
    import json
    import shutil

    cfg_path = os.path.join(args.output_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        rope_params = cfg.get("rope_parameters")
        if isinstance(rope_params, dict):
            cfg.setdefault("rope_theta", rope_params.get("rope_theta"))
            cfg.setdefault("rope_scaling", None)
        if "dtype" in cfg and "torch_dtype" not in cfg:
            cfg["torch_dtype"] = cfg["dtype"]
        cfg["transformers_version"] = "4.51.0"
        if fp8_writer is not None:
            cfg["quantization_config"] = build_quantization_config(
                args.fp8_strategy,
                fp8_block_size,
                fp8_writer.result.modules_to_not_convert,
            )
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"[convert] post-processed {cfg_path} for transformers 4.x compatibility")

    for fname in ("tokenizer_config.json", "vocab.json", "merges.txt"):
        src = os.path.join(args.origin_hf_dir, fname)
        dst = os.path.join(args.output_dir, fname)
        if os.path.isfile(src):
            shutil.copyfile(src, dst)
            print(f"[convert] copied {fname} from origin (tokenizer-compatibility)")

    if fp8_writer is not None:
        print(
            f"[convert] wrote {len(fp8_writer.result.weight_map)} FP8 checkpoint tensors "
            f"({fp8_writer.result.total_size} bytes)"
        )

    print("Done!")
