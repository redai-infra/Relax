# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reconcile Megatron-Bridge HuggingFace exports whose index lists ghost keys.

Megatron-Bridge's non-distributed ``save_generator`` has a bug: when the source
Megatron checkpoint lacks some tensors that the reference HF model expects — for
example an RL/SFT checkpoint trained without MTP, while the reference config
enables MTP — ``AutoBridge.export_ckpt(..., strict=False)`` writes the incomplete
shards correctly (skipping the absent tensors) but still records each incomplete
shard's *full expected* key set as "saved". The resulting
``model.safetensors.index.json`` therefore lists "ghost" keys that point at
shards which do not actually contain them.

Downstream loaders (transformers / SGLang) then fail: the index references
tensors that cannot be found. This is especially fatal when serving with EAGLE
speculative decoding, whose draft model *is* the MTP (nextn) layer.

Root cause (in the dependency, not in Relax):
``megatron/bridge/models/hf_pretrained/state.py`` does
``all_saved_keys.update(keys_for_file)`` in the ``strict=False`` incomplete-shard
fallback; it should be ``all_saved_keys.update(tensors_to_save.keys())`` (the
distributed save path already does the equivalent). Until that is fixed upstream,
this module reconciles the exported checkpoint on the Relax side.

The reconciliation:
1. Rebuilds ``model.safetensors.index.json`` from the tensors that are physically
   present in the shards, so the index can never reference a missing tensor.
2. Optionally supplements MTP tensors the checkpoint lacked from the reference HF
   model (``--origin-hf-dir``), so the exported model matches the reference
   structure and stays deployable (e.g. for EAGLE speculative decoding).
"""

from __future__ import annotations

import json
import os
import struct
from typing import Optional

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_INDEX_NAME = "model.safetensors.index.json"


def _read_safetensors_header(path: str) -> dict[str, int]:
    """Return ``{tensor_key: num_bytes}`` from a safetensors header.

    Only the 8-byte length prefix and the JSON header are read, so the cost is
    independent of the shard size (no tensor data is loaded or mmapped).
    """
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    sizes: dict[str, int] = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        begin, end = meta.get("data_offsets", (0, 0))
        sizes[key] = int(end) - int(begin)
    return sizes


def _scan_physical(model_dir: str) -> dict[str, tuple[str, int]]:
    """Map every tensor physically present in ``model_dir`` to ``(filename,
    num_bytes)``."""
    physical: dict[str, tuple[str, int]] = {}
    for fname in sorted(os.listdir(model_dir)):
        if fname.endswith(".safetensors"):
            for key, nbytes in _read_safetensors_header(os.path.join(model_dir, fname)).items():
                physical[key] = (fname, nbytes)
    return physical


def _load_weight_map(index_path: str) -> dict[str, str]:
    with open(index_path) as f:
        return json.load(f).get("weight_map", {})


def reconcile_hf_export_index(
    output_dir: str,
    reference_hf_dir: Optional[str] = None,
    supplement_mtp: bool = True,
    mtp_shard_name: str = "model-mtp-00001-of-00001.safetensors",
) -> dict:
    """Make an exported HF checkpoint's index consistent with its actual
    shards.

    Removes ghost entries (index keys not present in any shard) and, when
    ``supplement_mtp`` is set and ``reference_hf_dir`` is given, fills MTP tensors
    the Megatron checkpoint lacked from the reference model before rebuilding the
    index. The index is always rebuilt from what is physically present, so it can
    never reference a missing tensor afterwards.

    Args:
        output_dir: Exported HF checkpoint directory (modified in place).
        reference_hf_dir: Reference HF model directory (``--origin-hf-dir``) that
            provides MTP weights. Required to supplement MTP.
        supplement_mtp: If True, supplement ghost MTP keys from ``reference_hf_dir``.
        mtp_shard_name: Filename of the shard written for supplemented MTP tensors.

    Returns:
        Summary dict ``{"ghosts": [...], "supplemented": [...], "dropped": [...]}``.
        ``ghosts`` are index keys that were missing from the shards; ``supplemented``
        were filled from the reference; ``dropped`` were removed from the index.
    """
    index_path = os.path.join(output_dir, _INDEX_NAME)
    empty = {"ghosts": [], "supplemented": [], "dropped": []}
    if not os.path.isfile(index_path):
        # Single-file (model.safetensors) export has no index to reconcile.
        return empty

    weight_map = _load_weight_map(index_path)
    physical = _scan_physical(output_dir)
    ghosts = sorted(k for k in weight_map if k not in physical)
    if not ghosts:
        return empty

    logger.warning(
        "HF export index lists %d key(s) not present in any shard (Megatron-Bridge ghost-key bug); reconciling %s",
        len(ghosts),
        output_dir,
    )

    supplemented: list[str] = []
    if supplement_mtp and reference_hf_dir:
        ref_index_path = os.path.join(reference_hf_dir, _INDEX_NAME)
        ref_map = _load_weight_map(ref_index_path) if os.path.isfile(ref_index_path) else {}
        # Only MTP tensors are expected to be legitimately absent from the checkpoint
        # (RL/SFT does not train MTP). Restrict supplementation to those, sourced
        # from the reference model, so a genuine backbone-conversion bug is never
        # silently masked with reference weights.
        mtp_ghosts = [k for k in ghosts if "mtp" in k.lower() and k in ref_map]
        if mtp_ghosts:
            import safetensors
            import safetensors.torch

            by_file: dict[str, list[str]] = {}
            for k in mtp_ghosts:
                by_file.setdefault(ref_map[k], []).append(k)
            tensors = {}
            for ref_file, keys in by_file.items():
                with safetensors.safe_open(
                    os.path.join(reference_hf_dir, ref_file), framework="pt", device="cpu"
                ) as fh:
                    for k in keys:
                        tensors[k] = fh.get_tensor(k)
            safetensors.torch.save_file(tensors, os.path.join(output_dir, mtp_shard_name), metadata={"format": "pt"})
            supplemented = sorted(tensors.keys())
            logger.info(
                "Supplemented %d MTP tensor(s) from reference model %s",
                len(supplemented),
                reference_hf_dir,
            )
            del tensors

    # Rebuild the index purely from what is physically present (now including any
    # supplemented MTP shard). This drops every remaining ghost entry.
    physical = _scan_physical(output_dir)
    dropped = sorted(k for k in ghosts if k not in physical)
    if dropped:
        logger.warning(
            "Dropped %d unresolved ghost key(s) from index (not in checkpoint or reference): %s%s",
            len(dropped),
            dropped[:10],
            " ..." if len(dropped) > 10 else "",
        )

    new_weight_map = {key: fname for key, (fname, _nbytes) in physical.items()}
    total_size = sum(nbytes for _fname, nbytes in physical.values())
    with open(index_path) as f:
        index = json.load(f)
    index["weight_map"] = new_weight_map
    index.setdefault("metadata", {})["total_size"] = total_size
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    logger.info(
        "Reconciled HF export index: %d key(s) now consistent with shards (%d supplemented, %d dropped).",
        len(new_weight_map),
        len(supplemented),
        len(dropped),
    )
    return {"ghosts": ghosts, "supplemented": supplemented, "dropped": dropped}
