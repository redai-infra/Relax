#!/usr/bin/env python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Generate a REAL multimodal payload fixture for byte-exact acceptance.

The maintainer acceptance for the TransferQueue RDMA data plane requires
byte-exact consistency on *real* multimodal payloads, not synthetic
production-shaped tensors.  "Real" means: actual dataset images pushed through
the exact production preprocessing chain, producing the exact structure the
Relax data plane ships — ``multimodal_train_inputs`` as ``list[dict]``
(a tensordict ``NonTensorStack``, i.e. MooncakeStore's non-tensor slow path).

This script replicates the production chain line-for-line by calling the same
functions rollout uses:

  parquet row {prompt, image}
    -> relax.utils.data.data_utils.build_messages          (placeholder -> content)
    -> tokenizer.apply_chat_template                        (prompt string)
    -> relax.utils.data.processing_utils.process_vision_info (bytes -> PIL, resize)
    -> adapt_processor_kwargs -> HF processor -> strip input_ids/attention_mask
       -> numpy->torch -> remap_mm_train_inputs             (== sglang_rollout._run_processor)
    -> GRPO group expansion: n_samples_per_prompt byte-identical copies per
       prompt (production runs the processor once per sample; determinism is
       verified below so per-sample clones are byte-equivalent)

The fixture bundles the resulting ``train_data`` lists plus a leaf-level
SHA-256 manifest (see :mod:`relax.utils.payload_digest`).  Consumers:

  * tests/utils/test_tq_dataplane_behavior.py  (full tq.init/put/get link)
  * tests/utils/test_tq_failure_paths.py       (direct MooncakeStore client)
  * scripts/benchmarks/tq_cross_node_bench.py  (--payload-profiles real-multimodal)

The .pt file is machine-local (hundreds of MB; NOT committed).  Committable
provenance goes to ``--manifest-json``: generation args, dataset rows, patch
counts, and every leaf hash, so any regenerated fixture can be audited.

Example (defaults target the acceptance dataset used in issue #217):

  PYTHONPATH=. python scripts/benchmarks/make_multimodal_fixture.py \\
      --dataset /path/to/<multimodal-dataset>.parquet \\
      --model /path/to/<qwen-vl-model-dir> \\
      --num-prompts 6 --n-samples-per-prompt 2 \\
      --output tests/fixtures/tq_multimodal_fixture.pt \\
      --manifest-json tests/fixtures/tq_multimodal_fixture.manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Parquet file with 'prompt' and 'image' columns.")
    parser.add_argument("--model", required=True, help="HF model dir providing the processor + tokenizer.")
    parser.add_argument("--output", default="tests/fixtures/tq_multimodal_fixture.pt", help="Fixture .pt path.")
    parser.add_argument("--manifest-json", default="", help="Optional committable provenance JSON path.")
    parser.add_argument("--num-prompts", type=int, default=6, help="Distinct prompts (images) to process.")
    parser.add_argument(
        "--n-samples-per-prompt",
        type=int,
        default=2,
        help="GRPO group size: byte-identical copies per prompt (production F4 semantics).",
    )
    parser.add_argument("--row-offset", type=int, default=0, help="First dataset row to scan.")
    parser.add_argument(
        "--skip-determinism-check",
        action="store_true",
        help="Skip the double-run processor determinism verification (not recommended).",
    )
    return parser.parse_args()


def _load_rows(dataset_path: str, num_prompts: int, row_offset: int) -> list[dict[str, Any]]:
    """Sequentially collect rows that carry at least one image."""
    import pyarrow.parquet as pq

    table = pq.read_table(dataset_path, columns=["prompt", "image"])
    rows: list[dict[str, Any]] = []
    for index in range(row_offset, table.num_rows):
        images = table["image"][index].as_py()
        if not images:
            continue
        rows.append(
            {
                "row_index": index,
                "prompt": table["prompt"][index].as_py(),
                "image": images,
            }
        )
        if len(rows) >= num_prompts:
            return rows
    raise RuntimeError(
        f"Only found {len(rows)} usable rows (needed {num_prompts}) in {dataset_path} from offset {row_offset}."
    )


def _run_production_processor(row: dict[str, Any], tokenizer: Any, processor: Any) -> tuple[list[int], dict[str, Any]]:
    """Replicate ``sglang_rollout._run_image_processor``'s synchronous body.

    Every call below is the same production function the rollout worker uses;
    nothing is re-implemented here.
    """
    from relax.utils.data.data_utils import build_messages
    from relax.utils.data.processing_utils import (
        adapt_processor_kwargs,
        process_vision_info,
        remap_mm_train_inputs,
    )

    messages = build_messages(
        {"prompt": row["prompt"], "image": row["image"]},
        prompt_key="prompt",
        system_prompt=None,
        as_conversation=True,
        multimodal_keys={"image": "image"},
    )
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    multimodal_inputs = process_vision_info(messages, processor, use_audio_in_video=False)

    adapted = adapt_processor_kwargs(
        processor, multimodal_inputs, {"use_audio_in_video": False, "return_mm_token_type_ids": False}
    )
    processor_output = processor(text=prompt_str, **adapted)
    prompt_ids = processor_output["input_ids"][0]
    if isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids.tolist()
    train_inputs = {
        key: (torch.from_numpy(value) if isinstance(value, np.ndarray) else value)
        for key, value in processor_output.items()
        if key not in ["input_ids", "attention_mask"]
    } or None
    train_inputs = remap_mm_train_inputs(processor, train_inputs)
    if not isinstance(train_inputs, dict) or not train_inputs:
        raise RuntimeError(f"Processor produced no multimodal train inputs for row {row['row_index']}.")
    return list(prompt_ids), train_inputs


def _clone_train_inputs(train_inputs: dict[str, Any]) -> dict[str, Any]:
    """Independent storage per sample, byte-identical content (F4
    semantics)."""
    cloned: dict[str, Any] = {}
    for key, value in train_inputs.items():
        cloned[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return cloned


def main() -> int:
    from relax.utils.payload_digest import diff_digests, leaf_digests, total_leaf_bytes

    args = parse_args()

    from transformers import AutoProcessor, AutoTokenizer

    print(f"[fixture] loading processor/tokenizer from {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    rows = _load_rows(args.dataset, args.num_prompts, args.row_offset)
    print(f"[fixture] processing {len(rows)} dataset rows (offset {args.row_offset})")

    determinism: dict[str, Any] = {"checked": False, "byte_identical": None}
    if not args.skip_determinism_check:
        first_ids_a, first_mm_a = _run_production_processor(rows[0], tokenizer, processor)
        first_ids_b, first_mm_b = _run_production_processor(rows[0], tokenizer, processor)
        mismatches = diff_digests(leaf_digests(first_mm_a), leaf_digests(first_mm_b))
        determinism = {"checked": True, "byte_identical": not mismatches and first_ids_a == first_ids_b}
        if mismatches:
            # The GRPO group expansion below clones one processor run, which is
            # only equivalent to production's N independent runs if the
            # processor is byte-deterministic (RFC F4 caveat). Surface loudly.
            print("[fixture] WARNING: processor is NOT byte-deterministic across runs:")
            for line in mismatches[:8]:
                print(f"    {line}")
        else:
            print("[fixture] processor determinism verified (two runs byte-identical)")

    tokens: list[list[int]] = []
    multimodal_train_inputs: list[dict[str, Any]] = []
    group_index: list[int] = []
    prompt_meta: list[dict[str, Any]] = []
    for prompt_position, row in enumerate(rows):
        prompt_ids, train_inputs = _run_production_processor(row, tokenizer, processor)
        grid = train_inputs.get("image_grid_thw")
        prompt_meta.append(
            {
                "row_index": row["row_index"],
                "num_images": len(row["image"]),
                "prompt_len": len(prompt_ids),
                "mm_keys": sorted(train_inputs.keys()),
                "mm_shapes": {
                    key: list(value.shape) for key, value in train_inputs.items() if isinstance(value, torch.Tensor)
                },
                "image_grid_thw": grid.tolist() if isinstance(grid, torch.Tensor) else None,
            }
        )
        for _ in range(args.n_samples_per_prompt):
            tokens.append(list(prompt_ids))
            multimodal_train_inputs.append(_clone_train_inputs(train_inputs))
            group_index.append(prompt_position)
        print(
            f"[fixture]   row {row['row_index']}: prompt_len={len(prompt_ids)} "
            f"mm={prompt_meta[-1]['mm_shapes']} x{args.n_samples_per_prompt} samples"
        )

    num_samples = len(tokens)
    train_data = {"tokens": tokens, "multimodal_train_inputs": multimodal_train_inputs}

    manifest: dict[str, Any] = {}
    for sample_index in range(num_samples):
        manifest.update(
            leaf_digests(multimodal_train_inputs[sample_index], f"sample[{sample_index}].multimodal_train_inputs")
        )
        manifest.update(
            leaf_digests(torch.tensor(tokens[sample_index], dtype=torch.int64), f"sample[{sample_index}].tokens")
        )

    payload_bytes = sum(total_leaf_bytes(sample) for sample in multimodal_train_inputs)
    meta = {
        "schema": 1,
        "generated_at_unix": time.time(),
        "model": os.path.abspath(args.model),
        "processor_class": type(processor).__name__,
        "image_processor_class": type(processor.image_processor).__name__,
        "patch_size": getattr(processor.image_processor, "patch_size", None),
        "dataset": os.path.abspath(args.dataset),
        "row_offset": args.row_offset,
        "num_prompts": len(rows),
        "n_samples_per_prompt": args.n_samples_per_prompt,
        "num_samples": num_samples,
        "multimodal_payload_bytes": payload_bytes,
        "processor_determinism": determinism,
        "prompts": prompt_meta,
        "versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "tensordict": __import__("tensordict").__version__,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({"meta": meta, "train_data": train_data, "manifest": manifest}, args.output)
    print(
        f"[fixture] wrote {args.output}: {num_samples} samples, "
        f"{payload_bytes / 1024**2:.1f} MiB multimodal payload, {len(manifest)} manifest leaves"
    )

    if args.manifest_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.manifest_json)), exist_ok=True)
        with open(args.manifest_json, "w") as handle:
            json.dump({"meta": meta, "manifest": manifest}, handle, indent=1, sort_keys=True)
        print(f"[fixture] wrote provenance manifest {args.manifest_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
