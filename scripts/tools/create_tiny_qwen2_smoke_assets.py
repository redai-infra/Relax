#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from relax.utils.types import Sample


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
MODEL_VOCAB_SIZE = 256
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
NUM_HIDDEN_LAYERS = 2
NUM_ATTENTION_HEADS = 4
NUM_KEY_VALUE_HEADS = 2
MAX_POSITION_EMBEDDINGS = 128
PROMPT_LENGTH = 8
RESPONSE_LENGTH = 4
NUM_SAMPLES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create local tiny Qwen2 assets for Relax ROCm smoke tests.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write the generated assets into.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing assets in the output directory.",
    )
    return parser.parse_args()


def build_vocab() -> dict[str, int]:
    base_tokens = [f"tok_{i}" for i in range(MODEL_VOCAB_SIZE - len(SPECIAL_TOKENS))]
    vocab_tokens = SPECIAL_TOKENS + base_tokens
    return {token: idx for idx, token in enumerate(vocab_tokens)}


def create_tokenizer(tokenizer_dir: Path, *, force: bool) -> None:
    if tokenizer_dir.exists() and not force:
        return

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    vocab = build_vocab()

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    fast_tokenizer.model_max_length = MAX_POSITION_EMBEDDINGS
    fast_tokenizer.save_pretrained(tokenizer_dir)

    generation_config = {
        "bos_token_id": vocab["<bos>"],
        "eos_token_id": vocab["<eos>"],
        "pad_token_id": vocab["<pad>"],
        "transformers_version": "auto",
    }
    (tokenizer_dir / "generation_config.json").write_text(json.dumps(generation_config, indent=2), encoding="utf-8")


def create_model(model_dir: Path, *, force: bool) -> None:
    if model_dir.exists() and (model_dir / "config.json").exists() and not force:
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    vocab = build_vocab()
    config = Qwen2Config(
        vocab_size=len(vocab),
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        attention_bias=False,
        bos_token_id=vocab["<bos>"],
        eos_token_id=vocab["<eos>"],
        pad_token_id=vocab["<pad>"],
        torch_dtype="bfloat16",
    )
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(model_dir, safe_serialization=True)


def build_sample(index: int) -> Sample:
    prompt_tokens = [10 + index, 20 + index, 30 + index, 40 + index, 50 + index, 60 + index, 70 + index, 80 + index]
    response_tokens = [100 + index, 110 + index, 120 + index, 130 + index]
    tokens = prompt_tokens + response_tokens
    reward = [1.0, 0.0, 0.8, -0.2][index]

    return Sample(
        group_index=index // 2,
        index=index,
        prompt=f"prompt {index}",
        response=f"response {index}",
        tokens=tokens,
        rollout_tokens=tokens.copy(),
        response_length=RESPONSE_LENGTH,
        reward=reward,
        loss_mask=[1] * RESPONSE_LENGTH,
        status=Sample.Status.COMPLETED,
        label=f"label {index}",
        metadata={"raw_reward": reward},
        train_metadata={"source": "tiny_qwen2_smoke"},
    )


def create_debug_rollout(debug_dir: Path, *, force: bool) -> None:
    target_path = debug_dir / "0.pt"
    if target_path.exists() and not force:
        return

    debug_dir.mkdir(parents=True, exist_ok=True)
    samples = [build_sample(i).to_dict() for i in range(NUM_SAMPLES)]
    torch.save({"rollout_id": 0, "samples": samples}, target_path)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    tokenizer_dir = output_dir / "hf_model"
    debug_dir = output_dir / "debug_rollout"

    create_tokenizer(tokenizer_dir, force=args.force)
    create_model(tokenizer_dir, force=args.force)
    create_debug_rollout(debug_dir, force=args.force)

    manifest = {
        "hf_model": str(tokenizer_dir),
        "debug_rollout": str(debug_dir),
        "num_samples": NUM_SAMPLES,
        "prompt_length": PROMPT_LENGTH,
        "response_length": RESPONSE_LENGTH,
        "vocab_size": MODEL_VOCAB_SIZE,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
