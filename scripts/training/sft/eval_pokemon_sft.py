# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Send pokemon SFT data to an OpenAI-compatible SGLang server and dump
outputs.

Usage:
    python scripts/training/sft/eval_pokemon_sft.py \
        --model <served_model_name> \
        --data /path/to/pokemon_gpt4o_en.parquet [/path/to/pokemon_gpt4o_zh.parquet ...] \
        --num-samples 20 \
        --output eval_pokemon_sft.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import random
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from openai import AsyncOpenAI

from relax.utils.multimodal.image_utils import load_image, to_rgb


ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:30000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", required=True, help="served_model_name of the SGLang server")
    p.add_argument("--data", nargs="+", required=True, help="One or more parquet files")
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output", default="eval_pokemon_sft.jsonl")
    p.add_argument("--timeout", type=float, default=120.0)
    return p.parse_args()


def encode_image(image_field: Any) -> str:
    """Convert any supported image input to a data:image/png;base64,...

    URI.
    """
    img = load_image(image_field)
    img = to_rgb(img)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_messages(conversations: list[dict], images: list[Any] | None) -> tuple[list[dict], str]:
    """Split a conversation row into (input messages, reference assistant
    text).

    The last assistant turn is held out as the reference; everything before it
    is the prompt. Images are attached to the first user turn as image_url
    content blocks.
    """
    turns = [{"role": ROLE_MAP.get(c["from"], c["from"]), "content": c["value"]} for c in conversations]

    last_assistant_idx = None
    for i in range(len(turns) - 1, -1, -1):
        if turns[i]["role"] == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx is None:
        raise ValueError("conversation has no assistant turn")

    reference = turns[last_assistant_idx]["content"]
    input_turns = turns[:last_assistant_idx]

    image_parts = []
    if images is not None and len(images) > 0:
        for img in images:
            image_parts.append({"type": "image_url", "image_url": {"url": encode_image(img)}})

    if image_parts:
        for t in input_turns:
            if t["role"] == "user":
                t["content"] = image_parts + [{"type": "text", "text": t["content"]}]
                break

    return input_turns, reference


def load_samples(paths: list[str], num_samples: int, seed: int) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        table = pq.read_table(path, columns=["conversations", "images"])
        df = table.to_pandas()
        for _, row in df.iterrows():
            rows.append(
                {
                    "source": path,
                    "conversations": list(row["conversations"]),
                    "images": list(row["images"]) if row["images"] is not None else [],
                }
            )
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:num_samples]


async def one_request(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    args: argparse.Namespace,
    idx: int,
    sample: dict,
) -> dict:
    try:
        messages, reference = build_messages(sample["conversations"], sample["images"])
    except Exception as e:
        return {"idx": idx, "source": sample["source"], "error": f"build_messages: {e}"}

    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            output = resp.choices[0].message.content
            return {
                "idx": idx,
                "source": sample["source"],
                "input": messages,
                "reference": reference,
                "output": output,
                "error": None,
            }
        except Exception as e:
            return {
                "idx": idx,
                "source": sample["source"],
                "input": messages,
                "reference": reference,
                "output": None,
                "error": str(e),
            }


async def main_async(args: argparse.Namespace) -> None:
    samples = load_samples(args.data, args.num_samples, args.seed)
    print(f"Loaded {len(samples)} samples from {len(args.data)} parquet file(s).")

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    sem = asyncio.Semaphore(args.concurrency)
    tasks = [asyncio.create_task(one_request(client, sem, args, i, s)) for i, s in enumerate(samples)]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = 0
    fail = 0
    with out_path.open("w", encoding="utf-8") as f:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            if result.get("error") is None and "output" in result:
                ok += 1
                print(f"[{result['idx']:>3}] ok")
            else:
                fail += 1
                print(f"[{result['idx']:>3}] FAIL: {result.get('error')}")
    print(f"\nDone. ok={ok} fail={fail} -> {out_path}")


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
