# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""export EXP_DIR=/path/to/your/dir hf download Keven16/Qwen3-4B-Non-Thinking-
RL-Math-Step500 \

--local-dir ${EXP_DIR}/Qwen3-4B-Non-Thinking-RL-Math-Step500

hf download Qwen/Qwen3-4B --local-dir ${EXP_DIR}/Qwen3-4B

hf download --repo-type dataset Keven16/G-OPD-Training-Data \
--local-dir ${EXP_DIR}/G-OPD-Training-Data

python prepare_data.py ${EXP_DIR}/G-OPD-Training-Data/train.parquet \
${EXP_DIR}/G-OPD-Training-Data/train.jsonl
"""

import argparse
import json

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Convert parquet to jsonl for OPD training")
    parser.add_argument("in_path", type=str, help="Path to input parquet file")
    parser.add_argument("out_path", type=str, help="Path to output jsonl file")
    return parser.parse_args()


def to_messages(prompt):
    """verl prompt (numpy array / list of dict) -> list[{'role','content'}]"""
    msgs = []
    for m in list(prompt):
        msgs.append({"role": str(m["role"]), "content": str(m["content"])})
    return msgs


def to_label(reward_model):
    if isinstance(reward_model, dict):
        gt = reward_model.get("ground_truth")
    else:
        gt = reward_model["ground_truth"]
    return str(gt)


def main(in_path, out_path):
    df = pd.read_parquet(in_path)
    n = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for _, row in df.iterrows():
            obj = {
                "prompt": to_messages(row["prompt"]),
                "label": to_label(row["reward_model"]),
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    print(f"{in_path} -> {out_path}: {n} lines")


if __name__ == "__main__":
    args = parse_args()
    main(args.in_path, args.out_path)
