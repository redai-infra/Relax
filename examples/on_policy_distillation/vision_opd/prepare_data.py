# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare Vision-OPD-6K data for Relax.

The output schema is consumed by the vision OPD launch script:
  - messages: student-side conversation
  - images: student-side full images
  - bbox_images: teacher-side cropped images
  - label: ground-truth answer
  - metadata: auxiliary fields
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Sequence


BBOX_HINT = "Only focus on the objects inside the red bounding box in the image to answer this question."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Vision-OPD-6K to Relax jsonl.")
    parser.add_argument(
        "--data-dir",
        default="/root/Vision-OPD-6K",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--hf-repo",
        default="yuanqianhao/Vision-OPD-6K",
        help="HuggingFace dataset repository id.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download and extraction; only regenerate train.relax.jsonl.",
    )
    parser.add_argument(
        "--keep-bbox-hint",
        action="store_true",
        help="Keep the bbox hint in the student prompt.",
    )
    return parser.parse_args()


def _extract_concatenated_tar(tar_files: Sequence[str], images_dir: str) -> None:
    proc = subprocess.Popen(["tar", "-zxf", "-", "-C", images_dir], stdin=subprocess.PIPE)
    try:
        if proc.stdin is None:
            raise RuntimeError("failed to open tar stdin")
        with proc.stdin:
            for name in tar_files:
                with open(os.path.join(images_dir, name), "rb") as part:
                    shutil.copyfileobj(part, proc.stdin)
        return_code = proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        raise
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, proc.args)


def download_and_extract(repo_id: str, data_dir: str) -> None:
    subprocess.run(
        ["hf", "download", "--repo-type", "dataset", repo_id, "--local-dir", data_dir],
        check=True,
    )

    images_dir = os.path.join(data_dir, "images")
    if os.path.isdir(images_dir):
        tar_files = sorted(name for name in os.listdir(images_dir) if name.startswith("images.tar.gz"))
        if tar_files:
            _extract_concatenated_tar(tar_files, images_dir)
            for name in tar_files:
                os.remove(os.path.join(images_dir, name))

    teacher_dir = os.path.join(data_dir, "teacher_images")
    teacher_tar = os.path.join(teacher_dir, "teacher_images.tar.gz")
    if os.path.exists(teacher_tar):
        subprocess.run(["tar", "-xf", "teacher_images.tar.gz", "-C", "."], cwd=teacher_dir, check=True)
        os.remove(teacher_tar)


def _student_question(problem: str, keep_bbox_hint: bool) -> str:
    text = problem or ""
    if not keep_bbox_hint:
        text = text.replace(f"\n\n{BBOX_HINT}", "").replace(BBOX_HINT, "")
    return text.strip()


def build_record(item: dict, data_dir: str, keep_bbox_hint: bool) -> dict:
    image_path = os.path.join(data_dir, item["images"][0])
    teacher_image_path = os.path.join(data_dir, item["teacher_images"][0])
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"student image not found: {image_path}")
    if not os.path.exists(teacher_image_path):
        raise FileNotFoundError(f"teacher image not found: {teacher_image_path}")

    return {
        "messages": [{"role": "user", "content": _student_question(item.get("problem", ""), keep_bbox_hint)}],
        "images": [image_path],
        "bbox_images": [teacher_image_path],
        "label": item.get("answer", ""),
        "metadata": {
            "data_source": "vision-opd-6k",
            "original_problem": item.get("problem", ""),
            "extra_info": item.get("extra_info", {}),
        },
    }


def convert_to_jsonl(data_dir: str, keep_bbox_hint: bool) -> None:
    source = os.path.join(data_dir, "train.jsonl")
    target = os.path.join(data_dir, "train.relax.jsonl")
    if not os.path.exists(source):
        print(f"source file not found: {source}", file=sys.stderr)
        sys.exit(1)

    ok_count = 0
    skip_count = 0
    with open(source, encoding="utf-8") as f_in, open(target, "w", encoding="utf-8") as f_out:
        for line_no, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = build_record(json.loads(line), data_dir, keep_bbox_hint)
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                ok_count += 1
            except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
                print(f"line {line_no} skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
                skip_count += 1

    print(f"wrote {ok_count} samples to {target} (skipped {skip_count})")


def main() -> None:
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    if not args.skip_download:
        download_and_extract(args.hf_repo, data_dir)
    convert_to_jsonl(data_dir, keep_bbox_hint=args.keep_bbox_hint)


if __name__ == "__main__":
    main()
