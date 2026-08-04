# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Convert frozen HotpotQA/MemAgent files to ReLax JSONL input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET_ID = "BytedTsinghua-SIA/hotpotqa"
DEFAULT_DATASET_REVISION = "27275ff4fee67ac0acb6478e405e7ac07efbdc1a"


def _json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list, tuple)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _first_user_content(prompt: Any) -> str:
    prompt = _json_safe(prompt)
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        return str(first.get("content", "")) if isinstance(first, dict) else str(first)
    return ""


def convert_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize either training parquet rows or RULER-HQA JSON rows."""
    row = _json_safe(row)
    context = row.get("context", "")
    if isinstance(context, list):
        context = "\n\n".join(str(part) for part in context)
    context = str(context).strip()
    if not context:
        return None

    if row.get("input"):
        question = str(row["input"]).strip()
        ground_truth = row.get("answers", [])
        extra_info = row
    else:
        extra_info = row.get("extra_info") or {}
        question = _first_user_content(row.get("prompt")) or str(extra_info.get("question", ""))
        reward_model = row.get("reward_model") or {}
        if isinstance(reward_model, str):
            reward_model = json.loads(reward_model)
        ground_truth = reward_model.get("ground_truth", row.get("ground_truth", []))

    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    ground_truth = [str(answer) for answer in _json_safe(ground_truth) if str(answer).strip()]
    question = question.strip()
    if not question or not ground_truth:
        return None

    return {
        "prompt": question,
        "label": ground_truth[0],
        "metadata": {
            "question": question,
            "context": context,
            "ground_truth": ground_truth,
            "num_docs": int(extra_info.get("num_docs", row.get("num_docs", 0)) or 0),
            "data_source": str(row.get("data_source", "hotpotqa")),
        },
    }


def read_rows(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Reading parquet requires pandas and pyarrow.") from exc
        yield from pd.read_parquet(path).to_dict(orient="records")
        return
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)
        return
    if suffix == ".json":
        with path.open(encoding="utf-8") as source:
            payload = json.load(source)
        if isinstance(payload, list):
            yield from payload
        elif isinstance(payload, dict) and all(isinstance(value, dict) for value in payload.values()):
            yield from payload.values()
        else:
            yield payload
        return
    raise ValueError(f"Unsupported input format: {path}")


def convert_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as destination:
        for row in read_rows(input_path):
            converted = convert_row(row)
            if converted is None:
                skipped += 1
                continue
            destination.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1
    return written, skipped


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def update_manifest(
    manifest_path: Path,
    source_path: Path,
    filename: str,
    repo_id: str,
    revision: str,
    output_path: Path | None = None,
    written: int | None = None,
    skipped: int | None = None,
) -> None:
    manifest = {"artifacts": []}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as source:
            manifest = json.load(source)
    artifact = {
        "repo_id": repo_id,
        "resolved_revision": revision,
        "filename": filename,
        "local_path": str(source_path.resolve()),
        "size_bytes": source_path.stat().st_size,
        "sha256": sha256_file(source_path),
    }
    # Hash the normalized JSONL as well as the downloaded source. This makes
    # preprocessing changes reviewable instead of proving only the input blob.
    if output_path is not None:
        artifact["converted"] = {
            "local_path": str(output_path.resolve()),
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "written_rows": written,
            "skipped_rows": skipped,
        }
    artifacts = [item for item in manifest.get("artifacts", []) if item.get("filename") != filename]
    artifacts.append(artifact)
    manifest["artifacts"] = sorted(artifacts, key=lambda item: item["filename"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as destination:
        json.dump(manifest, destination, ensure_ascii=False, indent=2)
        destination.write("\n")


def download_hf_file(repo_id: str, revision: str, filename: str, cache_dir: Path | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Hugging Face download requires huggingface_hub.") from exc
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            cache_dir=None if cache_dir is None else str(cache_dir),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Local parquet/JSON/JSONL file")
    source.add_argument("--hf-file", help="Filename in the frozen Hugging Face dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    input_path = args.input
    source_filename = input_path.name if input_path else args.hf_file
    if input_path is None:
        input_path = download_hf_file(args.repo_id, args.revision, args.hf_file, args.cache_dir)
    written, skipped = convert_file(input_path, args.output)
    if args.manifest:
        update_manifest(
            args.manifest,
            input_path,
            source_filename,
            args.repo_id,
            args.revision,
            output_path=args.output,
            written=written,
            skipped=skipped,
        )
    print(f"Written={written} skipped={skipped} output={args.output}")


if __name__ == "__main__":
    main()
