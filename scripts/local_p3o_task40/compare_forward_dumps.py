#!/usr/bin/env python3

"""Compare Task40 forward dumps by global sample and token key."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


STAGE_ORDER = {
    "block.input": 0,
    "self_attention.input": 1,
    "qkv_projection.input": 2,
    "qkv_projection.output": 3,
    "attention_query": 4,
    "attention_key": 5,
    "attention_value": 6,
    "attention_output": 7,
    "self_attention.output": 8,
    "block.output": 9,
}


@dataclass(frozen=True)
class Capture:
    directory: Path
    metadata: dict[str, Any]


@dataclass
class DumpIndex:
    run_dir: Path
    captures: list[Capture]
    samples: dict[str, list[Capture]]
    stages: set[str]
    errors: list[str]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _find_dump_dir(run_dir: Path) -> Path:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        return run_dir / manifest.get("dump_dir", "dump")
    if (run_dir / "dump").is_dir():
        return run_dir / "dump"
    if list(run_dir.glob("rank*/micro*/metadata.json")):
        return run_dir
    raise ValueError(f"cannot find a forward dump under {run_dir}")


def _load_index(run_dir: Path) -> DumpIndex:
    dump_dir = _find_dump_dir(run_dir)
    captures: list[Capture] = []
    samples: dict[str, list[Capture]] = {}
    stages: set[str] = set()
    errors: list[str] = []
    for metadata_path in sorted(dump_dir.glob("rank*/micro*/metadata.json")):
        metadata = _load_json(metadata_path)
        capture = Capture(metadata_path.parent, metadata)
        captures.append(capture)
        if not metadata.get("complete", False):
            errors.append(f"incomplete capture: {metadata_path}")
        if metadata.get("phase") != "log_prob":
            errors.append(f"unexpected phase in {metadata_path}: {metadata.get('phase')}")
        required_stages = set(metadata.get("required_stages", []))
        actual_stages = set(metadata.get("stages", {}))
        if not required_stages or required_stages != actual_stages:
            errors.append(
                f"stage contract mismatch in {metadata_path}: "
                f"missing={sorted(required_stages - actual_stages)}, "
                f"unexpected={sorted(actual_stages - required_stages)}"
            )
        token_metadata_path = capture.directory / metadata.get("token_metadata_path", "token_metadata.pt")
        if not token_metadata_path.is_file():
            errors.append(f"missing token metadata: {token_metadata_path}")
        else:
            token_meta = _token_metadata(capture.directory)
            cu_q = token_meta.get("cu_seqlens_q")
            cu_kv = token_meta.get("cu_seqlens_kv")
            if cu_q is None or cu_kv is None or not torch.equal(cu_q, cu_kv):
                errors.append(f"missing or unequal cu_seqlens_q/kv: {metadata_path}")
            elif cu_q.numel() < 2 or bool((cu_q[1:] < cu_q[:-1]).any()):
                errors.append(f"non-monotonic cu_seqlens: {metadata_path}")
            if token_meta.get("position_ids_argument") is not None:
                errors.append(f"position_ids argument must be None for the frozen command: {metadata_path}")
            if not torch.equal(token_meta["derived_position_ids"], token_meta["local_token_indices"]):
                errors.append(f"derived position IDs disagree with token mapping: {metadata_path}")
        for sample_key in metadata.get("sample_keys", []):
            samples.setdefault(sample_key, []).append(capture)
        stages.update(metadata.get("stages", {}))
        for stage, stage_meta in metadata.get("stages", {}).items():
            if stage_meta.get("token_axis") is None:
                errors.append(f"stage has no token axis: {metadata_path}:{stage}")
            if not stage_meta.get("finite", False):
                errors.append(f"non-finite stage: {metadata_path}:{stage}")
            if not (capture.directory / stage_meta["path"]).is_file():
                errors.append(f"missing tensor: {metadata_path}:{stage}")
    if not captures:
        errors.append(f"no complete capture metadata under {dump_dir}")
    return DumpIndex(run_dir.resolve(), captures, samples, stages, errors)


@lru_cache(maxsize=256)
def _token_metadata(directory: Path) -> dict[str, Any]:
    return torch.load(directory / "token_metadata.pt", map_location="cpu", weights_only=False)


def _stage_sort_key(stage: str) -> tuple[int, int, str]:
    match = re.match(r"layer_(\d+)\.(.+)", stage)
    if not match:
        return (10**9, 0 if stage == "logits" else 1, stage)
    layer = int(match.group(1))
    suffix = match.group(2)
    return (layer, STAGE_ORDER.get(suffix, 100), suffix)


def _rows_for_sample(
    captures: list[Capture], sample_key: str, stage: str
) -> tuple[torch.Tensor, list[int], dict[int, int]]:
    row_tensors: list[torch.Tensor] = []
    token_positions: list[int] = []
    chunks_by_position: dict[int, int] = {}
    feature_shape: tuple[int, ...] | None = None
    for capture in captures:
        stage_meta = capture.metadata.get("stages", {}).get(stage)
        if stage_meta is None:
            continue
        token_meta = _token_metadata(capture.directory)
        sample_keys = token_meta["sample_keys"]
        matching_sample_indices = [index for index, key in enumerate(sample_keys) if key == sample_key]
        if len(matching_sample_indices) != 1:
            raise ValueError(
                f"{capture.directory}: sample key {sample_key} occurs {len(matching_sample_indices)} times; "
                "content-hash key is ambiguous"
            )
        sample_index = matching_sample_indices[0]
        local_sample_indices = token_meta["local_sample_indices"]
        local_token_indices = token_meta["local_token_indices"]
        local_chunk_indices = token_meta["local_chunk_indices"]
        local_real_mask = token_meta["local_real_mask"]
        selected = ((local_sample_indices == sample_index) & local_real_mask).nonzero().flatten()
        positions = local_token_indices[selected].to(torch.int64)
        chunks = local_chunk_indices[selected].to(torch.int64)

        tensor = torch.load(capture.directory / stage_meta["path"], map_location="cpu", weights_only=False)
        token_axis = int(stage_meta["token_axis"])
        tensor = tensor.movedim(token_axis, 0)
        if tensor.shape[0] != local_token_indices.numel():
            raise ValueError(
                f"{capture.directory}:{stage}: token axis has {tensor.shape[0]} rows, "
                f"metadata has {local_token_indices.numel()}"
            )
        current_feature_shape = tuple(int(size) for size in tensor.shape[1:])
        if feature_shape is None:
            feature_shape = current_feature_shape
        elif current_feature_shape != feature_shape:
            raise ValueError(f"{capture.directory}:{stage}: feature shape {current_feature_shape} != {feature_shape}")
        selected_rows = tensor.index_select(0, selected).reshape(selected.numel(), -1)
        row_tensors.append(selected_rows)
        for position, chunk in zip(positions.tolist(), chunks.tolist(), strict=True):
            if position in chunks_by_position:
                raise ValueError(f"duplicate global token key ({sample_key}, {position}) for stage {stage}")
            chunks_by_position[position] = chunk
            token_positions.append(position)

    if not row_tensors:
        raise ValueError(f"sample {sample_key} has no tensor for stage {stage}")
    rows = torch.cat(row_tensors, dim=0)
    order = sorted(range(len(token_positions)), key=token_positions.__getitem__)
    ordered_rows = rows.index_select(0, torch.tensor(order, dtype=torch.int64))
    ordered_positions = [token_positions[index] for index in order]
    return ordered_rows, ordered_positions, chunks_by_position


def _compare_rows(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    positions: list[int],
    candidate_chunks: dict[int, int],
    *,
    atol: float,
    rtol: float,
    max_chunk_elements: int,
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"row shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}")
    feature_size = int(reference.shape[1])
    rows_per_chunk = max(1, max_chunk_elements // max(1, feature_size))
    max_abs = 0.0
    diff_l2_sq = 0.0
    reference_l2_sq = 0.0
    first_bad_token: int | None = None
    finite = True
    by_cp_chunk: dict[int, float] = {}
    near_boundary: dict[int, float] = {}
    chunk_boundaries = sorted(
        position
        for previous, position in zip(positions, positions[1:])
        if candidate_chunks.get(previous) != candidate_chunks.get(position)
    )

    for start in range(0, reference.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, reference.shape[0])
        left = reference[start:stop].float()
        right = candidate[start:stop].float()
        difference = left - right
        finite = finite and bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        absolute = difference.abs()
        max_abs = max(max_abs, float(absolute.max()) if absolute.numel() else 0.0)
        diff_l2_sq += float(torch.sum(difference.double() * difference.double()))
        reference_l2_sq += float(torch.sum(left.double() * left.double()))
        tolerance = atol + rtol * torch.maximum(left.abs(), right.abs())
        row_bad = (absolute > tolerance).reshape(stop - start, -1).any(dim=1)
        row_max = absolute.reshape(stop - start, -1).amax(dim=1)
        for offset, token_max in enumerate(row_max.tolist()):
            position = positions[start + offset]
            cp_chunk = candidate_chunks[position]
            by_cp_chunk[cp_chunk] = max(by_cp_chunk.get(cp_chunk, 0.0), token_max)
            distance = min((abs(position - boundary) for boundary in chunk_boundaries), default=10**9)
            if distance <= 2:
                near_boundary[distance] = max(near_boundary.get(distance, 0.0), token_max)
        if first_bad_token is None and bool(row_bad.any()):
            first_local = int(row_bad.nonzero()[0])
            first_bad_token = positions[start + first_local]

    relative_l2 = math.sqrt(diff_l2_sq) / max(math.sqrt(reference_l2_sq), 1e-30)
    return {
        "finite": finite,
        "shape": list(reference.shape),
        "token_count": len(positions),
        "max_abs": max_abs,
        "relative_l2": relative_l2,
        "within_tolerance": finite and first_bad_token is None,
        "first_bad_global_token_index": first_bad_token,
        "max_abs_by_candidate_cp_chunk": {str(key): value for key, value in sorted(by_cp_chunk.items())},
        "max_abs_near_candidate_chunk_boundary": {
            f"distance_{key}": value for key, value in sorted(near_boundary.items())
        },
    }


def _metadata_checks(reference: DumpIndex, candidate: DumpIndex) -> dict[str, Any]:
    reference_samples = set(reference.samples)
    candidate_samples = set(candidate.samples)
    checks: dict[str, Any] = {
        "sample_keys_equal": reference_samples == candidate_samples,
        "reference_only_sample_keys": sorted(reference_samples - candidate_samples),
        "candidate_only_sample_keys": sorted(candidate_samples - reference_samples),
        "per_sample": {},
    }
    for sample_key in sorted(reference_samples & candidate_samples):
        reference_captures = reference.samples[sample_key]
        candidate_captures = candidate.samples[sample_key]
        reference_meta = _token_metadata(reference_captures[0].directory)
        candidate_meta = _token_metadata(candidate_captures[0].directory)
        reference_tokens = next(
            tokens
            for key, tokens in zip(reference_meta["sample_keys"], reference_meta["full_token_ids"], strict=True)
            if key == sample_key
        )
        candidate_tokens = next(
            tokens
            for key, tokens in zip(candidate_meta["sample_keys"], candidate_meta["full_token_ids"], strict=True)
            if key == sample_key
        )
        reference_positions = sorted(
            {
                int(position)
                for capture in reference_captures
                for key, position, real in zip(
                    (
                        _token_metadata(capture.directory)["sample_keys"][int(index)] if int(index) >= 0 else None
                        for index in _token_metadata(capture.directory)["local_sample_indices"]
                    ),
                    _token_metadata(capture.directory)["local_token_indices"],
                    _token_metadata(capture.directory)["local_real_mask"],
                    strict=True,
                )
                if key == sample_key and bool(real)
            }
        )
        candidate_positions = sorted(
            {
                int(position)
                for capture in candidate_captures
                for key, position, real in zip(
                    (
                        _token_metadata(capture.directory)["sample_keys"][int(index)] if int(index) >= 0 else None
                        for index in _token_metadata(capture.directory)["local_sample_indices"]
                    ),
                    _token_metadata(capture.directory)["local_token_indices"],
                    _token_metadata(capture.directory)["local_real_mask"],
                    strict=True,
                )
                if key == sample_key and bool(real)
            }
        )
        checks["per_sample"][sample_key] = {
            "token_ids_equal": torch.equal(reference_tokens, candidate_tokens),
            "global_token_positions_equal": reference_positions == candidate_positions,
            "reference_token_count": len(reference_positions),
            "candidate_token_count": len(candidate_positions),
        }
    checks["all_equal"] = checks["sample_keys_equal"] and all(
        sample["token_ids_equal"] and sample["global_token_positions_equal"]
        for sample in checks["per_sample"].values()
    )
    return checks


def _classification(first_stage: str | None, metadata_equal: bool) -> str | None:
    if not metadata_equal:
        return "INPUT_PACKING_BUG"
    if first_stage is None:
        return None
    if first_stage == "logits":
        return "LOGPROB_EXTRACTION_BUG"
    return "ATTENTION_KERNEL_ORDER"


def compare(
    reference_dir: Path,
    candidate_dir: Path,
    *,
    atol: float,
    rtol: float,
    max_chunk_elements: int,
) -> dict[str, Any]:
    """Compare two runs under the global sample/token contract."""
    reference = _load_index(reference_dir)
    candidate = _load_index(candidate_dir)
    metadata = _metadata_checks(reference, candidate)
    errors = reference.errors + candidate.errors
    if reference.stages != candidate.stages:
        errors.append(
            f"stage sets differ: reference_only={sorted(reference.stages - candidate.stages)}, "
            f"candidate_only={sorted(candidate.stages - reference.stages)}"
        )

    stage_reports: dict[str, Any] = {}
    first_divergence: dict[str, Any] | None = None
    common_samples = sorted(set(reference.samples) & set(candidate.samples))
    for stage in sorted(reference.stages & candidate.stages, key=_stage_sort_key):
        sample_reports: dict[str, Any] = {}
        for sample_key in common_samples:
            try:
                left, left_positions, _ = _rows_for_sample(reference.samples[sample_key], sample_key, stage)
                right, right_positions, right_chunks = _rows_for_sample(
                    candidate.samples[sample_key], sample_key, stage
                )
                if left_positions != right_positions:
                    raise ValueError(
                        f"global token coverage differs: reference={left_positions[:8]}... candidate={right_positions[:8]}..."
                    )
                result = _compare_rows(
                    left,
                    right,
                    left_positions,
                    right_chunks,
                    atol=atol,
                    rtol=rtol,
                    max_chunk_elements=max_chunk_elements,
                )
                sample_reports[sample_key] = result
                if not result["within_tolerance"] and first_divergence is None:
                    first_divergence = {
                        "stage": stage,
                        "sample_key": sample_key,
                        "global_token_index": result["first_bad_global_token_index"],
                        "max_abs": result["max_abs"],
                        "relative_l2": result["relative_l2"],
                    }
            except (OSError, RuntimeError, ValueError) as exc:
                message = f"{stage}:{sample_key}: {exc}"
                errors.append(message)
                sample_reports[sample_key] = {"error": str(exc), "within_tolerance": False}
                if first_divergence is None:
                    first_divergence = {"stage": stage, "sample_key": sample_key, "error": str(exc)}
        stage_reports[stage] = {
            "within_tolerance": bool(sample_reports)
            and all(sample.get("within_tolerance", False) for sample in sample_reports.values()),
            "samples": sample_reports,
        }

    passed = (
        metadata["all_equal"]
        and not errors
        and bool(stage_reports)
        and all(stage["within_tolerance"] for stage in stage_reports.values())
    )
    first_stage = None if first_divergence is None else first_divergence.get("stage")
    return {
        "verdict": "FORWARD_MATCH" if passed else "FORWARD_MISMATCH",
        "reference": str(reference.run_dir),
        "candidate": str(candidate.run_dir),
        "tolerances": {"atol": atol, "rtol": rtol},
        "metadata": metadata,
        "errors": errors,
        "stages": stage_reports,
        "first_divergence": first_divergence,
        "root_cause_classification": _classification(first_stage, metadata["all_equal"]),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Task40 CP forward comparison",
        "",
        f"**Verdict: `{report['verdict']}`**",
        "",
        f"- Reference: `{report['reference']}`",
        f"- Candidate: `{report['candidate']}`",
        f"- Metadata equal: `{report['metadata']['all_equal']}`",
        f"- Root-cause classification: `{report['root_cause_classification']}`",
        "",
        "## First divergence",
        "",
        "```json",
        json.dumps(report["first_divergence"], indent=2, sort_keys=True),
        "```",
        "",
        "## Stage summary",
        "",
        "| Stage | Within tolerance | Worst max-abs | Worst rel-L2 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for stage, stage_report in report["stages"].items():
        numeric = [sample for sample in stage_report["samples"].values() if "max_abs" in sample]
        worst_abs = max((sample["max_abs"] for sample in numeric), default=None)
        worst_rel = max((sample["relative_l2"] for sample in numeric), default=None)
        lines.append(f"| `{stage}` | `{stage_report['within_tolerance']}` | `{worst_abs}` | `{worst_rel}` |")
    if report["errors"]:
        lines.extend(["", "## Contract errors", "", "```json", json.dumps(report["errors"], indent=2), "```"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="DP4CP1 or other CP1 run directory")
    parser.add_argument("--candidate", type=Path, required=True, help="DP2CP2/DP2CP1 run directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--max-chunk-elements", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be non-negative")
    if args.max_chunk_elements <= 0:
        parser.error("--max-chunk-elements must be positive")

    report = compare(
        args.reference,
        args.candidate,
        atol=args.atol,
        rtol=args.rtol,
        max_chunk_elements=args.max_chunk_elements,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ROOT_CAUSE.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "ROOT_CAUSE.md").write_text(_render_markdown(report))
    sys.stdout.write(
        json.dumps({"verdict": report["verdict"], "first_divergence": report["first_divergence"]}, sort_keys=True)
        + "\n"
    )
    if report["verdict"] != "FORWARD_MATCH":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
