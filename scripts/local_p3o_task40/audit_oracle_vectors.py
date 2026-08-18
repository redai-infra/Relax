#!/usr/bin/env python3

"""Audit Task40 Step-0 oracle inputs across DP4CP1 and DP2CP2."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


QUANTILES = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
EXPECTED_TOPOLOGY_CONFIG_FIELDS = {"context_parallel_size"}
PRECISION_CONFIG_FIELDS = ("fp16", "bf16", "params_dtype", "autocast_dtype", "attention_softmax_in_fp32")
PARALLEL_CONFIG_FIELDS = (
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
    "sequence_parallel",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _token_key(tokens: torch.Tensor) -> str:
    return hashlib.sha256(tokens.to(torch.int64).numpy().tobytes()).hexdigest()


def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value)
    return float(value)


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "max_abs": None, "quantiles": {}}
    tensor = torch.tensor(values, dtype=torch.float64).abs()
    return {
        "count": tensor.numel(),
        "max_abs": float(tensor.max()),
        "mean_abs": float(tensor.mean()),
        "quantiles": {
            f"p{int(quantile * 100):02d}": float(torch.quantile(tensor, quantile)) for quantile in QUANTILES
        },
    }


def _response_indices(
    *, total_length: int, response_length: int, cp_size: int, cp_rank: int, max_seq_len: int
) -> list[tuple[int, int]]:
    """Return ``(response_index, global_chunk_index)`` in local output
    order."""
    if cp_size == 1:
        return [(index, 0) for index in range(response_length)]

    prompt_length = total_length - response_length
    chunk_size = math.ceil(max_seq_len / (2 * cp_size))
    chunks = (cp_rank, 2 * cp_size - cp_rank - 1)
    result = []
    for chunk_index in chunks:
        # Logits at sequence index i predict token i+1. Match
        # get_logits_and_tokens_offset_with_cp rather than slicing token IDs
        # directly, including the one-token shift at every CP chunk boundary.
        start = max(chunk_index * chunk_size, prompt_length - 1) + 1
        stop = min((chunk_index + 1) * chunk_size, total_length - 1) + 1
        for token_index in range(start, max(start, stop)):
            if prompt_length <= token_index < total_length:
                result.append((token_index - prompt_length, chunk_index))
    return result


def _runtime_by_rank(run_dir: Path) -> dict[int, dict[str, Any]]:
    records = [json.loads(path.read_text()) for path in sorted((run_dir / "oracle").glob("runtime_rank*.json"))]
    if not records:
        raise ValueError(f"no runtime_rank*.json files under {run_dir / 'oracle'}")
    return {int(record["rank"]): record for record in records}


def _new_sample(
    tokens: torch.Tensor,
    positions: torch.Tensor,
    loss_mask: torch.Tensor,
    total_length: int,
    response_length: int,
    max_seq_len: int,
    max_seq_len_source: str,
) -> dict[str, Any]:
    return {
        "tokens": tokens.to(torch.int64),
        "positions": positions.to(torch.int64),
        "loss_mask": loss_mask.to(torch.int64),
        "total_length": total_length,
        "response_length": response_length,
        "max_seq_len": max_seq_len,
        "max_seq_len_source": max_seq_len_source,
        "current": {},
        "rollout": {},
        "valid": {},
        "chunk": {},
        "cp_ranks": set(),
    }


def _metadata_mismatch(sample: dict[str, Any], candidate: dict[str, Any]) -> str | None:
    tensor_fields = ("tokens", "positions", "loss_mask")
    for field in tensor_fields:
        if not torch.equal(sample[field], candidate[field]):
            return field
    for field in ("total_length", "response_length", "max_seq_len"):
        if sample[field] != candidate[field]:
            return field
    return None


def _load_run(run_dir: Path) -> dict[str, Any]:
    runtimes = _runtime_by_rank(run_dir)
    paths = sorted((run_dir / "oracle").glob("vectors_rank*_micro*.pt"))
    if not paths:
        raise ValueError(f"no vectors_rank*_micro*.pt files under {run_dir / 'oracle'}")

    samples: dict[str, dict[str, Any]] = {}
    stats_by_rank: dict[int, dict[str, float]] = defaultdict(lambda: {"s1": 0.0, "s2": 0.0, "n": 0.0})
    duplicate_metadata_checks = 0
    vector_token_count = 0
    for path in paths:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        rank = int(artifact["rank"])
        runtime = runtimes[rank]
        cp_rank = int(runtime["cp_rank"])
        cp_size = int(runtime["cp_world_size"])
        current = artifact["current_log_probs"].to(torch.float64).flatten()
        rollout = artifact["rollout_log_probs"].to(torch.float64).flatten()
        valid = artifact["valid_mask"].bool().flatten()
        offset = 0
        raw_max_seq_lens = artifact.get("max_seq_lens")
        for sample_index, (tokens, positions, loss_mask, total_length, response_length) in enumerate(
            zip(
                artifact["token_ids"],
                artifact["position_ids"],
                artifact["loss_masks"],
                artifact["total_lengths"],
                artifact["response_lengths"],
                strict=True,
            )
        ):
            total_length = int(total_length)
            response_length = int(response_length)
            if raw_max_seq_lens is None:
                max_seq_len = total_length
                max_seq_len_source = "derived_from_total_length_for_thd_single_sample"
            else:
                max_seq_len = int(raw_max_seq_lens[sample_index])
                max_seq_len_source = "artifact"
            key = _token_key(tokens)
            candidate = _new_sample(
                tokens, positions, loss_mask, total_length, response_length, max_seq_len, max_seq_len_source
            )
            if key not in samples:
                samples[key] = candidate
            else:
                mismatch = _metadata_mismatch(samples[key], candidate)
                if mismatch is not None:
                    raise ValueError(f"{path}: duplicate sample {key} disagrees in {mismatch}")
                duplicate_metadata_checks += 1
            sample = samples[key]
            local_indices = _response_indices(
                total_length=total_length,
                response_length=response_length,
                cp_size=cp_size,
                cp_rank=cp_rank,
                max_seq_len=max_seq_len,
            )
            stop = offset + len(local_indices)
            if stop > current.numel():
                raise ValueError(f"{path}: local vector ends inside sample {key}")
            for local_index, (response_index, chunk_index) in enumerate(local_indices, start=offset):
                if response_index in sample["current"]:
                    raise ValueError(f"{path}: duplicate response index {response_index} for sample {key}")
                sample["current"][response_index] = float(current[local_index])
                sample["rollout"][response_index] = float(rollout[local_index])
                sample["valid"][response_index] = bool(valid[local_index])
                sample["chunk"][response_index] = chunk_index
            sample["cp_ranks"].add(cp_rank)
            offset = stop
            vector_token_count += len(local_indices)
        if not (offset == current.numel() == rollout.numel() == valid.numel()):
            raise ValueError(
                f"{path}: reconstructed={offset}, current={current.numel()}, rollout={rollout.numel()}, "
                f"valid={valid.numel()}"
            )
        stats_by_rank[rank]["s1"] += _as_float(artifact["local_s1"])
        stats_by_rank[rank]["s2"] += _as_float(artifact["local_s2"])
        stats_by_rank[rank]["n"] += _as_float(artifact["local_n"])

    cp_size = max(int(record["cp_world_size"]) for record in runtimes.values())
    reconstruction_failures = []
    position_failures = []
    for key, sample in samples.items():
        expected_response_indices = set(range(sample["response_length"]))
        if set(sample["current"]) != expected_response_indices:
            reconstruction_failures.append(
                {
                    "sample_key": key,
                    "missing": sorted(expected_response_indices - set(sample["current"])),
                    "unexpected": sorted(set(sample["current"]) - expected_response_indices),
                }
            )
        expected_positions = torch.arange(sample["total_length"], dtype=torch.int64)
        if not torch.equal(sample["positions"], expected_positions):
            position_failures.append(key)

        if cp_size > 1:
            chunk_size = math.ceil(sample["max_seq_len"] / (2 * cp_size))
            reconstructed = torch.cat(
                [
                    sample["tokens"][chunk * chunk_size : min((chunk + 1) * chunk_size, sample["total_length"])]
                    for chunk in range(2 * cp_size)
                ]
            )
            if not torch.equal(reconstructed, sample["tokens"]):
                reconstruction_failures.append({"sample_key": key, "reason": "ordered CP chunks != CP1 sequence"})

    stats_sum = {field: sum(rank_stats[field] for rank_stats in stats_by_rank.values()) for field in ("s1", "s2", "n")}
    runtime_contract = {
        "bf16_all_ranks": all(record.get("bf16") is True for record in runtimes.values()),
        "fp16_disabled_all_ranks": all(record.get("fp16") is False for record in runtimes.values()),
        "requested_precisions": sorted({record.get("requested_precision") for record in runtimes.values()}),
        "cp_world_sizes": sorted({int(record["cp_world_size"]) for record in runtimes.values()}),
        "dp_world_sizes": sorted({int(record["dp_world_size"]) for record in runtimes.values()}),
        "micro_batch_sizes": sorted({int(record["micro_batch_size"]) for record in runtimes.values()}),
        "global_batch_sizes": sorted({int(record["global_batch_size"]) for record in runtimes.values()}),
        "qkv_formats": sorted({record.get("qkv_format", "thd") for record in runtimes.values()}),
    }
    return {
        "run_dir": str(run_dir.resolve()),
        "runtime": list(runtimes.values()),
        "vector_file_count": len(paths),
        "sample_count": len(samples),
        "vector_token_count": vector_token_count,
        "duplicate_metadata_checks": duplicate_metadata_checks,
        "samples": samples,
        "position_global_monotonic": not position_failures,
        "position_failure_sample_keys": position_failures,
        "cp_reconstruction_pass": not reconstruction_failures,
        "cp_reconstruction_failures": reconstruction_failures,
        "runtime_contract": runtime_contract,
        "stats_by_rank": {str(rank): values for rank, values in sorted(stats_by_rank.items())},
        "local_stats_sum": stats_sum,
    }


def _first_tensor_difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any] | None:
    if left.shape != right.shape:
        return {"index": 0, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    difference = (left != right).flatten().nonzero()
    if difference.numel() == 0:
        return None
    index = int(difference[0])
    return {"index": index, "left": left.flatten()[index].item(), "right": right.flatten()[index].item()}


def _compare_inputs(left: dict[str, Any], right: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    left_keys = set(left["samples"])
    right_keys = set(right["samples"])
    mismatches: list[dict[str, Any]] = []
    if left_keys != right_keys:
        mismatches.append(
            {
                "field": "token_ids",
                "left_only_sample_keys": sorted(left_keys - right_keys),
                "right_only_sample_keys": sorted(right_keys - left_keys),
                "first_difference_token": 0,
            }
        )

    checks = {field: True for field in ("token_ids", "position_ids", "max_seq_lens", "loss_masks", "valid_masks")}
    rollout_errors = []
    current_errors = []
    current_by_chunk: dict[int, list[float]] = defaultdict(list)
    boundary_errors: dict[str, list[float]] = defaultdict(list)
    valid_token_count = 0
    for key in sorted(left_keys & right_keys):
        left_sample = left["samples"][key]
        right_sample = right["samples"][key]
        tensor_fields = {"token_ids": "tokens", "position_ids": "positions", "loss_masks": "loss_mask"}
        for public_field, internal_field in tensor_fields.items():
            difference = _first_tensor_difference(left_sample[internal_field], right_sample[internal_field])
            if difference is not None:
                checks[public_field] = False
                mismatches.append(
                    {
                        "sample_key": key,
                        "field": public_field,
                        "first_difference_token": difference["index"],
                        **difference,
                    }
                )
        left_valid = torch.tensor(
            [left_sample["valid"][index] for index in sorted(left_sample["valid"])], dtype=torch.bool
        )
        right_valid = torch.tensor(
            [right_sample["valid"][index] for index in sorted(right_sample["valid"])], dtype=torch.bool
        )
        valid_difference = _first_tensor_difference(left_valid, right_valid)
        if valid_difference is not None:
            checks["valid_masks"] = False
            mismatches.append(
                {
                    "sample_key": key,
                    "field": "valid_masks",
                    "first_difference_token": valid_difference["index"],
                    **valid_difference,
                }
            )
        if left_sample["max_seq_len"] != right_sample["max_seq_len"]:
            checks["max_seq_lens"] = False
            mismatches.append(
                {
                    "sample_key": key,
                    "field": "max_seq_lens",
                    "first_difference_token": 0,
                    "left": left_sample["max_seq_len"],
                    "right": right_sample["max_seq_len"],
                }
            )

        prompt_length = left_sample["total_length"] - left_sample["response_length"]
        chunk_size = math.ceil(right_sample["max_seq_len"] / 4)
        global_boundaries = [chunk_size, 2 * chunk_size, 3 * chunk_size]
        for response_index in sorted(set(left_sample["current"]) & set(right_sample["current"])):
            if not (left_sample["valid"][response_index] and right_sample["valid"][response_index]):
                continue
            valid_token_count += 1
            rollout_error = left_sample["rollout"][response_index] - right_sample["rollout"][response_index]
            current_error = left_sample["current"][response_index] - right_sample["current"][response_index]
            rollout_errors.append(rollout_error)
            current_errors.append(current_error)
            chunk_index = right_sample["chunk"][response_index]
            current_by_chunk[chunk_index].append(current_error)
            global_token_index = prompt_length + response_index
            for boundary in global_boundaries:
                distance = abs(global_token_index - boundary)
                if distance <= 2:
                    boundary_errors[f"distance_{distance}"].append(current_error)

    checks["token_ids"] &= left_keys == right_keys
    return (
        {
            "field_equality": checks,
            "valid_token_count": valid_token_count,
            "rollout_log_probs": _distribution(rollout_errors),
            "current_log_probs": {
                "overall": _distribution(current_errors),
                "by_cp_chunk": {
                    f"chunk_{chunk}": _distribution(values) for chunk, values in sorted(current_by_chunk.items())
                },
                "near_chunk_boundaries": {
                    name: _distribution(values) for name, values in sorted(boundary_errors.items())
                },
            },
        },
        mismatches,
    )


def _load_and_compare_configs(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = json.loads(left_path.read_text())
    right = json.loads(right_path.read_text())
    all_fields = sorted(set(left) | set(right))
    differences = {
        field: {"dp4cp1": left.get(field), "dp2cp2": right.get(field)}
        for field in all_fields
        if left.get(field) != right.get(field)
    }
    unexpected = {field: value for field, value in differences.items() if field not in EXPECTED_TOPOLOGY_CONFIG_FIELDS}
    return {
        "dp4cp1_path": str(left_path.resolve()),
        "dp2cp2_path": str(right_path.resolve()),
        "dp4cp1_sha256": _sha256(left_path),
        "dp2cp2_sha256": _sha256(right_path),
        "field_count": len(all_fields),
        "precision_fields": {
            field: {"dp4cp1": left.get(field), "dp2cp2": right.get(field)} for field in PRECISION_CONFIG_FIELDS
        },
        "parallel_fields": {
            field: {"dp4cp1": left.get(field), "dp2cp2": right.get(field)} for field in PARALLEL_CONFIG_FIELDS
        },
        "differences": differences,
        "expected_topology_difference_fields": sorted(EXPECTED_TOPOLOGY_CONFIG_FIELDS),
        "unexpected_differences": unexpected,
        "invariant_fields_identical": not unexpected,
    }


def _public_run(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key != "samples"}


def _render_markdown(report: dict[str, Any]) -> str:
    comparison = report["comparison"]
    config = report["provider_transformer_config"]
    lines = [
        "# Task40 P3O offline input audit",
        "",
        f"**Verdict: `{report['verdict']}`**",
        "",
        "## Input equality",
        "",
    ]
    for field, equal in comparison["field_equality"].items():
        lines.append(f"- `{field}`: {'equal' if equal else 'DIFFERENT'}")
    lines.extend(
        [
            f"- rollout log-prob max-abs: `{comparison['rollout_log_probs']['max_abs']}`",
            f"- valid response tokens compared: `{comparison['valid_token_count']}`",
            f"- DP2CP2 ordered chunk reconstruction: `{report['runs']['dp2cp2']['cp_reconstruction_pass']}`",
            f"- global monotonic positions: DP4CP1=`{report['runs']['dp4cp1']['position_global_monotonic']}`, "
            f"DP2CP2=`{report['runs']['dp2cp2']['position_global_monotonic']}`",
            f"- BF16 runtime contract: DP4CP1=`{report['runs']['dp4cp1']['runtime_contract']['bf16_all_ranks']}`, "
            f"DP2CP2=`{report['runs']['dp2cp2']['runtime_contract']['bf16_all_ranks']}`",
            "",
            "## Current log-prob error distribution",
            "",
            "```json",
            json.dumps(comparison["current_log_probs"], indent=2, sort_keys=True),
            "```",
            "",
            "## Local sufficient-stat sums",
            "",
            "```json",
            json.dumps(
                {name: run["local_stats_sum"] for name, run in report["runs"].items()}, indent=2, sort_keys=True
            ),
            "```",
            "",
            "## Provider transformer config",
            "",
            f"Compared `{config['field_count']}` top-level fields. Invariant fields identical: "
            f"`{config['invariant_fields_identical']}`. The only allowed topology delta is "
            "`context_parallel_size`.",
            "",
            "The BF16 oracle runs did not contain `provider_transformer_config.json`; these config files are from the "
            "same P0 batch's final successful FP32 diagnostic runs. BF16 precision itself is confirmed by every "
            "`runtime_rank*.json` record, but the provider-config comparison is explicitly a same-batch proxy.",
            "",
            "```json",
            json.dumps(
                {
                    "precision_fields": config["precision_fields"],
                    "parallel_fields": config["parallel_fields"],
                    "differences": config["differences"],
                    "unexpected_differences": config["unexpected_differences"],
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
        ]
    )
    if report["mismatches"]:
        lines.extend(["", "## First mismatch", "", "```json", json.dumps(report["mismatches"][0], indent=2), "```"])
    lines.extend(
        [
            "",
            "## Evidence limitation",
            "",
            "The THD vector hook stored full un-concatenated tokens and synthesized global position IDs, not the "
            "post-slice packed token tensor or `cu_seqlens`. This audit validates the recorded inputs, inferred "
            "single-sample `max_seq_lens`, CP ownership/coverage, and response-vector alignment; Batch 7 remains "
            "responsible for direct `cu_seqlens` and weight-identity capture.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp4cp1-run", type=Path, required=True)
    parser.add_argument("--dp2cp2-run", type=Path, required=True)
    parser.add_argument("--dp4cp1-config", type=Path, required=True)
    parser.add_argument("--dp2cp2-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runs = {"dp4cp1": _load_run(args.dp4cp1_run), "dp2cp2": _load_run(args.dp2cp2_run)}
    comparison, mismatches = _compare_inputs(runs["dp4cp1"], runs["dp2cp2"])
    config = _load_and_compare_configs(args.dp4cp1_config, args.dp2cp2_config)
    runtime_contract_pass = (
        runs["dp4cp1"]["runtime_contract"]["bf16_all_ranks"]
        and runs["dp2cp2"]["runtime_contract"]["bf16_all_ranks"]
        and runs["dp4cp1"]["runtime_contract"]["cp_world_sizes"] == [1]
        and runs["dp4cp1"]["runtime_contract"]["dp_world_sizes"] == [4]
        and runs["dp2cp2"]["runtime_contract"]["cp_world_sizes"] == [2]
        and runs["dp2cp2"]["runtime_contract"]["dp_world_sizes"] == [2]
    )
    input_equal = (
        all(comparison["field_equality"].values())
        and comparison["rollout_log_probs"]["max_abs"] == 0.0
        and all(run["position_global_monotonic"] and run["cp_reconstruction_pass"] for run in runs.values())
        and runtime_contract_pass
        and config["invariant_fields_identical"]
    )
    verdict = "INPUT_IDENTICAL" if input_equal else "INPUT_MISMATCH_FOUND"
    report = {
        "verdict": verdict,
        "mismatches": mismatches,
        "runs": {name: _public_run(run) for name, run in runs.items()},
        "comparison": comparison,
        "provider_transformer_config": config,
        "runtime_contract_pass": runtime_contract_pass,
        "evidence_notes": {
            "max_seq_lens": "derived from total_length because THD single-sample artifacts omit max_seq_lens",
            "bf16_provider_config": "not captured; same-batch final successful FP32 configs used as proxy",
            "actual_parameter_hash": "deferred to Batch 7 by plan",
            "cu_seqlens": "not captured by this oracle hook; direct capture deferred to Batch 7",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "offline_input_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "offline_input_audit.md").write_text(_render_markdown(report))
    print(json.dumps({"verdict": verdict, "mismatch_count": len(mismatches)}, sort_keys=True))


if __name__ == "__main__":
    main()
