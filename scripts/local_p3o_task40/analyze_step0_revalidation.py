#!/usr/bin/env python3

"""Apply the frozen Batch-7 three-stage Step-0 acceptance gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from scripts.local_p3o_task40.analyze_step0_oracle import _load_run, _token_key
from scripts.local_p3o_task40.step0_revalidation import EXPECTED_FIXTURE_SHA256


REL_TOL = 1e-6
NEAR_ZERO_ABS_TOL = 1e-9
LOGPROB_ABS_TOL = 1e-6
GRAD_REL_L2_TOL = 1e-6
GRAD_COSINE_MIN = 1.0 - 1e-9
COMMIT_UNDER_TEST = "347b9ef69b54b761247069f4e486b097c7ea93a1"
TOPOLOGIES = ("dp1", "dp4cp1", "dp2cp1", "dp2cp2")
METRIC_KEYS = {
    "normalized_ess": "train/p3o/normalized_ess",
    "adaptive_cap": "train/p3o/adaptive_cap",
    "ratio_mean": "train/p3o/ratio_mean",
    "ratio_std": "train/p3o/ratio_std",
    "cap_fraction": "train/p3o/cap_fraction",
    "clip_fraction": "train/p3o/clip_fraction",
    "behavior_kl_proxy": "train/p3o/behavior_kl_proxy",
    "adaptive_kl_loss": "train/p3o/adaptive_kl_loss",
    "reference_kl": "train/p3o/reference_kl",
    "score_loss": "train/p3o/score_loss",
    "entropy": "train/p3o/entropy",
    "total_loss": "train/p3o/total_loss",
    "loss": "train/loss",
}


def _json_write(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _plain(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


def _canonical_cu(total_lengths: list[int], pad_multiple: int) -> list[int]:
    boundaries = [0]
    for length in total_lengths:
        boundaries.append(boundaries[-1] + int(length))
    padding = (-boundaries[-1]) % pad_multiple
    if padding:
        boundaries.append(boundaries[-1] + padding)
    return boundaries


def _expected_source_cu(total_lengths: list[int], cp_size: int, pad_multiple: int) -> list[int]:
    boundaries = [0]
    for length in total_lengths:
        padded = int(length) if cp_size == 1 else math.ceil(int(length) / (2 * cp_size)) * (2 * cp_size)
        boundaries.append(boundaries[-1] + padded)
    local_length = boundaries[-1] // cp_size
    local_padding = (-local_length) % pad_multiple
    if local_padding:
        boundaries.append(boundaries[-1] + local_padding * cp_size)
    return boundaries


def _sample_metadata(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    runtimes = {
        int(record["rank"]): record
        for record in (
            json.loads(path.read_text()) for path in sorted((run_dir / "oracle").glob("runtime_rank*.json"))
        )
    }
    samples: dict[str, Any] = {}
    cu_records: dict[str, Any] = {}
    for path in sorted((run_dir / "oracle").glob("vectors_rank*_micro*.pt")):
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        runtime = runtimes[int(artifact["rank"])]
        cp_size = int(runtime["cp_world_size"])
        tokens_list = artifact["token_ids"]
        positions_list = artifact["position_ids"]
        loss_masks = artifact["loss_masks"]
        total_lengths = [int(value) for value in artifact["total_lengths"]]
        response_lengths = [int(value) for value in artifact["response_lengths"]]
        if len(tokens_list) != len(total_lengths):
            raise ValueError(f"token/length count mismatch in {path}")
        for index, (tokens, positions, total_length, response_length) in enumerate(
            zip(tokens_list, positions_list, total_lengths, response_lengths, strict=True)
        ):
            key = _token_key(tokens)
            loss_mask = loss_masks[index] if isinstance(loss_masks, (list, tuple)) else loss_masks
            record = {
                "token_ids": _plain(tokens.to(torch.int64)),
                "position_ids": _plain(positions.to(torch.int64)),
                "total_length": total_length,
                "response_length": response_length,
                "loss_mask": _plain(loss_mask),
            }
            if key in samples and samples[key] != record:
                raise ValueError(f"inconsistent per-token metadata for sample {key} in {run_dir}")
            samples[key] = record

        direct_q = _plain(artifact.get("cu_seqlens_q"))
        direct_kv = _plain(artifact.get("cu_seqlens_kv"))
        pad_multiple = int(artifact.get("relax_attention_pad_multiple") or 0)
        if direct_q is None or direct_kv is None or pad_multiple <= 0:
            raise ValueError(f"missing cu_seqlens or P3O pad metadata in {path}")
        sample_group_key = "+".join(_token_key(tokens) for tokens in tokens_list)
        canonical = _canonical_cu(total_lengths, pad_multiple)
        expected_source = _expected_source_cu(total_lengths, cp_size, pad_multiple)
        relation = {
            "sample_keys": sample_group_key.split("+"),
            "cp_size": cp_size,
            "pad_multiple": pad_multiple,
            "total_lengths": total_lengths,
            "direct_cu_seqlens_q": direct_q,
            "direct_cu_seqlens_kv": direct_kv,
            "relax_cu_seqlens_cpu": _plain(artifact.get("relax_cu_seqlens_cpu")),
            "expected_source_cu_seqlens": expected_source,
            "derived_canonical_cp1_cu_seqlens": canonical,
            "source_relation_pass": direct_q == direct_kv == expected_source,
            "derivation": (
                "CP1 keeps each real sample length unchanged. CP>1 pads each sample to 2*CP before zig-zag "
                "slicing; rank-local concatenation is then padded to pad_multiple and source boundaries are "
                "multiplied by CP. The Batch-6 adapter removes CP-only padding, concatenates total_lengths, "
                "and appends only CP1 tail padding."
            ),
        }
        if sample_group_key in cu_records and cu_records[sample_group_key] != relation:
            raise ValueError(f"inconsistent cu_seqlens relation for {sample_group_key} in {run_dir}")
        cu_records[sample_group_key] = relation
    return samples, cu_records


def _numeric_comparison(reference: float, candidate: float) -> dict[str, Any]:
    absolute_error = abs(reference - candidate)
    scale = max(abs(reference), abs(candidate))
    relative_error = absolute_error / max(scale, 1e-30)
    near_zero = scale <= NEAR_ZERO_ABS_TOL
    passed = absolute_error <= NEAR_ZERO_ABS_TOL if near_zero else relative_error <= REL_TOL
    return {
        "reference": reference,
        "candidate": candidate,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "near_zero_rule": near_zero,
        "threshold": NEAR_ZERO_ABS_TOL if near_zero else REL_TOL,
        "pass": passed,
    }


def _load_parameter_manifests(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text()) for path in sorted((run_dir / "oracle").glob("initial_parameters_rank*.json"))
    ]


def _load_gradient_summaries(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text()) for path in sorted((run_dir / "oracle" / "gradients").glob("summary_rank*.json"))
    ]


def _gradient_shard_map(run_dir: Path, summaries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    gradient_root = run_dir / "oracle" / "gradients"
    shards: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        for record in summary["tensors"]:
            if not record.get("present"):
                continue
            relative = record.get("file")
            if not relative:
                raise ValueError(f"gradient shard is missing a file for {record['name']} in {run_dir}")
            path = gradient_root / relative
            if not path.is_file():
                raise ValueError(f"missing gradient tensor file: {path}")
            enriched = dict(record)
            enriched["path"] = path
            enriched["rank"] = int(summary["rank"])
            shards.setdefault(record["name"], []).append(enriched)
    return shards


def _gradient_coverage(run_dir: Path, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    parameter_manifests = _load_parameter_manifests(run_dir)
    expected = {
        record["name"]: int(record["numel"])
        for record in parameter_manifests[0]["tensors"]
        if bool(record["requires_grad"])
    }
    shards = _gradient_shard_map(run_dir, summaries)
    errors = []
    if set(shards) != set(expected):
        errors.append(
            {
                "missing_parameters": sorted(set(expected) - set(shards)),
                "unexpected_parameters": sorted(set(shards) - set(expected)),
            }
        )
    for name in sorted(set(expected) & set(shards)):
        cursor = 0
        for record in sorted(shards[name], key=lambda item: (int(item["range_start"]), int(item["range_end"]))):
            start = int(record["range_start"])
            end = int(record["range_end"])
            if start != cursor or end <= start or int(record["numel"]) != end - start:
                errors.append(
                    {"name": name, "expected_start": cursor, "record": {**record, "path": str(record["path"])}}
                )
                break
            cursor = end
        if cursor != expected[name]:
            errors.append({"name": name, "covered_until": cursor, "parameter_numel": expected[name]})
    return {
        "pass": not errors,
        "errors": errors,
        "parameter_count": len(expected),
        "covered_parameter_count": len(shards),
        "aggregate_parameter_numel": sum(expected.values()),
        "aggregate_shard_numel": sum(int(record["numel"]) for records in shards.values() for record in records),
    }


def _reconstruct_gradient(records: list[dict[str, Any]]) -> torch.Tensor:
    ordered = sorted(records, key=lambda item: (int(item["range_start"]), int(item["range_end"])))
    parameter_numel = int(ordered[0]["parameter_numel"])
    gradient = torch.empty(parameter_numel, dtype=torch.float32)
    cursor = 0
    for record in ordered:
        start = int(record["range_start"])
        end = int(record["range_end"])
        if start != cursor or end <= start:
            raise ValueError(
                f"non-contiguous gradient shards for {record['name']}: expected {cursor}, got {start}:{end}"
            )
        shard = torch.load(record["path"], map_location="cpu", weights_only=False).reshape(-1).to(torch.float32)
        if shard.numel() != end - start:
            raise ValueError(f"gradient shard/file length mismatch for {record['name']}")
        gradient[start:end] = shard
        cursor = end
    if cursor != parameter_numel:
        raise ValueError(f"incomplete gradient reconstruction for {ordered[0]['name']}: {cursor}/{parameter_numel}")
    return gradient


def _compare_gradients(reference_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    reference_summaries = _load_gradient_summaries(reference_dir)
    candidate_summaries = _load_gradient_summaries(candidate_dir)
    reference_shards = _gradient_shard_map(reference_dir, reference_summaries)
    candidate_shards = _gradient_shard_map(candidate_dir, candidate_summaries)
    names_equal = set(reference_shards) == set(candidate_shards)
    if not names_equal:
        return {
            "pass": False,
            "parameter_names_equal": False,
            "missing_from_candidate": sorted(set(reference_shards) - set(candidate_shards)),
            "unexpected_in_candidate": sorted(set(candidate_shards) - set(reference_shards)),
        }

    diff_sq = 0.0
    reference_sq = 0.0
    candidate_sq = 0.0
    dot = 0.0
    per_tensor = []
    for name in sorted(reference_shards):
        reference = _reconstruct_gradient(reference_shards[name])
        candidate = _reconstruct_gradient(candidate_shards[name])
        if reference.shape != candidate.shape:
            per_tensor.append({"name": name, "shape_equal": False, "pass": False})
            continue
        tensor_diff_sq = 0.0
        tensor_reference_sq = 0.0
        tensor_candidate_sq = 0.0
        tensor_dot = 0.0
        for reference_chunk, candidate_chunk in zip(
            reference.split(8 * 1024 * 1024), candidate.split(8 * 1024 * 1024), strict=True
        ):
            reference64 = reference_chunk.to(torch.float64)
            candidate64 = candidate_chunk.to(torch.float64)
            delta = candidate64 - reference64
            tensor_diff_sq += float(torch.sum(delta * delta))
            tensor_reference_sq += float(torch.sum(reference64 * reference64))
            tensor_candidate_sq += float(torch.sum(candidate64 * candidate64))
            tensor_dot += float(torch.sum(reference64 * candidate64))
        tensor_rel_l2 = math.sqrt(tensor_diff_sq) / max(math.sqrt(tensor_reference_sq), 1e-30)
        if tensor_reference_sq == 0.0 and tensor_candidate_sq == 0.0:
            tensor_cosine = 1.0
        else:
            tensor_cosine = tensor_dot / max(math.sqrt(tensor_reference_sq * tensor_candidate_sq), 1e-30)
        tensor_pass = tensor_rel_l2 <= GRAD_REL_L2_TOL and tensor_cosine >= GRAD_COSINE_MIN
        per_tensor.append(
            {
                "name": name,
                "shape_equal": True,
                "relative_l2": tensor_rel_l2,
                "cosine": tensor_cosine,
                "pass": tensor_pass,
            }
        )
        diff_sq += tensor_diff_sq
        reference_sq += tensor_reference_sq
        candidate_sq += tensor_candidate_sq
        dot += tensor_dot

    relative_l2 = math.sqrt(diff_sq) / max(math.sqrt(reference_sq), 1e-30)
    cosine = 1.0 if reference_sq == candidate_sq == 0.0 else dot / max(math.sqrt(reference_sq * candidate_sq), 1e-30)
    all_shapes = all(record.get("shape_equal", False) for record in per_tensor)
    return {
        "parameter_names_equal": names_equal,
        "all_shapes_equal": all_shapes,
        "full_parameter_relative_l2": relative_l2,
        "full_parameter_cosine": cosine,
        "relative_l2_threshold": GRAD_REL_L2_TOL,
        "cosine_minimum": GRAD_COSINE_MIN,
        "per_tensor": per_tensor,
        "worst_per_tensor_relative_l2": max(
            (record.get("relative_l2", math.inf) for record in per_tensor), default=0.0
        ),
        "minimum_per_tensor_cosine": min((record.get("cosine", -math.inf) for record in per_tensor), default=1.0),
        "pass": all_shapes and relative_l2 <= GRAD_REL_L2_TOL and cosine >= GRAD_COSINE_MIN,
    }


def _artifact_contract(name: str, run_dir: Path, run: dict[str, Any]) -> dict[str, Any]:
    runtimes = run["runtime"]
    parameters = _load_parameter_manifests(run_dir)
    gradients = _load_gradient_summaries(run_dir)
    gradient_coverage = _gradient_coverage(run_dir, gradients) if gradients else {"pass": False, "errors": ["none"]}
    expected_world_size = int(runtimes[0]["world_size"]) if runtimes else 0
    expected_vectors = sum(
        int(runtime["global_batch_size"]) // (int(runtime["dp_world_size"]) * int(runtime["micro_batch_size"]))
        for runtime in runtimes
    )
    gradient_hashes = sorted({record["gradient_sha256"] for record in gradients})
    parameter_hashes = sorted({record["parameter_sha256"] for record in parameters})
    resolved = {
        "topology": name,
        "world_size": expected_world_size,
        "dp_world_size": sorted({int(runtime["dp_world_size"]) for runtime in runtimes}),
        "cp_world_size": sorted({int(runtime["cp_world_size"]) for runtime in runtimes}),
        "tp_world_size": sorted({int(runtime["tp_world_size"]) for runtime in runtimes}),
        "pp_world_size": sorted({int(runtime["pp_world_size"]) for runtime in runtimes}),
        "bf16": sorted({bool(runtime["bf16"]) for runtime in runtimes}),
        "fp16": sorted({bool(runtime["fp16"]) for runtime in runtimes}),
        "qkv_format": sorted({str(runtime["qkv_format"]) for runtime in runtimes}),
        "micro_batch_size": sorted({int(runtime["micro_batch_size"]) for runtime in runtimes}),
        "p3o_ess_scope": sorted({str(runtime["p3o_ess_scope"]) for runtime in runtimes}),
        "fixture_sha256": sorted({str(runtime["fixture_sha256"]) for runtime in runtimes}),
    }
    checks = {
        "exit_zero": run["exit_code"] == 0,
        "runtime_rank_count": len(runtimes) == expected_world_size,
        "oracle_vector_count": run["vector_artifact_count"] == expected_vectors,
        "global_stat_rank_count": len(list((run_dir / "oracle").glob("global_stats_rank*_sync0.pt")))
        == expected_world_size,
        "parameter_manifest_rank_count": len(parameters) == expected_world_size,
        "gradient_summary_rank_count": len(gradients) == expected_world_size,
        "fixture_sha": resolved["fixture_sha256"] == [EXPECTED_FIXTURE_SHA256],
        "bf16_only": resolved["bf16"] == [True] and resolved["fp16"] == [False],
        "thd": resolved["qkv_format"] == ["thd"],
        "mbs1": resolved["micro_batch_size"] == [1],
        "step_scope": resolved["p3o_ess_scope"] == ["step"],
        "position_ids_valid": bool(run["position_ids_valid"]),
        "global_stats_rank_agreement": bool(run["global_rank_agreement"]),
        "parameters_identical_within_topology": len(parameter_hashes) == 1,
        "gradient_shard_coverage": bool(gradient_coverage["pass"]),
        "no_missing_owned_gradients": all(not record["missing_owned_gradients"] for record in gradients),
        "all_gradients_finite": all(
            all((not tensor.get("present")) or tensor.get("finite", False) for tensor in record["tensors"])
            for record in gradients
        ),
    }
    return {
        "resolved_contract": resolved,
        "checks": checks,
        "parameter_sha256_values": parameter_hashes,
        "gradient_sha256_values": gradient_hashes,
        "gradient_shard_coverage": gradient_coverage,
        "pass": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for topology in TOPOLOGIES:
        parser.add_argument(f"--{topology}", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run_dirs = {topology: getattr(args, topology).resolve() for topology in TOPOLOGIES}
    runs = {topology: _load_run(run_dir) for topology, run_dir in run_dirs.items()}
    metadata = {}
    cu_relations = {}
    contracts = {}
    for topology in TOPOLOGIES:
        metadata[topology], cu_relations[topology] = _sample_metadata(run_dirs[topology])
        contracts[topology] = _artifact_contract(topology, run_dirs[topology], runs[topology])

    reference = runs["dp1"]
    reference_metadata = metadata["dp1"]
    reference_canonical_cu = {
        key: record["derived_canonical_cp1_cu_seqlens"] for key, record in cu_relations["dp1"].items()
    }
    parameter_hashes = {topology: contracts[topology]["parameter_sha256_values"] for topology in TOPOLOGIES}
    cross_topology_parameter_sha_pass = (
        all(len(values) == 1 for values in parameter_hashes.values())
        and len({values[0] for values in parameter_hashes.values()}) == 1
    )

    overall_pass = True
    topology_verdicts = {}
    for topology in TOPOLOGIES:
        run = runs[topology]
        key_match = set(reference["canonical"]) == set(run["canonical"])
        current_max_abs = float(torch.max(torch.abs(reference["current"] - run["current"]))) if key_match else math.inf
        behavior_max_abs = (
            float(torch.max(torch.abs(reference["behavior"] - run["behavior"]))) if key_match else math.inf
        )
        canonical_cu = {
            key: record["derived_canonical_cp1_cu_seqlens"] for key, record in cu_relations[topology].items()
        }
        stage1_checks = {
            "artifact_contract": contracts[topology]["pass"],
            "valid_token_keys_identical": key_match,
            "current_log_probs_max_abs": current_max_abs <= LOGPROB_ABS_TOL,
            "rollout_log_probs_exact": behavior_max_abs == 0.0,
            "per_token_metadata_equal": metadata[topology] == reference_metadata,
            "initial_parameter_sha_cross_topology": cross_topology_parameter_sha_pass,
            "source_cu_derivation_valid": all(
                record["source_relation_pass"] for record in cu_relations[topology].values()
            ),
            "derived_canonical_cu_equal": canonical_cu == reference_canonical_cu,
        }
        stage1 = {
            "status": "PASS" if all(stage1_checks.values()) else "FAIL",
            "checks": stage1_checks,
            "current_log_probs_max_abs": current_max_abs,
            "current_log_probs_max_abs_threshold": LOGPROB_ABS_TOL,
            "rollout_log_probs_max_abs": behavior_max_abs,
            "initial_parameter_sha256": parameter_hashes[topology],
            "cross_topology_initial_parameter_sha256": parameter_hashes,
            "cu_seqlens": cu_relations[topology],
        }

        stage2_values = {
            "s1": (float(reference["global_stats"]["s1"]), float(run["global_stats"]["s1"])),
            "s2": (float(reference["global_stats"]["s2"]), float(run["global_stats"]["s2"])),
            "n": (float(reference["global_stats"]["n"]), float(run["global_stats"]["n"])),
            **{
                name: (float(reference["metrics"][metric]), float(run["metrics"][metric]))
                for name, metric in METRIC_KEYS.items()
            },
        }
        stage2_comparisons = {
            name: _numeric_comparison(reference_value, candidate_value)
            for name, (reference_value, candidate_value) in stage2_values.items()
        }
        stage2 = {
            "status": "PASS" if all(record["pass"] for record in stage2_comparisons.values()) else "FAIL",
            "relative_tolerance": REL_TOL,
            "near_zero_absolute_tolerance": NEAR_ZERO_ABS_TOL,
            "comparisons": stage2_comparisons,
        }

        gradient_comparison = _compare_gradients(run_dirs["dp1"], run_dirs[topology])
        stage3 = {"status": "PASS" if gradient_comparison["pass"] else "FAIL", **gradient_comparison}
        stages_pass = stage1["status"] == stage2["status"] == stage3["status"] == "PASS"
        overall_pass &= stages_pass
        topology_verdicts[topology] = {
            "topology": topology,
            "reference": "dp1",
            "run_dir": str(run_dirs[topology]),
            "status": "PASS" if stages_pass else "FAIL",
            "artifact_contract": contracts[topology],
            "stage1": stage1,
            "stage2": stage2,
            "stage3": stage3,
        }
        topology_root = args.output_root / topology
        topology_root.mkdir(parents=True, exist_ok=True)
        _json_write(topology_root / "stage1_verdict.json", stage1)
        _json_write(topology_root / "stage2_verdict.json", stage2)
        _json_write(topology_root / "stage3_verdict.json", stage3)
        _json_write(topology_root / "cell_verdict.json", topology_verdicts[topology])

    failed_stages = {
        topology: [
            stage for stage in ("stage1", "stage2", "stage3") if topology_verdicts[topology][stage]["status"] != "PASS"
        ]
        for topology in TOPOLOGIES
        if topology_verdicts[topology]["status"] != "PASS"
    }
    input_identity_checks = {
        "artifact_contract",
        "valid_token_keys_identical",
        "rollout_log_probs_exact",
        "per_token_metadata_equal",
        "initial_parameter_sha_cross_topology",
        "source_cu_derivation_valid",
        "derived_canonical_cu_equal",
    }
    input_identity_failed = any(
        not verdict["stage1"]["checks"][check]
        for verdict in topology_verdicts.values()
        for check in input_identity_checks
    )
    route = "Batch 5" if input_identity_failed else ("Batch 6" if failed_stages else None)
    result = {
        "batch": 7,
        "commit": COMMIT_UNDER_TEST,
        "status": "COMPLETE" if overall_pass else "FAIL_ROUTING",
        "allow_batch8": overall_pass,
        "thresholds": {
            "current_log_probs_max_abs": LOGPROB_ABS_TOL,
            "stage2_relative": REL_TOL,
            "stage2_near_zero_absolute": NEAR_ZERO_ABS_TOL,
            "gradient_relative_l2": GRAD_REL_L2_TOL,
            "gradient_cosine_minimum": GRAD_COSINE_MIN,
        },
        "failed_stages": failed_stages,
        "route": route,
        "topologies": topology_verdicts,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _json_write(args.output_root / "BATCH7_VERDICT.json", result)
    print(json.dumps({"status": result["status"], "allow_batch8": overall_pass, "failed_stages": failed_stages}))
    raise SystemExit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
