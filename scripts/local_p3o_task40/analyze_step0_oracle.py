#!/usr/bin/env python3

"""Analyze fixed-input Step-0 topology runs and emit an auditable verdict."""

import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path

import torch


ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP_METRICS = re.compile(r"step 0: (\{.*\})")


def _token_key(tokens: torch.Tensor) -> str:
    return hashlib.sha256(tokens.to(torch.int64).numpy().tobytes()).hexdigest()


def _relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-30)


def _metrics(run_dir: Path) -> dict[str, float]:
    text = ANSI.sub("", (run_dir / "stdout_stderr.log").read_text(errors="replace"))
    matches = STEP_METRICS.findall(text)
    if len(matches) != 1:
        raise ValueError(f"expected one Step-0 metric record in {run_dir}, found {len(matches)}")
    return ast.literal_eval(matches[0])


def _load_run(run_dir: Path) -> dict:
    runtimes = [json.loads(path.read_text()) for path in sorted((run_dir / "oracle").glob("runtime_rank*.json"))]
    vector_paths = sorted((run_dir / "oracle").glob("vectors_rank*_micro*.pt"))
    global_paths = sorted((run_dir / "oracle").glob("global_stats_rank*_sync0.pt"))
    if not runtimes or not vector_paths or not global_paths:
        raise ValueError(f"missing oracle artifacts in {run_dir}")

    canonical = {}
    seen_tokens = set()
    position_ids_valid = True
    local_s1 = 0.0
    local_s2 = 0.0
    local_n = 0.0
    runtime_by_rank = {record["rank"]: record for record in runtimes}
    for path in vector_paths:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        runtime = runtime_by_rank[artifact["rank"]]
        cp_rank = runtime["cp_rank"]
        cp_size = runtime["cp_world_size"]
        qkv_format = runtime.get("qkv_format", "thd")
        current = artifact["current_log_probs"].to(torch.float64)
        behavior = artifact["rollout_log_probs"].to(torch.float64)
        valid = artifact["valid_mask"].bool()
        offset = 0
        for tokens, positions, response_length, total_length in zip(
            artifact["token_ids"],
            artifact["position_ids"],
            artifact["response_lengths"],
            artifact["total_lengths"],
            strict=True,
        ):
            token_key = _token_key(tokens)
            seen_tokens.add(token_key)
            position_ids_valid &= torch.equal(positions, torch.arange(len(tokens), dtype=torch.int64))
            if cp_size == 1:
                response_indices = list(range(response_length))
            else:
                prompt_length = total_length - response_length
                if qkv_format == "bshd":
                    max_seq_lens = artifact.get("max_seq_lens")
                    if not max_seq_lens:
                        raise ValueError(f"missing max_seq_lens for BSHD artifact {path}")
                    max_seq_len = int(max_seq_lens[0])
                    chunk_size = (max_seq_len + 2 * cp_size - 1) // (2 * cp_size)
                else:
                    chunk_size = (total_length + 2 * cp_size - 1) // (2 * cp_size)
                token_ranges = (
                    (
                        max(cp_rank * chunk_size, prompt_length - 1) + 1,
                        min((cp_rank + 1) * chunk_size, total_length - 1) + 1,
                    ),
                    (
                        max((2 * cp_size - cp_rank - 1) * chunk_size, prompt_length - 1) + 1,
                        min((2 * cp_size - cp_rank) * chunk_size, total_length - 1) + 1,
                    ),
                )
                response_indices = [
                    token_index - prompt_length
                    for start, stop in token_ranges
                    for token_index in range(start, max(start, stop))
                    if prompt_length <= token_index < total_length
                ]
            for local_index, response_index in enumerate(response_indices):
                flat_index = offset + local_index
                if bool(valid[flat_index]):
                    key = (token_key, response_index)
                    if key in canonical:
                        raise ValueError(f"duplicate valid token key {key} in {run_dir}")
                    canonical[key] = (float(current[flat_index]), float(behavior[flat_index]))
            offset += len(response_indices)
        if offset != current.numel() or current.shape != behavior.shape or current.shape != valid.shape:
            raise ValueError(f"unaligned token vectors in {path}")
        local_s1 += float(artifact["local_s1"])
        local_s2 += float(artifact["local_s2"])
        local_n += float(artifact["local_n"])

    current = torch.tensor([canonical[key][0] for key in sorted(canonical)], dtype=torch.float64)
    behavior = torch.tensor([canonical[key][1] for key in sorted(canonical)], dtype=torch.float64)
    # Match production exactly: subtraction is performed in float32 before
    # ESS moments are promoted to float64.
    log_ratio = (current.to(torch.float32) - behavior.to(torch.float32)).to(torch.float64)
    ratio = torch.exp(log_ratio)
    recomputed = {
        "s1": float(ratio.sum()),
        "s2": float(ratio.square().sum()),
        "n": int(ratio.numel()),
        "kl_sum": float((log_ratio + torch.exp(torch.clamp(-log_ratio, min=-10.0, max=10.0)) - 1.0).sum()),
    }
    recomputed["normalizer"] = recomputed["n"]
    recomputed["ess"] = recomputed["s1"] ** 2 / (recomputed["n"] * (recomputed["s2"] + 1e-8))

    global_records = [torch.load(path, map_location="cpu", weights_only=False) for path in global_paths]
    global_stats = {key: float(global_records[0][key]) for key in ("s1", "s2", "n")}
    global_rank_agreement = all(
        all(float(record[key]) == global_stats[key] for key in global_stats) for record in global_records
    )
    metrics = _metrics(run_dir)
    return {
        "run_dir": str(run_dir.resolve()),
        "exit_code": int((run_dir / "exit_code.txt").read_text().strip()),
        "runtime": runtimes,
        "vector_artifact_count": len(vector_paths),
        "token_stream_count": len(seen_tokens),
        "position_ids_valid": position_ids_valid,
        "canonical": canonical,
        "current": current,
        "behavior": behavior,
        "local_stats_sum": {"s1": local_s1, "s2": local_s2, "n": local_n},
        "global_stats": global_stats,
        "global_rank_agreement": global_rank_agreement,
        "recomputed": recomputed,
        "metrics": metrics,
    }


def _public(run: dict) -> dict:
    return {key: value for key, value in run.items() if key not in {"canonical", "current", "behavior"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp4-bf16", type=Path, required=True)
    parser.add_argument("--dp2cp2-bf16", type=Path, required=True)
    parser.add_argument("--dp4-fp32", type=Path, required=True)
    parser.add_argument("--dp2cp2-fp32", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = {
        "dp4_bf16": _load_run(args.dp4_bf16),
        "dp2cp2_bf16": _load_run(args.dp2cp2_bf16),
        "dp4_fp32": _load_run(args.dp4_fp32),
        "dp2cp2_fp32": _load_run(args.dp2cp2_fp32),
    }
    comparisons = {}
    for precision in ("bf16", "fp32"):
        left = runs[f"dp4_{precision}"]
        right = runs[f"dp2cp2_{precision}"]
        key_match = set(left["canonical"]) == set(right["canonical"])
        behavior_max_abs = float(torch.max(torch.abs(left["behavior"] - right["behavior"]))) if key_match else math.inf
        current_max_abs = float(torch.max(torch.abs(left["current"] - right["current"]))) if key_match else math.inf
        metric_keys = {
            "ess": "train/p3o/normalized_ess",
            "cap": "train/p3o/adaptive_cap",
            "loss": "train/loss",
            "gradient_l2_norm": "train/grad_norm",
        }
        comparisons[precision] = {
            "valid_token_keys_identical": key_match,
            "behavior_log_probs_max_abs": behavior_max_abs,
            "current_log_probs_max_abs": current_max_abs,
            "s1_relative_error": _relative_error(left["recomputed"]["s1"], right["recomputed"]["s1"]),
            "s2_relative_error": _relative_error(left["recomputed"]["s2"], right["recomputed"]["s2"]),
            "kl_sum_relative_error": _relative_error(left["recomputed"]["kl_sum"], right["recomputed"]["kl_sum"]),
            **{
                f"{name}_relative_error": _relative_error(left["metrics"][metric], right["metrics"][metric])
                for name, metric in metric_keys.items()
            },
        }

    required_artifacts_pass = all(
        run["exit_code"] == 0
        and run["position_ids_valid"]
        and run["global_rank_agreement"]
        and run["recomputed"]["n"] == 13635
        and abs(run["global_stats"]["s1"] - run["recomputed"]["s1"]) < 1e-9
        and abs(run["global_stats"]["s2"] - run["recomputed"]["s2"]) < 1e-9
        for run in runs.values()
    )
    fp32 = comparisons["fp32"]
    fp32_alignment_pass = (
        fp32["valid_token_keys_identical"]
        and fp32["behavior_log_probs_max_abs"] == 0.0
        and fp32["current_log_probs_max_abs"] <= 1e-7
        and fp32["ess_relative_error"] <= 1e-7
        and fp32["loss_relative_error"] <= 1e-7
        and fp32["gradient_l2_norm_relative_error"] <= 1e-7
    )
    verdict = {
        "criterion": "FP32 DP4CP1 vs DP2CP2 aligns to approximately 1e-7; BF16-only differences are numerical noise",
        "required_artifacts_pass": required_artifacts_pass,
        "fp32_alignment_pass": fp32_alignment_pass,
        "classification": (
            "bf16_numerical_noise"
            if required_artifacts_pass and fp32_alignment_pass
            else "cp_forward_mismatch_remains"
        ),
        "minimum_p0_passed": required_artifacts_pass and fp32_alignment_pass,
    }
    output = {
        "runs": {name: _public(run) for name, run in runs.items()},
        "comparisons": comparisons,
        "verdict": verdict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verdict, sort_keys=True))


if __name__ == "__main__":
    main()
