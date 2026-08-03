# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Classify SGLang selected-token log-probabilities with an HF oracle.

This standalone probe sends top-p=1 requests at temperatures 1.0 and 1.2 to an
already-running SGLang server. It then scores the exact returned token
sequences with the same local Hugging Face checkpoint and compares SGLang's
``output_token_logprobs`` against both the raw model distribution and the
temperature-scaled sampling distribution.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CALIBRATION_MEAN_ABS_TOL = 0.05
CALIBRATION_MAX_ABS_TOL = 0.25
CLASSIFICATION_MEAN_ABS_MARGIN = 0.05


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def _request_generation(
    server_url: str,
    input_ids: list[int],
    *,
    temperature: float,
    sampling_seed: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    output = _post_json(
        server_url.rstrip("/") + "/generate",
        {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": temperature,
                "top_p": 1.0,
                "top_k": -1,
                "max_new_tokens": max_new_tokens,
                "sampling_seed": sampling_seed,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
        },
    )
    pairs = output.get("meta_info", {}).get("output_token_logprobs")
    if not pairs:
        raise RuntimeError(f"SGLang response has no output_token_logprobs: {output}")
    token_ids = [int(item[1]) for item in pairs]
    log_probs = [float(item[0]) for item in pairs]
    if output.get("output_ids") is not None and list(output["output_ids"]) != token_ids:
        raise RuntimeError("SGLang output_ids disagree with output_token_logprobs token IDs")
    if not all(math.isfinite(value) for value in log_probs):
        raise RuntimeError("SGLang returned a non-finite selected-token log-probability")
    return {
        "temperature": temperature,
        "sampling_seed": sampling_seed,
        "token_ids": token_ids,
        "sglang_log_probs": log_probs,
        "finish_reason": output.get("meta_info", {}).get("finish_reason"),
        "text": output.get("text", ""),
    }


def _score_with_hf(
    model: torch.nn.Module,
    prompt_ids: list[int],
    response_ids: list[int],
    *,
    temperature: float,
) -> tuple[list[float], list[float]]:
    sequence = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=sequence, use_cache=False).logits[0]
    start = len(prompt_ids) - 1
    stop = start + len(response_ids)
    response_logits = logits[start:stop].float()
    targets = torch.tensor(response_ids, dtype=torch.long, device=model.device)
    raw = response_logits.log_softmax(dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
    scaled = (response_logits / temperature).log_softmax(dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
    return raw.cpu().tolist(), scaled.cpu().tolist()


def _errors(actual: list[float], expected: list[float]) -> dict[str, float]:
    if len(actual) != len(expected) or not actual:
        raise RuntimeError(f"invalid comparison lengths: actual={len(actual)}, expected={len(expected)}")
    absolute = [abs(left - right) for left, right in zip(actual, expected, strict=True)]
    squared = [(left - right) ** 2 for left, right in zip(actual, expected, strict=True)]
    return {
        "mean_abs": sum(absolute) / len(absolute),
        "max_abs": max(absolute),
        "rmse": math.sqrt(sum(squared) / len(squared)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite semantics result: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("HF oracle requires CUDA")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Compute 17 + 25 and explain the result briefly."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    requests = [
        _request_generation(
            args.server_url,
            prompt_ids,
            temperature=temperature,
            sampling_seed=seed,
            max_new_tokens=args.max_new_tokens,
        )
        for temperature, seed in ((1.0, 20260801), (1.2, 20260802))
    ]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to("cuda")
    model.eval()

    observations = []
    for request_result in requests:
        raw, scaled = _score_with_hf(
            model,
            prompt_ids,
            request_result["token_ids"],
            temperature=request_result["temperature"],
        )
        observations.append(
            {
                "temperature": request_result["temperature"],
                "sampling_seed": request_result["sampling_seed"],
                "response_tokens": len(request_result["token_ids"]),
                "finish_reason": request_result["finish_reason"],
                "raw_model_errors": _errors(request_result["sglang_log_probs"], raw),
                "temperature_scaled_errors": _errors(request_result["sglang_log_probs"], scaled),
                "mean_raw_vs_scaled_oracle_abs_difference": _errors(raw, scaled)["mean_abs"],
                "response_text": request_result["text"],
            }
        )

    calibration = observations[0]["raw_model_errors"]
    mismatch = observations[1]
    raw_mean_abs = mismatch["raw_model_errors"]["mean_abs"]
    scaled_mean_abs = mismatch["temperature_scaled_errors"]["mean_abs"]
    calibration_passed = (
        calibration["mean_abs"] <= CALIBRATION_MEAN_ABS_TOL and calibration["max_abs"] <= CALIBRATION_MAX_ABS_TOL
    )
    if scaled_mean_abs + CLASSIFICATION_MEAN_ABS_MARGIN <= raw_mean_abs:
        classification = "temperature_scaled_sampling_distribution"
    elif raw_mean_abs + CLASSIFICATION_MEAN_ABS_MARGIN <= scaled_mean_abs:
        classification = "unscaled_raw_model_distribution"
    else:
        classification = "ambiguous"

    result = {
        "scope": "installed SGLang server selected-token output log-prob semantics",
        "sglang_version": version("sglang"),
        "model_path": str(args.model_path),
        "server_url": args.server_url,
        "prompt_tokens": len(prompt_ids),
        "top_p": 1.0,
        "top_k": -1,
        "frozen_thresholds": {
            "calibration_mean_abs": CALIBRATION_MEAN_ABS_TOL,
            "calibration_max_abs": CALIBRATION_MAX_ABS_TOL,
            "classification_mean_abs_margin": CLASSIFICATION_MEAN_ABS_MARGIN,
        },
        "calibration_passed": calibration_passed,
        "classification": classification,
        "behavior_logprob_semantics_passed": calibration_passed
        and classification == "temperature_scaled_sampling_distribution",
        "observations": observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["behavior_logprob_semantics_passed"]:
        raise RuntimeError("SGLang behavior log-probability semantics gate did not pass")


if __name__ == "__main__":
    main()
