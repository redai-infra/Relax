# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare real-Qwen P3O replay in four-rank BF16 and single-GPU FP32.

The probe has two explicit phases. ``prepare`` samples an immutable batch from
the local Qwen checkpoint at temperature 1.2 and stores the selected-token
behavior log-probabilities, masks, MOPD rewards, and group-normalized
advantages. ``compare`` replays that exact batch for ten optimizer updates:
four ranks execute a BF16 model with FP32 master parameters and NCCL gradient
summation, while rank zero executes the same updates with a full-FP32 oracle.

This exercises real model forward/backward and real rollout data with Relax's
production P3O primitives. It deliberately does not claim full Megatron-step
parity: the production launcher has no verified FP32 mode and the formal runs
did not persist replayable optimizer windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from relax.engine.rewards.mopd import get_mopd_reward
from relax.utils.training.p3o_utils import (
    P3OSufficientStats,
    compute_p3o_sufficient_stats_unchecked,
    compute_p3o_token_terms,
    finalize_p3o_step_context,
)


WORLD_SIZE = 4
BEHAVIOR_TEMPERATURE = 1.2
BEHAVIOR_TOP_P = 1.0
OPTIMIZER_STEPS = 10
GRAD_CLIP = 1.0

# Frozen before this real-rollout comparison. These are the upward-rounded
# bounds from the 2026-07-31 synthetic four-A100 calibration.
ESS_RTOL = 1e-3
ESS_ATOL = 1e-3
LOSS_RTOL = 2e-3
LOSS_ATOL = 3e-3
GRAD_RELATIVE_L2_TOL = 2e-3

# Match the first ten learning rates of the eleven-step formal cosine schedule.
FORMAL_LEARNING_RATES = (
    9.090909090909091e-6,
    9.797464868072489e-6,
    9.118382907149164e-6,
    8.028048435688333e-6,
    6.635339816587109e-6,
    5.079329819174041e-6,
    3.5153981233586277e-6,
    2.09971545214401e-6,
    9.73648712344707e-7,
    2.4964441129527337e-7,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _response_mask(response_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Return a mask through the first EOS token, inclusive."""
    mask = torch.ones_like(response_ids, dtype=torch.bool)
    eos_positions = torch.nonzero(response_ids == eos_token_id, as_tuple=False)
    if eos_positions.numel() != 0:
        first_eos = int(eos_positions[0, 0])
        mask[first_eos + 1 :] = False
    return mask


def _normalize_group_rewards(rewards: list[float]) -> torch.Tensor:
    values = torch.tensor(rewards, dtype=torch.float32)
    centered = values - values.mean()
    return centered / (values.std() + 1e-6)


def _load_model(model_path: Path, *, dtype: torch.dtype, device: torch.device) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.train()
    return model.to(device)


def _sample_group(
    model: torch.nn.Module,
    tokenizer: Any,
    question: str,
    answer: str,
    *,
    samples_per_prompt: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)
    prompt_length = input_ids.shape[1]

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=BEHAVIOR_TEMPERATURE,
            top_p=BEHAVIOR_TOP_P,
            top_k=0,
            num_return_sequences=samples_per_prompt,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response_ids = generated.sequences[:, prompt_length:]
    if len(generated.scores) != response_ids.shape[1]:
        raise RuntimeError(
            f"generation score/token mismatch: scores={len(generated.scores)}, tokens={response_ids.shape[1]}"
        )
    behavior_log_probs = torch.stack(
        [
            score.float().log_softmax(dim=-1).gather(1, response_ids[:, step : step + 1]).squeeze(1)
            for step, score in enumerate(generated.scores)
        ],
        dim=1,
    ).cpu()

    records: list[dict[str, Any]] = []
    rewards: list[float] = []
    for sample_index in range(samples_per_prompt):
        mask = _response_mask(response_ids[sample_index], tokenizer.eos_token_id).cpu()
        valid_count = int(mask.sum())
        trimmed_response = response_ids[sample_index, :valid_count].cpu()
        response_text = tokenizer.decode(trimmed_response, skip_special_tokens=False)
        reward = get_mopd_reward(response_text, answer, {"data_source": "gsm8k"})
        rewards.append(reward)
        records.append(
            {
                "tokens": generated.sequences[sample_index, : prompt_length + valid_count].cpu(),
                "prompt_length": prompt_length,
                "behavior_log_probs": behavior_log_probs[sample_index, :valid_count].clone(),
                "response_text": response_text,
                "reward": reward,
            }
        )

    normalized_rewards = _normalize_group_rewards(rewards)
    for record, advantage in zip(records, normalized_rewards, strict=True):
        record["advantage"] = float(advantage)
    return records


def _pack_records(records: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    batch_size = len(records)
    max_total_length = max(len(record["tokens"]) for record in records)
    tokens = torch.full((batch_size, max_total_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_total_length), dtype=torch.long)
    response_mask = torch.zeros((batch_size, max_total_length - 1), dtype=torch.bool)
    behavior_log_probs = torch.zeros((batch_size, max_total_length - 1), dtype=torch.float32)
    advantages = torch.zeros((batch_size, max_total_length - 1), dtype=torch.float32)

    for row, record in enumerate(records):
        sample_tokens = record["tokens"]
        prompt_length = int(record["prompt_length"])
        response_length = len(record["behavior_log_probs"])
        total_length = len(sample_tokens)
        target_start = prompt_length - 1
        target_end = target_start + response_length
        tokens[row, :total_length] = sample_tokens
        attention_mask[row, :total_length] = 1
        response_mask[row, target_start:target_end] = True
        behavior_log_probs[row, target_start:target_end] = record["behavior_log_probs"]
        advantages[row, target_start:target_end] = float(record["advantage"])

    return {
        "tokens": tokens,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "behavior_log_probs": behavior_log_probs,
        "advantages": advantages,
    }


def prepare_rollout(args: argparse.Namespace) -> None:
    """Generate and persist a fixed real-Qwen rollout batch."""
    if not torch.cuda.is_available():
        raise RuntimeError("rollout preparation requires CUDA")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda", 0)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model(args.model_path, dtype=torch.bfloat16, device=device)

    table = parquet.read_table(args.dataset_path, columns=["question", "answer"])
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    candidate_indices = torch.randperm(len(table), generator=generator)[: args.max_candidate_prompts].tolist()

    selected_records: list[dict[str, Any]] = []
    selected_groups: list[dict[str, Any]] = []
    candidate_groups: list[dict[str, Any]] = []
    for dataset_index in candidate_indices:
        row = table.slice(dataset_index, 1).to_pylist()[0]
        group_records = _sample_group(
            model,
            tokenizer,
            row["question"],
            row["answer"],
            samples_per_prompt=args.samples_per_prompt,
            max_new_tokens=args.max_new_tokens,
        )
        rewards = [float(record["reward"]) for record in group_records]
        candidate_groups.append(
            {
                "dataset_index": dataset_index,
                "question": row["question"],
                "answer": row["answer"],
                "rewards": rewards,
                "responses": [record["response_text"] for record in group_records],
            }
        )
        if len(set(rewards)) == 1:
            continue
        group_index = len(selected_groups)
        for record in group_records:
            record["group_index"] = group_index
        selected_records.extend(group_records)
        selected_groups.append(
            {
                "group_index": group_index,
                "dataset_index": dataset_index,
                "question": row["question"],
                "answer": row["answer"],
                "rewards": rewards,
            }
        )
        if len(selected_groups) == args.num_prompts:
            break

    if len(selected_groups) != args.num_prompts:
        failure_path = args.rollout_path.with_suffix(".prepare_failure.json")
        if failure_path.exists():
            raise FileExistsError(f"refusing to overwrite preparation failure evidence: {failure_path}")
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(
                {
                    "reason": "insufficient mixed-reward groups",
                    "seed": args.seed,
                    "candidate_indices": candidate_indices,
                    "candidate_groups": candidate_groups,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"found only {len(selected_groups)} mixed-reward groups among "
            f"{len(candidate_indices)} deterministic candidates"
        )

    packed = _pack_records(selected_records, tokenizer.pad_token_id)
    if not torch.isfinite(packed["behavior_log_probs"][packed["response_mask"]]).all():
        raise RuntimeError("rollout preparation produced non-finite behavior log-probabilities")
    if not torch.any(packed["advantages"][packed["response_mask"]] != 0):
        raise RuntimeError("rollout preparation produced only zero advantages")

    payload = {
        **packed,
        "metadata": {
            "model_path": str(args.model_path),
            "dataset_path": str(args.dataset_path),
            "seed": args.seed,
            "behavior_temperature": BEHAVIOR_TEMPERATURE,
            "behavior_top_p": BEHAVIOR_TOP_P,
            "samples_per_prompt": args.samples_per_prompt,
            "max_new_tokens": args.max_new_tokens,
            "candidate_indices": candidate_indices,
            "selected_groups": selected_groups,
            "responses": [record["response_text"] for record in selected_records],
            "rewards": [float(record["reward"]) for record in selected_records],
            "sequence_advantages": [float(record["advantage"]) for record in selected_records],
            "valid_tokens": int(packed["response_mask"].sum()),
            "score_semantics": "Transformers processed generation scores after temperature/top-p",
        },
    }
    args.rollout_path.parent.mkdir(parents=True, exist_ok=True)
    if args.rollout_path.exists():
        raise FileExistsError(f"refusing to overwrite rollout artifact: {args.rollout_path}")
    torch.save(payload, args.rollout_path)
    manifest = {
        **payload["metadata"],
        "rollout_path": str(args.rollout_path),
        "rollout_sha256": _sha256(args.rollout_path),
        "batch_size": int(packed["tokens"].shape[0]),
        "padded_total_length": int(packed["tokens"].shape[1]),
    }
    manifest_path = args.rollout_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def _selected_log_probs(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids=tokens, attention_mask=attention_mask, use_cache=False).logits[:, :-1]
    targets = tokens[:, 1:]
    selected = logits.float().log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return torch.where(response_mask, selected, torch.zeros_like(selected))


def _accumulate_stats(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    *,
    micro_batch_size: int,
) -> tuple[P3OSufficientStats, torch.Tensor]:
    stats_vector = torch.zeros(3, dtype=torch.float64, device=tokens.device)
    invalid_flag = torch.zeros((), dtype=torch.float64, device=tokens.device)
    with torch.no_grad():
        for start in range(0, len(tokens), micro_batch_size):
            stop = start + micro_batch_size
            log_probs = _selected_log_probs(
                model,
                tokens[start:stop],
                attention_mask[start:stop],
                response_mask[start:stop],
            )
            stats, chunk_invalid_flag = compute_p3o_sufficient_stats_unchecked(
                log_probs,
                behavior_log_probs[start:stop],
                response_mask[start:stop],
            )
            stats_vector += stats.as_vector()
            invalid_flag += chunk_invalid_flag
    return P3OSufficientStats.from_vector(stats_vector), invalid_flag


def _reduce_real_model_gradients(
    model_parameters: list[torch.nn.Parameter],
    master_parameters: list[torch.nn.Parameter],
) -> None:
    for model_parameter, master_parameter in zip(model_parameters, master_parameters, strict=True):
        if model_parameter.grad is None:
            master_parameter.grad = None
            continue
        gradient = model_parameter.grad.detach().float()
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        master_parameter.grad = gradient
        model_parameter.grad = None


def _vector_comparison(actual: list[torch.Tensor | None], expected: list[torch.Tensor | None]) -> tuple[float, float]:
    difference_sq = torch.zeros((), dtype=torch.float64, device=expected[0].device)
    actual_sq = torch.zeros_like(difference_sq)
    expected_sq = torch.zeros_like(difference_sq)
    dot = torch.zeros_like(difference_sq)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        if actual_tensor is None or expected_tensor is None:
            if actual_tensor is not expected_tensor:
                raise RuntimeError("BF16 and FP32 runs disagree on whether a tensor is present")
            continue
        actual_float = actual_tensor.detach().float()
        expected_float = expected_tensor.detach().float()
        difference_sq += torch.sum((actual_float - expected_float).square(), dtype=torch.float64)
        actual_sq += torch.sum(actual_float.square(), dtype=torch.float64)
        expected_sq += torch.sum(expected_float.square(), dtype=torch.float64)
        dot += torch.sum(actual_float * expected_float, dtype=torch.float64)
    epsilon = torch.finfo(torch.float64).eps
    relative_l2 = torch.sqrt(difference_sq) / torch.sqrt(expected_sq).clamp_min(epsilon)
    cosine = dot / (torch.sqrt(actual_sq) * torch.sqrt(expected_sq)).clamp_min(epsilon)
    return float(relative_l2), float(cosine)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = max(abs(float(expected)), torch.finfo(torch.float64).eps)
    return abs(float(actual) - float(expected)) / denominator


def _set_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def compare_rollout(args: argparse.Namespace) -> None:
    """Run the distributed BF16 and rank-zero FP32 replay comparison."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"real-Qwen replay requires exactly {WORLD_SIZE} ranks, got {world_size}")
    if args.steps < 1 or args.steps > len(FORMAL_LEARNING_RATES):
        raise ValueError(f"steps must be in [1, {len(FORMAL_LEARNING_RATES)}]")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    payload = torch.load(args.rollout_path, map_location="cpu", weights_only=False)
    rollout_sha256 = _sha256(args.rollout_path) if rank == 0 else ""
    batch_size = int(payload["tokens"].shape[0])
    if batch_size % world_size != 0:
        raise RuntimeError(f"batch size {batch_size} is not divisible by world size {world_size}")
    local_indices = torch.arange(rank, batch_size, world_size)

    distributed_model = _load_model(args.model_path, dtype=torch.bfloat16, device=device)
    model_parameters = list(distributed_model.parameters())
    master_parameters = [
        torch.nn.Parameter(parameter.detach().float().clone(), requires_grad=True) for parameter in model_parameters
    ]
    distributed_optimizer = torch.optim.AdamW(
        master_parameters,
        lr=FORMAL_LEARNING_RATES[0],
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    local_tokens = payload["tokens"][local_indices].to(device)
    local_attention_mask = payload["attention_mask"][local_indices].to(device)
    local_response_mask = payload["response_mask"][local_indices].to(device)
    local_behavior = payload["behavior_log_probs"][local_indices].to(device)
    local_advantages = payload["advantages"][local_indices].to(device)

    reference_model = None
    reference_optimizer = None
    if rank == 0:
        reference_model = _load_model(args.model_path, dtype=torch.float32, device=device)
        with torch.no_grad():
            for reference_parameter, master_parameter in zip(
                reference_model.parameters(), master_parameters, strict=True
            ):
                reference_parameter.copy_(master_parameter)
        reference_optimizer = torch.optim.AdamW(
            reference_model.parameters(),
            lr=FORMAL_LEARNING_RATES[0],
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )
        full_tokens = payload["tokens"].to(device)
        full_attention_mask = payload["attention_mask"].to(device)
        full_response_mask = payload["response_mask"].to(device)
        full_behavior = payload["behavior_log_probs"].to(device)
        full_advantages = payload["advantages"].to(device)

    observations: list[dict[str, float | int]] = []
    initial_parameter_relative_l2 = None
    if rank == 0:
        assert reference_model is not None
        initial_parameter_relative_l2, _ = _vector_comparison(
            [parameter.detach() for parameter in master_parameters],
            [parameter.detach() for parameter in reference_model.parameters()],
        )

    for step in range(args.steps):
        learning_rate = FORMAL_LEARNING_RATES[step]
        distributed_model.zero_grad(set_to_none=True)
        distributed_optimizer.zero_grad(set_to_none=True)
        _set_lr(distributed_optimizer, learning_rate)

        local_stats, invalid_flag = _accumulate_stats(
            distributed_model,
            local_tokens,
            local_attention_mask,
            local_response_mask,
            local_behavior,
            micro_batch_size=len(local_tokens),
        )
        reduced = torch.cat([local_stats.as_vector(), invalid_flag.reshape(1)])
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        if bool(reduced[3] > 0):
            raise RuntimeError("distributed BF16 replay produced a non-finite valid-token ratio")
        distributed_context = finalize_p3o_step_context(P3OSufficientStats.from_vector(reduced[:3]))
        local_log_probs = _selected_log_probs(
            distributed_model,
            local_tokens,
            local_attention_mask,
            local_response_mask,
        )
        distributed_terms = compute_p3o_token_terms(
            local_log_probs,
            local_behavior,
            local_advantages,
            local_response_mask,
            distributed_context,
        )
        local_loss = (distributed_terms.score_loss + distributed_terms.adaptive_kl_loss).sum()
        local_loss = local_loss / distributed_context.valid_token_count
        local_loss.backward()
        _reduce_real_model_gradients(model_parameters, master_parameters)
        distributed_loss = local_loss.detach().clone()
        dist.all_reduce(distributed_loss, op=dist.ReduceOp.SUM)
        del distributed_terms, local_log_probs, local_loss

        reference_context = None
        reference_loss = None
        gradient_relative_l2 = None
        gradient_cosine = None
        if rank == 0:
            assert reference_model is not None and reference_optimizer is not None
            reference_optimizer.zero_grad(set_to_none=True)
            _set_lr(reference_optimizer, learning_rate)
            reference_stats, reference_invalid_flag = _accumulate_stats(
                reference_model,
                full_tokens,
                full_attention_mask,
                full_response_mask,
                full_behavior,
                micro_batch_size=1,
            )
            if bool(reference_invalid_flag > 0):
                raise RuntimeError("FP32 replay produced a non-finite valid-token ratio")
            reference_context = finalize_p3o_step_context(reference_stats)
            reference_loss = torch.zeros((), dtype=torch.float32, device=device)
            for sample_index in range(batch_size):
                sample_slice = slice(sample_index, sample_index + 1)
                reference_log_probs = _selected_log_probs(
                    reference_model,
                    full_tokens[sample_slice],
                    full_attention_mask[sample_slice],
                    full_response_mask[sample_slice],
                )
                reference_terms = compute_p3o_token_terms(
                    reference_log_probs,
                    full_behavior[sample_slice],
                    full_advantages[sample_slice],
                    full_response_mask[sample_slice],
                    reference_context,
                )
                sample_loss = (reference_terms.score_loss + reference_terms.adaptive_kl_loss).sum()
                sample_loss = sample_loss / reference_context.valid_token_count
                sample_loss.backward()
                reference_loss += sample_loss.detach()
            gradient_relative_l2, gradient_cosine = _vector_comparison(
                [parameter.grad for parameter in master_parameters],
                [parameter.grad for parameter in reference_model.parameters()],
            )

        dist.barrier()
        distributed_grad_norm = torch.nn.utils.clip_grad_norm_(master_parameters, GRAD_CLIP)
        distributed_optimizer.step()
        with torch.no_grad():
            for model_parameter, master_parameter in zip(model_parameters, master_parameters, strict=True):
                model_parameter.copy_(master_parameter.to(dtype=torch.bfloat16))
        if rank == 0:
            assert reference_model is not None and reference_optimizer is not None
            reference_grad_norm = torch.nn.utils.clip_grad_norm_(reference_model.parameters(), GRAD_CLIP)
            reference_optimizer.step()
        dist.barrier()

        if rank == 0:
            assert reference_model is not None
            assert reference_context is not None and reference_loss is not None
            assert gradient_relative_l2 is not None and gradient_cosine is not None
            parameter_relative_l2, parameter_cosine = _vector_comparison(
                [parameter.detach() for parameter in master_parameters],
                [parameter.detach() for parameter in reference_model.parameters()],
            )
            observations.append(
                {
                    "step": step + 1,
                    "learning_rate": learning_rate,
                    "ess_bf16_nccl": float(distributed_context.normalized_ess),
                    "ess_fp32": float(reference_context.normalized_ess),
                    "ess_abs_error": abs(
                        float(distributed_context.normalized_ess) - float(reference_context.normalized_ess)
                    ),
                    "ess_rel_error": _relative_error(
                        distributed_context.normalized_ess, reference_context.normalized_ess
                    ),
                    "loss_bf16_nccl": float(distributed_loss),
                    "loss_fp32": float(reference_loss.detach()),
                    "loss_abs_error": abs(float(distributed_loss) - float(reference_loss.detach())),
                    "loss_rel_error": _relative_error(distributed_loss, reference_loss),
                    "gradient_relative_l2": gradient_relative_l2,
                    "gradient_cosine": gradient_cosine,
                    "parameter_relative_l2": parameter_relative_l2,
                    "parameter_cosine": parameter_cosine,
                    "grad_norm_bf16_nccl": float(distributed_grad_norm),
                    "grad_norm_fp32": float(reference_grad_norm),
                }
            )
            print(json.dumps(observations[-1], sort_keys=True))
        dist.barrier()

    peak_allocated = torch.tensor(torch.cuda.max_memory_allocated(device), dtype=torch.float64, device=device)
    peak_reserved = torch.tensor(torch.cuda.max_memory_reserved(device), dtype=torch.float64, device=device)
    dist.all_reduce(peak_allocated, op=dist.ReduceOp.MAX)
    dist.all_reduce(peak_reserved, op=dist.ReduceOp.MAX)

    passed = torch.ones((), dtype=torch.int32, device=device)
    if rank == 0:
        summary: dict[str, Any] = {
            "scope": "real Qwen3-0.6B rollout replay with production P3O primitives; standalone HF model path",
            "full_megatron_step_parity": False,
            "world_size": world_size,
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(world_size)],
            "model_path": str(args.model_path),
            "rollout_path": str(args.rollout_path),
            "rollout_sha256": rollout_sha256,
            "steps": args.steps,
            "batch_size": batch_size,
            "valid_tokens": int(payload["response_mask"].sum()),
            "initial_parameter_relative_l2": initial_parameter_relative_l2,
            "peak_allocated_gb": float(peak_allocated) / 1024**3,
            "peak_reserved_gb": float(peak_reserved) / 1024**3,
            "max_ess_abs_error": max(item["ess_abs_error"] for item in observations),
            "max_ess_rel_error": max(item["ess_rel_error"] for item in observations),
            "max_loss_abs_error": max(item["loss_abs_error"] for item in observations),
            "max_loss_rel_error": max(item["loss_rel_error"] for item in observations),
            "max_gradient_relative_l2": max(item["gradient_relative_l2"] for item in observations),
            "min_gradient_cosine": min(item["gradient_cosine"] for item in observations),
            "max_parameter_relative_l2": max(item["parameter_relative_l2"] for item in observations),
            "frozen_tolerances": {
                "ess_rtol": ESS_RTOL,
                "ess_atol": ESS_ATOL,
                "loss_rtol": LOSS_RTOL,
                "loss_atol": LOSS_ATOL,
                "gradient_relative_l2": GRAD_RELATIVE_L2_TOL,
            },
            "observations": observations,
        }
        numeric_values = [
            value for observation in observations for value in observation.values() if isinstance(value, (float, int))
        ]
        if not all(math.isfinite(float(value)) for value in numeric_values):
            raise RuntimeError("real-Qwen replay produced a non-finite metric")
        summary["passed_frozen_synthetic_tolerances"] = (
            summary["max_ess_rel_error"] <= ESS_RTOL
            and summary["max_ess_abs_error"] <= ESS_ATOL
            and summary["max_loss_rel_error"] <= LOSS_RTOL
            and summary["max_loss_abs_error"] <= LOSS_ATOL
            and summary["max_gradient_relative_l2"] <= GRAD_RELATIVE_L2_TOL
        )
        passed.fill_(int(summary["passed_frozen_synthetic_tolerances"]))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite comparison output: {args.output}")
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))

    dist.broadcast(passed, src=0)
    dist.destroy_process_group()
    if not bool(passed):
        raise RuntimeError("real-Qwen replay exceeded the pre-frozen synthetic tolerances")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "compare"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--rollout-path", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--max-candidate-prompts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--steps", type=int, default=OPTIMIZER_STEPS)
    args = parser.parse_args()

    if args.mode == "prepare":
        if args.dataset_path is None:
            parser.error("--dataset-path is required in prepare mode")
        prepare_rollout(args)
    else:
        if args.output is None:
            parser.error("--output is required in compare mode")
        compare_rollout(args)


if __name__ == "__main__":
    main()
