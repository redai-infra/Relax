import logging
import math
from typing import Any, Literal

import numpy as np
import torch

from relax.utils.types import Sample


logger = logging.getLogger(__name__)


def dict_add_prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def compute_pass_rate(
    flat_rewards: list[float],
    group_size: int,
    num_groups: int | None = None,
):
    if group_size == 1:
        return {}

    if num_groups is None:
        # Caller doesn't know the group structure (e.g. streaming consumer whose
        # per-DP slice may not be group-aligned).  Derive whole groups and drop
        # any ragged remainder instead of asserting, so a non-multiple count
        # never crashes the caller.  An empty result means nothing to report.
        num_groups = len(flat_rewards) // group_size
        if num_groups == 0:
            logger.warning(
                "compute_pass_rate: %d rewards < group_size %d; skipping pass@k",
                len(flat_rewards),
                group_size,
            )
            return {}
        usable = num_groups * group_size
        if usable != len(flat_rewards):
            logger.warning(
                "compute_pass_rate: %d rewards not a multiple of group_size %d; truncating to %d for pass@k",
                len(flat_rewards),
                group_size,
                usable,
            )
            flat_rewards = flat_rewards[:usable]

    pass_rate_name_list = [2**i for i in range(int(math.log2(group_size)) + 1)]

    assert len(flat_rewards) == num_groups * group_size, f"{len(flat_rewards)=} {num_groups=} {group_size=}"
    rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)

    log_dict = {}
    for k in pass_rate_name_list:
        num_correct = np.sum(rewards_of_group == 1, axis=1)
        num_samples = np.full(num_groups, group_size)

        pass_k_estimates = _estimate_pass_at_k(num_samples, num_correct, k)

        pass_k = np.mean(pass_k_estimates)
        log_dict[f"pass@{k}"] = pass_k

    return log_dict


def _estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n, c, k):
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct, strict=False)])


def compute_statistics(values: list[float]) -> dict[str, float]:
    values = np.array(values)
    return {
        "mean": np.mean(values).item(),
        "median": np.median(values).item(),
        "max": np.max(values).item(),
        "min": np.min(values).item(),
    }


def is_rollout_numeric_metric_value(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def append_rollout_numeric_metric_values(metric_values: dict[str, list[float]], *, key: str, value) -> None:
    if isinstance(value, (list, tuple)):
        flattened = [float(item) for item in value if is_rollout_numeric_metric_value(item)]
        if flattened:
            metric_values.setdefault(key, []).extend(flattened)
        return
    if is_rollout_numeric_metric_value(value):
        metric_values.setdefault(key, []).append(float(value))


def finalize_rollout_explicit_metric_values(metric_values: dict[str, list[float]]) -> dict[str, float]:
    log_dict: dict[str, float] = {}
    for metric_name, values in metric_values.items():
        if values:
            log_dict |= dict_add_prefix(compute_statistics(values), f"{metric_name}/")
    return log_dict


def _compute_rloo_group_diagnostics(args, samples: list[Sample]) -> dict[str, float]:
    """Compute RLOO-specific diagnostics from raw rollout samples.

    These metrics are computed on the rollout side where both raw rewards and
    group structure are available. The leave-one-out baseline is recomputed
    here from raw rewards (mirroring ``compute_rloo_leave_one_out_rewards``)
    so the diagnostics are self-contained.

    Metrics:
        rloo/baseline_mean: mean of per-group LOO baseline means.
        rloo/adv_abs_mean: mean of per-group |advantage| means (signal strength;
            the plain mean of RLOO advantages is always 0 and carries no info).
        rloo/no_signal_frac: token-level fraction of tokens whose sample has
            zero advantage (group rewards all equal, or the sample lands on the
            group mean). Zero-advantage tokens contribute no gradient.
        rloo/empty_response_frac: fraction of samples with empty responses.
        rloo/zero_adv_group_frac: fraction of groups where all advantages are 0.
    """
    if getattr(args, "advantage_estimator", None) != "rloo":
        return {}

    from relax.utils.training.ppo_utils import compute_rloo_leave_one_out_rewards

    group_size = args.n_samples_per_prompt
    groups: dict[int, list[tuple[Sample, float]]] = {}
    for sample in samples:
        gi = sample.group_index
        if gi is None:
            continue
        groups.setdefault(gi, []).append((sample, sample.get_reward_value(args)))

    all_baseline_means: list[float] = []
    all_adv_abs_means: list[float] = []
    zero_adv_groups = 0
    total_groups = 0
    zero_adv_tokens = 0
    total_response_tokens = 0
    empty_count = 0

    for items in groups.values():
        if len(items) != group_size:
            continue
        total_groups += 1
        rewards = torch.tensor([r for _, r in items], dtype=torch.float32)
        adv = compute_rloo_leave_one_out_rewards(rewards)
        baselines = rewards - adv
        all_baseline_means.append(baselines.mean().item())
        all_adv_abs_means.append(adv.abs().mean().item())
        if bool(adv.abs().max().item() < 1e-12):
            zero_adv_groups += 1
        for (sample, _), a in zip(items, adv.tolist(), strict=False):
            resp_len = getattr(sample, "response_length", 0) or 0
            total_response_tokens += resp_len
            if resp_len == 0:
                empty_count += 1
            if abs(a) < 1e-12:
                zero_adv_tokens += resp_len

    total_samples = len(samples)
    return {
        "rloo/baseline_mean": sum(all_baseline_means) / len(all_baseline_means) if all_baseline_means else 0.0,
        "rloo/adv_abs_mean": sum(all_adv_abs_means) / len(all_adv_abs_means) if all_adv_abs_means else 0.0,
        "rloo/no_signal_frac": zero_adv_tokens / total_response_tokens if total_response_tokens > 0 else 0.0,
        "rloo/empty_response_frac": empty_count / total_samples if total_samples > 0 else 0.0,
        "rloo/zero_adv_group_frac": zero_adv_groups / total_groups if total_groups > 0 else 0.0,
    }


def compute_rollout_explicit_reward_metrics(args, samples: list[Sample]) -> dict[str, float]:
    reward_metric_values: dict[str, list[float]] = {}
    primary_reward_key = getattr(args, "reward_key", None)
    for sample in samples:
        reward = sample.reward
        if not isinstance(reward, dict):
            continue
        for key, value in reward.items():
            if (
                not isinstance(key, str)
                or not key
                or key == primary_reward_key
                or key == "raw_reward"
                or key.startswith("_")
            ):
                continue
            append_rollout_numeric_metric_values(reward_metric_values, key=key, value=value)
    log_dict = finalize_rollout_explicit_metric_values(reward_metric_values)
    if args.log_passrate:
        rewards = [sample.get_reward_value(args) for sample in samples if sample.reward is not None]
        if rewards:
            log_dict |= dict_add_prefix(
                compute_pass_rate(flat_rewards=rewards, group_size=args.n_samples_per_prompt),
                "passrate/",
            )

    log_dict |= _compute_rloo_group_diagnostics(args, samples)

    return log_dict


def compression_ratio(
    data: str | bytes,
    *,
    encoding: str = "utf-8",
    algorithm: Literal["zlib", "gzip", "bz2", "lzma"] = "zlib",
    level: int = 9,
) -> tuple[float, float]:
    if isinstance(data, str):
        raw = data.encode(encoding)
    else:
        raw = data

    original = len(raw)
    if original == 0:
        return float("inf"), 0.0

    if algorithm == "zlib":
        import zlib

        compressed = zlib.compress(raw, level)
    elif algorithm == "gzip":
        import gzip

        compressed = gzip.compress(raw, compresslevel=level)
    elif algorithm == "bz2":
        import bz2

        compressed = bz2.compress(raw, compresslevel=level)
    elif algorithm == "lzma":
        import lzma

        compressed = lzma.compress(raw, preset=level)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    comp_len = len(compressed)
    if comp_len == 0:
        return float("inf"), 100.0

    ratio = original / comp_len
    savings_pct = 100.0 * (1.0 - comp_len / original)
    return ratio, savings_pct


def has_repetition(text: str):
    if len(text) > 10000 and compression_ratio(text[-10000:])[0] > 10:
        return True
    else:
        return False


def compute_rollout_step(args, rollout_id):
    if args.wandb_always_use_train_step:
        return rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    return rollout_id
