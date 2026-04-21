# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import aiohttp
import torch

from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample


logger = get_logger(__name__)


def _get_opd_topk(args) -> int:
    """Resolve OPD top-k for overlap metrics.

    Priority:
    1) args.opd_log_prob_top_k (CLI/runtime)
    2) fallback 0 when missing (for backward compatibility)
    """
    top_k = getattr(args, "opd_log_prob_top_k", None)
    if top_k is None:
        return 0
    return max(int(top_k), 0)


def _extract_token_id(candidate) -> int | None:
    """Best-effort extraction of token id from a top-logprob candidate item."""
    if isinstance(candidate, dict):
        for key in ("token_id", "id"):
            if key in candidate and isinstance(candidate[key], int):
                return int(candidate[key])
        token_val = candidate.get("token")
        if isinstance(token_val, int):
            return int(token_val)
        return None

    if isinstance(candidate, (list, tuple)):
        for item in candidate:
            if isinstance(item, int):
                return int(item)
        return None

    return int(candidate) if isinstance(candidate, int) else None


def _extract_topk_ids_from_candidates(candidates, top_k: int) -> tuple[list[int], bool]:
    """Extract fixed-width token-id top-k list from a candidate container.

    Returns:
        (token_ids, has_valid_token_id)
    """
    token_ids: list[int] = []
    has_valid_token_id = False

    if isinstance(candidates, dict):
        candidates = candidates.get("top_logprobs") or candidates.get("candidates") or []

    if isinstance(candidates, (list, tuple)):
        for item in candidates:
            token_id = _extract_token_id(item)
            if token_id is not None:
                token_ids.append(token_id)
                has_valid_token_id = True
            if len(token_ids) >= top_k:
                break

    if len(token_ids) < top_k:
        token_ids.extend([-1] * (top_k - len(token_ids)))
    else:
        token_ids = token_ids[:top_k]

    return token_ids, has_valid_token_id


def _extract_teacher_topk_token_ids(teacher_resp: dict, response_length: int, top_k: int) -> list[list[int]] | None:
    """Extract response-aligned teacher top-k token ids from SGLang response.

    Returns None when top-k information is unavailable.
    """
    if top_k <= 0 or response_length <= 0:
        return None

    meta_info = teacher_resp.get("meta_info", {})

    top_logprobs = (
        meta_info.get("input_top_logprobs")
        or meta_info.get("input_token_top_logprobs")
        or teacher_resp.get("top_logprobs")
    )
    if isinstance(top_logprobs, list) and top_logprobs:
        per_token_candidates = top_logprobs[-response_length:]
        if len(per_token_candidates) >= response_length:
            teacher_topk_ids: list[list[int]] = []
            any_valid = False
            for candidates in per_token_candidates:
                token_ids, has_valid = _extract_topk_ids_from_candidates(candidates, top_k)
                any_valid = any_valid or has_valid
                teacher_topk_ids.append(token_ids)
            if any_valid:
                return teacher_topk_ids
    return None


def _fallback_teacher_topk_token_ids(response_length: int, top_k: int) -> list[list[int]]:
    """Build a fixed-shape fallback for teacher top-k ids.

    Uses -1 as sentinel token id so downstream flatten/reshape stays stable
    while clearly indicating unavailable teacher top-k content.
    """
    if response_length <= 0 or top_k <= 0:
        return []
    return [[-1] * top_k for _ in range(response_length)]


def _get_teacher_url(args) -> str:
    """Resolve OPD teacher URL from args with backward-compatible fallback.

    Priority:
    1) args.opd_teacher_url (explicit OPD teacher endpoint)
    2) args.rm_url (backward-compatible default)
    """
    return getattr(args, "opd_teacher_url", None) or args.rm_url


def create_teacher_client_session(args) -> aiohttp.ClientSession:
    """Create a reusable HTTP session for OPD teacher requests.

    Reusing one session per batch avoids per-sample TCP/TLS setup overhead.
    """
    timeout_s = float(getattr(args, "opd_teacher_timeout_s", 30.0))
    connector_limit = int(getattr(args, "opd_teacher_connector_limit", 256))
    connector = aiohttp.TCPConnector(limit=connector_limit)
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    return aiohttp.ClientSession(connector=connector, timeout=timeout)


def _fallback_teacher_log_probs(sample: Sample, response_length: int) -> list[float]:
    """Build length-aligned fallback teacher log-probs for degraded mode.

    Prefer rollout log-probs from the same sample to keep OPD signal scale
    close to training distribution. If unavailable, use a neutral constant
    typical for token log-prob magnitude to avoid exploding penalties.
    """
    rollout_log_probs = list(getattr(sample, "rollout_log_probs", []) or [])
    if len(rollout_log_probs) >= response_length:
        return [float(v) for v in rollout_log_probs[-response_length:]]

    if rollout_log_probs:
        pad_val = float(rollout_log_probs[0])
        pad_count = response_length - len(rollout_log_probs)
        return [pad_val] * pad_count + [float(v) for v in rollout_log_probs]

    # Rare fallback when rollout log-probs are unavailable.
    return [-0.3] * response_length


async def _post_teacher_request_with_diagnostics(
    args, payload: dict, session: aiohttp.ClientSession | None = None
) -> dict:
    """Send OPD teacher request and emit detailed diagnostics on HTTP errors.

    Logs status code, URL and a truncated response body to make 4xx/5xx
    failures actionable while keeping logs bounded.
    """

    async def _do_post(active_session: aiohttp.ClientSession) -> dict:
        teacher_url = _get_teacher_url(args)
        async with active_session.post(teacher_url, json=payload) as resp:
            if resp.status >= 400:
                response_text = await resp.text()
                logger.error(
                    "OPD teacher request failed: status=%s, url=%s, body=%s",
                    resp.status,
                    teacher_url,
                    response_text[:2048],
                )
                resp.raise_for_status()
            return await resp.json()

    if session is not None:
        return await _do_post(session)

    async with create_teacher_client_session(args) as own_session:
        return await _do_post(own_session)


async def fetch_teacher_log_probs(args, sample: Sample, session: aiohttp.ClientSession | None = None) -> None:
    """Fetch teacher log-probs from the external SGLang teacher server and
    store them in sample.teacher_log_probs.

    This function is called automatically by the framework when
    --use-opd --opd-type sglang is enabled. It does NOT occupy
    --custom-rm-path or --custom-reward-post-process-path, so users
    can freely use their own custom reward functions alongside OPD.

    Args:
        args: The global args namespace (must have args.rm_url set to the teacher server URL).
        sample: A Sample object that already has tokens and response_length populated.
    """
    response_length = int(sample.response_length or 0)
    if response_length <= 0:
        # Avoid Python slicing pitfall: x[-0:] == x[:], not empty.
        sample.teacher_log_probs = []
        sample.teacher_topk_token_ids = []
        return

    opd_top_k = _get_opd_topk(args)

    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    if opd_top_k > 0:
        payload["top_logprobs_num"] = opd_top_k

    try:
        teacher_resp = await _post_teacher_request_with_diagnostics(args, payload, session=session)
    except Exception as exc:
        # Do not crash the whole rollout on teacher API failure.
        # Avoid ABORTED status here because aborted samples enter a different
        # buffering path that may impact rollout progress under sustained errors.
        sample.teacher_log_probs = _fallback_teacher_log_probs(sample, response_length)
        sample.teacher_topk_token_ids = _fallback_teacher_topk_token_ids(response_length, opd_top_k)
        if sample.metadata is None:
            sample.metadata = {}
        sample.metadata["opd_teacher_error"] = f"{type(exc).__name__}: {str(exc)[:512]}"
        sample.metadata["opd_teacher_fallback"] = "rollout_log_probs"
        logger.error(
            "OPD teacher fetch failed for sample_index=%s, response_length=%s, url=%s, error=%s. "
            "Falling back to rollout log-probs.",
            getattr(sample, "index", None),
            response_length,
            _get_teacher_url(args),
            f"{type(exc).__name__}: {str(exc)[:256]}",
        )
        return

    # Extract teacher log-probs from the sglang response and trim to response length
    token_logprobs = teacher_resp.get("meta_info", {}).get("input_token_logprobs", None)
    if not token_logprobs:
        sample.teacher_log_probs = _fallback_teacher_log_probs(sample, response_length)
        sample.teacher_topk_token_ids = _fallback_teacher_topk_token_ids(response_length, opd_top_k)
        if sample.metadata is None:
            sample.metadata = {}
        sample.metadata["opd_teacher_error"] = "Missing meta_info.input_token_logprobs in teacher response"
        sample.metadata["opd_teacher_fallback"] = "rollout_log_probs"
        logger.error(
            "Invalid OPD teacher response for sample_index=%s: missing input_token_logprobs. Falling back to rollout log-probs.",
            getattr(sample, "index", None),
        )
        return

    all_log_probs = torch.tensor([item[0] for item in token_logprobs[1:]], dtype=torch.float32)
    if all_log_probs.numel() < response_length:
        sample.teacher_log_probs = _fallback_teacher_log_probs(sample, response_length)
        sample.teacher_topk_token_ids = _fallback_teacher_topk_token_ids(response_length, opd_top_k)
        if sample.metadata is None:
            sample.metadata = {}
        sample.metadata["opd_teacher_error"] = (
            f"Teacher log-prob length mismatch: got {all_log_probs.numel()}, expected >= {response_length}"
        )
        sample.metadata["opd_teacher_fallback"] = "rollout_log_probs"
        logger.error(
            "Teacher log-prob length mismatch for sample_index=%s: got=%s, expected>=%s. Falling back to rollout log-probs.",
            getattr(sample, "index", None),
            all_log_probs.numel(),
            response_length,
        )
        return

    teacher_log_probs = all_log_probs[-response_length:]
    sample.teacher_log_probs = teacher_log_probs.tolist()

    teacher_topk_token_ids = _extract_teacher_topk_token_ids(teacher_resp, response_length, opd_top_k)
    if teacher_topk_token_ids is None:
        if opd_top_k > 0:
            logger.warning(
                "OPD teacher response missing top-logprobs for sample_index=%s (opd_log_prob_top_k=%s). "
                "Using sentinel top-k ids.",
                getattr(sample, "index", None),
                opd_top_k,
            )
        teacher_topk_token_ids = _fallback_teacher_topk_token_ids(response_length, opd_top_k)
    sample.teacher_topk_token_ids = teacher_topk_token_ids
