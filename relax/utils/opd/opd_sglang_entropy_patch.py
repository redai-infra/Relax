# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Monkey-patch for SGLang LogitsProcessor: compute per-token entropy during
teacher prefill and return it via ``customized_info`` → ``meta_info``.

Why
---
EOPD (Entropy-aware On-Policy Distillation) gates its forward-KL loss on
per-token teacher entropy.  The SGLang HTTP prefill endpoint returns top-K
log-probs but **not** entropy.  This patch adds entropy computation inside
the SGLang server process, piggybacking on the full-vocab log-softmax that
is already computed for ``input_token_logprobs``.

How
---
Two methods on ``LogitsProcessor`` are wrapped:

1. ``process_input_logprobs`` — after the original runs, computes
   ``H = -sum(p * log_p, dim=-1)`` from ``input_logits`` and caches the
   per-request b64-encoded float32 arrays on ``self``.

2. ``forward`` — injects the cached arrays into
   ``LogitsProcessorOutput.customized_info["relax_input_entropy_b64"]``,
   which SGLang's scheduler propagates to ``meta_info`` in the HTTP JSON
   response.

Usage
-----
Applied by ``_launch_server_with_patches()`` in ``sglang_engine.py`` when
``RELAX_OPD_ENTROPY_PATCH=1``.  Idempotent: applying twice is a no-op.
"""

from __future__ import annotations

import logging
import os

import pybase64
import torch
import torch.nn.functional as F


logger = logging.getLogger("opd_sglang_entropy_patch")

_PATCH_FLAG = "_relax_entropy_patched"


def _compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-token entropy from raw logits.

    Args:
        logits: ``[N, vocab_size]`` raw (pre-softmax) logits.

    Returns:
        ``[N]`` entropy in nats.
    """
    with torch.no_grad():
        log_p = F.log_softmax(logits.float(), dim=-1)
        return -(log_p.exp() * log_p).sum(dim=-1)


def _entropy_to_b64(entropy: torch.Tensor) -> str:
    return pybase64.b64encode(entropy.cpu().to(torch.float32).numpy().tobytes()).decode("ascii")


def apply_patch() -> bool:
    """Patch ``LogitsProcessor`` to compute and propagate input-token entropy.

    Returns True if applied, False if already patched.
    """
    from sglang.srt.layers.logits_processor import LogitsProcessor

    if getattr(LogitsProcessor, _PATCH_FLAG, False):
        return False

    _orig_process = LogitsProcessor.process_input_logprobs
    _orig_forward = LogitsProcessor.forward

    def _patched_process(self, input_logits, logits_metadata):
        result = _orig_process(self, input_logits, logits_metadata)

        try:
            entropy = _compute_entropy(input_logits)

            pruned_lens = logits_metadata.extend_logprob_pruned_lens_cpu
            if pruned_lens:
                per_request = torch.split(entropy, list(pruned_lens))
            else:
                per_request = [entropy]

            self._relax_entropy_b64 = [_entropy_to_b64(e) for e in per_request]
        except Exception:
            logger.debug("entropy computation failed", exc_info=True)
            self._relax_entropy_b64 = None

        return result

    def _patched_forward(self, *args, **kwargs):
        self._relax_entropy_b64 = None
        output = _orig_forward(self, *args, **kwargs)

        if self._relax_entropy_b64 is not None:
            if output.customized_info is None:
                output.customized_info = {}
            output.customized_info["relax_input_entropy_b64"] = self._relax_entropy_b64
            self._relax_entropy_b64 = None

        return output

    LogitsProcessor.process_input_logprobs = _patched_process
    LogitsProcessor.forward = _patched_forward
    setattr(LogitsProcessor, _PATCH_FLAG, True)
    logger.info("[entropy-patch] applied to LogitsProcessor")
    return True


def apply_opd_entropy_patch() -> None:
    """Apply the entropy patch in the current (server) process."""
    try:
        apply_patch()
    except Exception as e:
        logger.warning("Failed to apply OPD entropy patch: %r", e)


if os.environ.get("RELAX_OPD_ENTROPY_PATCH", "0") == "1":
    apply_opd_entropy_patch()
