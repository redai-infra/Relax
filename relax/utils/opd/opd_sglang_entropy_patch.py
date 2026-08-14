# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Monkey-patch for SGLang LogitsProcessor: compute per-token entropy during
teacher prefill and return it via ``customized_info`` → ``meta_info``.

Strategy (v7): Patch ``_get_logits`` to compute entropy from the *actual* model
logits immediately after the TP all-gather.  Only inject entropy into
``customized_info`` during EXTEND mode with logprob return (i.e. when
``extend_logprob_pruned_lens_cpu`` is a list).  Skip decode-mode and non-
logprob extend calls entirely to avoid broken ``customized_info`` routing.
"""

from __future__ import annotations

import logging
import os

import pybase64
import torch
import torch.nn.functional as F


logger = logging.getLogger("opd_sglang_entropy_patch")

_PATCH_FLAG = "_relax_entropy_patched_v7"


def _compute_entropy(logits: torch.Tensor, chunk_size: int = 256) -> torch.Tensor:
    with torch.no_grad():
        parts = []
        for i in range(0, logits.shape[0], chunk_size):
            chunk = logits[i : i + chunk_size].float()
            log_p = F.log_softmax(chunk, dim=-1)
            parts.append(-(log_p.exp() * log_p).sum(dim=-1))
        return torch.cat(parts) if len(parts) > 1 else parts[0]


def _entropy_to_b64(entropy: torch.Tensor) -> str:
    return pybase64.b64encode(entropy.cpu().to(torch.float32).numpy().tobytes()).decode("ascii")


def apply_patch() -> bool:
    from sglang.srt.layers.logits_processor import LogitsProcessor

    if getattr(LogitsProcessor, _PATCH_FLAG, False):
        return False

    _orig_get_logits = LogitsProcessor._get_logits
    _orig_forward = LogitsProcessor.forward

    def _patched_get_logits(self, hidden_states, lm_head, logits_metadata, *args, **kwargs):
        logits = _orig_get_logits(self, hidden_states, lm_head, logits_metadata, *args, **kwargs)

        if getattr(self, "_relax_collecting_entropy", False):
            if torch.cuda.is_current_stream_capturing():
                return logits
            try:
                ent = _compute_entropy(logits)
                if not hasattr(self, "_relax_entropy_parts"):
                    self._relax_entropy_parts = []
                self._relax_entropy_parts.append(ent.cpu())
            except Exception:
                logger.warning("entropy computation in _get_logits failed", exc_info=True)

        return logits

    def _patched_forward(self, input_ids, hidden_states, lm_head, logits_metadata, *args, **kwargs):
        self._relax_collecting_entropy = True
        self._relax_entropy_parts = []

        output = _orig_forward(self, input_ids, hidden_states, lm_head, logits_metadata, *args, **kwargs)

        self._relax_collecting_entropy = False

        try:
            if not self._relax_entropy_parts:
                return output

            all_entropy = (
                torch.cat(self._relax_entropy_parts)
                if len(self._relax_entropy_parts) > 1
                else self._relax_entropy_parts[0]
            )

            from sglang.srt.layers.logits_processor import LogitsMetadata
            from sglang.srt.model_executor.forward_batch_info import ForwardBatch

            if isinstance(logits_metadata, ForwardBatch):
                lm = LogitsMetadata.from_forward_batch(logits_metadata)
            else:
                lm = logits_metadata

            pruned_lens = getattr(lm, "extend_logprob_pruned_lens_cpu", None)

            if not isinstance(pruned_lens, list) or not pruned_lens:
                self._relax_entropy_parts = []
                return output

            total_input = sum(pruned_lens)
            if total_input > all_entropy.shape[0]:
                self._relax_entropy_parts = []
                return output

            input_entropy = all_entropy[-total_input:]
            per_request = torch.split(input_entropy, list(pruned_lens))
            b64_list = [_entropy_to_b64(e) for e in per_request]

            if output.customized_info is None:
                output.customized_info = {}
            output.customized_info["relax_input_entropy_b64"] = b64_list
        except Exception:
            logger.warning("entropy injection in forward failed", exc_info=True)

        self._relax_entropy_parts = []
        return output

    LogitsProcessor._get_logits = _patched_get_logits
    LogitsProcessor.forward = _patched_forward
    setattr(LogitsProcessor, _PATCH_FLAG, True)
    logger.info("EOPD entropy patch applied to LogitsProcessor._get_logits + .forward")
    return True


def apply_opd_entropy_patch() -> None:
    try:
        apply_patch()
    except Exception as e:
        logger.warning("Failed to apply OPD entropy patch: %r", e)


if os.environ.get("RELAX_OPD_ENTROPY_PATCH", "0") == "1":
    apply_opd_entropy_patch()
