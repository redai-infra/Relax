# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Backport SGLang's deterministic-sampler uint32 endpoint fix.

SGLang 0.5.12.post1 maps a 32-bit hash to ``[0, 1]`` by dividing by
``uint32.max``.  A hash equal to ``0xffffffff`` therefore produces exactly
``x == 1`` and Gumbel noise ``-log(-log(x)) == +inf``.  That token then wins
the argmax regardless of its model probability.  Upstream clamps ``log(x)``
away from zero by one hash bucket; this module applies the same correction in
the scheduler subprocess for the affected local runtime.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import version
from inspect import signature
from typing import Any

import torch
from packaging.version import Version

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_AFFECTED_SGLANG_VERSIONS = frozenset({"0.5.12.post1"})
_PATCH_MARKER = "_relax_uint32_endpoint_fix"


def _installed_sglang_version() -> str:
    return Version(version("sglang")).public


def _uniform_hash_to_gumbel_(values: torch.Tensor) -> torch.Tensor:
    """Transform uniform hash fractions in place without infinite endpoints."""
    values.log_().clamp_(min=torch.finfo(values.dtype).min, max=-(2.0**-32)).neg_()
    values.log_().neg_()
    return values


def _build_safe_multinomial_with_seed(
    murmur_hash32: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    compile_function: Callable[..., Any] = torch.compile,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build the upstream-equivalent deterministic multinomial function."""

    def _safe_multinomial_with_seed(
        logprobs: torch.Tensor, seed: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        _, vocabulary_size = logprobs.shape
        seed = seed.to(torch.uint64)
        column_indices = torch.arange(vocabulary_size, device=logprobs.device)
        hashed = murmur_hash32(seed, positions, column_indices)
        gumbel = hashed.to(torch.float64) / torch.iinfo(torch.uint32).max
        _uniform_hash_to_gumbel_(gumbel)
        gumbel.add_(logprobs.to(torch.float64))
        return torch.argmax(gumbel, dim=1, keepdim=True)

    return compile_function(dynamic=True)(_safe_multinomial_with_seed)


def apply_deterministic_sampler_endpoint_patch() -> bool:
    """Patch the affected SGLang sampler before scheduler model initialization.

    Returns ``True`` only when this call installs the backport.  Unaffected
    versions and already-patched scheduler processes are left unchanged.
    """
    installed_version = _installed_sglang_version()
    if installed_version not in _AFFECTED_SGLANG_VERSIONS:
        logger.info(
            "SGLang deterministic sampler endpoint backport not required for version %s",
            installed_version,
        )
        return False

    from sglang.srt.layers import sampler
    from sglang.srt.layers.utils.hash import murmur_hash32

    current = sampler.multinomial_with_seed
    if getattr(current, _PATCH_MARKER, False):
        return False
    parameter_names = tuple(signature(current).parameters)
    if parameter_names != ("logprobs", "seed", "positions"):
        raise RuntimeError(
            "Affected SGLang multinomial_with_seed signature changed: "
            f"expected=('logprobs', 'seed', 'positions'): actual={parameter_names}"
        )

    replacement = _build_safe_multinomial_with_seed(murmur_hash32)
    setattr(replacement, _PATCH_MARKER, True)
    sampler.multinomial_with_seed = replacement
    logger.warning(
        "Applied SGLang %s deterministic sampler uint32 endpoint backport",
        installed_version,
    )
    return True
