# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch

from relax.backends.sglang import deterministic_sampler_patch as patch


def _identity_compile(*, dynamic):
    assert dynamic is True
    return lambda function: function


def _install_fake_sglang_modules(
    monkeypatch: pytest.MonkeyPatch,
    sampler: ModuleType,
    hash_module: ModuleType,
) -> None:
    sglang = ModuleType("sglang")
    srt = ModuleType("sglang.srt")
    layers = ModuleType("sglang.srt.layers")
    utils = ModuleType("sglang.srt.layers.utils")
    sglang.srt = srt
    srt.layers = layers
    layers.sampler = sampler
    layers.utils = utils
    utils.hash = hash_module
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers", layers)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.sampler", sampler)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.utils", utils)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.utils.hash", hash_module)


def test_uniform_hash_endpoint_has_finite_upstream_gumbel_cap():
    values = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)

    result = patch._uniform_hash_to_gumbel_(values)

    assert torch.isfinite(result).all()
    assert result[1].item() == pytest.approx(-torch.log(-torch.log(torch.tensor(0.5))).item())
    assert result[-1].item() == pytest.approx(-torch.log(torch.tensor(2.0**-32)).item())


def test_safe_multinomial_does_not_let_uint32_endpoint_override_logprob():
    uint32_max = torch.iinfo(torch.uint32).max

    def fake_hash(seed, positions, column_indices):
        assert seed.shape == positions.shape == (1,)
        assert column_indices.shape == (2,)
        return torch.tensor([[uint32_max, uint32_max // 2]], dtype=torch.uint32)

    sample = patch._build_safe_multinomial_with_seed(fake_hash, compile_function=_identity_compile)
    selected = sample(
        torch.tensor([[float("-inf"), 0.0]], dtype=torch.float64),
        torch.tensor([44], dtype=torch.int64),
        torch.tensor([179], dtype=torch.int64),
    )

    assert selected.tolist() == [[1]]


def test_apply_patch_is_version_gated_and_idempotent(monkeypatch):
    sampler = ModuleType("sglang.srt.layers.sampler")

    def original(logprobs, seed, positions):
        return logprobs, seed, positions

    sampler.multinomial_with_seed = original
    hash_module = ModuleType("sglang.srt.layers.utils.hash")
    hash_module.murmur_hash32 = object()
    _install_fake_sglang_modules(monkeypatch, sampler, hash_module)
    monkeypatch.setattr(patch, "_installed_sglang_version", lambda: "0.5.12.post1")
    replacement = lambda *_args: None
    monkeypatch.setattr(patch, "_build_safe_multinomial_with_seed", lambda _hash: replacement)

    assert patch.apply_deterministic_sampler_endpoint_patch() is True
    assert sampler.multinomial_with_seed is replacement
    assert getattr(replacement, patch._PATCH_MARKER) is True
    assert patch.apply_deterministic_sampler_endpoint_patch() is False


def test_apply_patch_leaves_unaffected_sglang_unchanged(monkeypatch):
    monkeypatch.setattr(patch, "_installed_sglang_version", lambda: "0.5.13")

    assert patch.apply_deterministic_sampler_endpoint_patch() is False


def test_local_version_suffix_resolves_to_affected_public_version(monkeypatch):
    monkeypatch.setattr(patch, "version", lambda _package: "0.5.12.post1+cu129")

    assert patch._installed_sglang_version() == "0.5.12.post1"


def test_affected_signature_drift_fails_closed(monkeypatch):
    sampler = ModuleType("sglang.srt.layers.sampler")
    sampler.multinomial_with_seed = lambda inputs, seed: (inputs, seed)
    hash_module = ModuleType("sglang.srt.layers.utils.hash")
    hash_module.murmur_hash32 = object()
    _install_fake_sglang_modules(monkeypatch, sampler, hash_module)
    monkeypatch.setattr(patch, "_installed_sglang_version", lambda: "0.5.12.post1")

    with pytest.raises(RuntimeError, match="signature changed"):
        patch.apply_deterministic_sampler_endpoint_patch()
