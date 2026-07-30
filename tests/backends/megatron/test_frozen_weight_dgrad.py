# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Alarm for the frozen-weight DGRAD fold in the Megatron image patch.

``docker/patch/megatron/20260506-85bced0ae.patch`` backports the
``LinearWithFrozenWeight.backward`` hunk from NVIDIA/Megatron-LM#5092; a
Megatron bump could drop it silently, so the first test guards the patch text
and runs in CI while the rest check the behaviour and need Megatron.
"""

import inspect
from pathlib import Path

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode


try:
    from megatron.core.tensor_parallel.layers import LinearWithFrozenWeight
except ImportError:  # CI installs no training dependencies
    LinearWithFrozenWeight = None

needs_megatron = pytest.mark.skipif(LinearWithFrozenWeight is None, reason="requires Megatron")

PATCH = Path(__file__).resolve().parents[3] / "docker" / "patch" / "latest" / "megatron.patch"
HINT = f"the LinearWithFrozenWeight.backward hunk from NVIDIA/Megatron-LM#5092 is missing from {PATCH}"
FOLD = "grad_output.reshape(-1, grad_output.size(-1))"


def _megatron_lacks_dgrad_fold() -> bool:
    """Whether the Megatron on the path predates the hunk.

    False when Megatron is absent; ``needs_megatron`` gives the better reason.
    """
    if LinearWithFrozenWeight is None:
        return False
    try:
        # Bytes: a decode error here would take the whole module down at import.
        return FOLD.encode() not in Path(inspect.getfile(LinearWithFrozenWeight)).read_bytes()
    except (OSError, TypeError):
        return True  # cannot tell -- skip rather than report a regression


needs_fold = pytest.mark.skipif(
    _megatron_lacks_dgrad_fold(),
    reason=f"the Megatron on the path predates the DGRAD fold; rebuild the image with {PATCH}",
)


def test_megatron_patch_carries_dgrad_fold():
    """Runs without Megatron installed, so CI covers it."""
    assert PATCH.is_file(), HINT
    patch = PATCH.read_text()
    assert "megatron/core/tensor_parallel/layers.py" in patch, HINT
    assert FOLD in patch, HINT
    assert "grad_input.reshape(*grad_output.shape[:-1], weight.size(1))" in patch, HINT


class _MatmulSpy(TorchDispatchMode):
    """Records which matmul-family aten op a block dispatches to."""

    def __init__(self):
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if "bmm" in name:
            self.ops.append("bmm")
        elif name.endswith("mm.default"):
            self.ops.append("mm")
        return func(*args, **(kwargs or {}))


def _run(batch, tokens, vocab, hidden, dtype=torch.float32, device="cpu"):
    """Drive the real autograd Function with the strides production produces.

    Returns the matmul ops dispatched, the DGRAD produced, and the DGRAD from
    ``grad_output.matmul(weight)`` -- an oracle independent of the patch, being verbatim
    the line the hunk replaced.
    """
    grad_output = torch.randn(batch, tokens, vocab, dtype=dtype, device=device).transpose(0, 1)
    weight = torch.randn(vocab, hidden, dtype=dtype, device=device)
    # Load-bearing: ATen folds unconditionally when the operand requires grad, so a
    # non-detached weight would pass the assertions below on an unpatched Megatron.
    assert not weight.requires_grad, "fixture must stay detached, like the output weight after .detach()"
    inp = torch.randn(tokens, batch, hidden, dtype=dtype, device=device, requires_grad=True)
    out = LinearWithFrozenWeight.apply(inp, weight, None, False, None)
    assert out.shape == grad_output.shape

    spy = _MatmulSpy()
    with spy:
        dgrad = torch.autograd.grad(out, inp, grad_outputs=grad_output)[0]
    return spy.ops, dgrad, grad_output.matmul(weight)


@needs_megatron
@needs_fold
@pytest.mark.parametrize("batch", [1, 3])
def test_frozen_weight_dgrad_folds_to_a_single_mm(batch):
    """batch 1 reshapes as a view, batch 3 has to copy."""
    torch.manual_seed(0)
    ops, dgrad, oracle = _run(batch, tokens=32, vocab=64, hidden=16)

    assert ops == ["mm"], f"dispatched {ops}; {HINT}"
    assert dgrad.shape == (32, batch, 16)
    assert torch.allclose(dgrad, oracle, atol=1e-5, rtol=1e-5)


@needs_megatron
def test_frozen_weight_dgrad_passes_2d_grad_through():
    """A 2-D ``grad_output`` keeps the original single-matmul path."""
    torch.manual_seed(0)
    grad_output = torch.randn(32, 64)
    weight = torch.randn(64, 16)
    inp = torch.randn(32, 16, requires_grad=True)
    out = LinearWithFrozenWeight.apply(inp, weight, None, False, None)

    spy = _MatmulSpy()
    with spy:
        dgrad = torch.autograd.grad(out, inp, grad_outputs=grad_output)[0]

    assert spy.ops == ["mm"]
    assert dgrad.shape == (32, 16)
    assert torch.allclose(dgrad, grad_output.matmul(weight), atol=1e-6, rtol=1e-6)


@needs_megatron
@needs_fold
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_frozen_weight_dgrad_folds_to_a_single_mm_in_bf16_on_cuda():
    """bf16 on device is what training actually runs."""
    torch.manual_seed(0)
    ops, dgrad, oracle = _run(1, tokens=256, vocab=512, hidden=128, dtype=torch.bfloat16, device="cuda")

    assert ops == ["mm"], f"dispatched {ops}; {HINT}"
    # Bitwise-identical where measured, but mm and bmm may accumulate differently elsewhere.
    assert torch.allclose(dgrad.float(), oracle.float(), atol=1e-2, rtol=1e-2)
