# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the SGLang EOPD entropy patch utilities."""

import numpy as np
import pybase64
import torch


def test_entropy_computation_correctness():
    """Verify H = -sum(p * log_p) on synthetic logits."""
    from relax.utils.opd.opd_sglang_entropy_patch import _compute_entropy

    logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    entropy = _compute_entropy(logits)
    assert entropy.shape == (2,)

    probs_0 = torch.softmax(logits[0], dim=-1)
    expected_0 = -(probs_0 * probs_0.log()).sum()
    assert torch.allclose(entropy[0], expected_0, atol=1e-5)

    expected_uniform = np.log(3.0)
    assert abs(entropy[1].item() - expected_uniform) < 1e-5


def test_entropy_b64_roundtrip():
    """Verify encode → decode preserves float32 entropy values."""
    from relax.utils.opd.opd_sglang_entropy_patch import _entropy_to_b64

    original = torch.tensor([0.5, 1.2, 0.0, 3.14], dtype=torch.float32)
    b64_str = _entropy_to_b64(original)

    decoded = np.frombuffer(pybase64.b64decode(b64_str), dtype=np.float32)
    np.testing.assert_allclose(decoded, original.numpy(), atol=1e-7)


def test_logprob_response_entropy_1d_from_b64():
    """LogprobResponse.entropy_1d() correctly parses meta_info with b64 entropy."""
    from relax.utils.opd.opd_main_worker import LogprobResponse

    values = np.array([0.1, 0.5, 1.0, 2.0], dtype=np.float32)
    b64_str = pybase64.b64encode(values.tobytes()).decode()

    resp = LogprobResponse({"meta_info": {"relax_input_entropy_b64": [b64_str]}})
    result = resp.entropy_1d()

    assert result is not None
    np.testing.assert_allclose(result, values, atol=1e-7)


def test_logprob_response_entropy_1d_bare_string():
    """LogprobResponse.entropy_1d() handles bare b64 string (not wrapped in list)."""
    from relax.utils.opd.opd_main_worker import LogprobResponse

    values = np.array([0.3, 0.7], dtype=np.float32)
    b64_str = pybase64.b64encode(values.tobytes()).decode()

    resp = LogprobResponse({"meta_info": {"relax_input_entropy_b64": b64_str}})
    result = resp.entropy_1d()

    assert result is not None
    np.testing.assert_allclose(result, values, atol=1e-7)


def test_logprob_response_entropy_1d_missing():
    """LogprobResponse.entropy_1d() returns None when field is absent."""
    from relax.utils.opd.opd_main_worker import LogprobResponse

    resp = LogprobResponse({"meta_info": {}})
    assert resp.entropy_1d() is None


def test_logprob_response_entropy_1d_empty():
    """LogprobResponse.entropy_1d() returns None for empty response."""
    from relax.utils.opd.opd_main_worker import LogprobResponse

    resp = LogprobResponse(None)
    assert resp.entropy_1d() is None
