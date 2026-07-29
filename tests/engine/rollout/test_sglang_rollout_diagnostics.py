# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""LogprobResponse rollout-side self-topk decoding (base64 path).

Refactored: the old module-level ``extract_sglang_topk_logprobs`` was replaced
by ``LogprobResponse.self_topk("rollout", ...)`` which decodes the sglang
base64 ``output_top_logprobs_*_b64`` fields into numpy ``(ids, logps)``.
"""

import numpy as np
import pybase64
import pytest

from relax.engine.rollout.sglang_rollout import _aggregate_rollout_timing
from relax.utils.opd.opd_main_worker import LogprobResponse
from relax.utils.types import Sample


def _b64(arr: np.ndarray) -> str:
    return pybase64.b64encode(arr.tobytes()).decode("utf-8")


def test_rollout_self_topk_keeps_token_id_zero_from_b64() -> None:
    # R=2, K=2; token_id 0 must be preserved (not treated as missing).
    vals = np.array([-0.1, -0.2, -0.3, -0.4], dtype=np.float32)
    ids = np.array([0, 9, 4, 7], dtype=np.int32)
    resp = {
        "meta_info": {
            "output_top_logprobs_val_b64": _b64(vals),
            "output_top_logprobs_idx_b64": _b64(ids),
        }
    }

    pair = LogprobResponse(resp).self_topk("rollout", top_k=2)

    assert pair is not None
    out_ids, out_lps = pair
    np.testing.assert_array_equal(out_ids, np.array([[0, 9], [4, 7]], dtype=np.int32))
    np.testing.assert_allclose(out_lps, np.array([[-0.1, -0.2], [-0.3, -0.4]], dtype=np.float32))


def test_rollout_self_topk_returns_none_when_absent() -> None:
    assert LogprobResponse({"meta_info": {}}).self_topk("rollout", top_k=2) is None


def test_rollout_timing_reports_request_permit_wait() -> None:
    first = Sample(prompt="first")
    first.metadata["_timing"] = {"request_permit_wait": 0.2, "generate": 1.0}
    second = Sample(prompt="second")
    second.metadata["_timing"] = {"request_permit_wait": 0.4, "generate": 2.0}

    metrics = _aggregate_rollout_timing([first, second], [])

    assert metrics["perf_detail/rollout/request_permit_wait_time/mean"] == pytest.approx(0.3)
    assert metrics["perf_detail/rollout/request_permit_wait_time/max"] == 0.4
