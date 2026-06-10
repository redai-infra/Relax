# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SFT end-to-end smoke test.

Skipped unless RELAX_RUN_SMOKE=1 and GPU available.
"""

import os
import subprocess
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "training" / "sft" / "run-sft-smoke.sh"


@pytest.mark.skipif(
    os.environ.get("RELAX_RUN_SMOKE", "0") != "1",
    reason="set RELAX_RUN_SMOKE=1 to enable",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_sft_smoke_messages_demo():
    """Run the SFT smoke script and assert it exits 0."""
    assert SCRIPT.exists(), f"smoke script missing: {SCRIPT}"
    assert os.access(SCRIPT, os.X_OK), f"smoke script not executable: {SCRIPT}"

    # Caller must export SFT_SMOKE_MODEL pointing at a small HF/Megatron checkpoint.
    if "SFT_SMOKE_MODEL" not in os.environ:
        pytest.skip("SFT_SMOKE_MODEL env var not set")

    result = subprocess.run([str(SCRIPT)], capture_output=True, text=True, timeout=1800)
    print("STDOUT:", result.stdout[-4000:])
    print("STDERR:", result.stderr[-2000:])
    assert result.returncode == 0, f"smoke test failed (rc={result.returncode})"
    assert "SFT smoke test PASSED" in result.stdout
