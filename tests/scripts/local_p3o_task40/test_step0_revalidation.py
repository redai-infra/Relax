# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.local_p3o_task40 import analyze_step0_revalidation as analysis  # noqa: E402
from scripts.local_p3o_task40 import step0_revalidation as hook  # noqa: E402


def test_cu_seqlens_derivation_makes_cp2_relation_explicit() -> None:
    assert analysis._canonical_cu([282], 128) == [0, 282, 384]
    assert analysis._expected_source_cu([282], 1, 128) == [0, 282, 384]
    assert analysis._expected_source_cu([282], 2, 128) == [0, 284, 512]
    assert analysis._expected_source_cu([219], 1, 128) == [0, 219, 256]


def test_numeric_comparison_uses_frozen_relative_and_near_zero_gates() -> None:
    assert analysis._numeric_comparison(2.0, 2.0 + 1e-6)["pass"]
    assert not analysis._numeric_comparison(2.0, 2.0 + 3e-6)["pass"]
    assert analysis._numeric_comparison(0.0, 1e-9)["pass"]
    assert not analysis._numeric_comparison(0.0, 2e-9)["pass"]


def test_tensor_bytes_hashes_bf16_losslessly() -> None:
    tensor = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
    cpu, raw = hook._tensor_bytes(tensor)
    assert torch.equal(cpu, tensor)
    assert len(raw) == tensor.numel() * tensor.element_size()


def test_parameter_hash_worker_is_joined_before_update(tmp_path: Path, monkeypatch) -> None:
    state = {
        **hook._STATE,
        "output_dir": tmp_path,
        "rank": 0,
        "parameters_captured": False,
        "parameter_capture_error": None,
        "parameter_capture_thread": None,
    }
    monkeypatch.setattr(hook, "_STATE", state)
    hook._start_initial_parameter_capture([torch.nn.Linear(2, 2)])
    hook._join_initial_parameter_capture()
    assert state["parameters_captured"]
    assert (tmp_path / "initial_parameters_rank0.json").is_file()


def test_gradient_comparator_reports_full_vector_rel_l2_and_cosine(tmp_path: Path) -> None:
    summaries = []
    for label, vector in (("reference", torch.tensor([1.0, 2.0])), ("candidate", torch.tensor([1.0, 2.0]))):
        run_dir = tmp_path / label
        vector_dir = run_dir / "oracle" / "gradients" / "rank00000"
        vector_dir.mkdir(parents=True)
        torch.save(vector, vector_dir / "shard_00000.pt")
        summary = {
            "rank": 0,
            "tensors": [
                {
                    "name": "chunk000.weight",
                    "present": True,
                    "file": "rank00000/shard_00000.pt",
                    "parameter_numel": 2,
                    "range_start": 0,
                    "range_end": 2,
                }
            ],
        }
        (run_dir / "oracle" / "gradients" / "summary_rank0.json").write_text(__import__("json").dumps(summary))
        summaries.append(run_dir)
    comparison = analysis._compare_gradients(*summaries)
    assert comparison["pass"]
    assert comparison["full_parameter_relative_l2"] == 0.0
    assert comparison["full_parameter_cosine"] == 1.0


def test_reconstruct_gradient_joins_distributed_optimizer_shards(tmp_path: Path) -> None:
    records = []
    for rank, (start, values) in enumerate(((0, torch.tensor([1.0, 2.0])), (2, torch.tensor([3.0, 4.0])))):
        path = tmp_path / f"rank{rank}.pt"
        torch.save(values, path)
        records.append(
            {
                "name": "chunk000.weight",
                "parameter_numel": 4,
                "range_start": start,
                "range_end": start + values.numel(),
                "path": path,
            }
        )
    assert torch.equal(analysis._reconstruct_gradient(records), torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_launcher_is_frozen_to_four_bf16_step_scope_cells() -> None:
    launcher = (REPO_ROOT / "scripts/local_p3o_task40/run_step0_revalidation.sh").read_text()
    for topology in ("dp1", "dp4cp1", "dp2cp1", "dp2cp2"):
        assert topology in launcher
    assert "P3O_ESS_SCOPE=step" in launcher
    assert "P3O_MICRO_BATCH_SIZE=1" in launcher
    assert "step0_revalidation.configure" in launcher
    assert "fp32" not in launcher.lower()


def test_retry_verdict_is_pinned_to_the_batch6_loop_commit() -> None:
    assert analysis.COMMIT_UNDER_TEST == "ef854a3cfb3eb692e083d14e12fbd96a8b1c7af5"
