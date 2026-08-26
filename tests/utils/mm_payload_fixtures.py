# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared multimodal payload sources for TransferQueue byte-exact tests.

Two tiers, selected automatically:

* **real** — a fixture produced by ``scripts/benchmarks/make_multimodal_fixture.py``
  (real dataset images through the production Qwen-VL processing chain).
  Found via ``$RELAX_MM_FIXTURE`` or ``tests/fixtures/tq_multimodal_fixture.pt``.
  The fixture's leaf manifest is re-verified on load so a corrupted file can
  never silently pass as ground truth.
* **synthetic** — production-*structured* fallback so CI (no dataset, no model
  weights) still exercises the exact container shape the data plane ships:
  ``multimodal_train_inputs`` as ``list[dict]`` with variable-length fp32
  ``pixel_values [patches, 1536]`` + int64 ``image_grid_thw [1, 3]`` and
  ``t*h*w == patches`` (Qwen3-VL patch-16 geometry), which is MooncakeStore's
  non-tensor msgpack slow path — NOT the dense-tensor fast path.

Tests must report which tier ran (the returned ``source`` string) so real-
payload acceptance is auditable in CI logs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from relax.utils.payload_digest import diff_digests, leaf_digests


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "tq_multimodal_fixture.pt"

# Grids observed on a real Qwen-VL processor (patch 16, spatial merge 2) over
# the acceptance dataset: mixed aspect ratios, ~0.6-3.7k patches/image.
_SYNTHETIC_GRIDS = ((1, 58, 64), (1, 34, 64), (1, 64, 64), (1, 26, 40), (1, 64, 58), (1, 40, 40))
_QWEN3_VL_PATCH_DIM = 1536  # channels(3) * temporal(2) * patch(16)^2


def fixture_path() -> Path:
    """Fixture location: ``$RELAX_MM_FIXTURE`` wins, else the repo default."""
    override = os.environ.get("RELAX_MM_FIXTURE", "")
    return Path(override) if override else _DEFAULT_FIXTURE


def load_real_fixture(max_samples: int | None = None) -> dict[str, Any] | None:
    """Load + integrity-check the real fixture; ``None`` when unavailable."""
    path = fixture_path()
    if not path.is_file():
        return None
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    train_data = bundle["train_data"]
    manifest = bundle["manifest"]
    recomputed: dict[str, Any] = {}
    for index, sample in enumerate(train_data["multimodal_train_inputs"]):
        recomputed.update(leaf_digests(sample, f"sample[{index}].multimodal_train_inputs"))
        recomputed.update(
            leaf_digests(torch.tensor(train_data["tokens"][index], dtype=torch.int64), f"sample[{index}].tokens")
        )
    problems = diff_digests(manifest, recomputed)
    if problems:
        raise RuntimeError(
            f"Multimodal fixture {path} failed its own manifest ({len(problems)} leaf mismatches); "
            f"regenerate it with scripts/benchmarks/make_multimodal_fixture.py. First: {problems[0]}"
        )
    if max_samples is not None:
        train_data = {
            "tokens": train_data["tokens"][:max_samples],
            "multimodal_train_inputs": train_data["multimodal_train_inputs"][:max_samples],
        }
    return {"meta": bundle["meta"], "train_data": train_data}


def synthetic_mm_train_data(num_samples: int, seed: int = 20260813) -> dict[str, list[Any]]:
    """Production-structured synthetic ``train_data`` (see module
    docstring)."""
    generator = torch.Generator().manual_seed(seed)
    tokens: list[list[int]] = []
    multimodal: list[dict[str, torch.Tensor]] = []
    for index in range(num_samples):
        t, h, w = _SYNTHETIC_GRIDS[index % len(_SYNTHETIC_GRIDS)]
        patches = t * h * w
        multimodal.append(
            {
                "pixel_values": torch.randn(patches, _QWEN3_VL_PATCH_DIM, dtype=torch.float32, generator=generator),
                "image_grid_thw": torch.tensor([[t, h, w]], dtype=torch.int64),
            }
        )
        prompt_len = 512 + 173 * index
        tokens.append(torch.randint(0, 151_000, (prompt_len,), generator=generator).tolist())
    return {"tokens": tokens, "multimodal_train_inputs": multimodal}


def mm_train_data(num_samples: int) -> tuple[dict[str, list[Any]], str]:
    """Real-fixture ``train_data`` when available, else synthetic.

    Returns ``(train_data, source)`` where source is ``"real"`` /
    ``"synthetic"``; tests embed it in assertion ids so acceptance logs show
    which tier actually ran.
    """
    bundle = load_real_fixture(max_samples=num_samples)
    if bundle is not None and len(bundle["train_data"]["tokens"]) >= num_samples:
        return bundle["train_data"], "real"
    return synthetic_mm_train_data(num_samples), "synthetic"
