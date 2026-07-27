# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for ``reconcile_hf_export_index``.

Megatron-Bridge's non-distributed ``export_ckpt(strict=False)`` can leave
"ghost" keys in ``model.safetensors.index.json``: keys (typically ``mtp.*`` for
a checkpoint trained without MTP) that are listed in the index but absent from
every shard. These tests build a synthetic buggy export and verify that the
reconciler either supplements the missing MTP tensors from a reference model or
drops the ghost entries, always leaving the index consistent with the shards on
disk.
"""

from __future__ import annotations

import json

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
import safetensors  # noqa: E402
import safetensors.torch  # noqa: E402

from relax.utils.hf_export import reconcile_hf_export_index  # noqa: E402


_INDEX = "model.safetensors.index.json"


def _write_shard(path, tensors):
    safetensors.torch.save_file(tensors, str(path), metadata={"format": "pt"})


def _write_index(path, weight_map):
    with open(path, "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f)


def _make_reference(dirpath):
    """A well-formed reference HF model: backbone + 2 MTP tensors across 2
    shards."""
    dirpath.mkdir(parents=True, exist_ok=True)
    _write_shard(
        dirpath / "model-00001-of-00002.safetensors",
        {"model.embed_tokens.weight": torch.zeros(4, 4), "model.layers.0.mlp.gate.weight": torch.zeros(2, 4)},
    )
    _write_shard(
        dirpath / "model-00002-of-00002.safetensors",
        {
            "mtp.fc.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "mtp.norm.weight": torch.arange(4, dtype=torch.float32),
        },
    )
    _write_index(
        dirpath / _INDEX,
        {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.mlp.gate.weight": "model-00001-of-00002.safetensors",
            "mtp.fc.weight": "model-00002-of-00002.safetensors",
            "mtp.norm.weight": "model-00002-of-00002.safetensors",
        },
    )


def _make_buggy_export(dirpath):
    """Simulate the Bridge bug: shards hold only backbone tensors, but the
    index also lists ``mtp.*`` ghost keys pointing at a shard that does not
    contain them."""
    dirpath.mkdir(parents=True, exist_ok=True)
    _write_shard(
        dirpath / "model-00001-of-00002.safetensors",
        {"model.embed_tokens.weight": torch.ones(4, 4), "model.layers.0.mlp.gate.weight": torch.ones(2, 4)},
    )
    _write_shard(dirpath / "model-00002-of-00002.safetensors", {"model.norm.weight": torch.ones(4)})
    _write_index(
        dirpath / _INDEX,
        {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.mlp.gate.weight": "model-00001-of-00002.safetensors",
            "model.norm.weight": "model-00002-of-00002.safetensors",
            "mtp.fc.weight": "model-00002-of-00002.safetensors",  # ghost
            "mtp.norm.weight": "model-00002-of-00002.safetensors",  # ghost
        },
    )


def _physical_and_index(dirpath):
    index = json.load(open(dirpath / _INDEX))["weight_map"]
    physical = set()
    for shard in set(index.values()):
        with safetensors.safe_open(str(dirpath / shard), framework="pt", device="cpu") as f:
            physical.update(f.keys())
    return physical, index


def test_supplement_mtp_from_reference(tmp_path):
    ref, out = tmp_path / "ref", tmp_path / "out"
    _make_reference(ref)
    _make_buggy_export(out)

    summary = reconcile_hf_export_index(str(out), reference_hf_dir=str(ref), supplement_mtp=True)

    assert set(summary["ghosts"]) == {"mtp.fc.weight", "mtp.norm.weight"}
    assert set(summary["supplemented"]) == {"mtp.fc.weight", "mtp.norm.weight"}
    assert summary["dropped"] == []

    physical, index = _physical_and_index(out)
    assert set(index) == physical  # index consistent with shards, no ghosts
    assert {"mtp.fc.weight", "mtp.norm.weight"} <= set(index)
    # supplemented values must equal the reference
    with safetensors.safe_open(str(out / index["mtp.fc.weight"]), framework="pt", device="cpu") as f:
        got = f.get_tensor("mtp.fc.weight")
    assert torch.equal(got, torch.arange(8, dtype=torch.float32).reshape(2, 4))
    # backbone tensors are untouched (still the trained values, i.e. ones)
    with safetensors.safe_open(str(out / index["model.embed_tokens.weight"]), framework="pt", device="cpu") as f:
        assert torch.equal(f.get_tensor("model.embed_tokens.weight"), torch.ones(4, 4))


def test_reconcile_only_drops_ghosts(tmp_path):
    out = tmp_path / "out"
    _make_buggy_export(out)

    summary = reconcile_hf_export_index(str(out), reference_hf_dir=None, supplement_mtp=False)

    assert set(summary["ghosts"]) == {"mtp.fc.weight", "mtp.norm.weight"}
    assert summary["supplemented"] == []
    assert set(summary["dropped"]) == {"mtp.fc.weight", "mtp.norm.weight"}
    physical, index = _physical_and_index(out)
    assert set(index) == physical
    assert not ({"mtp.fc.weight", "mtp.norm.weight"} & set(index))


def test_no_ghosts_is_noop(tmp_path):
    ref = tmp_path / "ref"
    _make_reference(ref)
    before = json.load(open(ref / _INDEX))

    summary = reconcile_hf_export_index(str(ref), reference_hf_dir=str(ref), supplement_mtp=True)

    assert summary == {"ghosts": [], "supplemented": [], "dropped": []}
    assert json.load(open(ref / _INDEX)) == before  # index left untouched
