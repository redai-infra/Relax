# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Safety contracts for optional external multimodal acceptance fixtures."""

import torch

from tests.utils import mm_payload_fixtures
from tests.utils.tq._payload_assertions import leaf_digests


def _bundle() -> dict:
    train_data = {
        "tokens": [[1, 2, 3]],
        "multimodal_train_inputs": [
            {
                "pixel_values": torch.arange(12, dtype=torch.float32).reshape(2, 6),
                "image_grid_thw": torch.tensor([[1, 1, 2]], dtype=torch.int64),
            }
        ],
    }
    manifest = {}
    manifest.update(leaf_digests(train_data["multimodal_train_inputs"][0], "sample[0].multimodal_train_inputs"))
    manifest.update(leaf_digests(torch.tensor(train_data["tokens"][0]), "sample[0].tokens"))
    return {"meta": {"torch_version": torch.__version__}, "train_data": train_data, "manifest": manifest}


def test_external_fixture_uses_safe_weights_only_loader(monkeypatch, tmp_path):
    path = tmp_path / "fixture.pt"
    torch.save(_bundle(), path)
    monkeypatch.setenv("RELAX_MM_FIXTURE", str(path))

    loaded = mm_payload_fixtures.load_real_fixture()

    assert loaded is not None
    assert set(loaded) == {"train_data"}
    assert loaded["train_data"]["tokens"] == [[1, 2, 3]]


def test_external_fixture_load_failure_does_not_leak_path_or_exception(monkeypatch, tmp_path):
    path = tmp_path / "private-host-and-model-path.pt"
    path.write_bytes(b"not a torch fixture")
    monkeypatch.setenv("RELAX_MM_FIXTURE", str(path))

    try:
        mm_payload_fixtures.load_real_fixture()
    except RuntimeError as error:
        message = str(error)
    else:  # pragma: no cover - corrupt input must never be accepted
        raise AssertionError("corrupt fixture unexpectedly loaded")

    assert str(path) not in message
    assert "private-host" not in message
    assert "pickle" not in message.lower()


def test_external_fixture_manifest_failure_does_not_leak_path(monkeypatch, tmp_path):
    path = tmp_path / "private-dataset-location.pt"
    bundle = _bundle()
    bundle["manifest"] = {}
    torch.save(bundle, path)
    monkeypatch.setenv("RELAX_MM_FIXTURE", str(path))

    try:
        mm_payload_fixtures.load_real_fixture()
    except RuntimeError as error:
        message = str(error)
    else:  # pragma: no cover - invalid manifest must never be accepted
        raise AssertionError("fixture with an invalid manifest unexpectedly loaded")

    assert "failed its manifest" in message
    assert str(path) not in message
