# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real local SimpleStorage connection and data-plane contracts."""

from __future__ import annotations

import importlib.util
import time
from typing import Any

import pytest
import torch

from relax.utils.tq.correctness import diff_digests, leaf_digests, payload_rows


def _has_real_submodule(dotted: str) -> bool:
    try:
        return importlib.util.find_spec(dotted) is not None
    except (ImportError, ValueError, TypeError):
        return False


pytestmark = pytest.mark.skipif(
    not (_has_real_submodule("transfer_queue.storage") and importlib.util.find_spec("ray")),
    reason="requires real TransferQueue, Ray, and a startable local CPU cluster",
)
_TQ_ACTOR = "TransferQueueController"
_TQ_NS = "transfer_queue"


def _wait_controller_gone(timeout: float = 20.0) -> bool:
    import ray

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ray.get_actor(_TQ_ACTOR, namespace=_TQ_NS)
        except ValueError:
            return True
        time.sleep(0.4)
    return False


def _force_kill_controller() -> None:
    import ray

    try:
        ray.kill(ray.get_actor(_TQ_ACTOR, namespace=_TQ_NS))
    except ValueError:
        pass


def _payload(samples: int, fields: list[str], columns: int, seed: int = 0) -> Any:
    from tensordict import TensorDict

    generator = torch.Generator().manual_seed(seed)
    return TensorDict(
        {field: torch.randn(samples, columns, generator=generator) for field in fields},
        batch_size=[samples],
    )


def _multimodal_payload(samples: int) -> dict[str, Any]:
    grids = ((1, 58, 64), (1, 34, 64), (1, 64, 64), (1, 26, 40))
    generator = torch.Generator().manual_seed(20260813)
    multimodal, tokens = [], []
    for index in range(samples):
        temporal, height, width = grids[index % len(grids)]
        multimodal.append(
            {
                "pixel_values": torch.randn(temporal * height * width, 1536, generator=generator),
                "image_grid_thw": torch.tensor([[temporal, height, width]], dtype=torch.int64),
            }
        )
        tokens.append(torch.randint(0, 151_000, (512 + 173 * index,), generator=generator).tolist())
    return {"tokens": tokens, "multimodal_train_inputs": multimodal}


def _get(client: Any, partition: str, fields: list[str], size: int) -> Any:
    meta = client.get_meta(
        data_fields=fields,
        batch_size=size,
        partition_id=partition,
        mode="fetch",
        task_name=partition,
    )
    return meta, client.get_data(meta)


@pytest.fixture(scope="module")
def _ray_cluster():
    import ray

    ray.init(ignore_reinit_error=True, logging_level="ERROR")
    yield
    ray.shutdown()


@pytest.fixture
def tq_factory(_ray_cluster):
    import transfer_queue as tq
    from omegaconf import OmegaConf
    from transfer_queue import GRPOGroupNSampler

    def reinit(capacity: int = 1024):
        tq.close()
        if not _wait_controller_gone():
            _force_kill_controller()
            assert _wait_controller_gone()
        conf = OmegaConf.create(
            {
                "controller": {"sampler": GRPOGroupNSampler(n_samples_per_prompt=1), "polling_mode": True},
                "backend": {"SimpleStorage": {"total_storage_size": capacity, "num_data_storage_units": 1}},
            },
            flags={"allow_objects": True},
        )
        tq.init(conf=conf)
        return tq.get_client()

    yield reinit
    tq.close()
    _wait_controller_gone()
    _force_kill_controller()
    _wait_controller_gone()


@pytest.mark.parametrize(
    ("fields", "columns", "samples", "seed"),
    [(["a", "b"], 8, 4, 0), (["img", "txt", "mask"], 16, 8, 42), (["pixel_values"], 1176, 4, 7)],
    ids=["connection", "multi-field", "multimodal-width"],
)
def test_dense_round_trip_is_byte_exact(tq_factory, fields: list[str], columns: int, samples: int, seed: int) -> None:
    client = tq_factory()
    payload = _payload(samples, fields, columns, seed)
    client.put(payload, partition_id="dense")
    _, received = _get(client, "dense", fields, samples)
    assert set(fields) <= set(received.keys())
    for field in fields:
        expected_rows = [leaf_digests(row) for row in payload_rows(payload[field])]
        actual_rows = [leaf_digests(row) for row in payload_rows(received[field])]
        assert len(actual_rows) == len(expected_rows)
        assert all(
            not diff_digests(expected, actual) for expected, actual in zip(expected_rows, actual_rows, strict=True)
        )


def test_backpressure_fails_without_publishing_data(tq_factory) -> None:
    client = tq_factory(capacity=4)
    with pytest.raises(RuntimeError, match="capacity"):
        client.put(_payload(8, ["a"], 4), partition_id="backpressure")
    meta, _ = _get(client, "backpressure", ["a"], 8)
    assert getattr(meta, "size", None) == 0


def test_empty_get_returns_without_hanging(tq_factory) -> None:
    meta, data = _get(tq_factory(), "empty", ["a"], 4)
    assert getattr(meta, "size", None) == 0 and list(data.keys()) == []


def test_repeat_put_stays_bounded_and_uncorrupted(tq_factory) -> None:
    client = tq_factory()
    first, second = _payload(4, ["a"], 4, 1), _payload(4, ["a"], 4, 2)
    client.put(first, partition_id="repeat")
    client.put(second, partition_id="repeat")
    meta, received = _get(client, "repeat", ["a"], 4)
    candidates = [leaf_digests(row) for row in [*payload_rows(first["a"]), *payload_rows(second["a"])]]
    assert getattr(meta, "size", None) == 4
    assert all(leaf_digests(row) in candidates for row in payload_rows(received["a"]))


def test_clear_partition_and_reinit_are_isolated(tq_factory) -> None:
    client = tq_factory()
    client.put(_payload(4, ["a"], 4), partition_id="cleanup")
    assert getattr(_get(client, "cleanup", ["a"], 4)[0], "size", None) == 4
    client.clear_partition("cleanup")
    assert getattr(_get(client, "cleanup", ["a"], 4)[0], "size", None) == 0
    assert getattr(_get(tq_factory(capacity=16), "cleanup", ["a"], 4)[0], "size", None) == 0


def test_multimodal_non_tensor_full_link_is_byte_exact(tq_factory, record_property) -> None:
    from relax.utils.utils import dict_to_tensordict

    samples = 4
    source = _multimodal_payload(samples)
    expected_mm = [leaf_digests(row) for row in source["multimodal_train_inputs"]]
    expected_tokens = [leaf_digests(torch.tensor(row, dtype=torch.int64)) for row in source["tokens"]]
    batch = dict_to_tensordict({**source, "sample_id": list(range(samples))}, batch_size=samples)
    assert type(batch.get("multimodal_train_inputs")).__name__ == "NonTensorStack"
    record_property("multimodal_payload_source", "synthetic")

    fields = ["sample_id", "tokens", "multimodal_train_inputs"]
    client = tq_factory()
    client.put(batch, partition_id="multimodal")
    _, received = _get(client, "multimodal", fields, samples)
    sample_ids = [int(value) for value in payload_rows(received["sample_id"])]
    assert sorted(sample_ids) == list(range(samples))
    for position, sample_id in enumerate(sample_ids):
        assert not diff_digests(expected_mm[sample_id], leaf_digests(payload_rows(received[fields[2]])[position]))
        assert not diff_digests(expected_tokens[sample_id], leaf_digests(payload_rows(received[fields[1]])[position]))
