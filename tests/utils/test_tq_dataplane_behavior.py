# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real local SimpleStorage data-plane contracts.

Covers connection, dense/multimodal byte identity, backpressure, empty get,
repeat put, and clear/reinit.  Cross-node transport and disconnect behavior
belong to the retained C0/C1/C2 benchmark.
"""

from __future__ import annotations

import importlib.util
import time

import pytest
import torch


def _has_real_submodule(dotted: str) -> bool:
    """Distinguish the real package from CI's single-file TQ stub."""
    try:
        return importlib.util.find_spec(dotted) is not None
    except (ImportError, ValueError, TypeError):
        return False


_TQ_OK = _has_real_submodule("transfer_queue.storage")
_RAY_OK = importlib.util.find_spec("ray") is not None

pytestmark = pytest.mark.skipif(
    not (_TQ_OK and _RAY_OK),
    reason=(
        "TransferQueue data-plane tests require the `transfer_queue` and `ray` "
        "packages plus a startable local Ray cluster (SimpleStorage; no GPU/RDMA)."
    ),
)

_TQ_ACTOR = "TransferQueueController"
_TQ_NS = "transfer_queue"


def _wait_controller_gone(timeout: float = 20.0) -> bool:
    """Poll until the named TQ controller is gone from the Ray GCS.

    Required between tests: after ``tq.close()`` kills the actor, its handle is
    still resolvable for a short window.  Re-init during that window would
    attach to a *dead* controller (ActorDiedError) — the F10 race.
    """
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


def _flat_values(t):
    """Flatten a dense tensor or NestedTensor to its comparable storage."""
    if type(t).__name__ == "NestedTensor":
        return t.values().reshape(-1)
    return t.reshape(-1)


def _row_value(column, row_position: int):
    """Extract one row from a dense, nested, or non-tensor column."""
    if isinstance(column, torch.Tensor) and column.is_nested:
        return column.unbind()[row_position]
    return column[row_position]


def _payload(n: int, fields: list[str], cols: int, dtype: str = "float32", seed: int = 0):
    """Build a TensorDict of ``n`` samples with ``fields`` of shape (n,
    cols)."""
    from tensordict import TensorDict

    dt = getattr(torch, dtype)
    g = torch.Generator().manual_seed(seed)
    data = {f: torch.randn(n, cols, dtype=dt, generator=g) for f in fields}
    return TensorDict(data, batch_size=[n])


def _multimodal_payload(num_samples: int) -> dict:
    """Deterministic Qwen3-VL-shaped payload for the non-tensor TQ path."""
    grids = ((1, 58, 64), (1, 34, 64), (1, 64, 64), (1, 26, 40))
    generator = torch.Generator().manual_seed(20260813)
    multimodal = []
    tokens = []
    for index in range(num_samples):
        t, h, w = grids[index % len(grids)]
        multimodal.append(
            {
                "pixel_values": torch.randn(t * h * w, 1536, generator=generator),
                "image_grid_thw": torch.tensor([[t, h, w]], dtype=torch.int64),
            }
        )
        tokens.append(torch.randint(0, 151_000, (512 + 173 * index,), generator=generator).tolist())
    return {"tokens": tokens, "multimodal_train_inputs": multimodal}


def _round_trip(client, payload, partition: str, fields: list[str], n: int):
    """put -> get_data and return the retrieved TensorDict."""
    client.put(payload, partition_id=partition)
    meta = client.get_meta(data_fields=fields, batch_size=n, partition_id=partition, mode="fetch", task_name=partition)
    return client.get_data(meta)


@pytest.fixture(scope="module")
def _ray_cluster():
    import ray

    ray.init(ignore_reinit_error=True, logging_level="ERROR")
    yield
    try:
        ray.shutdown()
    except Exception:
        pass


@pytest.fixture
def tq_factory(_ray_cluster):
    """Yield an F10-safe ``reinit(capacity, units) -> client`` factory."""
    import transfer_queue as tq
    from omegaconf import OmegaConf
    from transfer_queue import GRPOGroupNSampler

    def _reinit(capacity: int = 1024, units: int = 1):
        tq.close()
        if not _wait_controller_gone():
            _force_kill_controller()
            _wait_controller_gone()
        conf = OmegaConf.create(
            {
                "controller": {
                    "sampler": GRPOGroupNSampler(n_samples_per_prompt=1),
                    "polling_mode": True,
                },
                "backend": {
                    "SimpleStorage": {
                        "total_storage_size": capacity,
                        "num_data_storage_units": units,
                    }
                },
            },
            flags={"allow_objects": True},
        )
        tq.init(conf=conf)
        return tq.get_client()

    yield _reinit

    tq.close()
    _wait_controller_gone()
    _force_kill_controller()
    _wait_controller_gone()


class TestTqDataPlaneBehavior:
    @pytest.mark.parametrize(
        ("fields", "columns", "samples", "seed"),
        [
            (["a", "b"], 8, 4, 0),
            (["img", "txt", "mask"], 16, 8, 42),
            (["pixel_values"], 1176, 4, 7),
        ],
        ids=["connection", "multi-field", "multimodal-width"],
    )
    def test_dense_round_trip_is_byte_exact(self, tq_factory, fields, columns, samples, seed):
        client = tq_factory()
        payload = _payload(n=samples, fields=fields, cols=columns, seed=seed)
        got = _round_trip(client, payload, "dense", fields, samples)
        assert set(fields) <= set(got.keys())
        for field in fields:
            gv, av = _flat_values(got[field]), _flat_values(payload[field])
            assert gv.numel() == av.numel()
            assert gv.dtype == av.dtype
            assert torch.equal(gv, av), f"{field}: not byte-exact"

    def test_backpressure_raises_on_capacity_overflow(self, tq_factory):
        """A single put exceeding capacity raises rather than silently
        dropping."""
        client = tq_factory(capacity=4)
        payload = _payload(n=8, fields=["a"], cols=4)  # 8 samples > capacity 4
        with pytest.raises(RuntimeError, match="capacity"):
            client.put(payload, partition_id="bp")
        # Nothing was stored -> a subsequent get reports size 0 (no data).
        meta = client.get_meta(data_fields=["a"], batch_size=8, partition_id="bp", mode="fetch", task_name="bp")
        assert getattr(meta, "size", None) == 0

    def test_empty_get_returns_zero_size_without_hanging(self, tq_factory):
        """get on an empty partition returns size==0 + empty TensorDict (no
        hang)."""
        client = tq_factory()
        meta = client.get_meta(data_fields=["a"], batch_size=4, partition_id="empty", mode="fetch", task_name="empty")
        assert getattr(meta, "size", None) == 0
        data = client.get_data(meta)  # empty TensorDict; must not KeyError on access
        assert len(list(data.keys())) == 0

    def test_repeat_put_same_partition_is_safe(self, tq_factory):
        """Re-putting the same partition id is safe: no crash, no duplication,
        no corruption.

        TQ's overwrite-vs-sample semantics are sampler-dependent, so we assert
        the stable, observable contract -- the partition stays bounded at N and
        every returned sample is byte-identical to a sample we actually put
        (never garbage).
        """
        client = tq_factory()
        first = _payload(n=4, fields=["a"], cols=4, seed=1)
        second = _payload(n=4, fields=["a"], cols=4, seed=2)
        client.put(first, partition_id="rp")
        client.put(second, partition_id="rp")  # must not crash or hang
        meta = client.get_meta(data_fields=["a"], batch_size=4, partition_id="rp", mode="fetch", task_name="rp")
        assert getattr(meta, "size", None) == 4  # bounded; not duplicated to 8
        got = client.get_data(meta)
        gv = _flat_values(got["a"])
        assert gv.numel() == 16
        # Every returned row matches some row we put (first or second); order may
        # differ due to sampling, but no row may be corrupted.
        got_rows = gv.reshape(4, 4)
        candidates = torch.cat([first["a"], second["a"]], dim=0)  # (8, 4)
        for i in range(4):
            row = got_rows[i]
            assert torch.any(torch.all(candidates == row, dim=1)), f"row {i} matches no put sample (corrupted)"

    def test_cleanup_clear_partition_then_reinit_isolated(self, tq_factory):
        """clear_partition empties data; a reinit yields a fresh controller."""
        client = tq_factory()
        client.put(_payload(n=4, fields=["a"], cols=4), partition_id="cp")
        meta = client.get_meta(data_fields=["a"], batch_size=4, partition_id="cp", mode="fetch", task_name="cp")
        assert getattr(meta, "size", None) == 4
        client.clear_partition("cp")
        meta2 = client.get_meta(data_fields=["a"], batch_size=4, partition_id="cp", mode="fetch", task_name="cp2")
        assert getattr(meta2, "size", None) == 0

        # Reinit with a different capacity -> fresh controller, old partition gone.
        client2 = tq_factory(capacity=16)
        meta3 = client2.get_meta(data_fields=["a"], batch_size=4, partition_id="cp", mode="fetch", task_name="cp3")
        assert getattr(meta3, "size", None) == 0


class TestMultimodalFullLink:
    """Production NonTensorStack container survives the full link exactly."""

    def test_multimodal_list_dict_full_link_byte_exact(self, tq_factory, record_property):
        from relax.utils.utils import dict_to_tensordict
        from tests.utils.tq._payload_assertions import diff_digests, leaf_digests

        num_samples = 4
        train_data = _multimodal_payload(num_samples)
        source = "synthetic"
        record_property("multimodal_payload_source", source)
        train_data = dict(train_data)
        train_data["sample_id"] = list(range(num_samples))
        batch = dict_to_tensordict(train_data, batch_size=num_samples)
        assert type(batch.get("multimodal_train_inputs")).__name__ == "NonTensorStack", (
            "precondition: the multimodal column must be the production NonTensorStack container"
        )

        want_mm = [leaf_digests(sample) for sample in train_data["multimodal_train_inputs"]]
        want_tokens = [leaf_digests(torch.tensor(row, dtype=torch.int64)) for row in train_data["tokens"]]

        client = tq_factory()
        fields = ["sample_id", "tokens", "multimodal_train_inputs"]
        got = _round_trip(client, batch, "mmreal", fields, num_samples)

        got_ids = [int(v) for v in _flat_values(got["sample_id"])]
        assert sorted(got_ids) == list(range(num_samples)), f"[{source}] sample_id set mismatch: {got_ids}"
        for row_position, sample_id in enumerate(got_ids):
            mm_problems = diff_digests(
                want_mm[sample_id], leaf_digests(_row_value(got["multimodal_train_inputs"], row_position))
            )
            assert not mm_problems, (
                f"[{source}] sample {sample_id} multimodal leaves not byte-exact: {mm_problems[:4]}"
            )
            token_problems = diff_digests(
                want_tokens[sample_id], leaf_digests(_row_value(got["tokens"], row_position))
            )
            assert not token_problems, f"[{source}] sample {sample_id} tokens not byte-exact: {token_problems[:4]}"
