# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for placement-group node affinity."""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from relax.core.service import Service, _require_node_group_markers, create_placement_group


def test_missing_markers_raise_after_retries():
    with (
        patch("relax.core.service.ray.cluster_resources", return_value={"GPU": 8.0, "CPU": 64.0}),
        patch("relax.core.service.time.sleep") as mock_sleep,
    ):
        with pytest.raises(RuntimeError) as exc:
            _require_node_group_markers("stable", retries=3, retry_delay=0.01)

    msg = str(exc.value)
    assert "stable_gpu" in msg
    assert "stable_cpu" in msg
    assert "--no-enable-affinity" in msg
    # 3 attempts -> 2 sleeps between them.
    assert mock_sleep.call_count == 2


def test_present_markers_do_not_raise():
    resources = {"GPU": 8.0, "CPU": 64.0, "stable_gpu": 8.0, "stable_cpu": 8.0}
    with patch("relax.core.service.ray.cluster_resources", return_value=resources):
        _require_node_group_markers("stable", retries=3, retry_delay=0.01)  # no raise


def test_markers_appear_on_retry():
    """A stable node that registers its markers late must be tolerated."""
    calls = [
        {"GPU": 8.0, "CPU": 64.0},  # first probe: not yet registered
        {"GPU": 8.0, "CPU": 64.0, "stable_gpu": 8.0, "stable_cpu": 8.0},  # then present
    ]
    with (
        patch("relax.core.service.ray.cluster_resources", side_effect=calls),
        patch("relax.core.service.time.sleep"),
    ):
        _require_node_group_markers("stable", retries=3, retry_delay=0.01)  # no raise


def _run_create_placement_group(num_gpus=2, node_group_affinity=True, cluster_resources=None):
    """Create a placement group against a mocked Ray runtime."""
    captured = {}

    def _fake_placement_group(bundles, strategy="PACK"):
        captured["bundles"] = bundles
        captured["strategy"] = strategy
        return MagicMock(name="pg")

    def _fake_ray_get(arg):
        # Second call passes a list of get_ip_and_gpu_id futures -> return one
        # (ip, gpu_id) tuple per bundle. First call is pg.ready() (unused).
        if isinstance(arg, list):
            return [("10.0.0.1", i) for i in range(len(arg))]
        return None

    cr_mock = MagicMock(return_value=cluster_resources or {"GPU": 8.0, "CPU": 64.0})

    with (
        patch("relax.core.service.device_utils.get_ray_accelerator_name", return_value="GPU"),
        patch("relax.core.service.get_ray_accelerator_kwargs", return_value={"num_gpus": 1}),
        patch("relax.core.service.placement_group", side_effect=_fake_placement_group),
        patch("relax.core.service.PlacementGroupSchedulingStrategy", MagicMock()),
        patch("relax.core.service.InfoActor", MagicMock()),
        patch("relax.core.service.ray.get", side_effect=_fake_ray_get),
        patch("relax.core.service.ray.kill", MagicMock()),
        patch("relax.core.service.ray.cluster_resources", cr_mock),
        patch("relax.core.service.time.sleep"),
    ):
        pg, reordered_indices, reordered_gpu_ids = create_placement_group(
            num_gpus, node_group_affinity=node_group_affinity
        )
    return captured, cr_mock


def test_create_pg_stable_with_markers_adds_marker_bundles(monkeypatch):
    """env=stable + affinity=True + cluster declares markers -> every bundle
    carries the {group}_gpu/{group}_cpu markers and the PG builds without
    raising."""
    monkeypatch.setenv("RELAX_INITIAL_NODE_GROUP", "stable")
    resources = {"GPU": 8.0, "CPU": 64.0, "stable_gpu": 8.0, "stable_cpu": 8.0}
    captured, cr_mock = _run_create_placement_group(num_gpus=2, node_group_affinity=True, cluster_resources=resources)

    assert len(captured["bundles"]) == 2
    for bundle in captured["bundles"]:
        assert bundle["stable_gpu"] == 1
        assert bundle["stable_cpu"] == 1
        assert bundle["GPU"] == 1
        assert bundle["CPU"] == 1
    # Marker presence was actually probed.
    assert cr_mock.call_count >= 1


def test_create_pg_stable_missing_markers_raises(monkeypatch):
    """env=stable + affinity=True but the cluster never declares the markers ->
    RuntimeError after the retry loop (rather than a forever-hang on
    pg.ready())."""
    monkeypatch.setenv("RELAX_INITIAL_NODE_GROUP", "stable")
    with pytest.raises(RuntimeError) as exc:
        _run_create_placement_group(
            num_gpus=2,
            node_group_affinity=True,
            cluster_resources={"GPU": 8.0, "CPU": 64.0},  # no stable_* markers
        )
    assert "stable_gpu" in str(exc.value)
    assert "stable_cpu" in str(exc.value)


def test_create_pg_affinity_false_skips_marker_check(monkeypatch):
    """node_group_affinity=False (opt-out) even with env=stable -> markers are
    NOT probed and bundles carry NO marker, so a role can escape onto elastic
    nodes and a marker-less cluster is unaffected."""
    monkeypatch.setenv("RELAX_INITIAL_NODE_GROUP", "stable")
    captured, cr_mock = _run_create_placement_group(
        num_gpus=2,
        node_group_affinity=False,
        cluster_resources={"GPU": 8.0, "CPU": 64.0},  # no markers -> would raise if probed
    )

    for bundle in captured["bundles"]:
        assert "stable_gpu" not in bundle
        assert "stable_cpu" not in bundle
        assert bundle["GPU"] == 1
        assert bundle["CPU"] == 1
    # No marker probe at all when affinity is opted out.
    assert cr_mock.call_count == 0


def test_create_pg_no_env_is_plain_unconstrained(monkeypatch):
    """env unset -> ordinary PG: no marker probe, no marker bundle.

    Guarantees non-elastic clusters are completely unaffected by the affinity
    feature.
    """
    monkeypatch.delenv("RELAX_INITIAL_NODE_GROUP", raising=False)
    captured, cr_mock = _run_create_placement_group(
        num_gpus=3,
        node_group_affinity=True,  # affinity on, but env empty -> no-op
        cluster_resources={"GPU": 8.0, "CPU": 64.0},
    )

    assert len(captured["bundles"]) == 3
    for bundle in captured["bundles"]:
        assert bundle == {"GPU": 1, "CPU": 1}
    assert cr_mock.call_count == 0


def _build_service(config):
    """Construct a service without starting Ray Serve."""
    with (
        patch("relax.core.service.create_placement_group", return_value=("pg", [], [])) as mock_cpg,
        patch.object(Service, "_deploy", return_value=None),
    ):
        Service(cls=MagicMock(), role="actor", healthy=MagicMock(), config=config, num_gpus=2)
    return mock_cpg


def test_service_forwards_enable_affinity_false():
    """config.enable_affinity=False -> create_placement_group is called with
    node_group_affinity=False (the escape valve, independent of env)."""
    mock_cpg = _build_service(Namespace(enable_affinity=False))
    mock_cpg.assert_called_once_with(num_gpus=2, node_group_affinity=False)


def test_service_forwards_enable_affinity_true():
    """config.enable_affinity=True -> node_group_affinity=True."""
    mock_cpg = _build_service(Namespace(enable_affinity=True))
    mock_cpg.assert_called_once_with(num_gpus=2, node_group_affinity=True)
