# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for RolloutManager utility / helper methods:

- _normalize_engine_addr
- _parse_host_port
- _collect_existing_engine_addrs
- _collect_in_flight_engine_addrs
- EngineGroup / RolloutServer properties
"""

import pytest


try:
    from relax.distributed.ray.rollout import (
        RolloutManager,
        ScaleOutRequest,
        ScaleOutStatus,
    )

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from conftest import (
    create_test_manager,
    make_engine_group,
    make_mock_engine,
    make_rollout_server,
)


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


# ===================== _normalize_engine_addr ==============================


class TestNormalizeEngineAddr:
    def test_strip_http(self):
        assert RolloutManager._normalize_engine_addr("http://10.0.0.1:30000") == "10.0.0.1:30000"

    def test_strip_https(self):
        assert RolloutManager._normalize_engine_addr("https://10.0.0.1:30000") == "10.0.0.1:30000"

    def test_bare_addr_unchanged(self):
        assert RolloutManager._normalize_engine_addr("10.0.0.1:30000") == "10.0.0.1:30000"

    def test_ipv6_with_scheme(self):
        assert RolloutManager._normalize_engine_addr("http://[::1]:30000") == "[::1]:30000"

    def test_empty_string(self):
        assert RolloutManager._normalize_engine_addr("") == ""

    def test_hostname(self):
        assert RolloutManager._normalize_engine_addr("http://myhost:8080") == "myhost:8080"


# ======================== _parse_host_port =================================


class TestParseHostPort:
    def test_ipv4_basic(self):
        host, port = RolloutManager._parse_host_port("10.0.0.1:8000")
        assert host == "10.0.0.1"
        assert port == 8000

    def test_ipv6_bracket(self):
        host, port = RolloutManager._parse_host_port("[::1]:8000")
        assert host == "::1"
        assert port == 8000

    def test_with_http_scheme(self):
        host, port = RolloutManager._parse_host_port("http://10.0.0.1:8000")
        assert host == "10.0.0.1"
        assert port == 8000

    def test_with_https_scheme(self):
        host, port = RolloutManager._parse_host_port("https://host:9000")
        assert host == "host"
        assert port == 9000

    def test_invalid_format_too_many_colons(self):
        with pytest.raises(ValueError, match="Invalid address format"):
            RolloutManager._parse_host_port("a:b:c")

    def test_invalid_port_not_integer(self):
        with pytest.raises(ValueError, match="not a valid integer"):
            RolloutManager._parse_host_port("host:abc")

    def test_invalid_ipv6_no_bracket_close(self):
        with pytest.raises(ValueError, match="Invalid IPv6"):
            RolloutManager._parse_host_port("[::1:8000")

    def test_ipv6_bracket_no_port(self):
        with pytest.raises(ValueError, match="Invalid IPv6"):
            RolloutManager._parse_host_port("[::1]")

    def test_localhost(self):
        host, port = RolloutManager._parse_host_port("localhost:5000")
        assert host == "localhost"
        assert port == 5000


# =================== _collect_existing_engine_addrs ========================


class TestCollectExistingEngineAddrs:
    def test_all_live_engines(self, patch_ray_get):
        e1 = make_mock_engine(url="http://10.0.0.1:30000")
        e2 = make_mock_engine(url="http://192.0.2.2:30000")
        group = make_engine_group(engines=[e1, e2])
        srv = make_rollout_server(engine_groups=[group])
        manager = create_test_manager(servers={"default": srv})

        addrs = manager._collect_existing_engine_addrs(srv)
        assert addrs == {"10.0.0.1:30000", "192.0.2.2:30000"}

    def test_dead_engines_excluded(self, patch_ray_get):
        e1 = make_mock_engine(url="http://10.0.0.1:30000")
        group = make_engine_group(engines=[e1, None])
        srv = make_rollout_server(engine_groups=[group])
        manager = create_test_manager(servers={"default": srv})

        addrs = manager._collect_existing_engine_addrs(srv)
        assert addrs == {"10.0.0.1:30000"}

    def test_engine_returning_none_url(self, patch_ray_get):
        e1 = make_mock_engine(url=None)
        group = make_engine_group(engines=[e1])
        srv = make_rollout_server(engine_groups=[group])
        manager = create_test_manager(servers={"default": srv})

        addrs = manager._collect_existing_engine_addrs(srv)
        assert addrs == set()

    def test_engine_get_url_raises(self, patch_ray_get):
        e1 = make_mock_engine()
        e1.get_url.remote.side_effect = Exception("dead actor")
        group = make_engine_group(engines=[e1])
        srv = make_rollout_server(engine_groups=[group])
        manager = create_test_manager(servers={"default": srv})

        addrs = manager._collect_existing_engine_addrs(srv)
        assert addrs == set()

    def test_multiple_groups(self, patch_ray_get):
        e1 = make_mock_engine(url="http://a:1")
        e2 = make_mock_engine(url="http://b:2")
        g1 = make_engine_group(engines=[e1])
        g2 = make_engine_group(engines=[e2], is_scaled_out=True)
        srv = make_rollout_server(engine_groups=[g1, g2])
        manager = create_test_manager(servers={"default": srv})

        addrs = manager._collect_existing_engine_addrs(srv)
        assert addrs == {"a:1", "b:2"}


# =================== _collect_in_flight_engine_addrs =======================


class TestCollectInFlightEngineAddrs:
    def test_non_terminal_request_included(self):
        manager = create_test_manager()
        manager._scale_out_requests["r1"] = ScaleOutRequest(
            request_id="r1",
            status=ScaleOutStatus.CREATING,
            model_name="default",
            engine_urls=["10.0.0.1:30000"],
        )
        addrs = manager._collect_in_flight_engine_addrs("default")
        assert "10.0.0.1:30000" in addrs

    def test_terminal_request_excluded(self):
        manager = create_test_manager()
        for status in (ScaleOutStatus.ACTIVE, ScaleOutStatus.FAILED, ScaleOutStatus.CANCELLED):
            manager._scale_out_requests[f"r_{status.value}"] = ScaleOutRequest(
                request_id=f"r_{status.value}",
                status=status,
                model_name="default",
                engine_urls=["10.0.0.1:30000"],
            )
        addrs = manager._collect_in_flight_engine_addrs("default")
        assert len(addrs) == 0

    def test_different_model_excluded(self):
        manager = create_test_manager()
        manager._scale_out_requests["r1"] = ScaleOutRequest(
            request_id="r1",
            status=ScaleOutStatus.CREATING,
            model_name="reward",
            engine_urls=["10.0.0.1:30000"],
        )
        addrs = manager._collect_in_flight_engine_addrs("default")
        assert len(addrs) == 0

    def test_ray_native_request_no_urls(self):
        """ray_native requests have num_replicas > 0 but empty engine_urls."""
        manager = create_test_manager()
        manager._scale_out_requests["r1"] = ScaleOutRequest(
            request_id="r1",
            status=ScaleOutStatus.CREATING,
            model_name="default",
            num_replicas=3,
        )
        addrs = manager._collect_in_flight_engine_addrs("default")
        assert len(addrs) == 0

    def test_multiple_requests_union(self):
        manager = create_test_manager()
        manager._scale_out_requests["r1"] = ScaleOutRequest(
            request_id="r1",
            status=ScaleOutStatus.CONNECTING,
            model_name="default",
            engine_urls=["a:1"],
        )
        manager._scale_out_requests["r2"] = ScaleOutRequest(
            request_id="r2",
            status=ScaleOutStatus.HEALTH_CHECKING,
            model_name="default",
            engine_urls=["b:2"],
        )
        addrs = manager._collect_in_flight_engine_addrs("default")
        assert addrs == {"a:1", "b:2"}


# ================= EngineGroup / RolloutServer properties ==================


class TestEngineGroupProperties:
    def test_nodes_per_engine_single_node(self):
        args = type("A", (), {"num_gpus_per_node": 8})()
        g = make_engine_group(args=args, num_gpus_per_engine=2)
        assert g.nodes_per_engine == 1

    def test_nodes_per_engine_multi_node(self):
        args = type("A", (), {"num_gpus_per_node": 8})()
        g = make_engine_group(args=args, num_gpus_per_engine=16)
        assert g.nodes_per_engine == 2

    def test_engines_filters_by_nodes_per_engine(self):
        args = type("A", (), {"num_gpus_per_node": 8})()
        engines = [make_mock_engine() for _ in range(4)]
        g = make_engine_group(args=args, engines=engines, num_gpus_per_engine=16)
        # nodes_per_engine=2, so node-0 engines = all_engines[::2]
        assert len(g.engines) == 2

    def test_engines_single_node(self):
        args = type("A", (), {"num_gpus_per_node": 8})()
        engines = [make_mock_engine() for _ in range(4)]
        g = make_engine_group(args=args, engines=engines, num_gpus_per_engine=2)
        assert len(g.engines) == 4


class TestRolloutServerProperties:
    def test_engines_across_groups(self):
        args = type("A", (), {"num_gpus_per_node": 8})()
        g1 = make_engine_group(args=args, engines=[make_mock_engine(), make_mock_engine()])
        g2 = make_engine_group(args=args, engines=[make_mock_engine()])
        srv = make_rollout_server(engine_groups=[g1, g2])
        assert len(srv.engines) == 3

    def test_engine_gpu_counts(self):
        args = type("A", (), {"num_gpus_per_node": 8})()
        g1 = make_engine_group(args=args, engines=[make_mock_engine()], num_gpus_per_engine=2)
        g2 = make_engine_group(args=args, engines=[make_mock_engine()], num_gpus_per_engine=4)
        srv = make_rollout_server(engine_groups=[g1, g2])
        assert srv.engine_gpu_counts == [2, 4]

    def test_num_new_engines(self):
        args = type("A", (), {"num_gpus_per_node": 8})()
        g1 = make_engine_group(args=args)
        g1.num_new_engines = 2
        g2 = make_engine_group(args=args)
        g2.num_new_engines = 1
        srv = make_rollout_server(engine_groups=[g1, g2])
        assert srv.num_new_engines == 3

    def test_set_num_new_engines(self):
        args = type("A", (), {"num_gpus_per_node": 8})()
        g1 = make_engine_group(args=args)
        g2 = make_engine_group(args=args)
        srv = make_rollout_server(engine_groups=[g1, g2])
        srv.num_new_engines = 0
        assert g1.num_new_engines == 0
        assert g2.num_new_engines == 0
