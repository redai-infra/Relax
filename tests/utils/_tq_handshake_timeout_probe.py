# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Subprocess probe for one-shot Ray worker cleanup after attach timeout."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _process_is_running(pid: int, create_time: float) -> bool:
    import psutil

    try:
        process = psutil.Process(pid)
        return abs(process.create_time() - create_time) < 1e-3 and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _write_transfer_queue_stub(stub_dir: Path) -> None:
    """Create a TQ stub whose delayed init visibly mutates module state."""
    stub_dir.mkdir(parents=True, exist_ok=True)
    (stub_dir / "transfer_queue.py").write_text(
        """\
import os
import time
from pathlib import Path

import psutil

MUTATED = False


def init(conf=None):
    process = psutil.Process()
    Path(os.environ["RELAX_TEST_TQ_STARTED_MARKER"]).write_text(
        f"{process.pid},{process.create_time()}", encoding="utf-8"
    )
    time.sleep(float(os.environ["RELAX_TEST_TQ_LATE_MUTATION_DELAY"]))
    global MUTATED
    MUTATED = True
    Path(os.environ["RELAX_TEST_TQ_LATE_MARKER"]).write_text("dirty", encoding="utf-8")
""",
        encoding="utf-8",
    )


def main(probe_dir: Path) -> None:
    # Ray reads this switch at import time. Keep the probe isolated from both
    # uv parent-process discovery and any cluster selected by the caller.
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    os.environ.pop("RAY_ADDRESS", None)

    probe_dir.mkdir(parents=True, exist_ok=True)
    stub_dir = probe_dir / "stub"
    started_path = probe_dir / "timed-out-worker.started"
    mutation_path = probe_dir / "late-global-mutation"
    late_mutation_delay = 2.0
    _write_transfer_queue_stub(stub_dir)

    original_pythonpath = os.environ.get("PYTHONPATH", "")
    worker_pythonpath = os.pathsep.join(path for path in (str(stub_dir), original_pythonpath) if path)
    runtime_env = {
        "env_vars": {
            "PYTHONPATH": worker_pythonpath,
            "RELAX_TQ_ATTACH_TIMEOUT_SECONDS": "0.3",
            "RELAX_TEST_TQ_STARTED_MARKER": str(started_path),
            "RELAX_TEST_TQ_LATE_MARKER": str(mutation_path),
            "RELAX_TEST_TQ_LATE_MUTATION_DELAY": str(late_mutation_delay),
        }
    }

    import ray

    assert not ray.is_initialized()
    ray.init(
        address="local",
        num_cpus=1,
        include_dashboard=False,
        logging_level="ERROR",
        runtime_env=runtime_env,
        _temp_dir=str(probe_dir / "ray"),
    )
    try:
        from relax.utils.tq import lifecycle as tq_lifecycle

        conf = {"backend": {"storage_backend": "SimpleStorage"}, "controller": {}}

        @ray.remote(num_cpus=0)
        class _Controller:
            def __init__(self, config):
                self.config = config

            def get_config(self):
                return self.config

        controller = _Controller.options(
            name=tq_lifecycle.CONTROLLER_NAME,
            namespace=tq_lifecycle.CONTROLLER_NAMESPACE,
        ).remote(conf)
        assert ray.get(controller.get_config.remote()) == conf

        failures = tq_lifecycle.verify_cluster_attach(conf, timeout=0.3)
        assert len(failures) == 1
        # The public failure summary is deliberately scrubbed: the underlying
        # RayTaskError contains worker addresses, PIDs and traceback paths.
        assert failures[0].endswith("handshake task failed (RayTaskError)")
        assert started_path.exists(), "the production handshake never entered tq.init"
        timed_out_pid_text, timed_out_create_time_text = started_path.read_text(encoding="utf-8").split(",")
        timed_out_identity = (int(timed_out_pid_text), float(timed_out_create_time_text))

        # Without max_calls=1, the reusable task worker survives and its daemon
        # eventually dirties both the module global and this external marker.
        time.sleep(late_mutation_delay + 0.2)
        assert not mutation_path.exists(), "timed-out tq.init continued mutating state after task failure"

        exit_deadline = time.monotonic() + 10.0
        while _process_is_running(*timed_out_identity) and time.monotonic() < exit_deadline:
            time.sleep(0.05)
        assert not _process_is_running(*timed_out_identity), "one-shot handshake worker did not exit after timeout"

        @ray.remote(num_cpus=0, max_retries=0)
        def _clean_worker_state() -> tuple[int, float, bool, bool]:
            import os
            from pathlib import Path

            import psutil
            import transfer_queue

            process = psutil.Process()
            return (
                os.getpid(),
                process.create_time(),
                transfer_queue.MUTATED,
                Path(os.environ["RELAX_TEST_TQ_LATE_MARKER"]).exists(),
            )

        successor_pid, successor_create_time, successor_mutated, late_marker_exists = ray.get(
            _clean_worker_state.remote()
        )
        assert (successor_pid, successor_create_time) != timed_out_identity
        assert successor_mutated is False
        assert late_marker_exists is False
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
