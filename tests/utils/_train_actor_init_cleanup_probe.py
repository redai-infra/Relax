# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Subprocess probe for train-actor cleanup after one rank fails init."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path


def main(probe_dir: Path) -> None:
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    os.environ.pop("RAY_ADDRESS", None)

    import ray

    from relax.distributed.ray.actor_group import RayTrainGroup

    probe_dir.mkdir(parents=True, exist_ok=True)
    late_marker = probe_dir / "late-mutation"

    @ray.remote(max_restarts=0)
    class _InitActor:
        def __init__(self, fail: bool):
            self.fail = fail

        def init(self, _args, _role, **_kwargs):
            if not self.fail:
                return "ready"

            def mutate_late():
                time.sleep(2.0)
                late_marker.write_text("dirty", encoding="utf-8")

            threading.Thread(target=mutate_late, daemon=True).start()
            raise RuntimeError("expected initialization failure")

        def termination_probe(self):
            threading.Event().wait()

    ray.init(
        address="local",
        num_cpus=2,
        include_dashboard=False,
        logging_level="ERROR",
        _temp_dir=str(probe_dir / "ray"),
    )
    try:
        actors = [_InitActor.remote(True), _InitActor.remote(False)]
        group = object.__new__(RayTrainGroup)
        group._actor_handlers = actors

        try:
            group.init_and_wait(object(), "actor")
        except ray.exceptions.RayTaskError:
            pass
        else:
            raise AssertionError("rank initialization failure was not propagated")

        assert group._actor_handlers == []
        time.sleep(2.2)
        assert not late_marker.exists(), "killed actor completed a delayed process-global mutation"
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
