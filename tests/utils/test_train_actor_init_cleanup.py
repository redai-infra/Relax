# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only checks for fail-closed Ray train-actor initialization."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest
import ray

from relax.distributed.ray.actor_group import RayTrainGroup


class _RemoteMethod:
    def __init__(self, result=None, error: BaseException | None = None):
        self.result = result
        self.error = error

    def remote(self, *args, **kwargs):
        if self.error is not None:
            raise self.error
        return self.result


class _FakeActor:
    def __init__(self, probe_ref=None, probe_error: BaseException | None = None):
        self.termination_probe = _RemoteMethod(probe_ref, probe_error)


def _group(*actors) -> RayTrainGroup:
    group = object.__new__(RayTrainGroup)
    group._actor_handlers = list(actors)
    return group


def test_successful_initialization_preserves_actor_group(monkeypatch):
    actor = _FakeActor()
    group = _group(actor)
    refs = [object(), object()]
    values = {refs[0]: "rank-0", refs[1]: "rank-1"}
    monkeypatch.setattr(group, "async_init", lambda *args, **kwargs: refs)
    monkeypatch.setattr(ray, "wait", lambda pending, num_returns: ([pending[0]], pending[1:]))
    monkeypatch.setattr(ray, "get", lambda ref: values[ref])
    monkeypatch.setattr(ray, "kill", lambda *_args, **_kwargs: pytest.fail("success must not kill actors"))

    assert group.init_and_wait(object(), "actor") == ["rank-0", "rank-1"]
    assert group._actor_handlers == [actor]


def test_first_rank_failure_kills_and_confirms_every_actor(monkeypatch):
    actors = [_FakeActor("probe-0"), _FakeActor("probe-1")]
    group = _group(*actors)
    init_refs = ["init-0", "init-1"]
    killed = []
    monkeypatch.setattr(group, "async_init", lambda *args, **kwargs: init_refs)

    def fake_wait(refs, *, num_returns, timeout=None):
        if timeout is None:
            # rank 1 fails before rank 0 finishes; cleanup must begin without
            # waiting for rank 0.
            return ["init-1"], ["init-0"]
        return list(refs), []

    def fake_get(ref):
        if ref == "init-1":
            raise RuntimeError("rank initialization failed")
        if str(ref).startswith("probe-"):
            raise ray.exceptions.RayActorError(error_msg="actor terminated")
        pytest.fail(f"unexpected ray.get({ref!r})")

    monkeypatch.setattr(ray, "wait", fake_wait)
    monkeypatch.setattr(ray, "get", fake_get)
    monkeypatch.setattr(ray, "kill", lambda actor, no_restart: killed.append((actor, no_restart)))

    with pytest.raises(RuntimeError, match="rank initialization failed"):
        group.init_and_wait(object(), "actor")

    assert killed == [(actors[0], True), (actors[1], True)]
    assert group._actor_handlers == []


def test_kill_failure_still_attempts_every_actor_and_fails_closed(monkeypatch):
    actors = [_FakeActor("probe-0"), _FakeActor("probe-1")]
    group = _group(*actors)
    killed = []

    def fake_kill(actor, *, no_restart):
        killed.append(actor)
        if actor is actors[0]:
            raise RuntimeError("private control-plane detail")

    monkeypatch.setattr(ray, "kill", fake_kill)
    monkeypatch.setattr(ray, "wait", lambda refs, **kwargs: (list(refs), []))
    monkeypatch.setattr(
        ray,
        "get",
        lambda _ref: (_ for _ in ()).throw(ray.exceptions.RayActorError(error_msg="actor terminated")),
    )

    with pytest.raises(RuntimeError, match="Failed to confirm train actor cleanup") as excinfo:
        group._terminate_failed_init(timeout=0.1)

    assert killed == actors
    assert "private control-plane detail" not in str(excinfo.value)
    assert group._actor_handlers == actors


def test_pending_termination_probe_keeps_group_and_fails_closed(monkeypatch):
    actor = _FakeActor("probe")
    group = _group(actor)
    monkeypatch.setattr(ray, "kill", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ray, "wait", lambda refs, **kwargs: ([], list(refs)))

    with pytest.raises(RuntimeError, match=r"1 task\(s\) remained pending"):
        group._terminate_failed_init(timeout=0.1)

    assert group._actor_handlers == [actor]


def test_real_actor_failure_cannot_mutate_state_after_cleanup():
    """A killed actor process cannot complete a delayed daemon-thread write."""
    env = os.environ.copy()
    env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    env.pop("RAY_ADDRESS", None)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    with tempfile.TemporaryDirectory(prefix="train-init-ray-") as probe_dir:
        result = subprocess.run(
            [sys.executable, "-m", "tests.utils._train_actor_init_cleanup_probe", probe_dir],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    assert result.returncode == 0, f"probe stdout:\n{result.stdout}\nprobe stderr:\n{result.stderr}"
