# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for HTTP timeouts on the actor's rollout/actor_fwd service probes.

Regression target (deadlock): the actor's synchronous ``requests.get`` probes to
the rollout service had no ``timeout``. If the rollout service stalled (e.g. an
engine died mid weight-update handshake), the actor blocked forever on the
socket read. Fix#2 bounds these handshake/probe requests with the configurable
``rollout_http_timeout`` argument.

``/recover_rollout_engines`` is a long-running engine rebuild, so it uses a
separate, larger finite timeout rather than the short probe timeout.
"""

import pytest
import requests


pytest.importorskip("megatron.core")

try:
    from relax.backends.megatron import actor as actor_mod
except (ImportError, AssertionError) as _exc:
    pytest.skip(f"relax.backends.megatron.actor unavailable: {_exc}", allow_module_level=True)


MegatronTrainRayActor = actor_mod.MegatronTrainRayActor


class _Resp:
    def __init__(self, payload=False):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _record_get(calls, *, payload=False):
    def _get(url, *args, **kwargs):
        calls.append((url, kwargs))
        return _Resp(payload)

    return _get


def _shell():
    from argparse import Namespace

    shell = object.__new__(MegatronTrainRayActor)
    shell.args = Namespace(rollout_http_timeout=120.0, rollout_engine_init_timeout=3600.0)
    return shell


def test_weight_update_transaction_ids_are_monotonic_within_actor_session():
    shell = _shell()

    first = shell._new_rollout_weight_update_transaction_id()
    second = shell._new_rollout_weight_update_transaction_id()

    first_version, first_session, first_sequence = first.split(":")
    second_version, second_session, second_sequence = second.split(":")
    assert first_version == second_version == "relax-v1"
    assert first_session == second_session
    assert (first_sequence, second_sequence) == ("0", "1")


def _patch_resume_collective(monkeypatch):
    """Keep unit tests independent from an initialized distributed process."""
    monkeypatch.setattr(actor_mod.dist, "all_reduce", lambda *_a, **_k: None)
    monkeypatch.setattr(actor_mod, "get_gloo_group", lambda: None)


# ---------------------------------------------------------------------------
# _run_step_evaluation: /evaluate + /end_update_weight
# ---------------------------------------------------------------------------
def test_run_step_evaluation_probes_carry_timeout(monkeypatch):
    from argparse import Namespace

    calls = []
    monkeypatch.setattr(actor_mod, "get_serve_url", lambda *_a, **_k: "http://svc:0")
    monkeypatch.setattr(actor_mod.requests, "get", _record_get(calls))
    monkeypatch.setattr(actor_mod.dist, "get_rank", lambda *_a, **_k: 0)
    monkeypatch.setattr(actor_mod, "is_sft_mode", lambda _args: False)
    _patch_resume_collective(monkeypatch)

    shell = _shell()
    shell.args = Namespace(rollout_http_timeout=17.0, rollout_engine_init_timeout=3600.0)
    shell.rollout_manager = object()  # non-None -> has_rollout

    shell._run_step_evaluation(3, end_update_weight=True)

    by_url = dict(calls)
    assert any(u.endswith("/evaluate") for u in by_url)
    assert any(u.endswith("/end_update_weight") for u in by_url)
    for url, kwargs in calls:
        assert kwargs.get("timeout") == shell.args.rollout_http_timeout, url


def test_run_step_evaluation_ends_update_weight_even_if_evaluate_fails(monkeypatch):
    # Regression: /evaluate failing must NOT skip the /end_update_weight cleanup,
    # otherwise rollout + health monitoring stay paused forever.

    calls = []

    def _get(url, *args, **kwargs):
        calls.append(url)
        if url.endswith("/evaluate"):
            raise requests.exceptions.HTTPError("eval boom")
        return _Resp()

    monkeypatch.setattr(actor_mod, "get_serve_url", lambda *_a, **_k: "http://svc:0")
    monkeypatch.setattr(actor_mod.requests, "get", _get)
    monkeypatch.setattr(actor_mod.dist, "get_rank", lambda *_a, **_k: 0)
    monkeypatch.setattr(actor_mod, "is_sft_mode", lambda _args: False)
    _patch_resume_collective(monkeypatch)

    shell = _shell()
    shell.rollout_manager = object()  # non-None -> has_rollout

    shell._run_step_evaluation(3, end_update_weight=True)

    assert any(u.endswith("/evaluate") for u in calls)
    assert any(u.endswith("/end_update_weight") for u in calls), "cleanup must run even when /evaluate fails"


def test_run_step_evaluation_swallows_timeout(monkeypatch):

    def _timeout_get(url, *_a, **_k):
        if url.endswith("/evaluate"):
            raise requests.exceptions.Timeout("boom")
        return _Resp()

    monkeypatch.setattr(actor_mod, "get_serve_url", lambda *_a, **_k: "http://svc:0")
    monkeypatch.setattr(actor_mod.requests, "get", _timeout_get)
    monkeypatch.setattr(actor_mod.dist, "get_rank", lambda *_a, **_k: 0)
    monkeypatch.setattr(actor_mod, "is_sft_mode", lambda _args: False)
    _patch_resume_collective(monkeypatch)

    shell = _shell()
    shell.rollout_manager = object()

    # Evaluation failure is non-fatal, but cleanup must still complete.
    shell._run_step_evaluation(3, end_update_weight=True)


# ---------------------------------------------------------------------------
# _check_services_health: /can_do..., /recover..., /get_step
# ---------------------------------------------------------------------------
def _patch_health_common(monkeypatch, get_fn):
    monkeypatch.setattr(actor_mod, "get_serve_url", lambda *_a, **_k: "http://svc:0")
    monkeypatch.setattr(actor_mod.requests, "get", get_fn)
    monkeypatch.setattr(actor_mod.dist, "get_rank", lambda *_a, **_k: 0)
    monkeypatch.setattr(actor_mod.dist, "all_reduce", lambda *_a, **_k: None)
    monkeypatch.setattr(actor_mod, "get_gloo_group", lambda *_a, **_k: None)
    monkeypatch.setattr(actor_mod.time, "sleep", lambda *_a, **_k: None)


def test_check_services_health_can_do_and_get_step_carry_timeout(monkeypatch):
    from argparse import Namespace

    calls = []
    # payload truthy -> can_do returns 1 so the while loop proceeds to recover.
    _patch_health_common(monkeypatch, _record_get(calls, payload=True))

    shell = _shell()
    shell.args = Namespace(
        true_on_policy_mode=False,
        hybrid=False,
        rollout_http_timeout=17.0,
        rollout_engine_init_timeout=1800.0,
    )

    shell._check_services_health()

    can_do = [(u, k) for u, k in calls if u.endswith("/can_do_update_weight_for_async")]
    recover = [(u, k) for u, k in calls if u.endswith("/recover_rollout_engines")]
    get_step = [(u, k) for u, k in calls if u.endswith("/get_step")]

    assert can_do and all(k.get("timeout") == shell.args.rollout_http_timeout for _u, k in can_do)
    assert get_step and all(k.get("timeout") == shell.args.rollout_http_timeout for _u, k in get_step)

    assert recover and all(k.get("timeout") == shell.args.rollout_engine_init_timeout for _u, k in recover)


def test_check_services_health_aborts_when_pause_resume_is_uncertain(monkeypatch):
    from argparse import Namespace

    def _timeout_get(url, *_a, **_k):
        if url.endswith(("/can_do_update_weight_for_async", "/end_update_weight")):
            raise requests.exceptions.Timeout("boom")
        return _Resp()

    _patch_health_common(monkeypatch, _timeout_get)

    shell = _shell()
    shell.args = Namespace(
        true_on_policy_mode=False,
        hybrid=False,
        rollout_http_timeout=120.0,
        rollout_engine_init_timeout=3600.0,
    )

    # A timeout after the pause request is sent leaves the rollout state
    # uncertain. If cancellation/resume also cannot be confirmed, continuing
    # could leave Rollout paused after the Actor has moved on.
    with pytest.raises(RuntimeError, match="Failed to resume Rollout"):
        shell._check_services_health()


def test_check_services_health_degrades_when_recovery_times_out_but_resume_succeeds(monkeypatch):
    from argparse import Namespace
    from unittest.mock import Mock

    def _get(url, *_args, **_kwargs):
        if url.endswith("/can_do_update_weight_for_async"):
            return _Resp(True)
        if url.endswith("/recover_rollout_engines"):
            raise requests.exceptions.Timeout("recovery timed out")
        return _Resp()

    _patch_health_common(monkeypatch, _get)

    shell = _shell()
    shell.args = Namespace(
        true_on_policy_mode=False,
        hybrid=True,
        rollout_http_timeout=120.0,
        rollout_engine_init_timeout=3600.0,
    )
    shell._end_rollout_weight_update = Mock()

    rollout_only, actor_fwd_only = shell._check_services_health()

    assert rollout_only is True
    assert actor_fwd_only is True
    shell._end_rollout_weight_update.assert_called_once()
