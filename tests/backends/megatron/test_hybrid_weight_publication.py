# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


pytest.importorskip("megatron", reason="Megatron is an optional test dependency")

from relax.backends.megatron.actor import _should_publish_hybrid_weights, _warn_if_hybrid_publication_degraded


def test_hybrid_weight_publication_interval_one_preserves_existing_behavior() -> None:
    assert all(_should_publish_hybrid_weights(step, 1, 5) for step in range(5))


def test_hybrid_weight_publication_interval_two_uses_completed_step_boundary() -> None:
    decisions = [_should_publish_hybrid_weights(step, 2, 6) for step in range(6)]

    assert decisions == [False, True, False, True, False, True]


def test_hybrid_weight_publication_forces_final_step() -> None:
    assert _should_publish_hybrid_weights(4, 3, 5)


def test_hybrid_weight_publication_forces_configured_evaluation_when_epoch_is_unknown() -> None:
    assert _should_publish_hybrid_weights(
        0,
        2,
        20,
        evaluation_configured=True,
        eval_interval=10,
        num_rollout_per_epoch=None,
    )


def test_hybrid_weight_publication_uses_known_evaluation_boundaries() -> None:
    assert not _should_publish_hybrid_weights(
        2,
        5,
        20,
        evaluation_configured=True,
        eval_interval=10,
        num_rollout_per_epoch=4,
    )
    assert _should_publish_hybrid_weights(
        3,
        5,
        20,
        evaluation_configured=True,
        eval_interval=10,
        num_rollout_per_epoch=4,
    )


def test_hybrid_weight_publication_non_global_dataset_only_uses_eval_interval() -> None:
    assert not _should_publish_hybrid_weights(
        0,
        5,
        20,
        evaluation_configured=True,
        eval_interval=10,
        num_rollout_per_epoch=None,
        evaluation_at_epoch_boundary=False,
    )
    assert _should_publish_hybrid_weights(
        9,
        20,
        20,
        evaluation_configured=True,
        eval_interval=10,
        num_rollout_per_epoch=None,
        evaluation_at_epoch_boundary=False,
    )


def test_hybrid_weight_publication_rejects_nonpositive_interval() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _should_publish_hybrid_weights(0, 0, 20)


def test_hybrid_weight_publication_warns_once_when_evaluation_forces_interval_one(caplog) -> None:
    args = type(
        "Args",
        (),
        {
            "eval_interval": 10,
            "eval_prompt_data": ["aime", "/data/aime.jsonl"],
            "num_rollout_per_epoch": None,
            "update_weights_interval": 2,
            "rollout_global_dataset": True,
        },
    )()

    warned = _warn_if_hybrid_publication_degraded(args, already_warned=False)
    warned = _warn_if_hybrid_publication_degraded(args, already_warned=warned)

    assert warned is True
    assert caplog.text.count("Hybrid weight publication interval is disabled") == 1


def test_hybrid_weight_publication_non_global_dataset_does_not_warn(caplog) -> None:
    args = SimpleNamespace(
        eval_interval=10,
        eval_prompt_data=["aime", "/data/aime.jsonl"],
        num_rollout_per_epoch=None,
        update_weights_interval=2,
        rollout_global_dataset=False,
    )

    warned = _warn_if_hybrid_publication_degraded(args, already_warned=False)

    assert warned is False
    assert "Hybrid weight publication interval is disabled" not in caplog.text


def test_hybrid_weight_publication_resumes_rollout_when_recovery_fails(monkeypatch) -> None:
    from relax.backends.megatron import actor as actor_module

    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(true_on_policy_mode=False, hybrid=True, rollout_http_timeout=30)
    actor._end_rollout_weight_update = Mock()

    can_update_response = Mock()
    can_update_response.json.return_value = 1
    recover_response = Mock()
    recover_response.raise_for_status.side_effect = RuntimeError("recover failed")
    request_get = Mock(side_effect=[can_update_response, recover_response])

    monkeypatch.setattr(actor_module, "get_serve_url", lambda role: "http://rollout")
    monkeypatch.setattr(actor_module.requests, "get", request_get)
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(actor_module.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: None)

    rollout_only, actor_fwd_only = actor._check_services_health(7)

    assert rollout_only is True
    assert actor_fwd_only is True
    actor._end_rollout_weight_update.assert_called_once_with("http://rollout", 7)
    assert request_get.call_args_list[0].kwargs["timeout"] == 30
    assert "timeout" not in request_get.call_args_list[1].kwargs


def test_hybrid_weight_publication_resumes_when_pause_result_is_uncertain(monkeypatch) -> None:
    from relax.backends.megatron import actor as actor_module

    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(true_on_policy_mode=False, hybrid=True, rollout_http_timeout=30)
    actor._end_rollout_weight_update = Mock()

    can_update_response = Mock()
    can_update_response.raise_for_status.side_effect = RuntimeError("pause failed")

    monkeypatch.setattr(actor_module, "get_serve_url", lambda role: "http://rollout")
    monkeypatch.setattr(actor_module.requests, "get", Mock(return_value=can_update_response))
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(actor_module.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: None)

    _, actor_fwd_only = actor._check_services_health(7)

    assert actor_fwd_only is True
    actor._end_rollout_weight_update.assert_called_once_with("http://rollout", 7)


def test_hybrid_weight_publication_does_not_resume_before_pause_request(monkeypatch) -> None:
    from relax.backends.megatron import actor as actor_module

    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(true_on_policy_mode=False, hybrid=True, rollout_http_timeout=30)
    actor._end_rollout_weight_update = Mock()

    monkeypatch.setattr(actor_module, "get_serve_url", Mock(side_effect=RuntimeError("url unavailable")))
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(actor_module.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: None)

    _, actor_fwd_only = actor._check_services_health(7)

    assert actor_fwd_only is True
    actor._end_rollout_weight_update.assert_not_called()


def test_end_rollout_weight_update_retries_then_succeeds(monkeypatch) -> None:
    from relax.backends.megatron import actor as actor_module

    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(rollout_http_timeout=30)
    failed_response = Mock()
    failed_response.raise_for_status.side_effect = RuntimeError("temporary failure")
    successful_response = Mock()
    request_get = Mock(side_effect=[failed_response, failed_response, successful_response])
    monkeypatch.setattr(actor_module.requests, "get", request_get)
    monkeypatch.setattr(actor_module.time, "sleep", Mock())

    actor._end_rollout_weight_update("http://rollout", 7)

    assert request_get.call_count == 3


def test_end_rollout_weight_update_raises_after_retries(monkeypatch) -> None:
    from relax.backends.megatron import actor as actor_module

    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(rollout_http_timeout=30)
    failed_response = Mock()
    failed_response.raise_for_status.side_effect = RuntimeError("persistent failure")
    monkeypatch.setattr(actor_module.requests, "get", Mock(return_value=failed_response))
    monkeypatch.setattr(actor_module.time, "sleep", Mock())

    with pytest.raises(RuntimeError, match="Failed to resume rollout"):
        actor._end_rollout_weight_update("http://rollout", 7)


def test_end_rollout_weight_update_retries_url_resolution(monkeypatch) -> None:
    from relax.backends.megatron import actor as actor_module

    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(rollout_http_timeout=30)
    get_url = Mock(side_effect=[RuntimeError("discovery failed"), RuntimeError("discovery failed"), "http://rollout"])
    monkeypatch.setattr(actor_module, "get_serve_url", get_url)
    monkeypatch.setattr(actor_module.requests, "get", Mock(return_value=Mock()))
    monkeypatch.setattr(actor_module.time, "sleep", Mock())

    actor._end_rollout_weight_update(None, 7)

    assert get_url.call_count == 3


def test_normal_resume_failure_is_synchronized_before_raising(monkeypatch) -> None:
    from relax.backends.megatron import actor as actor_module

    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor._end_rollout_weight_update = Mock(side_effect=RuntimeError("resume failed"))
    all_reduce = Mock()
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(actor_module.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: None)

    with pytest.raises(RuntimeError, match="Failed to resume Rollout"):
        actor._resume_rollout_weight_update("http://rollout", 7)

    all_reduce.assert_called_once()


def test_nonzero_rank_raises_when_normal_resume_failure_is_broadcast(monkeypatch) -> None:
    from relax.backends.megatron import actor as actor_module

    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor._end_rollout_weight_update = Mock()

    def broadcast_failure(flag, **kwargs):
        flag[0] = 1

    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(actor_module.dist, "all_reduce", broadcast_failure)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: None)

    with pytest.raises(RuntimeError, match="Failed to resume Rollout"):
        actor._resume_rollout_weight_update(None, 7)

    actor._end_rollout_weight_update.assert_not_called()


def test_weight_update_failure_resumes_and_aborts_all_ranks(monkeypatch) -> None:
    from relax.backends.megatron import actor as actor_module

    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor._resume_rollout_weight_update = Mock()
    update = Mock(side_effect=RuntimeError("update failed"))
    all_reduce = Mock()
    monkeypatch.setattr(actor_module.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: None)

    with pytest.raises(RuntimeError, match="Weight update failed"):
        actor._run_weight_update_with_resume_on_error(7, update)

    actor._resume_rollout_weight_update.assert_called_once_with(None, 7)
    all_reduce.assert_called_once()


def test_successful_rank_resumes_when_peer_weight_update_fails(monkeypatch) -> None:
    from relax.backends.megatron import actor as actor_module

    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor._resume_rollout_weight_update = Mock()
    update = Mock()

    def broadcast_failure(flag, **kwargs):
        flag[0] = 1

    monkeypatch.setattr(actor_module.dist, "all_reduce", broadcast_failure)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: None)

    with pytest.raises(RuntimeError, match="Weight update failed"):
        actor._run_weight_update_with_resume_on_error(7, update)

    update.assert_called_once_with()
    actor._resume_rollout_weight_update.assert_called_once_with(None, 7)


def test_hybrid_weight_publication_aborts_all_ranks_when_resume_fails(monkeypatch) -> None:
    from relax.backends.megatron import actor as actor_module

    actor = actor_module.MegatronTrainRayActor.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(true_on_policy_mode=False, hybrid=True, rollout_http_timeout=30)
    actor._end_rollout_weight_update = Mock(side_effect=RuntimeError("resume failed"))

    can_update_response = Mock()
    can_update_response.raise_for_status.side_effect = RuntimeError("pause result lost")
    all_reduce = Mock()
    monkeypatch.setattr(actor_module, "get_serve_url", lambda role: "http://rollout")
    monkeypatch.setattr(actor_module.requests, "get", Mock(return_value=can_update_response))
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(actor_module.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: None)

    with pytest.raises(RuntimeError, match="Failed to resume Rollout"):
        actor._check_services_health(7)

    all_reduce.assert_called_once()
