# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest


pytest.importorskip("megatron")

from relax.backends.megatron import actor as actor_module


def _actor_for_sleep():
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(offload_train=True, use_critic=False, colocate=True)
    actor.role = "actor"
    actor._training_pg_state = actor_module.PG_ACTIVE
    actor._preserve_pg_for_weight_sync_ready = True
    actor._preserved_pg_canary_passed = True
    actor._torch_memory_saver_enabled = True
    actor._all_ranks_agree = lambda value: value
    return actor


def test_sleep_preserves_live_groups_only_for_immediate_weight_sync(monkeypatch):
    actor = _actor_for_sleep()
    calls = []
    monkeypatch.setattr(actor_module, "clear_memory", lambda **kwargs: calls.append("clear"))
    monkeypatch.setattr(actor_module, "print_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(actor_module, "all_process_groups_active", lambda: True)
    monkeypatch.setattr(actor_module.dist, "barrier", lambda **kwargs: calls.append("barrier"))
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: object())
    monkeypatch.setattr(actor_module, "destroy_process_groups", lambda **kwargs: calls.append("destroy"))
    monkeypatch.setattr(actor_module.torch_memory_saver, "pause", lambda: calls.append("pause"))

    actor_module.MegatronTrainRayActor.sleep.__wrapped__(actor, preserve_process_groups_for_weight_sync=True)

    assert actor._training_pg_state == actor_module.PG_PAUSED_LIVE
    assert calls == ["clear", "barrier", "pause"]


def test_sleep_falls_back_to_destroy_when_post_pause_canary_fails(monkeypatch):
    actor = _actor_for_sleep()
    actor._preserved_pg_canary_passed = False
    actor._run_preserved_pg_canary = lambda: False
    calls = []
    monkeypatch.setattr(actor_module, "clear_memory", lambda **kwargs: None)
    monkeypatch.setattr(actor_module, "print_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(actor_module, "all_process_groups_active", lambda: True)
    monkeypatch.setattr(actor_module.dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: object())
    monkeypatch.setattr(actor_module, "destroy_process_groups", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(actor_module.torch_memory_saver, "pause", lambda: None)

    actor_module.MegatronTrainRayActor.sleep.__wrapped__(actor, preserve_process_groups_for_weight_sync=True)

    assert actor._training_pg_state == actor_module.PG_DESTROYED
    assert not actor._preserve_pg_for_weight_sync_ready
    assert calls == [{}]


def test_update_exception_destroys_preserved_groups_without_cooldown(monkeypatch):
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(debug_train_only=False, debug_rollout_only=False)
    actor._training_pg_state = actor_module.PG_PAUSED_LIVE
    actor._update_weights_impl = lambda preserved: (_ for _ in ()).throw(RuntimeError("sync failed"))
    calls = []
    monkeypatch.setattr(actor_module, "destroy_process_groups", lambda **kwargs: calls.append(kwargs))

    with pytest.raises(RuntimeError, match="sync failed"):
        actor_module.MegatronTrainRayActor.update_weights.__wrapped__(actor)

    assert actor._training_pg_state == actor_module.PG_DESTROYED
    assert calls == [{"post_destroy_delay": 0}]
