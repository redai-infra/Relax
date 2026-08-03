# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.utils import reloadable_process_group as rpg


def test_deferred_port_release_wait_only_sleeps_for_remaining_time(monkeypatch):
    clock = iter((10.0, 11.25))
    sleeps = []
    monkeypatch.setattr(rpg.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(rpg.time, "sleep", sleeps.append)
    rpg._port_release_deadline_by_pid.clear()

    rpg._defer_port_release_wait(pid=7, delay=2.0)
    rpg._wait_for_deferred_port_release(pid=7)

    assert sleeps == [0.75]
    assert 7 not in rpg._port_release_deadline_by_pid


def test_elapsed_deferred_wait_does_not_sleep(monkeypatch):
    clock = iter((10.0, 13.0))
    sleeps = []
    monkeypatch.setattr(rpg.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(rpg.time, "sleep", sleeps.append)
    rpg._port_release_deadline_by_pid.clear()

    rpg._defer_port_release_wait(pid=7, delay=2.0)
    rpg._wait_for_deferred_port_release(pid=7)

    assert sleeps == []


def test_later_destroy_extends_existing_deadline(monkeypatch):
    clock = iter((10.0, 10.5, 11.0))
    sleeps = []
    monkeypatch.setattr(rpg.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(rpg.time, "sleep", sleeps.append)
    rpg._port_release_deadline_by_pid.clear()

    rpg._defer_port_release_wait(pid=7, delay=2.0)
    rpg._defer_port_release_wait(pid=7, delay=2.0)
    rpg._wait_for_deferred_port_release(pid=7)

    assert sleeps == [1.5]


def test_all_process_groups_active_requires_nonempty_live_registry(monkeypatch):
    pid = 123
    monkeypatch.setattr(rpg.os, "getpid", lambda: pid)
    rpg.ReloadableProcessGroup.GROUPS[pid] = []
    assert not rpg.all_process_groups_active()

    live = type("Group", (), {"group": object()})()
    dead = type("Group", (), {"group": None})()
    rpg.ReloadableProcessGroup.GROUPS[pid] = [live]
    assert rpg.all_process_groups_active()
    rpg.ReloadableProcessGroup.GROUPS[pid] = [live, dead]
    assert not rpg.all_process_groups_active()

    del rpg.ReloadableProcessGroup.GROUPS[pid]
