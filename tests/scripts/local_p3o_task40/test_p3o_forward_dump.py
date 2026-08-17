import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.local_p3o_task40 import p3o_forward_dump  # noqa: E402


def test_install_get_batch_observers_updates_both_imported_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    def original_get_batch() -> None:
        return None

    state = p3o_forward_dump.DumpState()
    monkeypatch.setattr(p3o_forward_dump, "_STATE", state)
    model_backend = SimpleNamespace(get_batch=original_get_batch)
    p3o_step_backend = SimpleNamespace(get_batch=original_get_batch)

    p3o_forward_dump._install_get_batch_observers(model_backend, p3o_step_backend)

    assert state.original_get_batch is original_get_batch
    assert model_backend.get_batch is p3o_forward_dump._observed_get_batch
    assert p3o_step_backend.get_batch is p3o_forward_dump._observed_get_batch


def test_install_get_batch_observers_rejects_mismatched_original_aliases() -> None:
    model_backend = SimpleNamespace(get_batch=lambda: None)
    p3o_step_backend = SimpleNamespace(get_batch=lambda: None)

    with pytest.raises(RuntimeError, match="do not share"):
        p3o_forward_dump._install_get_batch_observers(model_backend, p3o_step_backend)


@pytest.mark.parametrize(
    ("use_rollout_logprobs", "captures", "expected_phase"),
    [
        (True, [], p3o_forward_dump.CAPTURE_PHASE),
        (True, ["rank00000/micro00000"], "train"),
        (False, [], "train"),
    ],
)
def test_before_train_step_selects_stats_forward_only_when_actor_forward_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
    use_rollout_logprobs: bool,
    captures: list[str],
    expected_phase: str,
) -> None:
    state = p3o_forward_dump.DumpState(capture_directories=captures)
    monkeypatch.setattr(p3o_forward_dump, "_STATE", state)
    args = SimpleNamespace(use_rollout_logprobs=use_rollout_logprobs)

    p3o_forward_dump.before_train_step(args, 0, 0, None, None, None)

    assert state.phase == expected_phase
