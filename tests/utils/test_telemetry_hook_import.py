# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock


def test_telemetry_hook_import_swallows_non_import_errors(monkeypatch) -> None:
    import relax.utils as relax_utils

    logger = SimpleNamespace(warning=Mock())
    monkeypatch.setenv("RELAX_TELEMETRY_HOOK", "broken_hook")

    def fail_import(name):
        assert name == "broken_hook"
        raise RuntimeError("boom")

    monkeypatch.setattr(relax_utils.importlib, "import_module", fail_import)

    relax_utils.try_import_telemetry_hook(logger)

    logger.warning.assert_called_once()


def test_telemetry_hook_import_is_noop_when_unset(monkeypatch) -> None:
    import relax.utils as relax_utils

    monkeypatch.delenv("RELAX_TELEMETRY_HOOK", raising=False)
    import_module = Mock()
    monkeypatch.setattr(relax_utils.importlib, "import_module", import_module)

    relax_utils.try_import_telemetry_hook()

    import_module.assert_not_called()


def test_telemetry_hook_import_does_not_preload_telemetry(monkeypatch) -> None:
    import relax.utils as relax_utils

    if hasattr(relax_utils, "telemetry"):
        monkeypatch.delattr(relax_utils, "telemetry", raising=False)
    monkeypatch.delitem(sys.modules, "relax.utils.telemetry", raising=False)

    seen = {}
    monkeypatch.setenv("RELAX_TELEMETRY_HOOK", "late_hook")

    def fake_import(name):
        assert name == "late_hook"
        seen["telemetry_loaded"] = "relax.utils.telemetry" in sys.modules

    monkeypatch.setattr(relax_utils.importlib, "import_module", fake_import)

    relax_utils.try_import_telemetry_hook()

    assert seen["telemetry_loaded"] is False
