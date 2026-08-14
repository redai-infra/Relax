# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Import-time megatron stubs for CPU-only P3O tests.

``relax.backends.megatron.{loss,model,p3o_step}`` import ``megatron.core`` at
module scope, but CI installs no megatron package (see
``.github/workflows/ci.yml``). The P3O logic under test is pure tensor math plus
collectives, so the megatron surface can be replaced by ``MagicMock`` for the
duration of the import.

Without this, the four P3O test modules raise ``ModuleNotFoundError`` during
collection, and because CI runs ``pytest tests/ -x`` that aborts the *entire*
suite rather than skipping a few tests.

Stubbing only spans the ``with`` block: the previous ``sys.modules`` entries are
restored afterwards, so a real megatron install is never shadowed and these
tests exercise the same code path on a GPU machine.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
from unittest.mock import MagicMock


#: Top-level packages replaced while the context manager is active. Any
#: submodule below these is synthesized on demand, so the P3O import chain does
#: not have to be enumerated here.
STUBBED_ROOTS = ("megatron",)
_MISSING = object()


@contextmanager
def temporarily_stub_module(name: str, module: ModuleType) -> Iterator[None]:
    """Expose one import stub without resetting unrelated import-cache entries.

    ``unittest.mock.patch.dict(sys.modules, ...)`` restores the complete module
    cache when its context exits. Imports performed inside that context can
    include native Torch or Ray modules, so clearing them makes a later import
    reinitialize process-global extension state. Preserve and restore only the
    requested entry instead.
    """
    previous_module = sys.modules.get(name, _MISSING)
    sys.modules[name] = module
    try:
        yield
    finally:
        if previous_module is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous_module


class _MagicModule(ModuleType):
    """Module whose unknown attributes resolve to ``MagicMock``.

    A plain ``MagicMock`` cannot stand in for a package -- ``import a.b`` fails
    with "is not a package" -- so a real module object is used and attribute
    lookup is delegated to a mock.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__path__: list[str] = []
        self._mock = MagicMock(name=name)

    def __getattr__(self, item: str) -> object:
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        return getattr(self._mock, item)


class _StubLoader(Loader):
    def create_module(self, spec: ModuleSpec) -> ModuleType:
        return _MagicModule(spec.name)

    def exec_module(self, module: ModuleType) -> None:  # noqa: D102 - nothing to execute
        return None


class _StubFinder(MetaPathFinder):
    """Resolve any ``<root>`` or ``<root>.*`` name to a synthetic module."""

    def __init__(self, roots: tuple[str, ...]) -> None:
        self._roots = roots

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> ModuleSpec | None:
        root = fullname.split(".", 1)[0]
        if root not in self._roots:
            return None
        return ModuleSpec(fullname, _StubLoader(), is_package=True)


@contextmanager
def stubbed_megatron_modules(roots: tuple[str, ...] = STUBBED_ROOTS) -> Iterator[None]:
    """Make ``megatron`` importable as a stub, restoring prior state on exit.

    No-op for roots that are genuinely installed, so a GPU machine with real
    megatron exercises the production import path unchanged.
    """
    missing = tuple(root for root in roots if _is_missing(root))
    if not missing:
        yield
        return

    finder = _StubFinder(missing)
    sys.meta_path.insert(0, finder)
    created_before = set(sys.modules)
    try:
        yield
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        for name in set(sys.modules) - created_before:
            if isinstance(sys.modules.get(name), _MagicModule):
                del sys.modules[name]


def _is_missing(root: str) -> bool:
    if root in sys.modules:
        return False
    try:
        from importlib.util import find_spec

        return find_spec(root) is None
    except (ImportError, ValueError, ModuleNotFoundError):
        return True
