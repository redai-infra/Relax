# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Model-source descriptors and optional provider registration."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSource:
    """Serializable description of where a model comes from."""

    uri: str
    endpoint: str | None = None
    credential_mode: str = "default"
    addressing_style: str = "auto"
    provider_name: str = "cli"

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValueError("ModelSource.uri must be non-empty")
        if self.credential_mode not in {"default", "placeholder"}:
            raise ValueError(f"Unsupported model credential mode: {self.credential_mode!r}")
        if self.addressing_style not in {"auto", "path"}:
            raise ValueError(f"Unsupported model addressing style: {self.addressing_style!r}")


@dataclass(frozen=True)
class LocalModel:
    """Node-local view of a model source."""

    source: ModelSource
    path: str
    completeness: str


@dataclass(frozen=True)
class SGLangLoadPlan:
    """Resolved model path and load format for one SGLang engine group."""

    model_path: str
    load_format: str
    source: ModelSource


ModelSourceProvider = Callable[[Sequence[str]], ModelSource | None]

_PROVIDERS: dict[str, ModelSourceProvider] = {}
_FROZEN = False


def register_model_source_provider(name: str, provider: ModelSourceProvider) -> None:
    """Register a provider before the first model-source resolution."""
    global _FROZEN

    if _FROZEN:
        raise RuntimeError(f"Model source provider registry is frozen; cannot register {name!r}")
    if not name:
        raise ValueError("Model source provider name must be non-empty")
    existing = _PROVIDERS.get(name)
    if existing is provider:
        return
    if existing is not None:
        raise RuntimeError(f"Model source provider {name!r} is already registered")
    _PROVIDERS[name] = provider


def resolve_model_source(argv: Sequence[str]) -> ModelSource | None:
    """Resolve exactly one optional provider result and freeze registration."""
    global _FROZEN

    _FROZEN = True
    resolved: list[tuple[str, ModelSource]] = []
    for name in sorted(_PROVIDERS):
        try:
            source = _PROVIDERS[name](tuple(argv))
            if source is not None and not isinstance(source, ModelSource):
                raise TypeError(f"expected ModelSource or None, got {type(source).__name__}")
        except Exception as exc:
            raise RuntimeError(f"Model source provider {name!r} failed: {exc}") from exc
        if source is not None:
            resolved.append((name, source))
    if len(resolved) > 1:
        names = ", ".join(name for name, _ in resolved)
        raise RuntimeError(f"Multiple model source providers matched: {names}")
    if not resolved:
        return None
    name, source = resolved[0]
    if source.provider_name != name:
        source = ModelSource(
            uri=source.uri,
            endpoint=source.endpoint,
            credential_mode=source.credential_mode,
            addressing_style=source.addressing_style,
            provider_name=name,
        )
    return source


def apply_model_source_to_argv(argv: Sequence[str], source: ModelSource) -> list[str]:
    """Return a copied argv with one canonical ``--hf-checkpoint`` value."""
    updated: list[str] = []
    prefix = "--hf-checkpoint="
    index = 0
    while index < len(argv):
        item = argv[index]
        if item.startswith(prefix):
            index += 1
            continue
        if item == "--hf-checkpoint":
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ValueError("--hf-checkpoint requires a value")
            index += 2
            continue
        updated.append(item)
        index += 1
    updated.extend(("--hf-checkpoint", source.uri))
    return updated
