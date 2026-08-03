# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


REDACTED = "<redacted>"

_SENSITIVE_NAMES = {
    "agent_env",
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "credential",
    "credentials",
    "notify_urls",
    "password",
    "passwd",
    "private_key",
    "secret",
    "token",
    "wandb_key",
}

_STRUCTURED_SECRET_OPTIONS = {"train_env_vars"}


def sanitize_argv(argv: list[str], *, known_secrets: set[str] | None = None) -> tuple[list[str], set[str]]:
    sanitized = []
    secret_values = set(known_secrets or ())
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--agent-env":
            sanitized.append(item)
            index += 1
            while index < len(argv) and not argv[index].startswith("--"):
                sanitized_item, secret = _sanitize_env_item(argv[index])
                sanitized.append(sanitized_item)
                if secret:
                    secret_values.add(secret)
                index += 1
            continue

        if item.startswith("--") and "=" in item:
            option, value = item.split("=", 1)
            if is_sensitive_name(option):
                sanitized.append(f"{option}={REDACTED}")
                if value:
                    secret_values.add(value)
            elif _normalize_name(option) in _STRUCTURED_SECRET_OPTIONS:
                sanitized.append(f"{option}={_sanitize_json_mapping(value, secret_values)}")
            else:
                sanitized.append(item)
            index += 1
            continue

        sanitized.append(item)
        if item.startswith("--") and index + 1 < len(argv):
            value = argv[index + 1]
            if is_sensitive_name(item):
                sanitized.append(REDACTED)
                if value:
                    secret_values.add(value)
                index += 2
                continue
            if _normalize_name(item) in _STRUCTURED_SECRET_OPTIONS:
                sanitized.append(_sanitize_json_mapping(value, secret_values))
                index += 2
                continue
        index += 1
    return [sanitize_text(item, secret_values) or "" for item in sanitized], secret_values


def sanitize_config(config: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    secret_values: set[str] = set()
    sanitized = _sanitize_value(config, secret_values=secret_values)
    return sanitized, secret_values


def sanitize_text(value: str | None, secret_values: set[str]) -> str | None:
    if value is None:
        return None
    sanitized = value
    for secret in sorted(secret_values, key=len, reverse=True):
        if secret:
            sanitized = sanitized.replace(secret, REDACTED)
    return sanitized


def sanitize_details(value: Any, secret_values: set[str]) -> Any:
    sanitized = _sanitize_value(value, secret_values=set())
    return _replace_secret_literals(sanitized, secret_values)


def is_sensitive_name(name: str) -> bool:
    normalized = _normalize_name(name)
    if normalized in _SENSITIVE_NAMES:
        return True
    parts = normalized.split("_")
    if any(part in {"password", "passwd", "secret", "token", "credential", "credentials"} for part in parts):
        return True
    return "api" in parts and "key" in parts or "private" in parts and "key" in parts


def _normalize_name(name: str) -> str:
    return name.lstrip("-").replace("-", "_").lower()


def _sanitize_json_mapping(value: str, secret_values: set[str]) -> str:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return sanitize_text(value, secret_values) or ""
    if not isinstance(parsed, Mapping):
        return sanitize_text(value, secret_values) or ""
    sanitized = _sanitize_value(parsed, secret_values=secret_values)
    return json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sanitize_value(value: Any, *, secret_values: set[str], key: str | None = None) -> Any:
    if key == "agent_env":
        sanitized_items = []
        for item in value or []:
            sanitized_item, secret = _sanitize_env_item(str(item))
            sanitized_items.append(sanitized_item)
            if secret:
                secret_values.add(secret)
        return sanitized_items
    if key is not None and is_sensitive_name(key):
        _collect_scalar_secrets(value, secret_values)
        return REDACTED
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize_value(item_value, secret_values=secret_values, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize_value(item, secret_values=secret_values) for item in value]
    return value


def _sanitize_env_item(item: str) -> tuple[str, str | None]:
    if "=" not in item:
        return REDACTED, item or None
    key, value = item.split("=", 1)
    return f"{key}={REDACTED}", value or None


def _collect_scalar_secrets(value: Any, secret_values: set[str]) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _collect_scalar_secrets(item, secret_values)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _collect_scalar_secrets(item, secret_values)
    elif value is not None:
        secret_values.add(str(value))


def _replace_secret_literals(value: Any, secret_values: set[str]) -> Any:
    if isinstance(value, str):
        return sanitize_text(value, secret_values)
    if isinstance(value, Mapping):
        return {key: _replace_secret_literals(item, secret_values) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_secret_literals(item, secret_values) for item in value]
    return value
