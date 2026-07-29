# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class DiagnosticResult:
    rule_id: str
    severity: Severity
    message: str
    fix: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
            "fix": self.fix,
            "details": self.details,
        }


@dataclass(frozen=True)
class DoctorContext:
    argv: list[str]
    args: Any | None
    parse_error: str | None
    config: dict[str, Any]
    topology: dict[str, Any]
    command: list[str]


@dataclass(frozen=True)
class DoctorReport:
    ok: bool
    argv: list[str]
    command: list[str]
    diagnostics: list[DiagnosticResult]
    config: dict[str, Any]
    topology: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "argv": self.argv,
            "command": self.command,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "config": self.config,
            "topology": self.topology,
        }
