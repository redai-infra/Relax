# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
from typing import Any

from relax.utils.doctor.models import DiagnosticResult, DoctorContext, DoctorReport
from relax.utils.doctor.rules import get_rules
from relax.utils.doctor.topology import build_topology_plan, serialize_config


def build_expected_command(argv: list[str]) -> list[str]:
    return ["python", "-m", "relax.entrypoints.train", *argv]


def run_doctor(
    *,
    argv: list[str],
    args: Any | None,
    parse_error: str | None = None,
    strict_warnings: bool = False,
) -> DoctorReport:
    config = serialize_config(args)
    topology = build_topology_plan(args)
    context = DoctorContext(
        argv=argv,
        args=args,
        parse_error=parse_error,
        config=config,
        topology=topology,
        command=build_expected_command(argv),
    )

    diagnostics: list[DiagnosticResult] = []
    for rule in get_rules():
        diagnostics.extend(rule.check(context))

    targeted_errors = any(item.severity == "error" and item.rule_id != "CONFIG_PARSE_ERROR" for item in diagnostics)
    if targeted_errors:
        diagnostics = [item for item in diagnostics if item.rule_id != "CONFIG_PARSE_ERROR"]

    if strict_warnings:
        diagnostics = [
            DiagnosticResult(
                rule_id=item.rule_id,
                severity="error" if item.severity == "warning" else item.severity,
                message=item.message,
                fix=item.fix,
                details=item.details,
            )
            for item in diagnostics
        ]

    ok = not any(item.severity == "error" for item in diagnostics)
    return DoctorReport(
        ok=ok,
        argv=argv,
        command=context.command,
        diagnostics=diagnostics,
        config=config,
        topology=topology,
    )


def render_json(report: DoctorReport) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)


def render_text(report: DoctorReport) -> str:
    lines = [
        "Relax Config Doctor",
        f"Status: {'PASS' if report.ok else 'FAIL'}",
        "",
        "Expected launch command:",
        "  " + " ".join(_shell_quote(item) for item in report.command),
        "",
        "Diagnostics:",
    ]
    if not report.diagnostics:
        lines.append("  No errors or warnings.")
    else:
        for item in report.diagnostics:
            lines.extend(
                [
                    f"  [{item.severity.upper()}] {item.rule_id}",
                    f"    {item.message}",
                    f"    Fix: {item.fix}",
                ]
            )
            if item.details:
                details = json.dumps(item.details, ensure_ascii=False, sort_keys=True)
                lines.append(f"    Details: {details}")

    lines.extend(
        [
            "",
            "Role topology:",
            _indent_json(report.topology),
            "",
            "Final merged config:",
            _indent_json(report.config),
        ]
    )
    return "\n".join(lines)


def _indent_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    return "\n".join("  " + line for line in payload.splitlines())


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=.,/:@%")
    if all(ch in safe for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"
