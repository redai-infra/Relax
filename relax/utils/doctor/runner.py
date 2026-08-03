# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
from typing import Any

from relax.utils.doctor.models import DiagnosticResult, DoctorContext, DoctorReport
from relax.utils.doctor.rules import get_rules
from relax.utils.doctor.sanitizer import (
    sanitize_argv,
    sanitize_config,
    sanitize_details,
    sanitize_text,
)
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
    config_state = "unavailable" if args is None else ("partial" if parse_error is not None else "validated")
    config, config_secrets = sanitize_config(serialize_config(args))
    safe_argv, argv_secrets = sanitize_argv(argv, known_secrets=config_secrets)
    secret_values = config_secrets | argv_secrets
    safe_parse_error = sanitize_text(parse_error, secret_values)
    topology = build_topology_plan(args) if config_state == "validated" else {}
    context = DoctorContext(
        argv=safe_argv,
        args=args,
        parse_error=safe_parse_error,
        config_state=config_state,
        config=config,
        topology=topology,
        command=build_expected_command(safe_argv),
    )

    diagnostics: list[DiagnosticResult] = []
    for rule in get_rules():
        if config_state != "validated" and not rule.supports_partial:
            continue
        try:
            diagnostics.extend(rule.check(context))
        except Exception as exc:  # noqa: BLE001 - reports must remain structured for malformed input
            diagnostics.append(
                DiagnosticResult(
                    rule_id="DOCTOR_RULE_EXECUTION_ERROR",
                    severity="error",
                    message=f"diagnostic rule {rule.rule_id} failed: {type(exc).__name__}: {exc}",
                    fix="Fix the reported input first; if the failure persists, report the rule id to Relax maintainers.",
                    details={"failed_rule_id": rule.rule_id},
                )
            )

    diagnostics = [
        DiagnosticResult(
            rule_id=item.rule_id,
            severity=item.severity,
            message=sanitize_text(item.message, secret_values) or "",
            fix=sanitize_text(item.fix, secret_values) or "",
            details=sanitize_details(item.details, secret_values),
        )
        for item in diagnostics
    ]

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
        argv=safe_argv,
        command=context.command,
        diagnostics=diagnostics,
        config_state=config_state,
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
            {
                "validated": "Final merged config:",
                "partial": "Partial parsed config:",
                "unavailable": "Config unavailable:",
            }[report.config_state],
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
