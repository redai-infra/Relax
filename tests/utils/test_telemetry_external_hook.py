# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import importlib
import json
import os
import sys

import pytest

from relax.utils import telemetry


def test_external_telemetry_hook_writes_jsonl_dump(tmp_path, monkeypatch) -> None:
    hook_name = os.environ.get("RELAX_TELEMETRY_HOOK")
    log_env_name = os.environ.get("RELAX_TELEMETRY_TEST_LOG_ENV")
    expected_flags = os.environ.get("RELAX_TELEMETRY_TEST_EXPECTED_FLAGS")
    if not hook_name or not log_env_name or not expected_flags:
        pytest.skip("set external telemetry hook test env to run integration")

    log_file = tmp_path / "relax-telemetry.jsonl"
    monkeypatch.setenv(log_env_name, str(log_file))
    telemetry.register_backend(None)

    for name in list(sys.modules):
        if name == hook_name or name.startswith(f"{hook_name}."):
            sys.modules.pop(name, None)

    importlib.import_module(hook_name)

    try:
        telemetry.mark_start(fields={"train_batch_size": 2}, run_name="cpu_integration")
        with telemetry.step(1, role="actor", tokens=128) as span:
            span.update(mfu=0.42, flops=123456789.0)
        telemetry.mark_end(status="success")

        raw_dump = log_file.read_text()
        print("RELAX_TELEMETRY_JSONL_DUMP_START")
        print(raw_dump.rstrip())
        print("RELAX_TELEMETRY_JSONL_DUMP_END")

        events = [json.loads(line) for line in raw_dump.splitlines()]
    finally:
        telemetry.register_backend(None)

    assert [str(event["flag"]) for event in events] == expected_flags.split(",")
    assert events[0]["args"]["framework"] == "relax"
    assert events[0]["args"]["run_name"] == "cpu_integration"
    assert events[1]["args"]["role"] == "actor"
    assert events[1]["args"]["tokens"] == 128
    assert events[2]["args"]["mfu"] == 0.42
    assert events[2]["args"]["flops"] == 123456789.0
    assert events[3]["args"]["status"] == "success"
