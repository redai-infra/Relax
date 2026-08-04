#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""No-GPU contract check for the Task 22 SGLang calibration dependency."""

from __future__ import annotations

import inspect
import math
import pickle
from types import SimpleNamespace


def main() -> None:
    from sglang.srt.managers import scheduler_runtime_checker_mixin
    from sglang.srt.observability import scheduler_metrics_mixin
    from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats
    from sglang.srt.utils import request_logger, scheduler_status_logger

    scheduler = SchedulerReqTimeStats()
    scheduler.enable_metrics = True
    scheduler.wait_queue_entry_time = 100.0
    scheduler.forward_entry_time = 101.25
    scheduler.prefill_finished_time = 103.5
    second_hop = pickle.loads(pickle.dumps(pickle.loads(pickle.dumps(scheduler))))
    timing = second_hop.convert_to_output_meta_info()
    if not math.isclose(timing.get("queue_time", -1.0), 1.25):
        raise RuntimeError("SGLang two-hop queue timing transport is unavailable")
    if timing.get("prefill_finished_time", 0.0) <= timing.get("forward_entry_time", 0.0):
        raise RuntimeError("SGLang two-hop prefill timing transport is unavailable")

    status_source = inspect.getsource(scheduler_status_logger.SchedulerStatusLogger)
    metrics_source = inspect.getsource(scheduler_metrics_mixin.SchedulerMetricsMixin)
    required_fields = (
        "timestamp_epoch",
        "idle",
        "decode_tokens_cumulative",
        "prefill_tokens_cumulative",
        "cached_tokens_cumulative",
        "forward_mode",
        "running_rids",
        "running_seq_lens",
        "running_origin_input_lens",
        "running_output_lens",
        "queued_rids",
        "queued_origin_input_lens",
        "queued_output_lens",
    )
    missing = [field for field in required_fields if field not in status_source]
    if missing:
        raise RuntimeError(f"SGLang scheduler shape observability is incomplete: {missing}")
    if not any(
        marker in metrics_source
        for marker in ("RELAX_REQUEST_SHAPE_OBSERVABILITY", "TASK22_REQUEST_SHAPE_PREFILL_STATUS")
    ):
        raise RuntimeError("SGLang scheduler status logger is not called from the scheduler hot path")
    if "decode_tokens=decode_tokens" not in metrics_source:
        raise RuntimeError("SGLang exact decode-token accumulation is not connected to the decode hot path")
    if "prefill_tokens=prefill_stats.log_input_tokens" not in metrics_source:
        raise RuntimeError("SGLang exact prefill-token accumulation is not connected to the prefill hot path")
    if "cached_tokens=prefill_stats.log_hit_tokens" not in metrics_source:
        raise RuntimeError("SGLang exact cached-token accumulation is not connected to the prefill hot path")

    idle_source = inspect.getsource(scheduler_runtime_checker_mixin.SchedulerRuntimeCheckerMixin.on_idle)
    if "RELAX_SCHEDULER_IDLE_HEARTBEAT" not in idle_source or "maybe_dump(None" not in idle_source:
        raise RuntimeError("SGLang idle-transition state is not connected to scheduler idle handling")
    request_logger_source = inspect.getsource(request_logger.RequestLogger)
    if "RELAX_RID_ONLY_REQUEST_LOGGING" not in request_logger_source:
        raise RuntimeError("SGLang request logger lacks RID-only calibration mode")

    events: list[dict] = []
    original_log_json = scheduler_status_logger.log_json
    scheduler_status_logger.log_json = lambda _loggers, _event, payload: events.append(payload)
    try:
        status = scheduler_status_logger.SchedulerStatusLogger(targets=[], dump_interval=3600.0)
        request = SimpleNamespace(
            rid="smoke-rid",
            seqlen=8,
            origin_input_ids=[1, 2, 3],
            output_ids=[4, 5],
        )
        batch = SimpleNamespace(reqs=[request], forward_mode="DECODE", decoding_reqs=[request])
        status.maybe_dump(batch, [], decode_tokens=3, prefill_tokens=11, cached_tokens=5)
        status.maybe_dump(batch, [], decode_tokens=4, prefill_tokens=7, cached_tokens=3)
        status.maybe_dump(None, [])
    finally:
        scheduler_status_logger.log_json = original_log_json
    if len(events) != 2:
        raise RuntimeError(f"SGLang state transitions should force two dumps, got {len(events)}")
    if events[0].get("idle") is not False or events[1].get("idle") is not True:
        raise RuntimeError("SGLang active-to-idle transition state is incorrect")
    if events[1].get("decode_tokens_cumulative") != 7:
        raise RuntimeError("SGLang decode-token accumulator lost rate-limited iterations")
    if events[1].get("prefill_tokens_cumulative") != 18:
        raise RuntimeError("SGLang prefill-token accumulator lost rate-limited iterations")
    if events[1].get("cached_tokens_cumulative") != 8:
        raise RuntimeError("SGLang cached-token accumulator lost rate-limited iterations")
    if not isinstance(events[1].get("timestamp_epoch"), float):
        raise RuntimeError("SGLang scheduler status lacks an explicit epoch timestamp")
    print("TASK22_SGLANG_CALIBRATION_PREFLIGHT verdict=PASS")


if __name__ == "__main__":
    main()
