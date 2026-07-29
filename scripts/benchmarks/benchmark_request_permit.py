# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU async benchmark for session-level versus request-level permits."""

import argparse
import asyncio
import json
import statistics
from dataclasses import asdict, dataclass
from time import perf_counter

from relax.engine.rollout.request_permit import RequestPermitPool


@dataclass(frozen=True)
class ScenarioResult:
    mode: str
    total_seconds: float
    short_request_seconds: float
    peak_model_requests: int
    completion_order: tuple[str, ...]


async def _run_scenario(
    mode: str,
    *,
    turns: int,
    model_seconds: float,
    environment_seconds: float,
) -> ScenarioResult:
    permits = RequestPermitPool(1)
    first_model_completed = asyncio.Event()
    active_model_requests = 0
    peak_model_requests = 0
    completion_order: list[str] = []
    short_request_seconds = 0.0

    async def model_request() -> None:
        nonlocal active_model_requests, peak_model_requests
        active_model_requests += 1
        peak_model_requests = max(peak_model_requests, active_model_requests)
        try:
            await asyncio.sleep(model_seconds)
        finally:
            active_model_requests -= 1

    async def long_session() -> None:
        if mode == "session-level":
            async with permits.acquire():
                for turn in range(turns):
                    await model_request()
                    if turn == 0:
                        first_model_completed.set()
                    await asyncio.sleep(environment_seconds)
        else:
            for turn in range(turns):
                async with permits.acquire():
                    await model_request()
                if turn == 0:
                    first_model_completed.set()
                await asyncio.sleep(environment_seconds)
        completion_order.append("long")

    async def short_session() -> None:
        nonlocal short_request_seconds
        await first_model_completed.wait()
        started = perf_counter()
        async with permits.acquire():
            await model_request()
        short_request_seconds = perf_counter() - started
        completion_order.append("short")

    started = perf_counter()
    await asyncio.gather(long_session(), short_session())
    return ScenarioResult(
        mode=mode,
        total_seconds=perf_counter() - started,
        short_request_seconds=short_request_seconds,
        peak_model_requests=peak_model_requests,
        completion_order=tuple(completion_order),
    )


async def _benchmark(args: argparse.Namespace) -> dict:
    modes = ("session-level", "request-level")
    for _ in range(args.warmups):
        for mode in modes:
            await _run_scenario(
                mode,
                turns=args.turns,
                model_seconds=args.model_ms / 1000,
                environment_seconds=args.environment_ms / 1000,
            )

    results: dict[str, list[ScenarioResult]] = {mode: [] for mode in modes}
    for run_idx in range(args.runs):
        run_modes = modes if run_idx % 2 == 0 else tuple(reversed(modes))
        for mode in run_modes:
            results[mode].append(
                await _run_scenario(
                    mode,
                    turns=args.turns,
                    model_seconds=args.model_ms / 1000,
                    environment_seconds=args.environment_ms / 1000,
                )
            )

    expected_orders = {
        "session-level": ("long", "short"),
        "request-level": ("short", "long"),
    }
    for mode, runs in results.items():
        if any(run.peak_model_requests != 1 for run in runs):
            raise RuntimeError(f"{mode} exceeded the configured model-request concurrency limit")
        if any(run.completion_order != expected_orders[mode] for run in runs):
            raise RuntimeError(f"{mode} produced an unexpected completion order")

    summary = {
        "configuration": {
            "permit_limit": 1,
            "turns": args.turns,
            "model_ms": args.model_ms,
            "environment_ms": args.environment_ms,
            "warmups": args.warmups,
            "runs": args.runs,
        },
        "results": {},
    }
    for mode, runs in results.items():
        summary["results"][mode] = {
            "median_total_ms": statistics.median(run.total_seconds for run in runs) * 1000,
            "median_short_request_ms": statistics.median(run.short_request_seconds for run in runs) * 1000,
            "peak_model_requests": max(run.peak_model_requests for run in runs),
            "completion_orders": [list(run.completion_order) for run in runs],
            "samples": [asdict(run) for run in runs],
        }
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--model-ms", type=float, default=20.0)
    parser.add_argument("--environment-ms", type=float, default=100.0)
    parser.add_argument("--json", action="store_true", help="Print the complete machine-readable result.")
    args = parser.parse_args()
    if args.runs <= 0 or args.warmups < 0 or args.turns <= 0 or args.model_ms < 0 or args.environment_ms < 0:
        parser.error("runs and turns must be positive; warmups and durations must be non-negative")
    return args


def main() -> None:
    args = _parse_args()
    summary = asyncio.run(_benchmark(args))
    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print("mode\tmedian total (ms)\tmedian short request (ms)\tpeak model requests")
    for mode in ("session-level", "request-level"):
        result = summary["results"][mode]
        print(
            f"{mode}\t{result['median_total_ms']:.2f}\t"
            f"{result['median_short_request_ms']:.2f}\t{result['peak_model_requests']}"
        )


if __name__ == "__main__":
    main()
