# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Evaluate MaaS Chat Completions models through the R2E-Gym Gateway."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import signal
import sys
import uuid
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

client_module = importlib.import_module("examples.nemo_gym_agentic.app.client")
protocol_module = importlib.import_module("examples.nemo_gym_agentic.app.protocol")
ClientConfig = client_module.ClientConfig
GatewayClient = client_module.GatewayClient
GatewayRequestError = client_module.GatewayRequestError
TrialCancelled = client_module.TrialCancelled
TrialFailed = client_module.TrialFailed
run_trial = client_module.run_trial
TrialRequest = protocol_module.TrialRequest


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} must be set")
    return value


def _read_tasks(path: Path, limit: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            task = json.loads(line)
            metadata = task.get("responses_create_params", {}).get("metadata", {})
            if not isinstance(metadata, dict) or "instance_dict" not in metadata:
                raise ValueError(f"{path}:{line_number} is not a prepared R2E-Gym row")
            tasks.append(task)
            if len(tasks) >= limit:
                break
    if not tasks:
        raise ValueError(f"No R2E-Gym rows found in {path}")
    return tasks


def _instance_id(task: dict[str, Any]) -> str:
    metadata = task["responses_create_params"]["metadata"]
    if metadata.get("instance_id"):
        return str(metadata["instance_id"])
    instance_dict = metadata.get("instance_dict", {})
    if isinstance(instance_dict, str):
        try:
            instance_dict = json.loads(instance_dict)
        except json.JSONDecodeError:
            instance_dict = {}
    if isinstance(instance_dict, dict) and instance_dict.get("instance_id"):
        return str(instance_dict["instance_id"])
    return "unknown"


def _trial_payload(
    task: dict[str, Any],
    *,
    request_id: str,
    model_base_url: str,
    model: str,
    api_key: str,
    user_email: str,
    app_id: str,
    max_tokens: int,
    temperature: float,
    deadline_s: float,
    lease_s: float,
) -> dict[str, Any]:
    return {
        "protocol_version": "relax-nemo-gym/v1",
        "request_id": request_id,
        "session": {
            "session_id": uuid.uuid4().hex,
            "group_id": "r2e-gym-maas-eval",
            "rollout_mode": "eval",
            "attempt": 1,
        },
        "environment": {
            "name": "r2e_gym",
            "config": "r2e-gym-v1",
            "task": task,
        },
        "model_endpoint": {
            "base_url": model_base_url.rstrip("/"),
            "api_key": api_key,
            "model": model,
            "api_key_header": "api-key",
            "api_key_prefix": "",
            "headers": {
                "x-maas-user-email": user_email,
                "x-maas-app-id": app_id,
            },
        },
        "generation": {
            "sampling_params": {
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        },
        "interrupt_policy": "protected",
        "deadline_s": deadline_s,
        "lease_s": lease_s,
        "metadata": {
            "integration": "maas-eval",
            "instance_id": _instance_id(task),
            "capture_artifacts": True,
        },
    }


async def _run_trial(
    client: GatewayClient,
    semaphore: asyncio.Semaphore,
    stop_event: asyncio.Event,
    *,
    index: int,
    task: dict[str, Any],
    model_base_url: str,
    model: str,
    api_key: str,
    user_email: str,
    app_id: str,
    max_tokens: int,
    temperature: float,
    deadline_s: float,
    lease_s: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    request_id = f"r2e-maas-{uuid.uuid4().hex}"
    instance_id = _instance_id(task)
    async with semaphore:
        try:
            request = TrialRequest.from_payload(
                _trial_payload(
                    task,
                    request_id=request_id,
                    model_base_url=model_base_url,
                    model=model,
                    api_key=api_key,
                    user_email=user_email,
                    app_id=app_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    deadline_s=deadline_s,
                    lease_s=lease_s,
                )
            )
            result = await run_trial(
                client=client,
                request=request,
                stop_event=stop_event,
                poll_interval_s=poll_interval_s,
            )
            artifact = _read_artifact(result.artifact_ref)
            return {
                "index": index,
                "instance_id": instance_id,
                "request_id": request_id,
                "result": {
                    "status": result.status.value,
                    "reward": result.reward,
                    "metrics": result.metrics,
                    "artifact_ref": result.artifact_ref,
                    "error": result.error,
                },
                "artifact": artifact,
            }
        except (GatewayRequestError, TrialCancelled, TrialFailed) as exc:
            return {
                "index": index,
                "instance_id": instance_id,
                "request_id": request_id,
                "result": {
                    "status": "client_error",
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                },
            }


def _read_artifact(artifact_ref: str | None) -> dict[str, Any] | None:
    if artifact_ref is None:
        return None
    artifact_path = Path(artifact_ref)
    if not artifact_path.is_absolute() or not artifact_path.is_file():
        return None
    if artifact_path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"Artifact manifest is unexpectedly large: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact manifest must contain a JSON object: {artifact_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--task-jsonl", type=Path, required=True)
    parser.add_argument("--model-base-url", default=os.environ.get("MAAS_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("MAAS_MODEL"))
    parser.add_argument("--limit", type=_positive_int, default=1)
    parser.add_argument("--max-concurrency", type=_positive_int, default=1)
    parser.add_argument("--max-tokens", type=_positive_int, default=8192)
    parser.add_argument("--temperature", type=_non_negative_float, default=0.7)
    parser.add_argument("--deadline-s", type=_positive_float, default=7200.0)
    parser.add_argument("--lease-s", type=_positive_float, default=120.0)
    parser.add_argument("--poll-interval-s", type=_positive_float, default=2.0)
    parser.add_argument("--output-jsonl", type=Path)
    args = parser.parse_args()
    if not args.model_base_url:
        parser.error("--model-base-url or MAAS_BASE_URL must be set")
    if not args.model:
        parser.error("--model or MAAS_MODEL must be set")
    return args


async def async_main(args: argparse.Namespace) -> list[dict[str, Any]]:
    api_key = _required_env("MAAS_API_KEY")
    user_email = _required_env("MAAS_USER_EMAIL")
    app_id = _required_env("MAAS_APP_ID")
    tasks = _read_tasks(args.task_jsonl, args.limit)
    gateway_url = args.gateway_url.rstrip("/")
    semaphore = asyncio.Semaphore(args.max_concurrency)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_name, stop_event.set)
            installed_signals.append(signal_name)
        except NotImplementedError:
            pass

    output_handle = None
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.output_jsonl.open("w", encoding="utf-8")
        os.chmod(args.output_jsonl, 0o600)

    client_config = ClientConfig(
        gateway_url=gateway_url,
        poll_interval_s=args.poll_interval_s,
    )
    results: list[dict[str, Any]] = []
    try:
        async with GatewayClient(client_config) as client:
            pending = [
                _run_trial(
                    client,
                    semaphore,
                    stop_event,
                    index=index,
                    task=task,
                    model_base_url=args.model_base_url,
                    model=args.model,
                    api_key=api_key,
                    user_email=user_email,
                    app_id=app_id,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    deadline_s=args.deadline_s,
                    lease_s=args.lease_s,
                    poll_interval_s=args.poll_interval_s,
                )
                for index, task in enumerate(tasks)
            ]
            for completed in asyncio.as_completed(pending):
                result = await completed
                results.append(result)
                if output_handle is not None:
                    output_handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                    output_handle.flush()
    finally:
        if output_handle is not None:
            output_handle.close()
        for signal_name in installed_signals:
            loop.remove_signal_handler(signal_name)
    return results


def _scalar_reward(result: dict[str, Any]) -> float | None:
    reward = result.get("reward")
    if isinstance(reward, bool):
        return None
    if isinstance(reward, (int, float)):
        return float(reward)
    if isinstance(reward, dict):
        scalar = reward.get("scalar")
        if not isinstance(scalar, bool) and isinstance(scalar, (int, float)):
            return float(scalar)
    return None


def main() -> None:
    args = parse_args()
    try:
        results = asyncio.run(async_main(args))
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    completed = sum(result["result"]["status"] == "completed" for result in results)
    rewards = [
        reward
        for result in results
        if result["result"]["status"] == "completed"
        if (reward := _scalar_reward(result["result"])) is not None
    ]
    summary = {
        "trials": len(results),
        "completed": completed,
        "reward_sum": sum(rewards),
        "reward_mean": sum(rewards) / len(rewards) if rewards else None,
        "output_jsonl": str(args.output_jsonl) if args.output_jsonl is not None else None,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if any(result["result"]["status"] not in {"completed", "truncated"} for result in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
