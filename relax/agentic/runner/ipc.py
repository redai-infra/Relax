# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import argparse
import asyncio
import base64
import fcntl
import json
import os
import re
import signal
import socket
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_DEFAULT_BOOTSTRAP_TIMEOUT_S = 30.0
_DEFAULT_REQUEST_RETRY_ATTEMPTS = 3
_LENGTH_PREFIX_STRUCT = struct.Struct("!I")
# One rollout can burst hundreds of concurrent launch RPCs into this single UDS.
# asyncio/uvloop defaults backlog to 100, which drops connect() with EAGAIN under that load.
_LAUNCHER_SERVER_BACKLOG = max(1024, socket.SOMAXCONN)
_LAUNCH_SEMAPHORE: asyncio.BoundedSemaphore | None = None


class LauncherProtocolError(RuntimeError):
    pass


def encode_message(payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _LENGTH_PREFIX_STRUCT.pack(len(body)) + body


async def read_message(reader) -> dict[str, Any]:
    raw_size = await reader.readexactly(_LENGTH_PREFIX_STRUCT.size)
    (size,) = _LENGTH_PREFIX_STRUCT.unpack(raw_size)
    if size <= 0:
        raise LauncherProtocolError("Launcher protocol received non-positive message size.")
    body = await reader.readexactly(size)
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise LauncherProtocolError("Launcher protocol payload must be a JSON object.")
    return payload


async def write_message(writer, payload: dict[str, Any]) -> None:
    writer.write(encode_message(payload))
    await writer.drain()


def _ping_socket(socket_path: str) -> None:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.settimeout(1.0)
        client.connect(socket_path)
        client.sendall(encode_message({"op": "ping"}))
        raw_size = client.recv(_LENGTH_PREFIX_STRUCT.size)
        if len(raw_size) != _LENGTH_PREFIX_STRUCT.size:
            raise RuntimeError("Launcher daemon ping did not return a valid length prefix.")
        (size,) = _LENGTH_PREFIX_STRUCT.unpack(raw_size)
        payload = b""
        while len(payload) < size:
            chunk = client.recv(size - len(payload))
            if not chunk:
                raise RuntimeError("Launcher daemon ping response closed early.")
            payload += chunk
        response = json.loads(payload.decode("utf-8"))
        if not isinstance(response, dict) or response.get("ok") is not True:
            raise RuntimeError(f"Launcher daemon ping failed: {response!r}")
    finally:
        client.close()


class LauncherClient:
    def __init__(self, *, socket_path: str):
        self._socket_path = socket_path

    def _recover_default_launcher_socket(self) -> None:
        expected_socket_path = launcher_socket_path()
        if self._socket_path != expected_socket_path:
            return
        self._socket_path = ensure_local_launcher_daemon()

    @staticmethod
    def _request_retry_backoff_s(*, attempt: int) -> float:
        return min(0.5, 0.1 * (2**attempt))

    async def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(_DEFAULT_REQUEST_RETRY_ATTEMPTS):
            try:
                reader, writer = await asyncio.open_unix_connection(self._socket_path)
                try:
                    await write_message(writer, payload)
                    response = await read_message(reader)
                finally:
                    writer.close()
                    try:
                        await writer.wait_closed()
                    except Exception:
                        pass
                error = response.get("error")
                if isinstance(error, dict):
                    message = error.get("message")
                    if not isinstance(message, str) or not message:
                        message = "Launcher daemon request failed."
                    raise RuntimeError(message)
                return response
            except (
                OSError,
                asyncio.TimeoutError,
                asyncio.IncompleteReadError,
                json.JSONDecodeError,
                LauncherProtocolError,
            ) as exc:
                last_error = exc
                if attempt >= _DEFAULT_REQUEST_RETRY_ATTEMPTS - 1:
                    break
                try:
                    self._recover_default_launcher_socket()
                except Exception as recovery_exc:
                    logger.warning(
                        "Launcher daemon recovery attempt %d failed for %s: %s",
                        attempt + 1,
                        self._socket_path,
                        recovery_exc,
                    )
                await asyncio.sleep(self._request_retry_backoff_s(attempt=attempt))
        raise RuntimeError(
            f"Launcher daemon request failed after {_DEFAULT_REQUEST_RETRY_ATTEMPTS} attempts."
        ) from last_error

    async def ping(self) -> dict[str, Any]:
        return await self._request({"op": "ping"})

    async def launch(
        self,
        *,
        command: str,
        cwd: str | None,
        env: dict[str, str],
    ) -> dict[str, Any]:
        return await self._request({"op": "launch", "command": command, "cwd": cwd, "env": env})

    async def wait(self, *, handle: str) -> dict[str, Any]:
        return await self._request({"op": "wait", "handle": handle})

    async def kill(self, *, handle: str, signal_value: int = 15, forget: bool = True) -> dict[str, Any]:
        return await self._request({"op": "kill", "handle": handle, "signal": int(signal_value), "forget": forget})

    async def kill_all(self, *, signal_value: int = 15) -> dict[str, Any]:
        return await self._request({"op": "kill_all", "signal": int(signal_value)})

    async def shutdown(self) -> dict[str, Any]:
        return await self._request({"op": "shutdown"})


def _sanitize_token(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "-", value).strip("-") or "unknown"


def _launcher_launch_concurrency() -> int:
    raw_value = os.environ.get("RELAX_AGENTIC_LAUNCHER_CONCURRENCY")
    if raw_value is not None:
        concurrency = int(raw_value)
        if concurrency <= 0:
            raise ValueError("RELAX_AGENTIC_LAUNCHER_CONCURRENCY must be a positive integer.")
        return concurrency
    return max(16, min(128, os.cpu_count() or 16))


def _launcher_launch_semaphore() -> asyncio.BoundedSemaphore:
    global _LAUNCH_SEMAPHORE
    if _LAUNCH_SEMAPHORE is None:
        _LAUNCH_SEMAPHORE = asyncio.BoundedSemaphore(_launcher_launch_concurrency())
    return _LAUNCH_SEMAPHORE


def _launcher_job_token() -> str:
    explicit_namespace = os.environ.get("RELAX_LAUNCHER_NAMESPACE")
    if explicit_namespace:
        return _sanitize_token(explicit_namespace)
    job_id = os.environ.get("RAY_JOB_ID")
    if job_id:
        return _sanitize_token(job_id)
    repo_root = os.environ.get("RELAX")
    if repo_root:
        return _sanitize_token(repo_root)
    return "standalone"


def _launcher_user_token() -> str:
    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    if user:
        return _sanitize_token(user)
    return str(os.getuid())


def launcher_socket_path() -> str:
    return f"/tmp/relax-agentic-launcher-{_launcher_user_token()}-{_launcher_job_token()}.sock"


def _launcher_log_dir() -> str:
    return f"/tmp/relax-agentic-launcher-{_launcher_user_token()}-{_launcher_job_token()}-logs"


def _launcher_lock_path(socket_path: str) -> str:
    return f"{socket_path}.lock"


class _LauncherBootstrapLock:
    def __init__(self, *, socket_path: str):
        self._lock_path = _launcher_lock_path(socket_path)
        self._fh = None

    def __enter__(self):
        self._fh = open(self._lock_path, "a+", encoding="utf-8")
        fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        if self._fh is None:
            return
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None


def _wait_for_socket(socket_path: str, *, timeout_s: float = _DEFAULT_BOOTSTRAP_TIMEOUT_S) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            _ping_socket(socket_path)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"Launcher daemon did not become ready at {socket_path}: {last_error}")


def _spawn_daemon_process(*, socket_path: str, log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    node_name = _sanitize_token(socket.gethostname())
    log_path = Path(log_dir) / f"launcher-{node_name}.log"
    log_fh = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[3])
    try:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "relax.agentic.runner.ipc",
                "--sock",
                socket_path,
            ],
            cwd=repo_root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_fh.close()


def ensure_local_launcher_daemon() -> str:
    socket_path = launcher_socket_path()
    with _LauncherBootstrapLock(socket_path=socket_path):
        if os.path.exists(socket_path):
            try:
                _wait_for_socket(socket_path, timeout_s=0.2)
                return socket_path
            except Exception:
                try:
                    os.unlink(socket_path)
                except OSError:
                    pass
        _spawn_daemon_process(socket_path=socket_path, log_dir=_launcher_log_dir())
        _wait_for_socket(socket_path)
    return socket_path


@dataclass
class _CompletedProcess:
    exit_code: int
    stdout_b64: str
    stderr_b64: str
    started_at: float
    exited_at: float


@dataclass
class _ProcHandle:
    proc: asyncio.subprocess.Process
    started_at: float
    completed: _CompletedProcess | None = None

    async def wait(self) -> _CompletedProcess:
        if self.completed is not None:
            return self.completed
        stdout, stderr = await self.proc.communicate()
        exited_at = time.time()
        self.completed = _CompletedProcess(
            exit_code=int(self.proc.returncode if self.proc.returncode is not None else -1),
            stdout_b64=base64.b64encode(stdout).decode("ascii"),
            stderr_b64=base64.b64encode(stderr).decode("ascii"),
            started_at=self.started_at,
            exited_at=exited_at,
        )
        return self.completed


_HANDLES: dict[str, _ProcHandle] = {}


def _kill_process_group(pid: int | None, signal_value: int) -> None:
    if pid is None:
        return
    try:
        os.killpg(os.getpgid(pid), signal_value)
    except Exception:
        return


async def _launch(request: dict[str, object]) -> dict[str, object]:
    command = request.get("command")
    cwd = request.get("cwd")
    env = request.get("env")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("Launcher daemon requires a non-empty command string.")
    if cwd is not None and not isinstance(cwd, str):
        raise TypeError("Launcher daemon cwd must be a string when provided.")
    if not isinstance(env, dict):
        raise TypeError("Launcher daemon env must be a JSON object.")
    merged_env = {**os.environ, **{str(key): str(value) for key, value in env.items()}}
    queue_entered_at = time.time()
    async with _launcher_launch_semaphore():
        permit_acquired_at = time.time()
        started_at = time.time()
        proc = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-lc",
            command,
            cwd=cwd,
            env=merged_env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            close_fds=True,
        )
    spawn_returned_at = time.time()
    handle = uuid4().hex
    _HANDLES[handle] = _ProcHandle(proc=proc, started_at=started_at)
    return {
        "handle": handle,
        "pid": proc.pid,
        "launch_queue_entered_at": queue_entered_at,
        "launch_permit_acquired_at": permit_acquired_at,
        "started_at": started_at,
        "spawn_returned_at": spawn_returned_at,
    }


async def _wait(request: dict[str, object]) -> dict[str, object]:
    handle = request.get("handle")
    if not isinstance(handle, str) or not handle:
        raise ValueError("Launcher daemon wait requires a non-empty handle.")
    proc_handle = _HANDLES.get(handle)
    if proc_handle is None:
        raise KeyError(f"Unknown launcher handle: {handle}")
    try:
        completed = await proc_handle.wait()
        return {
            "handle": handle,
            "exit_code": completed.exit_code,
            "stdout_b64": completed.stdout_b64,
            "stderr_b64": completed.stderr_b64,
            "started_at": completed.started_at,
            "exited_at": completed.exited_at,
        }
    finally:
        _HANDLES.pop(handle, None)


async def _kill(request: dict[str, object]) -> dict[str, object]:
    handle = request.get("handle")
    signal_value = request.get("signal", signal.SIGTERM)
    forget = request.get("forget", True)
    if not isinstance(handle, str) or not handle:
        raise ValueError("Launcher daemon kill requires a non-empty handle.")
    if not isinstance(signal_value, int):
        raise TypeError("Launcher daemon signal must be an integer.")
    if not isinstance(forget, bool):
        raise TypeError("Launcher daemon forget flag must be a boolean.")
    proc_handle = _HANDLES.get(handle)
    if proc_handle is None:
        raise KeyError(f"Unknown launcher handle: {handle}")
    try:
        _kill_process_group(proc_handle.proc.pid, signal_value)
        return {"handle": handle, "killed": True}
    finally:
        if forget:
            _HANDLES.pop(handle, None)


async def _kill_all(request: dict[str, object]) -> dict[str, object]:
    signal_value = request.get("signal", signal.SIGTERM)
    if not isinstance(signal_value, int):
        raise TypeError("Launcher daemon signal must be an integer.")
    handles = list(_HANDLES.items())
    for handle, proc_handle in handles:
        try:
            _kill_process_group(proc_handle.proc.pid, signal_value)
        finally:
            _HANDLES.pop(handle, None)
    return {"killed": len(handles)}


async def _shutdown() -> dict[str, object]:
    await _kill_all({"signal": int(signal.SIGTERM)})
    asyncio.get_running_loop().call_soon(asyncio.get_running_loop().stop)
    return {"shutdown": True}


async def _dispatch(request: dict[str, object]) -> dict[str, object]:
    op = request.get("op")
    if op == "ping":
        return {"ok": True}
    if op == "launch":
        return await _launch(request)
    if op == "wait":
        return await _wait(request)
    if op == "kill":
        return await _kill(request)
    if op == "kill_all":
        return await _kill_all(request)
    if op == "shutdown":
        return await _shutdown()
    raise ValueError(f"Unknown launcher daemon operation: {op!r}")


async def _handle_client(reader, writer) -> None:
    response: dict[str, object] | None = None
    try:
        request = await read_message(reader)
        response = await _dispatch(request)
    except asyncio.IncompleteReadError:
        response = None
    except Exception as exc:
        logger.exception("Launcher daemon request failed: %s", exc)
        response = {"error": {"type": type(exc).__name__, "message": str(exc)}}
    try:
        if response is not None:
            await write_message(writer, response)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            return


async def _run_server(*, socket_path: str) -> None:
    socket_file = Path(socket_path)
    socket_file.parent.mkdir(parents=True, exist_ok=True)
    if socket_file.exists():
        socket_file.unlink()
    server = await asyncio.start_unix_server(_handle_client, path=socket_path, backlog=_LAUNCHER_SERVER_BACKLOG)
    async with server:
        await server.serve_forever()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Relax agentic launcher daemon")
    parser.add_argument("--sock", required=True, help="Unix domain socket path for the launcher daemon.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    logger.info(
        "Starting launcher daemon on socket %s with backlog=%d launch_concurrency=%d",
        args.sock,
        _LAUNCHER_SERVER_BACKLOG,
        _launcher_launch_concurrency(),
    )
    asyncio.run(_run_server(socket_path=args.sock))


if __name__ == "__main__":
    main()
