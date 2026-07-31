# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import enum
import importlib
import importlib.metadata
import ipaddress
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_MAJOR = 1
REDACTED = "<redacted>"

_PACKAGE_NAMES = ("relax", "ray", "torch", "sglang", "transformers", "megatron-core")
_IMAGE_ENV_KEYS = ("RELAX_CONTAINER_IMAGE", "CONTAINER_IMAGE", "IMAGE_NAME")
_INPUT_FIELDS = (
    "hf_checkpoint",
    "load",
    "ref_load",
    "prompt_data",
    "eval_prompt_data",
    "custom_dataset_class_path",
)
_PARALLEL_FIELDS = (
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
    "virtual_pipeline_model_parallel_size",
    "sequence_parallel",
)
_ROLE_FIELDS = {
    "actor": ("actor_num_nodes", "actor_num_gpus_per_node"),
    "critic": ("critic_num_nodes", "critic_num_gpus_per_node"),
    "rollout": (None, "rollout_num_gpus"),
    "genrm": (None, "genrm_num_gpus"),
}
_SECRET_KEY_RE = re.compile(
    r"(?:^|_)(?:token|password|passwd|secret|api_?key|access_?key|private_?key|credential|authorization|auth)(?:$|_)",
    re.IGNORECASE,
)
_ADDRESS_KEY_RE = re.compile(
    r"(?:^|_)(?:address|addresses|addr|addrs|host|hostname|ip|endpoint|url)(?:$|_)", re.IGNORECASE
)
_IPV4_RE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_NETWORK_LOCATION_RE = re.compile(r"(?P<scheme>\b[a-z][a-z0-9+.-]*://)(?P<location>[^/\s]+)", re.IGNORECASE)
_ASSIGNMENT_SECRET_RE = re.compile(
    r"(?i)\b(token|password|passwd|secret|api[-_]?key|access[-_]?key|authorization)=([^\s&]+)"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")
_SAFE_RESOURCE_NAMES = {"CPU", "GPU", "memory", "object_store_memory"}


class ManifestError(ValueError):
    """Raised when a manifest cannot be read or safely replayed."""


def _normalise_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")


def _is_sensitive_key(key: Any) -> bool:
    return bool(_SECRET_KEY_RE.search(_normalise_key(key)))


def _is_address_key(key: Any) -> bool:
    return bool(_ADDRESS_KEY_RE.search(_normalise_key(key)))


def _redact_private_ip(match: re.Match[str]) -> str:
    value = match.group(0)
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return value
    if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
        return REDACTED
    return value


def _redact_network_location(match: re.Match[str]) -> str:
    scheme = match.group("scheme")
    location = match.group("location")
    host_port = location.rsplit("@", 1)[-1]
    host = host_port.rsplit(":", 1)[0].strip("[]").lower()
    internal_hostname = host in {"localhost", "0.0.0.0"} or "." not in host
    internal_hostname = internal_hostname or host.endswith((".local", ".internal", ".cluster.local"))
    try:
        address = ipaddress.ip_address(host)
        internal_hostname = internal_hostname or address.is_private or address.is_loopback or address.is_link_local
    except ValueError:
        pass
    if "@" in location or internal_hostname:
        return f"{scheme}{REDACTED}"
    return match.group(0)


def _sanitize_string(value: str) -> str:
    value = _BEARER_RE.sub(f"Bearer {REDACTED}", value)
    value = _ASSIGNMENT_SECRET_RE.sub(lambda match: f"{match.group(1)}={REDACTED}", value)
    value = _NETWORK_LOCATION_RE.sub(_redact_network_location, value)
    return _IPV4_RE.sub(_redact_private_ip, value)


def sanitize_value(value: Any, *, key: Any = None) -> Any:
    """Convert a value to JSON-safe data while removing secrets and addresses."""
    if key is not None and (_is_sensitive_key(key) or _is_address_key(key)):
        return REDACTED
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, enum.Enum):
        return sanitize_value(value.value, key=key)
    if isinstance(value, Path):
        return _sanitize_string(str(value))
    if isinstance(value, str):
        return _sanitize_string(value)
    if isinstance(value, Mapping):
        return {str(item_key): sanitize_value(item_value, key=item_key) for item_key, item_value in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [sanitize_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((sanitize_value(item) for item in value), key=str)
    if callable(value):
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
        return f"{module}.{name}".lstrip(".")
    return _sanitize_string(str(value))


def sanitize_argv(argv: Sequence[str]) -> list[str]:
    """Redact sensitive flag values without treating `tokenizer` as `token`."""
    sanitized: list[str] = []
    redact_next = False
    for token in argv:
        if redact_next:
            sanitized.append(REDACTED)
            redact_next = False
            continue
        if token.startswith("--"):
            option, separator, option_value = token.partition("=")
            option_key = option[2:]
            if _is_sensitive_key(option_key) or _is_address_key(option_key):
                if separator:
                    sanitized.append(f"{option}={REDACTED}")
                else:
                    sanitized.append(option)
                    redact_next = True
                continue
        sanitized.append(_sanitize_string(str(token)))
    return sanitized


def _run_git(arguments: Sequence[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _collect_code(cwd: Path) -> dict[str, Any]:
    repository_root = _run_git(("rev-parse", "--show-toplevel"), cwd)
    git_cwd = Path(repository_root) if repository_root else cwd
    status = _run_git(("status", "--porcelain"), git_cwd) if repository_root else None
    return {
        "repository": git_cwd.name if repository_root else None,
        "commit": _run_git(("rev-parse", "HEAD"), git_cwd),
        "branch": _run_git(("branch", "--show-current"), git_cwd),
        "dirty": bool(status) if status is not None else None,
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in _PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _sanitize_image_reference(value: str) -> str:
    sanitized = _sanitize_string(value)
    parts = sanitized.split("/", 1)
    if len(parts) == 2 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
        return f"{REDACTED}/{parts[1]}"
    return sanitized


def _collect_environment(*, probe_framework: bool = False) -> dict[str, Any]:
    image = next((os.environ.get(key) for key in _IMAGE_ENV_KEYS if os.environ.get(key)), None)
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": _package_versions(),
        "container_image": _sanitize_image_reference(image) if image else None,
    }
    torch_module = sys.modules.get("torch")
    if torch_module is None and probe_framework:
        try:
            torch_module = importlib.import_module("torch")
        except Exception:
            torch_module = None
    if torch_module is not None:
        torch_version = getattr(torch_module, "version", None)
        environment["cuda"] = getattr(torch_version, "cuda", None)
    return environment


def _system_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _probe_gpus() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    gpus = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        name, driver, memory_mib = fields
        try:
            memory_value: int | None = int(memory_mib)
        except ValueError:
            memory_value = None
        gpus.append({"name": name, "driver": driver, "memory_mib": memory_value})
    return gpus


def _collect_hardware() -> dict[str, Any]:
    return {
        "cpu_count": os.cpu_count(),
        "memory_bytes": _system_memory_bytes(),
        "gpus": _probe_gpus(),
    }


def _safe_resources(resources: Mapping[str, Any] | None) -> dict[str, Any]:
    if not resources:
        return {}
    safe = {}
    for name, value in resources.items():
        if name in _SAFE_RESOURCE_NAMES or name.startswith("accelerator_type:"):
            safe[name] = sanitize_value(value)
    return dict(sorted(safe.items()))


def _collect_topology(args: Any) -> dict[str, Any]:
    config = vars(args) if args is not None and hasattr(args, "__dict__") else {}
    parallelism = {
        field: sanitize_value(config[field], key=field) for field in _PARALLEL_FIELDS if config.get(field) is not None
    }
    roles: dict[str, Any] = {}
    if isinstance(config.get("resource"), Mapping):
        roles["resource_config"] = sanitize_value(config["resource"], key="resource")
    for role, (node_field, gpu_field) in _ROLE_FIELDS.items():
        nodes = config.get(node_field) if node_field else None
        gpus = config.get(gpu_field)
        if nodes is None and gpus is None:
            continue
        role_data: dict[str, Any] = {}
        if nodes is not None:
            role_data["nodes"] = sanitize_value(nodes)
        if gpus is not None:
            role_data["gpus"] = sanitize_value(gpus)
        roles[role] = role_data
    return {"parallelism": parallelism, "roles": roles}


def _collect_runtime(args: Any, ray_module: Any = None) -> dict[str, Any]:
    initialized = False
    nodes: list[Mapping[str, Any]] = []
    cluster_resources: Mapping[str, Any] = {}
    if ray_module is not None:
        try:
            initialized = bool(ray_module.is_initialized())
            if initialized:
                nodes = [node for node in ray_module.nodes() if node.get("Alive")]
                cluster_resources = ray_module.cluster_resources()
        except Exception:
            initialized = False
            nodes = []
            cluster_resources = {}
    node_summaries = [{"resources": _safe_resources(node.get("Resources"))} for node in nodes]
    node_summaries.sort(key=lambda item: json.dumps(item, sort_keys=True))
    mode = "local"
    if initialized:
        mode = "single_node_ray" if len(nodes) <= 1 else "multi_node_ray"
    return {
        "mode": mode,
        "node_count": len(nodes) if initialized else 1,
        "cluster_resources": _safe_resources(cluster_resources),
        "nodes": node_summaries,
        "topology": _collect_topology(args),
    }


def _describe_input(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_describe_input(item) for item in value]
    if not isinstance(value, (str, Path)):
        return sanitize_value(value)
    identifier = _sanitize_string(str(value))
    descriptor: dict[str, Any] = {"identifier": identifier}
    try:
        path = Path(value).expanduser()
        if path.is_file():
            descriptor.update({"kind": "file", "size_bytes": path.stat().st_size})
        elif path.is_dir():
            descriptor["kind"] = "directory"
        else:
            descriptor["kind"] = "identifier"
    except (OSError, RuntimeError, ValueError):
        descriptor["kind"] = "unavailable"
    return descriptor


def _collect_inputs(args: Any) -> dict[str, Any]:
    config = vars(args) if args is not None and hasattr(args, "__dict__") else {}
    return {field: _describe_input(config[field]) for field in _INPUT_FIELDS if config.get(field) is not None}


def build_manifest(
    args: Any = None,
    runtime_env: Mapping[str, Any] | None = None,
    *,
    argv: Sequence[str] | None = None,
    cwd: str | Path | None = None,
    ray_module: Any = None,
    probe_framework: bool = False,
) -> dict[str, Any]:
    """Build a shareable manifest without importing optional training dependencies."""
    started_at = time.perf_counter()
    working_directory = Path(cwd or Path.cwd()).resolve()
    command = list(argv) if argv is not None else [sys.executable, *sys.argv]
    config = vars(args) if args is not None and hasattr(args, "__dict__") else {}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code": _collect_code(working_directory),
        "command": {"argv": sanitize_argv(command), "working_directory": "."},
        "config": {
            "arguments": sanitize_value(config),
            "runtime_env": sanitize_value(runtime_env or {}),
        },
        "environment": _collect_environment(probe_framework=probe_framework),
        "hardware": _collect_hardware(),
        "runtime": _collect_runtime(args, ray_module),
        "inputs": _collect_inputs(args),
    }
    manifest["collection_duration_ms"] = round((time.perf_counter() - started_at) * 1000, 3)
    return manifest


def _schema_major(version: Any) -> int:
    if isinstance(version, int):
        return version
    try:
        return int(str(version).split(".", 1)[0])
    except (TypeError, ValueError) as error:
        raise ManifestError(f"Invalid manifest schema version: {version!r}") from error


def normalize_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Read any v1 manifest, accepting missing optional fields and future v1 minors."""
    if "schema_version" not in manifest:
        raise ManifestError("Manifest is missing schema_version")
    major = _schema_major(manifest["schema_version"])
    if major != SUPPORTED_SCHEMA_MAJOR:
        raise ManifestError(
            f"Unsupported manifest schema major version {major}; supported major is {SUPPORTED_SCHEMA_MAJOR}"
        )
    normalized = dict(manifest)
    normalized["schema_version"] = str(manifest["schema_version"])
    for section in ("code", "command", "config", "environment", "hardware", "runtime", "inputs"):
        normalized.setdefault(section, {})
    return normalized


def write_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    """Validate and atomically write a manifest."""
    normalized = normalize_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as temporary:
            json.dump(normalized, temporary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def load_manifest(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open(encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"Unable to read manifest {path}: {error}") from error
    if not isinstance(manifest, Mapping):
        raise ManifestError("Manifest root must be a JSON object")
    return normalize_manifest(manifest)


def default_manifest_path(args: Any = None) -> Path:
    override = os.environ.get("RELAX_MANIFEST_PATH")
    if override:
        return Path(override).expanduser()
    run_name = None
    if args is not None:
        run_name = getattr(args, "tb_experiment_name", None) or getattr(args, "wandb_group", None)
    run_name = run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_run_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(run_name)).strip("-.") or "run"
    return Path.cwd() / "relax_runs" / safe_run_name[:120] / "experiment-manifest.json"


def write_experiment_manifest(
    args: Any,
    runtime_env: Mapping[str, Any] | None,
    *,
    path: str | Path | None = None,
    ray_module: Any = None,
) -> Path | None:
    """Best-effort training hook: manifest failures never block the run."""
    try:
        destination = Path(path) if path is not None else default_manifest_path(args)
        manifest = build_manifest(args, runtime_env, ray_module=ray_module)
        written_path = write_manifest(manifest, destination)
        logger.info("Reproducibility manifest saved to %s", written_path)
        return written_path
    except Exception as error:
        logger.warning("Unable to write reproducibility manifest; training will continue: %s", error)
        return None


def _get_path(data: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = data
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
    return current


def compare_environment(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return actionable differences while allowing extra fields in newer v1 manifests."""
    differences = []
    paths = (
        ("code", "commit"),
        ("environment", "python"),
        ("environment", "platform", "system"),
        ("environment", "platform", "machine"),
        ("environment", "packages"),
        ("environment", "cuda"),
        ("hardware", "cpu_count"),
        ("hardware", "gpus"),
    )
    for path in paths:
        expected_value = _get_path(expected, path)
        actual_value = _get_path(actual, path)
        if expected_value is None:
            continue
        if expected_value != actual_value:
            differences.append(
                {
                    "field": ".".join(path),
                    "expected": expected_value,
                    "actual": actual_value,
                    "suggestion": _suggest(path),
                }
            )
    return differences


def _suggest(path: Sequence[str]) -> str:
    field = ".".join(path)
    if field == "code.commit":
        return "Check out the recorded commit before replaying."
    if field == "environment.packages":
        return "Use the recorded image or align the listed package versions."
    if field.startswith("hardware.gpus"):
        return "Use hosts with the recorded GPU model, driver, and memory."
    return f"Align {field} with the recorded value or replay with an explicit override."


def inspect_environment(manifest: Mapping[str, Any], *, cwd: str | Path | None = None) -> list[dict[str, Any]]:
    current = build_manifest(argv=[], cwd=cwd, probe_framework=True)
    return compare_environment(normalize_manifest(manifest), current)


def replay_command(manifest: Mapping[str, Any]) -> list[str]:
    normalized = normalize_manifest(manifest)
    argv = normalized.get("command", {}).get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise ManifestError("Manifest command.argv must be a non-empty string array")
    if any(REDACTED in item for item in argv):
        raise ManifestError("Manifest command contains redacted values; supply credentials outside the manifest")
    return argv


def execute_manifest(manifest: Mapping[str, Any], *, cwd: str | Path | None = None) -> int:
    """Replay using argv directly; shell expansion is deliberately disabled."""
    command = replay_command(manifest)
    try:
        result = subprocess.run(command, cwd=cwd or Path.cwd(), check=False)
    except OSError as error:
        raise ManifestError(f"Unable to execute recorded command: {error}") from error
    return result.returncode
