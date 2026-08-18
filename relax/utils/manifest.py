# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Small, best-effort experiment manifests for reproducible Relax runs."""

import argparse
import hashlib
import importlib.metadata
import ipaddress
import json
import math
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import yaml

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

SCHEMA_VERSION = "1.0"
COLLECTION_TIMEOUT_SECONDS = 4.0
MAX_MANIFEST_BYTES = 100 * 1024
REDACTED = "<redacted>"

_PACKAGES = ("relax", "torch", "ray", "sglang", "transformers", "megatron-core", "numpy")
_ENV_PREFIXES = ("CUDA_", "NCCL_", "RAY_", "SLURM_", "WANDB_", "HF_", "RELAX_", "SGLANG_", "TQ_")
_REPRO_ENV_PREFIXES = _ENV_PREFIXES + ("TOKENIZERS_", "CLEARML_")
_REPRO_EXACT_ENV = {"PYTHONPATH", "PYTHONBUFFERED"}
_SECRET_KEY = re.compile(
    r"(?:^|_)(?:api_?key|access_?key|private_?key|token|password|passwd|secret|credential|authorization|auth)(?:$|_)",
    re.IGNORECASE,
)
_LOCATION_KEY = re.compile(
    r"(?:^address$|^host(?:name)?$|(?:^|_)(?:ray|master|head|node|worker|redis|gcs|dashboard|wandb|"
    r"external_engine)(?:_manager)?_(?:address(?:es)?|addr(?:s)?|ip|host(?:name)?|node|name|url)(?:$|_)|"
    r"(?:^|_)nodelist(?:$|_))",
    re.IGNORECASE,
)
_ASSIGNMENT = re.compile(r"(?i)(\b[A-Z_][A-Z0-9_-]*\b\s*[:=]\s*)(?:\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*'|[^\s,;&]+)")
_AUTHORIZATION = re.compile(
    r"(?i)(\bAuthorization\s*:\s*)(?:Basic|Bearer|Token|Digest)\s+(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_BEARER = re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]+")
_URL_PASSWORD = re.compile(r"(?i)([a-z][a-z0-9+.-]*://[^/@\s:]+:)([^/@\s]+)(@)")
_URL_QUERY_PARAMETER = re.compile(r"([?&;])([^=&#;\s]+)(=)([^&#;\s]*)")
_IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_IPV6 = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])")
_INTERNAL_HOST = re.compile(r"(?i)(?<![\w.-])(?:[A-Z0-9-]+\.)+(?:internal|corp|local|cluster\.local)(?::\d+)?")
_BARE_HOST_PORT = re.compile(r"(?i)(?<![\w.-])(?:localhost|[a-z][a-z0-9-]{1,62}):\d{2,5}(?!\d)")
_HOME_PATH = re.compile(r"(?i)(?:/(?:lustre/)?home/[^/\s]+|/root|C:\\Users\\[^\\\s]+)(?=[/\\])")
_SAFE_RAY_RESOURCES = {"CPU", "GPU", "memory", "object_store_memory"}
_SAFE_PLACEHOLDERS = {"<host>", "<internal-host>", "<internal-address>", "<slurm-nodes>"}
_INPUT_METADATA_SCRIPT = """
import hashlib, json, os, sys
results = []
for path in json.loads(sys.argv[1]):
    result = {"size_bytes": None}
    try:
        stat = os.stat(path)
        if os.path.isfile(path):
            result["size_bytes"] = stat.st_size
            if stat.st_size <= 1024 * 1024:
                with open(path, "rb") as stream:
                    content = stream.read(1024 * 1024 + 1)
                if len(content) <= 1024 * 1024 and os.stat(path).st_size == stat.st_size:
                    result["sha256"] = hashlib.sha256(content).hexdigest()
    except OSError:
        pass
    results.append(result)
print(json.dumps(results))
"""


class ManifestError(ValueError):
    """Raised for invalid or unsafe manifest operations."""


def _mask_non_public_ip(match: re.Match) -> str:
    candidate = match.group(0)
    try:
        is_public = ipaddress.ip_address(candidate).is_global
    except ValueError:
        is_public = True
    return candidate if is_public else "<internal-address>"


def _mask_sensitive_assignment(match: re.Match) -> str:
    key = re.sub(r"[^A-Za-z0-9]+", "_", match.group(1)).strip("_")
    return f"{match.group(1)}{REDACTED}" if _SECRET_KEY.search(key) else match.group(0)


def _mask_sensitive_url_parameter(match: re.Match) -> str:
    key = re.sub(r"[^A-Za-z0-9]+", "_", match.group(2)).strip("_")
    return f"{match.group(1)}{match.group(2)}{match.group(3)}{REDACTED}" if _SECRET_KEY.search(key) else match.group(0)


def _sanitize_string(value: str) -> str:
    stripped = value.strip()
    if stripped[:1] in {"{", "["}:
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass
        else:
            return json.dumps(sanitize_value(parsed), separators=(",", ":"), sort_keys=True)
    value = _AUTHORIZATION.sub(r"\1<redacted>", value)
    value = _BEARER.sub(r"\1<redacted>", value)
    value = _URL_PASSWORD.sub(r"\1<redacted>\3", value)
    value = _URL_QUERY_PARAMETER.sub(_mask_sensitive_url_parameter, value)
    value = _ASSIGNMENT.sub(_mask_sensitive_assignment, value)
    value = _INTERNAL_HOST.sub("<internal-host>", value)
    value = _BARE_HOST_PORT.sub("<internal-host>", value)
    value = _IPV4.sub(_mask_non_public_ip, value)
    value = _IPV6.sub(_mask_non_public_ip, value)
    return _HOME_PATH.sub("<home>", value)


def _normalize_key(key: str) -> str:
    with_word_boundaries = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    return re.sub(r"[^A-Za-z0-9]+", "_", with_word_boundaries)


def sanitize_value(value: Any, key: Optional[str] = None) -> Any:
    """Convert arbitrary values to JSON-safe data while recursively redacting
    secrets."""
    normalized_key = _normalize_key(key) if key else ""
    if normalized_key and _SECRET_KEY.search(normalized_key):
        return REDACTED
    if normalized_key and _LOCATION_KEY.search(normalized_key):
        if value in (None, ""):
            return value
        return value if isinstance(value, str) and value in _SAFE_PLACEHOLDERS else "<internal-host>"
    if isinstance(value, float):
        return value if math.isfinite(value) else f"<non-finite:{value}>"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return _sanitize_string(value)
    if isinstance(value, Path):
        return _sanitize_string(str(value))
    if isinstance(value, Mapping):
        cleaned: Dict[str, Any] = {}
        for item_key in value:
            cleaned[str(item_key)] = sanitize_value(value[item_key], str(item_key))
        return cleaned
    if isinstance(value, (list, tuple, set, frozenset)):
        cleaned_items = []
        for item in value:
            cleaned_items.append(sanitize_value(item))
        return sorted(cleaned_items, key=str) if isinstance(value, (set, frozenset)) else cleaned_items
    if callable(value):
        owner = getattr(value, "__module__", "")
        label = getattr(value, "__qualname__", type(value).__name__)
        return ".".join(part for part in (owner, label) if part)
    return _sanitize_string(str(value))


def sanitize_argv(argv: Sequence[str]) -> List[str]:
    """Redact sensitive option values while preserving the argv structure."""
    result: List[str] = []
    replacement_next: Optional[str] = None
    for raw_token in argv:
        token = str(raw_token)
        if replacement_next is not None:
            result.append(replacement_next)
            replacement_next = None
            continue
        if token.startswith("--"):
            option, separator, option_value = token.partition("=")
            option_key = _normalize_key(option[2:])
            if _SECRET_KEY.search(option_key):
                result.append(f"{option}={REDACTED}" if separator else option)
                replacement_next = None if separator else REDACTED
                continue
            if _LOCATION_KEY.search(option_key):
                result.append(f"{option}=<internal-host>" if separator else option)
                replacement_next = None if separator else "<internal-host>"
                continue
        result.append(_sanitize_string(token))
    return result


def _canonical_argv(argv: Optional[Sequence[str]] = None) -> List[str]:
    if argv is not None:
        return [str(item) for item in argv]
    original = getattr(sys, "orig_argv", None)
    if isinstance(original, list) and "-m" in original:
        canonical = [str(item) for item in original]
        canonical[0] = "python"
        return canonical
    current = list(sys.argv)
    script = Path(current[0]) if current else Path()
    if script.name == "train.py" and "entrypoints" in script.parts and "relax" in script.parts:
        return ["python", "-m", "relax.entrypoints.train", *current[1:]]
    return ["python", *current]


def _run(command: Sequence[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        output = subprocess.check_output(
            list(command),
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return output.decode("utf-8", errors="replace").strip()


def _git_repository(path: Path) -> Dict[str, Any]:
    root = _run(("git", "rev-parse", "--show-toplevel"), path)
    if not root:
        return {}
    repository = Path(root)
    status = _run(("git", "status", "--porcelain"), repository)
    tracked_status = _run(("git", "status", "--porcelain", "--untracked-files=no"), repository)
    untracked = _run(("git", "ls-files", "--others", "--exclude-standard"), repository)
    return {
        "commit": _run(("git", "rev-parse", "HEAD"), repository),
        "branch": _run(("git", "branch", "--show-current"), repository),
        "dirty": bool(status),
        "tracked_dirty": bool(tracked_status),
        "untracked_files": sanitize_value(untracked.splitlines() if untracked else []),
    }


def _collect_code() -> Dict[str, Any]:
    source_root = Path(__file__).resolve().parents[2]
    result = {"relax": _git_repository(source_root) or _git_repository(Path.cwd())}
    candidates = [os.environ.get("MEGATRON_LM_PATH")]
    candidates.extend(os.environ.get("PYTHONPATH", "").split(os.pathsep))
    for candidate in candidates:
        path = Path(candidate) if candidate else None
        if path and (os.environ.get("MEGATRON_LM_PATH") == candidate or "megatron" in str(path).lower()):
            result["megatron"] = _git_repository(path)
            break
    return result


def _collect_environment() -> Dict[str, Any]:
    packages: Dict[str, str] = {}
    for package in _PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    environment = {
        "python_version": platform.python_version(),
        "os": platform.system(),
        "os_version": platform.release(),
        "machine": platform.machine(),
        "packages": packages,
    }
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        environment["cuda_version"] = getattr(getattr(torch_module, "version", None), "cuda", None)
        try:
            nccl_version = torch_module.cuda.nccl.version()
        except Exception:
            nccl_version = None
        environment["nccl_version"] = nccl_version
    else:
        smi = _run(("nvidia-smi",))
        cuda = re.search(r"CUDA Version:\s*([0-9.]+)", smi or "")
        environment["cuda_version"] = cuda.group(1) if cuda else None
        for distribution in ("nvidia-nccl-cu12", "nvidia-nccl-cu11"):
            try:
                environment["nccl_version"] = importlib.metadata.version(distribution)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
    environment.setdefault("cuda_version", None)
    environment.setdefault("nccl_version", None)
    return environment


def _collect_hardware() -> Dict[str, Any]:
    gpus: List[Dict[str, Any]] = []
    output = _run(("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"))
    for line in output.splitlines() if output else []:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        name, memory_mib, driver = fields
        try:
            memory = int(memory_mib)
        except ValueError:
            memory = None
        gpus.append({"model": name, "memory_mib": memory, "driver": driver})
    memory_gb = None
    try:
        memory_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3), 2)
    except (AttributeError, OSError, ValueError):
        pass
    numa_root = Path("/sys/devices/system/node")
    numa_nodes = len(list(numa_root.glob("node[0-9]*"))) if numa_root.is_dir() else None
    return {
        "cpu_count": os.cpu_count(),
        "cpu_model": platform.processor() or None,
        "memory_gb": memory_gb,
        "numa_nodes": numa_nodes,
        "gpu_count": len(gpus),
        "gpu_model": gpus[0]["model"] if gpus else None,
        "gpu_memory_gb": round(gpus[0]["memory_mib"] / 1024, 2) if gpus and gpus[0]["memory_mib"] else None,
        "driver_version": gpus[0]["driver"] if gpus else None,
        "gpus": gpus,
    }


def _safe_resources(resources: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        name: sanitize_value(value)
        for name, value in resources.items()
        if name in _SAFE_RAY_RESOURCES or name.startswith("accelerator_type:")
    }


def _ray_node_role(node: Mapping[str, Any]) -> str:
    resources = node.get("Resources", {})
    if node.get("IsHeadNode") is True or "node:__internal_head__" in resources:
        return "head"
    if node.get("IsHeadNode") is False or any(str(name).startswith("node:") for name in resources):
        return "worker"
    return "unknown"


def _parallel_topology(args: Any) -> Dict[str, Any]:
    fields = {
        "tensor": "tensor_model_parallel_size",
        "pipeline": "pipeline_model_parallel_size",
        "context": "context_parallel_size",
        "expert": "expert_model_parallel_size",
        "data": "data_parallel_size",
    }
    return {
        name: getattr(args, attribute)
        for name, attribute in fields.items()
        if args is not None and getattr(args, attribute, None) is not None
    }


def _collect_runtime(args: Any = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {"mode": "local", "node_count": 1, "nodes": []}
    ray_module = sys.modules.get("ray")
    try:
        initialized = bool(ray_module and ray_module.is_initialized())
    except Exception:
        initialized = False
    if initialized:
        try:
            nodes = [node for node in ray_module.nodes() if node.get("Alive")]
            result.update(
                {
                    "mode": (
                        "ray_initialized_unknown"
                        if not nodes
                        else "single_node_ray"
                        if len(nodes) == 1
                        else "multi_node_ray"
                    ),
                    "node_count": len(nodes),
                    "nodes": [
                        {
                            "role": _ray_node_role(node),
                            "resources": _safe_resources(node.get("Resources", {})),
                        }
                        for node in nodes
                    ],
                    "cluster_resources": _safe_resources(ray_module.cluster_resources()),
                }
            )
        except Exception as exc:
            logger.warning(f"Failed to collect Ray topology: {exc}")
    if os.environ.get("SLURM_JOB_ID"):
        result["slurm"] = sanitize_value(
            {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
                "nodelist": "<slurm-nodes>" if os.environ.get("SLURM_JOB_NODELIST") else None,
                "node_count": os.environ.get("SLURM_JOB_NUM_NODES"),
            }
        )
    result["parallel_topology"] = _parallel_topology(args)
    return result


def _first_argument(args: Any, names: Sequence[str]) -> Any:
    for name in names:
        value = getattr(args, name, None) if args is not None else None
        if value not in (None, "", []):
            return value
    return None


def _collect_training(args: Any) -> Dict[str, Any]:
    return sanitize_value(
        {
            "parallel_topology": _parallel_topology(args),
            "global_batch_size": _first_argument(args, ("global_batch_size",)),
            "micro_batch_size": _first_argument(args, ("micro_batch_size",)),
            "algorithm": _first_argument(args, ("algorithm", "advantage_estimator")),
        }
    )


def _input_metadata(value: Any) -> Any:
    sequence = list(value) if isinstance(value, (list, tuple)) else [value]
    limited = sequence[:32]
    details = [{"identifier": sanitize_value(item), "size_bytes": None} for item in limited]
    paths = [str(item) if isinstance(item, (str, Path)) else "" for item in limited]
    try:
        output = _run((sys.executable, "-c", _INPUT_METADATA_SCRIPT, json.dumps(paths)))
        if output:
            for detail, metadata in zip(details, json.loads(output)):
                detail.update(metadata)
    except (json.JSONDecodeError, TypeError):
        pass
    if len(sequence) > len(limited):
        details.append({"truncated_items": len(sequence) - len(limited)})
    return details if isinstance(value, (list, tuple)) else details[0]


def _collect_inputs(args: Any) -> Dict[str, Any]:
    model = _first_argument(args, ("hf_checkpoint", "actor_model_path", "load"))
    tokenizer = _first_argument(args, ("tokenizer_model", "tokenizer_path"))
    dataset = _first_argument(args, ("prompt_data", "data_path", "dataset"))
    return sanitize_value(
        {
            "model": model,
            "tokenizer": tokenizer,
            "dataset": dataset,
            "model_metadata": _input_metadata(model),
            "dataset_metadata": _input_metadata(dataset),
        }
    )


def _collect_config(args: Any, runtime_env: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    env_path = Path(__file__).resolve().parents[2] / "configs" / "env.yaml"
    configured_env: Dict[str, Any] = {}
    if env_path.is_file():
        try:
            configured_env = (yaml.safe_load(env_path.read_text(encoding="utf-8")) or {}).get("env_vars", {})
        except (OSError, yaml.YAMLError, AttributeError) as exc:
            logger.warning(f"Failed to read configs/env.yaml for manifest: {exc}")
    keys = {key for key in os.environ if key.startswith(_ENV_PREFIXES)} | set(configured_env)
    environment = {key: os.environ.get(key, configured_env.get(key)) for key in sorted(keys)}
    hashes = (
        {"configs/env.yaml": f"sha256:{hashlib.sha256(env_path.read_bytes()).hexdigest()}"}
        if env_path.is_file()
        else {}
    )
    config_path = Path(getattr(args, "config", "")) if args is not None and getattr(args, "config", None) else None
    if config_path and config_path.is_file():
        hashes[config_path.name] = f"sha256:{hashlib.sha256(config_path.read_bytes()).hexdigest()}"
    arguments = vars(args) if args is not None and hasattr(args, "__dict__") else {}
    return {
        "arguments": sanitize_value(arguments),
        "environment": sanitize_value(environment),
        "runtime_env": sanitize_value(runtime_env or {}),
        "files": hashes,
    }


def _run_collectors(collectors: Mapping[str, Callable[[], Any]], timeout: float) -> Dict[str, Any]:
    executor = ThreadPoolExecutor(max_workers=len(collectors), thread_name_prefix="manifest")
    futures: Dict[Future, str] = {executor.submit(function): name for name, function in collectors.items()}
    done, pending = wait(futures, timeout=timeout)
    result: Dict[str, Any] = {}
    for future in done:
        name = futures[future]
        try:
            result[name] = future.result()
        except Exception as exc:
            logger.warning(f"Manifest collector {name} failed: {exc}")
    for future in pending:
        future.cancel()
        logger.warning(f"Manifest collector {futures[future]} exceeded the {timeout:.1f}s deadline")
    executor.shutdown(wait=False)
    return result


def collect_manifest(
    args: Any = None,
    runtime_env: Optional[Mapping[str, Any]] = None,
    argv: Optional[Sequence[str]] = None,
    timeout: float = COLLECTION_TIMEOUT_SECONDS,
    include_runtime: bool = True,
) -> Dict[str, Any]:
    """Collect a sanitized manifest within one best-effort deadline."""
    run_id = f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:8]}"
    collectors = {
        "code": _collect_code,
        "environment": _collect_environment,
        "hardware": _collect_hardware,
        "config": lambda: _collect_config(args, runtime_env),
    }
    collectors["inputs"] = lambda: _collect_inputs(args)
    collectors["training"] = lambda: _collect_training(args)
    if include_runtime:
        collectors["runtime"] = lambda: _collect_runtime(args)
    sections = _run_collectors(collectors, timeout)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "user": "<user>"
        if os.environ.get("USER") or os.environ.get("LOGNAME") or os.environ.get("USERNAME")
        else None,
        "hostname": "<host>" if socket.gethostname() else None,
        "command": {"argv": sanitize_argv(_canonical_argv(argv)), "working_directory": "."},
        **sections,
    }
    manifest = sanitize_value(manifest)
    verify_no_secrets(manifest)
    return manifest


def verify_no_secrets(manifest: Mapping[str, Any]) -> None:
    """Reject serialized output that still contains obvious private addresses
    or credentials."""
    serialized = json.dumps(manifest, sort_keys=True)
    if (
        _AUTHORIZATION.search(serialized)
        or _BEARER.search(serialized)
        or _INTERNAL_HOST.search(serialized)
        or _BARE_HOST_PORT.search(serialized)
    ):
        raise ManifestError("Manifest still contains a credential or internal hostname")
    for match in _ASSIGNMENT.finditer(serialized):
        key = re.sub(r"[^A-Za-z0-9]+", "_", match.group(1)).strip("_")
        if _SECRET_KEY.search(key) and REDACTED not in match.group(0):
            raise ManifestError("Manifest still contains a credential assignment")
    for pattern in (_IPV4, _IPV6):
        for match in pattern.finditer(serialized):
            if _mask_non_public_ip(match) != match.group(0):
                raise ManifestError("Manifest still contains a private network address")


def normalize_manifest(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize any compatible v1 manifest and reject malformed sections."""
    if not isinstance(raw, Mapping):
        raise ManifestError("Manifest root must be an object")
    version = raw.get("schema_version")
    version_match = re.fullmatch(r"(\d+)\.(\d+)", str(version))
    if not version_match:
        raise ManifestError(f"Invalid schema version: {version!r}")
    major = int(version_match.group(1))
    if major != 1:
        raise ManifestError(f"Unsupported schema major version: {major}")
    manifest = dict(raw)
    if "cli_args" in manifest and "command" not in manifest:
        manifest["command"] = {"argv": manifest.pop("cli_args"), "working_directory": "."}
    object_sections = ("code", "command", "config", "environment", "hardware", "runtime", "inputs", "training")
    for section in object_sections:
        value = manifest.setdefault(section, {})
        if not isinstance(value, Mapping):
            raise ManifestError(f"Manifest section {section!r} must be an object")
    nested_objects = {
        "code": ("relax", "megatron"),
        "config": ("arguments", "environment", "runtime_env", "files"),
        "environment": ("packages",),
        "training": ("parallel_topology",),
    }
    for section, fields in nested_objects.items():
        for field in fields:
            value = manifest[section].get(field)
            if field in manifest[section] and not isinstance(value, Mapping):
                raise ManifestError(f"Manifest field {section}.{field} must be an object")
    argv = manifest["command"].get("argv", [])
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ManifestError("Manifest command.argv must be a list of strings")
    manifest["schema_version"] = SCHEMA_VERSION
    return manifest


def load_manifest(path: str) -> Dict[str, Any]:
    """Load and validate a manifest JSON file."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Cannot read manifest {path}: {exc}") from exc
    return normalize_manifest(data)


def is_primary_process() -> bool:
    """Return whether the current process owns the single manifest write."""
    for variable in ("RANK", "SLURM_PROCID"):
        value = os.environ.get(variable)
        if value is not None:
            try:
                return int(value) == 0
            except ValueError:
                logger.warning(f"Ignoring invalid {variable}={value!r}")
    return True


def collect_and_save_manifest(
    args: Any,
    runtime_env: Optional[Mapping[str, Any]] = None,
    output: Optional[str] = None,
    include_runtime: bool = True,
) -> Optional[str]:
    """Collect and atomically save one manifest; failures never escape to
    training."""
    if not is_primary_process():
        return None
    try:
        manifest = collect_manifest(args, runtime_env, include_runtime=include_runtime)
        directory = Path(getattr(args, "tensorboard_dir", None) or Path.cwd())
        destination = Path(output) if output else directory / f"manifest_{manifest['run_id']}.json"
        _write_manifest(destination, manifest)
        logger.info(f"Experiment manifest saved to: {destination}")
        return str(destination)
    except Exception as exc:
        logger.warning(f"Failed to generate experiment manifest: {exc}", exc_info=True)
        return None


def _write_manifest(destination: Path, manifest: Mapping[str, Any]) -> None:
    payload = json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ManifestError(f"Manifest is too large: {len(payload)} bytes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def update_manifest_runtime(
    path: Optional[str], args: Any = None, runtime_env: Optional[Mapping[str, Any]] = None
) -> Optional[str]:
    """Best-effort update of one initial manifest after Ray initialization."""
    if not path or not is_primary_process():
        return None
    try:
        manifest = load_manifest(path)
        updates = _run_collectors({"runtime": lambda: _collect_runtime(args)}, COLLECTION_TIMEOUT_SECONDS)
        if "runtime" not in updates:
            return path
        manifest["runtime"] = sanitize_value(updates["runtime"])
        verify_no_secrets(manifest)
        _write_manifest(Path(path), manifest)
        return path
    except Exception as exc:
        logger.warning(f"Failed to update experiment manifest: {exc}", exc_info=True)
        return None


def _flatten(value: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            result.update(_flatten(item, path))
        else:
            result[path] = item
    return result


def diff_manifests(old: Mapping[str, Any], new: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return reproducibility-relevant field differences."""
    old_flat, new_flat = _flatten(old), _flatten(new)
    ignored = {"generated_at", "run_id", "command.working_directory"}
    return [
        {"field": field, "old": old_flat.get(field), "new": new_flat.get(field)}
        for field in sorted(set(old_flat) | set(new_flat))
        if field not in ignored and old_flat.get(field) != new_flat.get(field)
    ]


def _render_diff(differences: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# Manifest Diff", "", "| Field | Old | New |", "| --- | --- | --- |"]
    lines.extend(f"| `{item['field']}` | `{item['old']}` | `{item['new']}` |" for item in differences)
    return "\n".join(lines) + "\n"


def _exclude_manifest_output(code: Mapping[str, Any], manifest_path: Optional[str]) -> Dict[str, Any]:
    result = dict(code)
    relax = dict(result.get("relax", {}))
    if not manifest_path or not relax:
        return result
    root = _run(("git", "rev-parse", "--show-toplevel"), Path.cwd())
    try:
        relative = Path(manifest_path).resolve().relative_to(Path(root).resolve()).as_posix() if root else None
    except ValueError:
        relative = None
    if relative:
        untracked = [item for item in relax.get("untracked_files", []) if Path(item).as_posix() != relative]
        relax["untracked_files"] = untracked
        relax["dirty"] = bool(relax.get("tracked_dirty") or untracked)
        result["relax"] = relax
    return result


def validate_environment(manifest: Mapping[str, Any], manifest_path: Optional[str] = None) -> Dict[str, Any]:
    """Compare a recorded manifest with the current environment."""
    expected = normalize_manifest(manifest)
    current = collect_manifest(argv=expected["command"].get("argv", []))
    expected_environment = {
        "code": expected.get("code", {}),
        "environment": expected.get("environment", {}),
        "hardware": expected.get("hardware", {}),
        "configured_environment": expected.get("config", {}).get("environment", {}),
    }
    current_environment = {
        "code": _exclude_manifest_output(current.get("code", {}), manifest_path),
        "environment": current.get("environment", {}),
        "hardware": current.get("hardware", {}),
        "configured_environment": current.get("config", {}).get("environment", {}),
    }
    expected_runtime = expected.get("runtime", {})
    current_runtime = current.get("runtime", {})
    runtime_unavailable = (
        expected_runtime.get("mode") not in (None, "local") and current_runtime.get("mode") == "local"
    )
    if runtime_unavailable:
        expected_environment["runtime_observation"] = expected_runtime.get("mode")
        current_environment["runtime_observation"] = None
    elif expected_runtime.get("mode") == "local" or current_runtime.get("mode") != "local":
        expected_environment["runtime"] = expected_runtime
        current_environment["runtime"] = current_runtime
    expected_runtime_vars = expected.get("config", {}).get("runtime_env", {}).get("env_vars", {})
    if isinstance(expected_runtime_vars, Mapping):
        observed_vars = current.get("config", {}).get("environment", {})
        missing_runtime_vars = {
            key: value for key, value in expected_runtime_vars.items() if observed_vars.get(key) != value
        }
        if missing_runtime_vars:
            expected_environment["runtime_environment_observation"] = missing_runtime_vars
            current_environment["runtime_environment_observation"] = {
                key: observed_vars.get(key) for key in missing_runtime_vars
            }
    differences = diff_manifests(expected_environment, current_environment)
    missing = any(item["old"] is None or item["new"] is None for item in differences)
    warning_only = bool(differences) and all(
        item["field"] == "runtime_observation"
        or (item["field"].startswith("runtime_environment_observation.") and item["new"] is None)
        or _is_compatible_version_difference(item)
        for item in differences
    )
    return {
        "status": "PASS" if not differences else "WARN" if warning_only else "FAIL",
        "match_status": "MATCH" if not differences else "MISSING" if missing else "DIFF",
        "differences": differences,
    }


def _is_compatible_version_difference(difference: Mapping[str, Any]) -> bool:
    field = str(difference["field"])
    if field != "environment.python_version" and not field.startswith("environment.packages."):
        return False
    old = re.match(r"^(\d+)\.(\d+)", str(difference["old"]))
    new = re.match(r"^(\d+)\.(\d+)", str(difference["new"]))
    return bool(old and new and old.groups() == new.groups())


def build_reproduction_script(manifest: Mapping[str, Any], manifest_path: str) -> str:
    """Build a dry-run shell script, refusing any redacted command argument."""
    normalized = normalize_manifest(manifest)
    argv = normalized["command"].get("argv", [])
    if not argv or any(
        REDACTED in item or "<internal-" in item or any(character in item for character in ("\x00", "\r", "\n"))
        for item in argv
    ):
        raise ManifestError("Recorded command is missing or contains redacted values")
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    commit = normalized.get("code", {}).get("relax", {}).get("commit")
    if commit:
        lines.append(f"git checkout {shlex.quote(str(commit))}")
    packages = normalized.get("environment", {}).get("packages", {})
    if packages:
        specs = [f"{name}=={version}" for name, version in sorted(packages.items())]
        lines.append(f"python -m pip install {' '.join(shlex.quote(spec) for spec in specs)}")
    for name, value in _reproduction_environment(normalized).items():
        lines.append(f"export {name}={shlex.quote(value)}")
    lines.append(f"python -m relax.utils.manifest validate {shlex.quote(manifest_path)}")
    command = []
    for item in argv:
        if item.startswith("<home>"):
            suffix = item[len("<home>") :].replace("\\", "/")
            command.append(f'"${{HOME}}"{shlex.quote(suffix)}')
        else:
            command.append(shlex.quote(item))
    lines.append(" ".join(command))
    return "\n".join(lines) + "\n"


def _reproduction_environment(manifest: Mapping[str, Any]) -> Dict[str, str]:
    config = manifest.get("config", {})
    candidates = dict(config.get("environment", {}))
    runtime_vars = config.get("runtime_env", {}).get("env_vars", {})
    if isinstance(runtime_vars, Mapping):
        candidates.update(runtime_vars)
    safe: Dict[str, str] = {}
    for name, value in candidates.items():
        text_value = str(value)
        normalized_name = _normalize_key(str(name))
        if (
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(name))
            or (not str(name).startswith(_REPRO_ENV_PREFIXES) and str(name) not in _REPRO_EXACT_ENV)
            or _SECRET_KEY.search(normalized_name)
            or _LOCATION_KEY.search(normalized_name)
            or any(character in text_value for character in ("\x00", "\r", "\n"))
        ):
            continue
        if name == "PYTHONPATH" and not _safe_pythonpath(text_value):
            continue
        if name == "PYTHONBUFFERED" and not re.fullmatch(r"[0-9]+", text_value):
            continue
        if REDACTED in text_value or "<internal-" in text_value or text_value == "<slurm-nodes>":
            continue
        safe[str(name)] = text_value.replace("<home>", str(Path.home()))
    return safe


def _safe_pythonpath(value: str) -> bool:
    parts = re.split(r"[;:]", value)
    if not parts or any(not part for part in parts):
        return False
    for part in parts:
        normalized = part.replace("\\", "/")
        if not normalized.startswith("<home>/") or ".." in normalized[len("<home>/") :].split("/"):
            return False
    return True


def execute_reproduction(manifest: Mapping[str, Any]) -> int:
    """Execute a validated argv array after an explicit CLI confirmation."""
    normalized = normalize_manifest(manifest)
    argv = normalized["command"].get("argv", [])
    if not argv or any(
        REDACTED in item or "<internal-" in item or any(character in item for character in ("\x00", "\r", "\n"))
        for item in argv
    ):
        raise ManifestError("Recorded command is missing or contains redacted values")
    command = [item.replace("<home>", str(Path.home())) for item in argv]
    environment = os.environ.copy()
    environment.update(_reproduction_environment(normalized))
    try:
        return subprocess.run(command, check=False, env=environment).returncode
    except (OSError, ValueError) as exc:
        raise ManifestError(f"Cannot execute recorded command: {exc}") from exc


def _emit(value: Any) -> None:
    sys.stdout.write(f"{value}\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the four-command manifest CLI."""
    parser = argparse.ArgumentParser(description="Relax experiment manifest tool")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--output", required=True)
    generate.add_argument("--config")
    for name in ("validate", "check"):
        validator = commands.add_parser(name)
        validator.add_argument("manifest")
    differ = commands.add_parser("diff")
    differ.add_argument("old")
    differ.add_argument("new")
    differ.add_argument("--output")
    for name in ("reproduce", "rerun"):
        reproduce = commands.add_parser(name)
        reproduce.add_argument("manifest")
        mode = reproduce.add_mutually_exclusive_group(required=True)
        mode.add_argument("--dry-run", action="store_true")
        mode.add_argument("--confirm", action="store_true")
    parsed = parser.parse_args(argv)
    try:
        if parsed.command == "generate":
            path = collect_and_save_manifest(
                argparse.Namespace(tensorboard_dir=None, config=parsed.config), output=parsed.output
            )
            return 0 if path else 1
        if parsed.command in ("validate", "check"):
            report = validate_environment(load_manifest(parsed.manifest), parsed.manifest)
            _emit(json.dumps(report, indent=2))
            return 1 if report["status"] == "FAIL" else 0
        if parsed.command == "diff":
            report = diff_manifests(load_manifest(parsed.old), load_manifest(parsed.new))
            rendered = _render_diff(report)
            if parsed.output:
                Path(parsed.output).write_text(rendered, encoding="utf-8")
            else:
                _emit(rendered)
            return 0
        loaded = load_manifest(parsed.manifest)
        if parsed.dry_run:
            _emit(build_reproduction_script(loaded, parsed.manifest).rstrip())
            return 0
        report = validate_environment(loaded, parsed.manifest)
        if report["status"] == "FAIL":
            _emit(json.dumps(report, indent=2))
            return 1
        if report["status"] == "WARN":
            _emit(json.dumps(report, indent=2))
        return execute_reproduction(loaded)
    except ManifestError as exc:
        logger.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ManifestError",
    "SCHEMA_VERSION",
    "build_reproduction_script",
    "collect_and_save_manifest",
    "collect_manifest",
    "diff_manifests",
    "execute_reproduction",
    "load_manifest",
    "main",
    "normalize_manifest",
    "sanitize_argv",
    "sanitize_value",
    "update_manifest_runtime",
    "validate_environment",
    "verify_no_secrets",
]
