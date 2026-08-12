# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Generic S3-to-memory model loading utilities.

The canonical model source is passed across actors.
Consumers that require local files use an idempotent node-local download,
while SGLang can select direct streaming based on ``load_format``.

This implementation assumes ample SHM capacity (approximately 900 GB). It does
not reserve capacity across distinct downloads or use a root-level capacity
lock. _download_model_to_shm_once only checks the missing bytes for one model
while holding that model's lock. On the required one-Pod-one-job deployment,
Relax removes full weight shards after startup while retaining metadata needed
during training.
"""

import fcntl
import hashlib
import json
import os
import re
import shutil
import time

from relax.utils.logging_utils import get_logger
from relax.utils.model_source import LocalModel, ModelSource


logger = get_logger(__name__)

_MARKER_PREFIX = "relax_model"
_MODEL_WEIGHT_FILENAMES = {
    "adapter_model.bin",
    "flax_model.msgpack",
    "model.bin",
    "model.ckpt",
    "model.pt",
    "model.pth",
    "pytorch_model.bin",
    "tf_model.h5",
}
_MODEL_WEIGHT_SUFFIXES = (".distcp", ".onnx", ".safetensors")
_MODEL_WEIGHT_FILENAME_PATTERNS = (
    re.compile(r"(?:adapter_model|model|pytorch_model)-\d{5}-of-\d{5}\.(?:bin|ckpt|pt|pth)"),
    re.compile(r"consolidated(?:\.\d+)?\.(?:bin|pt|pth)"),
    re.compile(r"mp_rank_\d+(?:_\d+)?_model_states\.pt"),
    re.compile(r"(?:optim|optimizer)(?:_states?)?\.(?:bin|ckpt|distcp|pt|pth)"),
    re.compile(r"(?:rng|scheduler|training|trainer)[_-]state(?:_\d+)?\.(?:bin|ckpt|distcp|json|pt|pth)"),
)
_MANIFEST_VERSION = 1
_CLEANUP_LOCK_TIMEOUT_SECONDS = 300.0
_CLEANUP_LOCK_POLL_INTERVAL_SECONDS = 0.1


def is_s3_uri(uri) -> bool:
    return isinstance(uri, str) and uri.lower().startswith("s3://")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    rest = uri[len("s3://") :]
    bucket, _, prefix = rest.partition("/")
    if not bucket or not prefix:
        raise ValueError(f"S3 model URI must include a non-empty bucket and prefix: {uri!r}")
    return bucket, prefix


def _normalize_prefix(prefix: str) -> str:
    """Add a trailing slash so listing and relative-path stripping use the same
    prefix."""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return prefix


def _cache_identity(uri: str, endpoint: str | None) -> str:
    return uri if endpoint is None else f"{uri}\nendpoint={endpoint}"


def _shm_dest_dir(uri: str, shm_root: str, endpoint: str | None = None) -> str:
    digest = hashlib.sha1(_cache_identity(uri, endpoint).encode()).hexdigest()[:16]
    return os.path.join(shm_root, f"{_MARKER_PREFIX}_{digest}")


def _safe_join(root: str, rel: str) -> str:
    root = os.path.normpath(root)  # Normalize trailing separators before the containment check.
    dest = os.path.normpath(os.path.join(root, rel))
    if not (dest == root or dest.startswith(root + os.sep)):
        raise ValueError(f"unsafe path escapes cache root: {rel!r}")
    return dest


def _make_s3_client(*, endpoint, use_placeholder_credentials=False, use_path_style=False):
    import boto3
    from botocore.config import Config

    config_kwargs = dict(
        retries={"max_attempts": 10, "mode": "standard"},
        max_pool_connections=64,
        proxies={},  # Disable proxies for this client without mutating os.environ.
    )
    if use_path_style:
        config_kwargs["s3"] = {"addressing_style": "path"}
    client_kwargs = dict(endpoint_url=endpoint, config=Config(**config_kwargs))
    if use_placeholder_credentials:
        # Some S3-compatible gateways require signed requests while ignoring
        # the credential values. This behavior must be selected explicitly.
        client_kwargs.update(aws_access_key_id="mock", aws_secret_access_key="mock")
    return boto3.client("s3", **client_kwargs)


def _list_objects_with_size(cli, bucket, prefix):
    objs = []
    for page in cli.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            if not o["Key"].endswith("/"):
                objs.append((o["Key"], o["Size"]))
    return objs


def _list_objects(cli, bucket, prefix):
    return [k for k, _ in _list_objects_with_size(cli, bucket, prefix)]


def _missing_bytes(objs, prefix: str, dest: str) -> int:
    """Return remote bytes not already present locally with the expected
    size."""
    missing = 0
    for key, size in objs:
        rel = key[len(prefix) :]
        try:
            local = _safe_join(dest, rel)
        except ValueError:
            missing += size
            continue
        if os.path.exists(local) and os.path.getsize(local) == size:
            continue
        missing += size
    return missing


def _indexed_weight_paths(dest: str) -> set[str]:
    paths = set()
    for current_root, _dirs, files in os.walk(dest):
        for filename in files:
            if not filename.endswith(".index.json"):
                continue
            index_path = os.path.join(current_root, filename)
            try:
                with open(index_path) as index_file:
                    weight_map = json.load(index_file).get("weight_map", {})
            except (AttributeError, OSError, json.JSONDecodeError):
                continue
            if not isinstance(weight_map, dict):
                continue
            index_dir = os.path.relpath(current_root, dest)
            for shard in weight_map.values():
                if not isinstance(shard, str):
                    continue
                rel = os.path.normpath(os.path.join(index_dir, shard)) if index_dir != "." else shard
                try:
                    _safe_join(dest, rel)
                except ValueError:
                    continue
                paths.add(rel)
    return paths


def _is_model_weight_path(rel: str, indexed_weights: set[str]) -> bool:
    if rel in indexed_weights:
        return True
    filename = os.path.basename(rel).lower()
    if filename.endswith(_MODEL_WEIGHT_SUFFIXES):
        return True
    if filename in _MODEL_WEIGHT_FILENAMES:
        return True
    return any(pattern.fullmatch(filename) is not None for pattern in _MODEL_WEIGHT_FILENAME_PATTERNS)


def _write_model_manifest(dest: str, identity: str, objs, prefix: str) -> None:
    indexed_weights = _indexed_weight_paths(dest)
    files = []
    for key, size in objs:
        rel = key[len(prefix) :]
        _safe_join(dest, rel)
        files.append(
            {
                "path": rel,
                "size": size,
                "kind": "weight" if _is_model_weight_path(rel, indexed_weights) else "metadata",
            }
        )
    manifest = {"version": _MANIFEST_VERSION, "identity": identity, "files": files}
    path = dest + ".manifest.json"
    tmp = path + ".tmp"
    with open(tmp, "w") as manifest_file:
        json.dump(manifest, manifest_file, sort_keys=True)
        manifest_file.flush()
        os.fsync(manifest_file.fileno())
    os.replace(tmp, path)
    _fsync_parent(path)


def _read_model_manifest(dest: str, identity: str) -> dict | None:
    try:
        with open(dest + ".manifest.json") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    if manifest.get("version") != _MANIFEST_VERSION or manifest.get("identity") != identity:
        return None
    if not isinstance(manifest.get("files"), list):
        return None
    return manifest


def _write_ready_marker(path: str, identity: str) -> None:
    """Publish cache readiness only after its preceding writes are durable."""
    tmp = path + ".tmp"
    with open(tmp, "w") as marker_file:
        marker_file.write(identity)
        marker_file.flush()
        os.fsync(marker_file.fileno())
    os.replace(tmp, path)
    _fsync_parent(path)


def _fsync_parent(path: str) -> None:
    directory = os.path.dirname(path) or "."
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _manifest_files_complete(dest: str, manifest: dict) -> bool:
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            return False
        try:
            path = _safe_join(dest, entry["path"])
            expected_size = entry["size"]
        except (KeyError, TypeError, ValueError):
            return False
        if not isinstance(expected_size, int) or expected_size < 0:
            return False
        if not os.path.isfile(path) or os.path.getsize(path) != expected_size:
            return False
    return True


def _acquire_cleanup_lock(lock_file) -> None:
    deadline = time.monotonic() + _CLEANUP_LOCK_TIMEOUT_SECONDS
    while True:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for the S3 model SHM cache lock") from exc
            time.sleep(min(_CLEANUP_LOCK_POLL_INTERVAL_SECONDS, remaining))


def _download_one(cli, bucket, key, prefix, dest, buffer_size=8 * 1024 * 1024):
    rel = key[len(prefix) :]
    local = _safe_join(dest, rel)
    head = cli.head_object(Bucket=bucket, Key=key)
    size = head["ContentLength"]
    if os.path.exists(local) and os.path.getsize(local) == size:
        return
    os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
    tmp = local + ".part"
    try:
        resp = cli.get_object(Bucket=bucket, Key=key)
        body = resp["Body"]
        try:
            with open(tmp, "wb") as f:
                for chunk in body.iter_chunks(chunk_size=buffer_size):
                    f.write(chunk)
        finally:
            body.close()
        actual = os.path.getsize(tmp)
        if actual != size:
            raise IOError(f"size mismatch {key}: {actual} != {size}")
        os.replace(tmp, local)  # Publish the completed file atomically.
    except BaseException:
        # Remove partial files after stream, validation, or publish failures.
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _download_prefix(
    bucket, prefix, dest, *, endpoint, workers, retries, use_placeholder_credentials=False, use_path_style=False
):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cli = _make_s3_client(
        endpoint=endpoint,
        use_placeholder_credentials=use_placeholder_credentials,
        use_path_style=use_path_style,
    )
    keys = _list_objects(cli, bucket, prefix)
    if not keys:
        raise RuntimeError(f"no objects under s3://{bucket}/{prefix}")
    remaining = list(keys)
    for _ in range(retries + 1):
        failed = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_download_one, cli, bucket, k, prefix, dest): k for k in remaining}
            for fu in as_completed(futs):
                try:
                    fu.result()
                except Exception as e:
                    logger.warning(f"download failed {futs[fu]}: {e}")
                    failed.append(futs[fu])
        if not failed:
            logger.info(f"downloaded {len(keys)} objects -> {dest}")
            return
        remaining = failed
    raise RuntimeError(f"download failed after {retries} retries: {remaining}")


def _download_selected_objects(cli, bucket, keys, prefix, dest, *, workers, retries, description) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    remaining = list(keys)
    for _ in range(retries + 1):
        failed = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_download_one, cli, bucket, key, prefix, dest): key for key in remaining}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    logger.warning(f"{description} download failed {futures[future]}: {exc}")
                    failed.append(futures[future])
        if not failed:
            return
        remaining = failed
    raise RuntimeError(f"{description} download failed after {retries} retries: {remaining}")


def _download_model_metadata_to_shm_once(
    uri, dest, *, endpoint, workers, retries, use_placeholder_credentials=False, use_path_style=False
):
    """Download only files needed to construct a dummy-loaded SGLang model."""
    marker = dest + ".metadata.done"
    full_marker = dest + ".done"
    lock = dest + ".lock"
    os.makedirs(os.path.dirname(dest) or "/", exist_ok=True)
    with open(lock, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        for ready_marker in (full_marker, marker):
            if os.path.isdir(dest) and os.path.isfile(ready_marker):
                with open(ready_marker) as marker_file:
                    if marker_file.read().strip() == _cache_identity(uri, endpoint):
                        return

        bucket, prefix = _parse_s3_uri(uri)
        prefix = _normalize_prefix(prefix)
        cli = _make_s3_client(
            endpoint=endpoint,
            use_placeholder_credentials=use_placeholder_credentials,
            use_path_style=use_path_style,
        )
        objects = _list_objects_with_size(cli, bucket, prefix)
        if not objects:
            raise RuntimeError(f"no objects under s3://{bucket}/{prefix}")
        index_objects = [(key, size) for key, size in objects if key.lower().endswith(".index.json")]
        _download_selected_objects(
            cli,
            bucket,
            [key for key, _size in index_objects],
            prefix,
            dest,
            workers=workers,
            retries=retries,
            description="model index",
        )
        indexed_weights = _indexed_weight_paths(dest)
        metadata_objects = [
            (key, size) for key, size in objects if not _is_model_weight_path(key[len(prefix) :], indexed_weights)
        ]
        if not metadata_objects:
            raise RuntimeError(f"no model metadata under s3://{bucket}/{prefix}")
        metadata_bytes = _missing_bytes(metadata_objects, prefix, dest)
        free = _free_bytes(os.path.dirname(dest) or "/")
        if metadata_bytes > free * 0.95:
            raise RuntimeError(f"shm capacity is insufficient for model metadata: need={metadata_bytes} free={free}")

        keys = [key for key, _ in metadata_objects]
        _download_selected_objects(
            cli,
            bucket,
            keys,
            prefix,
            dest,
            workers=workers,
            retries=retries,
            description="metadata",
        )
        _write_ready_marker(marker, _cache_identity(uri, endpoint))
        logger.info(f"downloaded {len(keys)} model metadata objects -> {dest}")


def _download_model_to_shm_once(
    uri,
    dest,
    *,
    endpoint,
    workers,
    retries,
    capacity_margin=0.95,
    use_placeholder_credentials=False,
    use_path_style=False,
):
    marker = dest + ".done"
    lock = dest + ".lock"
    identity = _cache_identity(uri, endpoint)
    os.makedirs(os.path.dirname(dest) or "/", exist_ok=True)
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        marker_matches = False
        if os.path.exists(marker) and os.path.isdir(dest):
            with open(marker) as mf:
                marker_matches = mf.read().strip() == identity
            if marker_matches:
                manifest = _read_model_manifest(dest, identity)
                if manifest is not None and _manifest_files_complete(dest, manifest):
                    logger.info(f"shm cache hit: {dest}")
                    return
        bucket, prefix = _parse_s3_uri(uri)
        prefix = _normalize_prefix(prefix)
        cli = _make_s3_client(
            endpoint=endpoint,
            use_placeholder_credentials=use_placeholder_credentials,
            use_path_style=use_path_style,
        )
        objs = _list_objects_with_size(cli, bucket, prefix)
        if not objs:
            raise RuntimeError(f"no objects under s3://{bucket}/{prefix}")
        if marker_matches and _missing_bytes(objs, prefix, dest) == 0:
            _write_model_manifest(dest, identity, objs, prefix)
            logger.info(f"migrated legacy shm cache manifest: {dest}")
            return
        # Check resumable missing bytes under the model lock after marker validation.
        need = _missing_bytes(objs, prefix, dest)
        free = _free_bytes(os.path.dirname(dest) or "/")
        if need > free * capacity_margin:
            raise RuntimeError(
                f"Insufficient SHM capacity (need={need / 1e9:.1f}GB "
                f"free={free / 1e9:.1f}GB at {os.path.dirname(dest)}); "
                "contact the Relax team for model-specific support for oversized checkpoints"
            )
        _download_prefix(
            bucket,
            prefix,
            dest,
            endpoint=endpoint,
            workers=workers,
            retries=retries,
            use_placeholder_credentials=use_placeholder_credentials,
            use_path_style=use_path_style,
        )
        _write_model_manifest(dest, identity, objs, prefix)
        _write_ready_marker(marker, identity)


def _free_bytes(path: str) -> int:
    return shutil.disk_usage(path).free


def _resolve_endpoint(args):
    return args.model_source.endpoint


def _resolve_shm_root(args) -> str:
    """Resolve an existing SHM cache root without falling back to disk."""
    root = getattr(args, "s3_model_shm_root", None) or "/dev/shm"
    if not os.path.isdir(root):
        raise RuntimeError(
            f"S3 model SHM root does not exist or is not a directory: {root!r}. "
            "Create the memory-backed directory before launch; disk fallback is intentionally disabled."
        )
    return root


def maybe_resolve_s3_model_to_shm(uri: str, args) -> str:
    if not is_s3_uri(uri):
        return uri
    config = getattr(args, "model_source", None)
    if config is None or uri != config.uri:
        return uri
    root = _resolve_shm_root(args)
    endpoint = _resolve_endpoint(args)
    workers = getattr(args, "s3_model_download_workers", None) or 20
    dest = _shm_dest_dir(uri, root, endpoint)
    _download_model_to_shm_once(
        uri,
        dest,
        endpoint=endpoint,
        workers=workers,
        retries=3,
        use_placeholder_credentials=config.credential_mode == "placeholder",
        use_path_style=config.addressing_style == "path",
    )
    logger.info(f"model s3->shm resolved: {uri} -> {dest}")
    return dest


def resolve_s3_model_metadata_to_shm(uri: str, args) -> str:
    """Resolve an S3 URI to a local directory without downloading weights."""
    if not is_s3_uri(uri):
        return uri
    config = getattr(args, "model_source", None)
    if config is None or uri != config.uri:
        return uri
    root = _resolve_shm_root(args)
    dest = _shm_dest_dir(uri, root, config.endpoint)
    workers = getattr(args, "s3_model_download_workers", None) or 20
    _download_model_metadata_to_shm_once(
        uri,
        dest,
        endpoint=_resolve_endpoint(args),
        workers=workers,
        retries=3,
        use_placeholder_credentials=config.credential_mode == "placeholder",
        use_path_style=config.addressing_style == "path",
    )
    logger.info(f"model metadata s3->shm resolved: {uri} -> {dest}")
    return dest


def get_s3_model_cached_path(uri: str, args) -> str | None:
    """Return a complete node-local S3 model path without downloading."""
    if not is_s3_uri(uri):
        return None
    config = getattr(args, "model_source", None)
    if config is None or uri != config.uri:
        return None
    root = getattr(args, "s3_model_shm_root", None) or "/dev/shm"
    if not os.path.isdir(root):
        return None
    dest = _shm_dest_dir(uri, root, config.endpoint)
    marker = dest + ".done"
    if not os.path.isdir(dest) or not os.path.isfile(marker):
        return None
    try:
        with open(marker) as marker_file:
            return dest if marker_file.read().strip() == _cache_identity(uri, config.endpoint) else None
    except OSError:
        return None


def cleanup_s3_model_weights_from_shm(args) -> tuple[int, int]:
    """Remove downloaded weight shards while preserving model metadata.

    The full-cache marker is removed before any shard so an interrupted cleanup
    can never be mistaken for a complete cache hit. The metadata marker keeps
    dummy loading and later config/tokenizer reads usable. Repeated calls are
    safe and finish an interrupted cleanup.

    Returns:
        A ``(removed_files, removed_bytes)`` tuple.
    """
    if getattr(args, "disable_s3_model_cleanup", False):
        return 0, 0
    config = getattr(args, "model_source", None)
    if config is None or not is_s3_uri(config.uri):
        return 0, 0
    root = getattr(args, "s3_model_shm_root", None) or "/dev/shm"
    if not os.path.isdir(root):
        return 0, 0
    dest = _shm_dest_dir(config.uri, root, config.endpoint)
    if not os.path.isdir(dest):
        return 0, 0

    identity = _cache_identity(config.uri, config.endpoint)
    full_marker = dest + ".done"
    metadata_marker = dest + ".metadata.done"
    lock = dest + ".lock"
    removed_files = 0
    removed_bytes = 0

    with open(lock, "w") as lock_file:
        _acquire_cleanup_lock(lock_file)
        marker_matches = False
        for marker in (full_marker, metadata_marker):
            try:
                with open(marker) as marker_file:
                    marker_matches = marker_file.read().strip() == identity
            except OSError:
                continue
            if marker_matches:
                break
        if not marker_matches:
            return 0, 0

        manifest = _read_model_manifest(dest, identity)
        if manifest is None:
            logger.warning(f"skip S3 model SHM cleanup because the download manifest is missing or invalid: {dest}")
            return 0, 0
        for entry in manifest["files"]:
            if not isinstance(entry, dict):
                logger.warning(f"skip S3 model SHM cleanup because the manifest is invalid: {dest}")
                return 0, 0
            if entry.get("kind") != "metadata":
                continue
            try:
                path = _safe_join(dest, entry["path"])
                expected_size = entry["size"]
            except (KeyError, TypeError, ValueError):
                logger.warning(f"skip S3 model SHM cleanup because the manifest is invalid: {dest}")
                return 0, 0
            if not isinstance(expected_size, int) or expected_size < 0:
                logger.warning(f"skip S3 model SHM cleanup because the manifest is invalid: {dest}")
                return 0, 0
            if not os.path.isfile(path) or os.path.getsize(path) != expected_size:
                logger.warning(f"skip S3 model SHM cleanup because metadata is incomplete: {path}")
                return 0, 0

        _write_ready_marker(metadata_marker, identity)

        try:
            os.remove(full_marker)
        except FileNotFoundError:
            pass

        for entry in manifest["files"]:
            if entry.get("kind") != "weight":
                continue
            try:
                path = _safe_join(dest, entry["path"])
            except (KeyError, TypeError, ValueError):
                logger.warning(f"ignore invalid weight entry in S3 model manifest: {entry!r}")
                continue
            try:
                size = os.path.getsize(path)
                os.remove(path)
            except FileNotFoundError:
                continue
            removed_files += 1
            removed_bytes += size

    if removed_files:
        logger.info(f"cleaned {removed_files} S3 model weight shards ({removed_bytes / 1e9:.1f} GB) from SHM: {dest}")
    return removed_files, removed_bytes


def prepare_local_model(args, *, completeness: str = "full") -> LocalModel:
    """Prepare a full or metadata-only node-local model view."""
    if completeness not in {"full", "metadata"}:
        raise ValueError(f"Unsupported local model completeness: {completeness!r}")
    source = getattr(args, "model_source", None)
    if source is None:
        source = ModelSource(uri=args.hf_checkpoint)
    resolver = maybe_resolve_s3_model_to_shm if completeness == "full" else resolve_s3_model_metadata_to_shm
    path = resolver(source.uri, args)
    return LocalModel(source=source, path=path, completeness=completeness)


def prepare_model_maybe_update_args(args, *, completeness: str = "full") -> LocalModel:
    """Prepare the model and update process-private model arguments in
    place."""
    local_model = prepare_local_model(args, completeness=completeness)
    source_uri = local_model.source.uri
    local_path = local_model.path
    if local_path == source_uri:
        return local_model

    args.hf_checkpoint = local_path
    for name in ("tokenizer_model",):
        if getattr(args, name, None) == source_uri:
            setattr(args, name, local_path)
    if completeness == "full" and getattr(args, "load", None) == source_uri:
        args.load = local_path
    return local_model
