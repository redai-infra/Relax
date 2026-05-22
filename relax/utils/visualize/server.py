# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Rollout result viewer server.

A lightweight FastAPI app for previewing the JSONL files written by
:func:`relax.utils.training.train_dump_utils.save_rollout_result_jsonl`
and :func:`save_eval_summary_jsonl`. Adapted from rlsp's
``jsonl_server.py`` with two changes:

- Auto-discovers ``train/`` and ``eval/`` subdirs under the data dir
  and exposes a ``/api/jsonl/subdirs`` endpoint so the UI can render
  a toggle. Falls back to a single anonymous bucket when the data dir
  is flat.
- Uses :mod:`relax.utils.logging_utils` for logging.

Usage::

    python -m relax.utils.visualize <save>/rollout_result --port 8080
"""

import argparse
import gc
import json
import re
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from relax.utils.logging_utils import get_logger
from relax.utils.visualize.templates import get_jsonl_viewer_html


logger = get_logger(__name__)

# Used when the data dir is flat (no train/eval subdirs).
_DEFAULT_SUBDIR = "default"


@dataclass
class CacheEntry:
    """A single cache entry with size tracking."""

    data: Any
    size_bytes: int
    access_count: int = 0


class LRUDataCache:
    """Memory-bounded LRU cache for loaded JSONL files."""

    def __init__(self, max_memory_mb: int = 2048, max_entries: int = 20):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_memory_bytes = max_memory_mb * 1024 * 1024
        self._max_entries = max_entries
        self._current_memory_bytes = 0
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def _estimate_size(self, obj: Any, seen: Optional[set] = None) -> int:
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, dict):
            size += sum(self._estimate_size(k, seen) + self._estimate_size(v, seen) for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            size += sum(self._estimate_size(item, seen) for item in obj)
        elif isinstance(obj, str):
            size += len(obj)
        return size

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key].access_count += 1
                self._stats["hits"] += 1
                return self._cache[key].data
            self._stats["misses"] += 1
            return None

    def put(self, key: str, data: Any, file_size_bytes: int) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return
            estimated_size = max(file_size_bytes, self._estimate_size(data))
            while self._cache and (
                self._current_memory_bytes + estimated_size > self._max_memory_bytes
                or len(self._cache) >= self._max_entries
            ):
                self._evict_one()
            self._cache[key] = CacheEntry(data=data, size_bytes=estimated_size)
            self._current_memory_bytes += estimated_size

    def _evict_one(self) -> None:
        if not self._cache:
            return
        key, entry = self._cache.popitem(last=False)
        self._current_memory_bytes -= entry.size_bytes
        self._stats["evictions"] += 1
        entry.data = None
        del entry
        logger.debug(f"Evicted cache entry: {key}")

    def clear(self) -> None:
        with self._lock:
            for entry in self._cache.values():
                entry.data = None
            self._cache.clear()
            self._current_memory_bytes = 0
        gc.collect()
        logger.info("Cache cleared and garbage collected")

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._cache),
                "memory_mb": round(self._current_memory_bytes / (1024 * 1024), 2),
                "max_memory_mb": round(self._max_memory_bytes / (1024 * 1024), 2),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
            }


def _discover_subdirs(data_path: Path) -> Dict[str, Path]:
    """Discover JSONL buckets under ``data_path``.

    Returns a mapping ``{name: directory}``. If ``train/`` and/or ``eval/``
    exist they become the named buckets; otherwise the data dir itself is
    exposed under :data:`_DEFAULT_SUBDIR`.
    """
    found = {name: data_path / name for name in ("train", "eval") if (data_path / name).is_dir()}
    if found:
        return found
    return {_DEFAULT_SUBDIR: data_path}


def create_app(
    data_dir: str,
    cache_memory_mb: int = 4096,
    cache_max_entries: int = 20,
    base_path: str = "",
) -> FastAPI:
    """Create the FastAPI application for the rollout result viewer."""

    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_path}")

    if base_path:
        if not base_path.startswith("/"):
            base_path = f"/{base_path}"
        if base_path.endswith("/"):
            base_path = base_path.rstrip("/")

    data_cache = LRUDataCache(max_memory_mb=cache_memory_mb, max_entries=cache_max_entries)

    app = FastAPI(
        title="Relax Rollout Result Viewer",
        description="Lightweight preview for Relax rollout_result/*.jsonl files",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.data_dir = data_path
    app.state.data_cache = data_cache
    app.state.base_path = base_path

    @app.on_event("shutdown")
    async def cleanup():
        logger.info("Shutting down - clearing cache...")
        data_cache.clear()
        gc.collect()

    def _resolve_subdir(subdir: str) -> Path:
        buckets = _discover_subdirs(data_path)
        if subdir not in buckets:
            raise HTTPException(status_code=404, detail=f"Subdir not found: {subdir}")
        return buckets[subdir]

    def _get_steps(subdir: str) -> List[Dict[str, Any]]:
        directory = _resolve_subdir(subdir)
        steps: List[Dict[str, Any]] = []
        pattern = re.compile(r"^(\d+)\.jsonl$")
        for f in sorted(directory.iterdir()):
            if f.is_file() and f.suffix == ".jsonl":
                match = pattern.match(f.name)
                if match:
                    steps.append(
                        {
                            "filename": f.name,
                            "step": int(match.group(1)),
                            "size_bytes": f.stat().st_size,
                        }
                    )
        steps.sort(key=lambda x: x["step"])
        return steps

    def _load_jsonl_file(subdir: str, filename: str) -> Dict[str, Any]:
        directory = _resolve_subdir(subdir)
        file_path = directory / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        cache_key = f"{subdir}/{filename}"
        cached_data = data_cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        logger.info(f"Loading JSONL file: {file_path}")
        try:
            file_size = file_path.stat().st_size
            samples = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {filename}: {e}")
                        continue

            result = {"filename": filename, "subdir": subdir, "samples": samples, "total": len(samples)}
            data_cache.put(cache_key, result, file_size)
            logger.info(f"Cached: {cache_key} ({file_size / (1024 * 1024):.1f} MB, {len(samples)} samples)")
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load file: {e}")

    # ============================ Routes ============================

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return get_jsonl_viewer_html(str(data_path), base_path)

    @app.get("/api/jsonl/subdirs")
    async def list_subdirs():
        buckets = _discover_subdirs(data_path)
        return {"data_dir": str(data_path), "subdirs": list(buckets.keys())}

    @app.get("/api/jsonl/{subdir}/steps")
    async def list_steps(subdir: str):
        steps = _get_steps(subdir)
        if not steps:
            raise HTTPException(status_code=404, detail=f"No JSONL files found in {subdir}")
        return {"subdir": subdir, "steps": steps, "total": len(steps)}

    @app.get("/api/jsonl/{subdir}/file/{filename}")
    async def get_file(subdir: str, filename: str):
        return _load_jsonl_file(subdir, filename)

    @app.get("/api/jsonl/{subdir}/file/{filename}/samples")
    async def get_samples(
        subdir: str,
        filename: str,
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=1000),
    ):
        data = _load_jsonl_file(subdir, filename)
        samples = data.get("samples", [])
        total = len(samples)
        start = (page - 1) * page_size
        end = min(start + page_size, total)
        return {
            "subdir": subdir,
            "filename": filename,
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size,
            "samples": samples[start:end],
        }

    @app.get("/api/cache/stats")
    async def cache_stats():
        return data_cache.get_stats()

    @app.post("/api/cache/clear")
    async def clear_cache():
        data_cache.clear()
        return {"status": "ok", "message": "Cache cleared"}

    return app


def main():
    """Entry point for ``python -m relax.utils.visualize``."""
    parser = argparse.ArgumentParser(
        description="Relax Rollout Result Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Web viewer (default)
    python -m relax.utils.visualize <save>/rollout_result

    # Web viewer with custom port and base path (for reverse proxy)
    python -m relax.utils.visualize <save>/rollout_result \\
        --port 8081 --base-path /absproxy/8081

    # Terminal viewer (textual TUI)
    python -m relax.utils.visualize <save>/rollout_result --tui
        """,
    )
    parser.add_argument(
        "data_dir",
        type=str,
        metavar="DATA_DIR",
        help="Path to the rollout_result directory (containing train/ and/or eval/ subdirs, "
        "or a flat directory of {step}.jsonl files).",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Launch the terminal UI instead of the web viewer (requires 'textual' and 'rich').",
    )
    parser.add_argument(
        "--mask-str",
        type=str,
        default=r"<\|image_pad\|>|<\|imgpad\|>|<\|audio_comp_pad\|>",
        help="Regex of substrings to mask with '*' in the TUI (default: multimodal pad tokens).",
    )
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to (default: 0.0.0.0)")
    parser.add_argument(
        "--cache-memory", type=int, default=4096, help="Maximum cache memory in MB (default: 4096 = 4GB)"
    )
    parser.add_argument("--cache-entries", type=int, default=20, help="Maximum number of cached files (default: 20)")
    parser.add_argument(
        "--base-path",
        type=str,
        default="",
        help="Base URL path for reverse proxy support (e.g., '/absproxy/8080')",
    )

    args = parser.parse_args()

    data_path = Path(args.data_dir).resolve()
    if not data_path.exists():
        print(f"Error: Data directory does not exist: {data_path}")
        return 1

    if args.tui:
        import traceback as _tb

        from relax.utils.visualize.tui import run as run_tui

        try:
            run_tui(str(data_path), mask_str=args.mask_str)
        except (ImportError, ValueError) as e:
            # Expected, user-facing errors — short message is enough.
            print(f"\nTUI failed to start: {e}")
            return 1
        except Exception:
            # Anything else (textual mount/render bug, JSON edge case, ...).
            # Print after textual has restored the terminal so it stays visible.
            print("\nTUI crashed. Full traceback:")
            _tb.print_exc()
            return 1
        return 0

    buckets = _discover_subdirs(data_path)

    print("=" * 60)
    print("Relax Rollout Result Viewer")
    print("=" * 60)
    print(f"Data directory: {data_path}")
    for name, path in buckets.items():
        n_files = len(list(path.glob("*.jsonl")))
        marker = "(auto)" if name == _DEFAULT_SUBDIR else ""
        print(f"  {name:<8} {marker:<7} {n_files} jsonl file(s)  ({path})")
    print()
    print("Cache settings:")
    print(f"  Max memory:   {args.cache_memory} MB")
    print(f"  Max entries:  {args.cache_entries} files")
    print()
    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"Open in browser:  http://localhost:{args.port}")
    if args.base_path:
        print(f"Base path:        {args.base_path}")
    print("=" * 60)

    app = create_app(
        str(data_path),
        cache_memory_mb=args.cache_memory,
        cache_max_entries=args.cache_entries,
        base_path=args.base_path,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
