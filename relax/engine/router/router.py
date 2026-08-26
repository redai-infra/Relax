import argparse
import asyncio
import json
import time
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from relax.engine.router.request_version_ledger import PublicationError, RequestVersionLedger
from relax.utils.logging_utils import get_logger
from relax.utils.misc import load_function


logger = get_logger(__name__)

PUBLICATION_PREFIX = "_relax/weight_publication"


def run_router(args):
    """Run the Slime router with the specified configuration."""
    # Initialize the router with tokenizer and lazy worker initialization
    slime_router = SlimeRouter(args, verbose=False)

    # Start the server
    uvicorn.run(slime_router.app, host=args.sglang_router_ip, port=args.sglang_router_port, log_level="info")


class SlimeRouter:
    def __init__(self, args, verbose=False):
        """Initialize the slime-router with SGLang router address."""
        self.args = args
        self.verbose = verbose

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            asyncio.create_task(self._health_check_loop())
            yield

        self.app = FastAPI(lifespan=lifespan)

        # URL -> Active Request Count (load state)
        self.worker_request_counts: dict[str, int] = {}
        # URL -> Consecutive Failures
        self.worker_failure_counts: dict[str, int] = {}
        # Quarantined workers excluded from routing pool
        self.dead_workers: set[str] = set()
        self.max_weight_version = None
        self.targeted_publication_enabled = bool(
            getattr(args, "hybrid_dcs_weight_sync", False)
            and getattr(args, "enable_cross_version_kv_continuation", False)
        )
        self.request_version_ledger = RequestVersionLedger()

        # Sticky-session routing: pin a routing key (read from ``sticky_header``) to a
        # worker URL so repeated requests for the same key reuse that worker's prefix/KV
        # cache. Each entry is ``[worker_url, last_seen_monotonic]``; idle entries are
        # evicted on the health-check cadence (see ``_evict_idle_sticky``).
        self.sticky_enabled = getattr(args, "slime_router_sticky", False)
        self.sticky_header = "X-SMG-Routing-Key"
        self.sticky_idle_secs = getattr(args, "slime_router_sticky_idle_secs", 600.0)
        self.sticky_map: dict[str, list] = {}
        self.sticky_stats = {"hit": 0, "assigned": 0, "remap": 0, "evicted": 0, "no_routing_key": 0}

        max_connections = getattr(args, "slime_router_max_connections", None)
        if max_connections is None:
            max_connections = (
                args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
            )

        timeout = getattr(args, "slime_router_timeout", None)

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_connections),
            timeout=httpx.Timeout(timeout),
        )
        retirement_timeout = float(getattr(args, "targeted_retirement_timeout_seconds", 15.0))
        self.control_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max(max_connections, 64)),
            timeout=httpx.Timeout(min(retirement_timeout, 5.0)),
        )

        self._setup_routes()

        for middleware_path in args.slime_router_middleware_paths or []:
            if self.verbose:
                print(f"[slime-router] Loading middleware from: {middleware_path}")
            middleware = load_function(middleware_path)
            self.app.add_middleware(middleware, router=self)

    def _setup_routes(self):
        """Setup all the HTTP routes."""
        # sglang-router api
        self.app.post("/add_worker")(self.add_worker)
        self.app.get("/list_workers")(self.list_workers)
        self.app.post("/retrieve_from_text")(self.retrieve_from_text)
        if self.targeted_publication_enabled:
            self.app.get(f"/{PUBLICATION_PREFIX}/state")(self.publication_state)
            self.app.post(f"/{PUBLICATION_PREFIX}/prepare")(self.prepare_publication)
            self.app.post(f"/{PUBLICATION_PREFIX}/commit")(self.commit_publication)
            self.app.post(f"/{PUBLICATION_PREFIX}/fail")(self.fail_publication)
        # Catch-all route for proxying to SGLang - must be registered LAST
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)

    async def _check_worker_health(self, url):
        """Encapsulated health check logic for better maintainability."""
        try:
            response = await self.client.get(f"{url}/health", timeout=5.0)
            if response.status_code == 200:
                return url, True
            logger.debug(f"[slime-router] Worker {url} is unhealthy (Status: {response.status_code})")
        except Exception as e:
            logger.debug(f"[slime-router] Worker {url} health check failed: {e}")
        return url, False

    async def _health_check_loop(self):
        """Background loop to monitor worker health and adjust routing pool."""
        interval = self.args.rollout_health_check_interval
        threshold = self.args.slime_router_health_check_failure_threshold

        while True:
            try:
                await asyncio.sleep(interval)

                self._evict_idle_sticky()

                urls = [u for u in self.worker_request_counts if u not in self.dead_workers]
                if not urls:
                    continue

                results = await asyncio.gather(*(self._check_worker_health(url) for url in urls))

                for url, is_healthy in results:
                    if not is_healthy:
                        failures = self.worker_failure_counts.get(url, 0) + 1
                        self.worker_failure_counts[url] = failures

                        if failures >= threshold:
                            logger.warning(
                                f"[slime-router] Worker {url} failed {threshold} consecutive health checks. Marking as DEAD."
                            )
                            self.dead_workers.add(url)
                            # TODO (chenyang): Connect back 'dead' workers requires a mechanism to sync
                            # model versions to avoid off-policy issues from stale weights, since these
                            # dead workers' parameters may not be refitted.
                    else:
                        self.worker_failure_counts[url] = 0

                logger.debug(
                    f"[slime-router] Health check complete. {len(self.worker_request_counts) - len(self.dead_workers)} workers healthy."
                )

                if self.sticky_enabled:
                    logger.info(
                        f"[slime-router] Sticky: keys={len(self.sticky_map)} hit={self.sticky_stats['hit']} "
                        f"assigned={self.sticky_stats['assigned']} remap={self.sticky_stats['remap']} "
                        f"evicted={self.sticky_stats['evicted']} no_routing_key={self.sticky_stats['no_routing_key']}"
                    )

            except asyncio.CancelledError:
                logger.warning("[slime-router] Background health check loop is being cancelled.")
                raise
            except Exception as e:
                logger.error(f"[slime-router] Unexpected error in health check loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def proxy(self, request: Request, path: str):
        """Proxy all other requests to the SGLang router."""
        # Forward all other paths to SGLang router
        routing_key = request.headers.get(self.sticky_header) if self.sticky_enabled else None
        use_request_version = self.targeted_publication_enabled and path == "generate"
        body = await request.body()
        rid = None
        payload = None
        if use_request_version:
            try:
                payload = json.loads(body)
            except (json.JSONDecodeError, TypeError):
                return JSONResponse(status_code=400, content={"error": "targeted publication requires a JSON body"})
            rid = payload.get("rid") if isinstance(payload, dict) else None
            if not isinstance(rid, str) or not rid:
                return JSONResponse(
                    status_code=400, content={"error": "targeted publication requires a non-empty rid"}
                )
            try:
                request_version = await self.request_version_ledger.register_selected(
                    rid,
                    lambda: self._use_url(routing_key),
                )
            except PublicationError as exc:
                return JSONResponse(status_code=503, content={"error": str(exc), "retryable": False})
            worker_url = request_version.worker
            base_extra_key = payload.get("extra_key", "")
            if base_extra_key is not None and not isinstance(base_extra_key, str):
                await self.request_version_ledger.unregister(rid)
                self._finish_url(worker_url)
                return JSONResponse(status_code=400, content={"error": "extra_key must be a string"})
            payload["extra_key"] = f"{base_extra_key}:weight-version:{request_version.kv_epoch_version}"
            body = json.dumps(payload, separators=(",", ":")).encode()
        else:
            worker_url = self._use_url(routing_key)
        url = f"{worker_url}/{path}"

        try:
            headers = dict(request.headers)
            headers.pop("content-length", None)
            response = await self.client.request(request.method, url, content=body, headers=headers)
            # Eagerly read content so we can return JSON (not streaming)
            content = await response.aread()
            content_type = response.headers.get("content-type", "")

            # Strip hop-by-hop and content-length headers from the upstream response.
            # When we re-serialize (especially JSON), the body size may differ from the
            # original Content-Length, causing uvicorn to raise
            # "Response content longer than Content-Length".
            # Let Starlette recompute Content-Length from the actual response body.
            excluded_headers = {"content-length", "transfer-encoding", "content-encoding"}
            filtered_headers = {k: v for k, v in response.headers.items() if k.lower() not in excluded_headers}

            try:
                # Prefer parsing JSON if possible
                data = json.loads(content)
                return JSONResponse(
                    content=data,
                    status_code=response.status_code,
                    headers=filtered_headers,
                )
            except Exception:
                # Fall back to raw body with original content type
                return Response(
                    content=content,
                    status_code=response.status_code,
                    headers=filtered_headers,
                    media_type=content_type or None,
                )
            finally:
                if response is not None:
                    await response.aclose()

        except httpx.TransportError as exc:
            logger.warning(
                "Upstream transport failed for %s %s; returning retryable 502: %s",
                request.method,
                path,
                type(exc).__name__,
            )
            return JSONResponse(
                status_code=502,
                content={"error": "upstream transport failure", "retryable": True},
            )
        finally:
            if rid is not None:
                await self.request_version_ledger.unregister(rid)
            self._finish_url(worker_url)

    async def publication_state(self):
        return await self.request_version_ledger.snapshot()

    async def prepare_publication(self, request: Request):
        payload = await request.json()
        target_version = int(payload.get("target_version", -1))
        target_actor_step = int(payload.get("target_actor_step", -1))
        max_gap = int(payload.get("max_gap", -1))
        max_actor_step_gap = int(payload.get("max_actor_step_gap", -1))
        timeout_seconds = float(payload.get("timeout_seconds", 15.0))
        publication_id = str(payload.get("publication_id", "")) or None
        started_at = time.monotonic()

        async def abort_request(worker: str, rid: str) -> None:
            async with self.control_client.stream("POST", f"{worker}/abort_request", json={"rid": rid}) as response:
                response.raise_for_status()
                await response.aread()

        plan = await self.request_version_ledger.prepare(
            target_version=target_version,
            target_actor_step=target_actor_step,
            max_gap=max_gap,
            max_actor_step_gap=max_actor_step_gap,
            abort_request=abort_request,
            timeout_seconds=timeout_seconds,
            publication_id=publication_id,
        )
        result = plan.to_dict()
        result["prepare_seconds"] = time.monotonic() - started_at
        logger.info(
            "DCS_TARGETED_RETIRE event=prepare publication_id=%s target_version=%s target_actor_step=%s "
            "active_requests=%s expired_requests=%s publication_gap_expired=%s "
            "actor_step_gap_expired=%s safe_requests=%s workers=%s prepare_seconds=%.6f",
            plan.publication_id,
            plan.target_version,
            plan.target_actor_step,
            plan.active_request_count,
            len(plan.expired_rids),
            len(plan.publication_gap_expired_rids),
            len(plan.actor_step_gap_expired_rids),
            plan.active_request_count - len(plan.expired_rids),
            len(plan.expired_by_worker),
            result["prepare_seconds"],
        )
        return result

    async def commit_publication(self, request: Request):
        payload = await request.json()
        plan = await self.request_version_ledger.commit(
            str(payload.get("publication_id", "")),
            int(payload.get("target_version", -1)),
            int(payload.get("target_actor_step", -1)),
        )
        logger.info(
            "DCS_TARGETED_RETIRE event=commit publication_id=%s target_version=%s "
            "target_actor_step=%s expired_requests=%s",
            plan.publication_id,
            plan.target_version,
            plan.target_actor_step,
            len(plan.expired_rids),
        )
        return {"status": "committed", **plan.to_dict()}

    async def fail_publication(self, request: Request):
        payload = await request.json()
        plan = await self.request_version_ledger.fail(
            str(payload.get("publication_id", "")),
            int(payload.get("target_version", -1)),
            int(payload.get("target_actor_step", -1)),
            str(payload.get("reason", "weight publication failed")),
        )
        logger.error(
            "DCS_TARGETED_RETIRE event=fail publication_id=%s target_version=%s reason=%s",
            plan.publication_id,
            plan.target_version,
            payload.get("reason", "weight publication failed"),
        )
        return {"status": "failed", **plan.to_dict()}

    async def add_worker(self, request: Request):
        """Add a new worker to the router.
        Supports providing the URL via query string or JSON body.
        Examples:
        - POST /add_worker?url=http://127.0.0.1:10090
        - POST /add_worker  with body {"url": "http://127.0.0.1:10090"}
        """
        # 1) Prefer query param
        worker_url = request.query_params.get("url") or request.query_params.get("worker_url")

        # 2) Fallback to JSON body
        if not worker_url:
            body = await request.body()
            payload = json.loads(body) if body else {}
            worker_url = payload.get("url") or payload.get("worker_url")

        if not worker_url:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )

        # Add if new, keep a simple request count per worker
        if worker_url not in self.worker_request_counts:
            self.worker_request_counts[worker_url] = 0
            self.worker_failure_counts[worker_url] = 0
            if self.verbose:
                print(f"[slime-router] Added new worker: {worker_url}")

        return {"status": "success", "worker_urls": self.worker_request_counts}

    async def list_workers(self, request: Request):
        """List all registered workers."""
        return {"urls": list(self.worker_request_counts.keys())}

    async def retrieve_from_text(self, request: Request):
        """Get token information from text input."""
        body = await request.body()
        payload = json.loads(body) if body else {}

        text = payload.get("text", "")

        # Use radix tree's retrieve_from_text method (no need to fetch weight version here)
        token_ids, logp, loss_mask = self.radix_tree.retrieve_from_text(text, return_logprob=True)

        # Handle the result based on whether logp was requested
        result = {
            "tokens": token_ids,  # token IDs
            "response": text,  # The input text
            "loss_mask": loss_mask,  # Loss mask for the tokens
            "token_length": len(token_ids),
            "loss_mask_length": len(loss_mask),
            "rollout_logp": logp,
        }

        return result

    def _select_least_loaded(self):
        """Return the worker URL with the fewest active requests (no
        bookkeeping)."""
        if not self.dead_workers:
            # Healthy path: select from all workers
            return min(self.worker_request_counts, key=self.worker_request_counts.get)
        # Degraded path: select from workers not in dead_workers
        valid_workers = (worker for worker in self.worker_request_counts if worker not in self.dead_workers)
        try:
            return min(valid_workers, key=self.worker_request_counts.get)
        except ValueError:
            raise RuntimeError("No healthy workers available in the pool") from None

    def _pick_sticky_url(self, routing_key):
        """Resolve a routing key to a worker, pinning new keys and remapping
        dead ones.

        True-sticky semantics: a live pin is never redistributed (even when
        workers are added); it is only remapped when its worker leaves the
        healthy set.
        """
        now = time.monotonic()
        entry = self.sticky_map.get(routing_key)
        if entry is not None and entry[0] in self.worker_request_counts and entry[0] not in self.dead_workers:
            entry[1] = now  # refresh last_seen so the assignment survives idle eviction
            self.sticky_stats["hit"] += 1
            return entry[0]

        # New key, or the pinned worker left the healthy set -> (re)assign via fallback.
        url = self._select_least_loaded()
        self.sticky_stats["remap" if entry is not None else "assigned"] += 1
        self.sticky_map[routing_key] = [url, now]
        return url

    def _use_url(self, routing_key=None):
        """Select a worker URL and account for the new in-flight request."""
        if self.sticky_enabled and routing_key:
            url = self._pick_sticky_url(routing_key)
        else:
            if self.sticky_enabled:
                self.sticky_stats["no_routing_key"] += 1
            url = self._select_least_loaded()
        self.worker_request_counts[url] += 1
        return url

    def _evict_idle_sticky(self):
        """Drop sticky assignments idle longer than ``sticky_idle_secs``.

        Bounds the map against unbounded routing-key cardinality. An evicted
        key is simply re-pinned (via fallback) the next time it is seen.
        """
        if not self.sticky_enabled or not self.sticky_map:
            return
        now = time.monotonic()
        stale = [key for key, (_, last_seen) in self.sticky_map.items() if now - last_seen > self.sticky_idle_secs]
        for key in stale:
            del self.sticky_map[key]
        if stale:
            self.sticky_stats["evicted"] += len(stale)

    def _finish_url(self, url):
        """Mark the request to the given URL as finished."""
        assert url in self.worker_request_counts, f"URL {url} not recognized"
        self.worker_request_counts[url] -= 1
        assert self.worker_request_counts[url] >= 0, f"URL {url} count went negative"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--sglang-host", type=str, required=True)
    parser.add_argument("--sglang-port", type=int, required=True)
    parser.add_argument("--tokenizer-name", type=str, help="Name of the tokenizer to use for tokenization")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Run the router
    run_router(args)
