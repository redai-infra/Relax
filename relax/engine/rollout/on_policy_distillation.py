# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence

import aiohttp
import numpy as np

from relax.utils.logging_utils import get_logger
from relax.utils.opd import opd_main_worker, opd_opsd_worker
from relax.utils.types import Sample


try:
    import orjson

    def _dumps_to_bytes(payload: dict) -> bytes:
        return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
except ImportError:  # pragma: no cover - orjson is normally available via sglang

    def _dumps_to_bytes(payload: dict) -> bytes:
        return json.dumps(payload).encode("utf-8")


logger = get_logger(__name__)

EncodeMultimodalInputs = Callable[[dict], Awaitable[tuple[dict, float]]]


def _aiohttp_json_post_kwargs(payload: dict) -> dict:
    """Bypass aiohttp's ``json=`` kwarg: serialize via orjson and ship as raw
    ``data=``."""
    return {
        "data": _dumps_to_bytes(payload),
        "headers": {"Content-Type": "application/json"},
    }


def is_opd_enabled(args, evaluation: bool = False) -> bool:
    return not evaluation and getattr(args, "use_opd", False) and getattr(args, "opd_type", None) == "sglang"


def _create_teacher_client_session(args) -> aiohttp.ClientSession:
    connector_limit = int(getattr(args, "opd_teacher_connector_limit", 256))
    connector = aiohttp.TCPConnector(limit=connector_limit)
    timeout = aiohttp.ClientTimeout(total=float(args.opd_teacher_timeout_s))
    return aiohttp.ClientSession(connector=connector, timeout=timeout)


# --- Teacher URL selection: MOPD routing (by data_source) x replica round-robin ---
# Two layers:
#   1. Routing (MOPD): when ``args.opd_teacher_routes_map`` is set, pick the
#      teacher for this sample by ``sample.metadata[args.opd_teacher_key]``.
#      Each route value is a LIST of that teacher's replica URLs.
#   2. Replica round-robin: spread requests across a teacher's replicas with a
#      per-teacher in-process counter (single-threaded asyncio makes ``+= 1`` safe).
# Single-teacher path falls back to ``args.opd_teacher_urls`` / ``opd_teacher_url``.
_TEACHER_URL_RR: dict[str, int] = {}


def _round_robin(urls: list[str], key: str) -> str:
    i = _TEACHER_URL_RR.get(key, 0)
    _TEACHER_URL_RR[key] = i + 1
    return urls[i % len(urls)]


def _pick_teacher_url(args, sample=None) -> str:
    routes_map = getattr(args, "opd_teacher_routes_map", None)
    if routes_map and sample is not None:
        key_field = getattr(args, "opd_teacher_key", None) or "data_source"
        metadata = getattr(sample, "metadata", None) or {}
        routing_value = metadata.get(key_field)
        if routing_value is None:
            raise ValueError(
                f"MOPD routing: sample missing key '{key_field}' in metadata. "
                f"Available metadata keys: {list(metadata.keys())}. "
                f"Ensure the dataset has a '{key_field}' column and it is surfaced "
                "via --metadata-key or the data pipeline."
            )
        replicas = routes_map.get(routing_value)
        if not replicas:
            raise KeyError(
                f"MOPD routing: no teacher route for '{key_field}={routing_value}'. "
                f"Available routes: {list(routes_map.keys())}."
            )
        return _round_robin(replicas, routing_value)
    # Single-teacher path: round-robin over replicas if configured.
    urls = getattr(args, "opd_teacher_urls", None)
    if urls and len(urls) > 1:
        return _round_robin(urls, "__single__")
    return args.opd_teacher_url


class OpdManager:
    def __init__(self, args):
        self.args = args
        self.topk_worker: opd_main_worker.TopkWorker | None = None
        self.sampled_worker: opd_main_worker.SampledTokenWorker | None = None  # 仅 student_sampled
        self.opsd_worker: opd_opsd_worker.OpsdWorker | None = None

        token_selection = args.opd_token_selection
        if token_selection != "student_sampled":
            self.topk_worker = opd_main_worker.TopkWorker.from_args(args)
        else:
            self.sampled_worker = opd_main_worker.SampledTokenWorker.from_args(args)

        opsd_worker = opd_opsd_worker.OpsdWorker.from_args(args)
        self.opsd_worker = opsd_worker if opsd_worker.is_opsd else None

    @property
    def is_topk(self) -> bool:
        return self.topk_worker is not None

    @property
    def is_opsd(self) -> bool:
        return self.opsd_worker is not None

    def schema_opd_transfer_data(self) -> list[str]:
        eopd = getattr(self.args, "use_eopd", False)
        fields: list[str] = []
        if self.topk_worker is not None:
            fields.extend(self.topk_worker.topk_transfer_fields(eopd=eopd))
            if self.topk_worker.is_advantage or getattr(self.args, "opd_teacher_advantage", False):
                fields.append(opd_main_worker.SampledTokenWorker.TRANSFER_TEACHER_LOG_PROBS)
        if self.sampled_worker is not None:
            fields.extend(self.sampled_worker.sampled_transfer_fields(eopd=eopd))
        return fields

    def produce_opd_transfer_data(self, samples: list[Sample], train_data: dict) -> None:
        if self.topk_worker is not None:
            schema_fields = set(self.topk_worker.topk_transfer_fields(eopd=getattr(self.args, "use_eopd", False)))
            for field_name in opd_main_worker.TopkWorker.TRANSFER_FIELDS:
                has_any = any(getattr(s, field_name, None) is not None for s in samples)
                if not has_any and field_name not in schema_fields:
                    continue
                flat: list = []
                for s in samples:
                    v = getattr(s, field_name, None)
                    if v is None:
                        flat.append([])
                    else:
                        flat.append(v.reshape(-1).tolist())
                train_data[field_name] = flat
            kl_field = opd_main_worker.TopkWorker.TRANSFER_K_LENGTHS
            if self.topk_worker.spec.name == "union" and any(getattr(s, kl_field, None) is not None for s in samples):
                train_data[kl_field] = [
                    getattr(s, kl_field).tolist() if getattr(s, kl_field, None) is not None else [] for s in samples
                ]
            if self.topk_worker.is_advantage or getattr(self.args, "opd_teacher_advantage", False):
                train_data[opd_main_worker.SampledTokenWorker.TRANSFER_TEACHER_LOG_PROBS] = [
                    s.teacher_log_probs if s.teacher_log_probs is not None else [] for s in samples
                ]
        elif self.sampled_worker is not None:
            train_data[opd_main_worker.SampledTokenWorker.TRANSFER_TEACHER_LOG_PROBS] = [
                s.teacher_log_probs if s.teacher_log_probs is not None else [] for s in samples
            ]

        if getattr(self.args, "use_eopd", False):
            train_data["teacher_entropy"] = [
                s.teacher_entropy if s.teacher_entropy is not None else [] for s in samples
            ]

    def before_rollout(self, payload: dict) -> None:
        if self.topk_worker is None:
            return
        fields = self.topk_worker.student_rollout_payload()
        if fields:
            payload.update(fields)

    def parse_rollout_logprobs(self, meta_info: dict, tokens: list, log_probs: list) -> tuple[list, list]:
        if self.topk_worker is None:
            return tokens, log_probs
        val_b64 = meta_info.get("output_token_logprobs_val_b64")
        if val_b64 is None:
            return tokens, log_probs
        import numpy as np
        import pybase64

        val = np.frombuffer(pybase64.b64decode(val_b64), dtype=np.float32)
        idx_b64 = meta_info.get("output_token_logprobs_idx_b64")
        idx = np.frombuffer(pybase64.b64decode(idx_b64), dtype=np.int32) if idx_b64 else np.array([], dtype=np.int32)
        n_expected = len(tokens)
        if idx.size == n_expected and val.size == n_expected:
            out_tokens = idx.tolist()
            out_log_probs = val.tolist()
        else:
            out_tokens = tokens
            out_log_probs = log_probs
        return out_tokens, out_log_probs

    def after_rollout(self, sample: Sample, output: dict) -> None:
        if self.topk_worker is None:
            return
        pair = opd_main_worker.LogprobResponse(output).self_topk("rollout", self.topk_worker.top_k)
        if pair is None:
            return
        token_ids, log_probs = pair
        if sample.student_topk_token_ids is None:
            sample.student_topk_token_ids = token_ids
            sample.student_topk_log_probs = log_probs
        else:  # multi-turn
            sample.student_topk_token_ids = np.vstack([sample.student_topk_token_ids, token_ids])
            sample.student_topk_log_probs = np.vstack([sample.student_topk_log_probs, log_probs])

    async def prefill(
        self,
        samples: Sample | Sequence[Sample],
        encode_multimodal_inputs: EncodeMultimodalInputs | None = None,
        *,
        include_student: bool = True,
    ) -> None:
        sample_list = list(samples) if isinstance(samples, Sequence) else [samples]

        if self.opsd_worker is not None:
            await asyncio.gather(*[self.opsd_worker.build_teacher_inputs(self.args, s) for s in sample_list])

        async with _create_teacher_client_session(self.args) as session:
            fetch_results = await asyncio.gather(*[self._teacher_prefill(s, session) for s in sample_list])
            self._raise_if_all_failed(sample_list, fetch_results)

            if include_student and self.topk_worker is not None and self.topk_worker.spec.student_at_teacher:
                await asyncio.gather(
                    *[self._student_prefill(s, session, encode_multimodal_inputs) for s in sample_list]
                )

        self._assemble_transfer(sample_list)

    async def _resolve_direct_engine_urls(self) -> list[str]:
        """Query the SGLang router for direct worker URLs to bypass the router
        for heavy token_ids_logprob requests (which block the engine event loop
        and cause the router's health-check circuit breaker to trip)."""
        router_base = f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}"
        try:
            import sglang_router
            from packaging.version import parse

            if parse(sglang_router.__version__) <= parse("0.2.1") or getattr(self.args, "use_slime_router", False):
                async with aiohttp.ClientSession() as s:
                    async with s.get(f"{router_base}/list_workers", timeout=aiohttp.ClientTimeout(total=5)) as r:
                        data = await r.json()
                        return data.get("urls", [])
            else:
                async with aiohttp.ClientSession() as s:
                    async with s.get(f"{router_base}/workers", timeout=aiohttp.ClientTimeout(total=5)) as r:
                        data = await r.json()
                        return [w["url"] for w in data.get("workers", [])]
        except Exception as e:
            logger.warning("Failed to resolve direct engine URLs from router: %s", e)
            return []

    async def prefill_student(
        self,
        samples: Sample | Sequence[Sample],
        encode_multimodal_inputs: EncodeMultimodalInputs | None = None,
        max_concurrent: int = 4,
    ) -> None:
        """Run student-at-teacher prefill only.  Call after all rollout generation
        is complete to avoid sending token_ids_logprob requests to the student
        engine while it is still decoding rollout responses."""
        if self.topk_worker is None or not self.topk_worker.spec.student_at_teacher:
            return
        sample_list = list(samples) if isinstance(samples, Sequence) else [samples]
        sem = asyncio.Semaphore(max_concurrent)

        direct_urls = await self._resolve_direct_engine_urls()
        student_url_override = f"{direct_urls[0]}/generate" if direct_urls else None
        if student_url_override:
            logger.info("[OPD] Bypassing router for student prefill: %s", student_url_override)

        async def _limited(s: Sample, session: aiohttp.ClientSession) -> None:
            async with sem:
                await self._student_prefill(s, session, encode_multimodal_inputs, student_url_override)

        async with _create_teacher_client_session(self.args) as session:
            await asyncio.gather(*[_limited(s, session) for s in sample_list])
        self._assemble_transfer(sample_list)

    async def _post_logprob(
        self,
        session: aiohttp.ClientSession,
        url: str,
        payload: dict,
        sample: Sample,
        err_tag: str,
    ) -> opd_main_worker.LogprobResponse | None:
        try:
            async with session.post(url, **_aiohttp_json_post_kwargs(payload)) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error("OPD %s failed: status=%s, url=%s, body=%s", err_tag, resp.status, url, body[:2048])
                    resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            logger.error(
                "OPD %s fetch failed for sample_index=%s, error=%s",
                err_tag,
                getattr(sample, "index", None),
                f"{type(exc).__name__}: {str(exc)[:256]}",
            )
            return None

        return opd_main_worker.LogprobResponse(data)

    async def _teacher_prefill(self, sample: Sample, session: aiohttp.ClientSession) -> bool:
        from relax.utils.opd.opd_utils import build_teacher_preexpanded_image_data

        response_length = int(sample.response_length or 0)
        if response_length <= 0:
            return True

        # OPSD: expanded teacher_tokens + image_data；
        if self.opsd_worker is not None:
            image_data = await build_teacher_preexpanded_image_data(sample)
            teacher_input_ids = self.opsd_worker.teacher_input_ids(sample, response_length)
            prompt_length = self.opsd_worker.teacher_prompt_len(sample, response_length)
        else:
            image_data = None
            teacher_input_ids = sample.rollout_tokens or sample.tokens
            prompt_length = len(sample.tokens) - response_length

        if getattr(self.args, "opd_eos_replace", False):
            _ENDOFTEXT, _IM_END = 151643, 151645
            teacher_input_ids = list(teacher_input_ids)
            for j in range(prompt_length, len(teacher_input_ids)):
                if teacher_input_ids[j] == _ENDOFTEXT:
                    teacher_input_ids[j] = _IM_END
                    break

        logprob_start_len = max(prompt_length - 1, 0)

        mm_fields = {"image_data": image_data} if image_data is not None else None
        if self.topk_worker is not None:
            payload = self.topk_worker.build_teacher_payload(
                input_ids=teacher_input_ids,
                logprob_start_len=logprob_start_len,
                student_topk_ids=sample.student_topk_token_ids,
                response_length=response_length,
                mm_fields=mm_fields,
            )
        else:
            payload = opd_main_worker.build_prefill_payload_base(teacher_input_ids, logprob_start_len)
            if mm_fields:
                payload.update(mm_fields)
        if getattr(self.args, "use_eopd", False):
            payload["sampling_params"]["max_new_tokens"] = 1
        teacher_url = _pick_teacher_url(self.args, sample)
        resp_obj = await self._post_logprob(session, teacher_url, payload, sample, "teacher prefill")
        if resp_obj is None:
            return False

        token_logprobs = resp_obj.base_logprobs_1d()
        if token_logprobs is None or len(token_logprobs) == 0:
            logger.error(
                "Invalid OPD teacher response for sample_index=%s: missing input_token_logprobs.",
                getattr(sample, "index", None),
            )
            return False

        if len(token_logprobs) != response_length + 1:
            logger.error(
                "Teacher log-prob length mismatch for sample_index=%s: got=%s expected=%s",
                getattr(sample, "index", None),
                len(token_logprobs) - 1,
                response_length,
            )
            return False

        needs_teacher_lp = (
            self.sampled_worker is not None
            or (self.topk_worker is not None and self.topk_worker.is_advantage)
            or getattr(self.args, "opd_teacher_advantage", False)
        )
        if needs_teacher_lp:
            sample.teacher_log_probs = [float(v) for v in token_logprobs[1 : 1 + response_length]]

        if self.topk_worker is not None:
            if self.topk_worker.spec.teacher_self_topk:
                pair = self.topk_worker.parse_prefill_self_topk(resp_obj, response_length)
                sample.teacher_topk_token_ids = pair[0] if pair else None
                sample.teacher_topk_log_probs = pair[1] if pair else None
            if self.topk_worker.spec.teacher_at_student:
                sample.teacher_at_student_topk_log_probs = self.topk_worker.parse_prefill_other_topk(
                    resp_obj, response_length
                )

        if getattr(self.args, "use_eopd", False):
            entropy = resp_obj.entropy_1d()
            if entropy is not None and len(entropy) >= response_length + 1:
                sample.teacher_entropy = [float(v) for v in entropy[1 : 1 + response_length]]
            elif entropy is not None and len(entropy) == response_length:
                sample.teacher_entropy = [float(v) for v in entropy]
            else:
                top_k = int(getattr(self.args, "opd_log_prob_top_k", 0) or 0)
                vocab_size = int(getattr(self.args, "opd_teacher_vocab_size", 151936) or 151936)
                entropy_topk = resp_obj.entropy_from_topk(top_k, vocab_size, response_length)
                if entropy_topk is not None and len(entropy_topk) == response_length:
                    sample.teacher_entropy = [float(v) for v in entropy_topk]
                else:
                    logger.warning(
                        "EOPD entropy missing for sample_index=%s (got %s elements, need %d). "
                        "Falling back to zeros (EOPD gate disabled for this sample).",
                        getattr(sample, "index", None),
                        len(entropy) if entropy is not None else "None",
                        response_length,
                    )
                    sample.teacher_entropy = [0.0] * response_length

        return True

    async def _student_prefill(
        self,
        sample: Sample,
        session: aiohttp.ClientSession,
        encode_mm_fn: EncodeMultimodalInputs | None,
        url_override: str | None = None,
    ) -> None:

        from relax.utils.opd.opd_utils import build_student_preexpanded_image_data

        response_length = int(sample.response_length or 0)
        teacher_topk_ids = sample.teacher_topk_token_ids

        prompt_length = len(sample.tokens) - response_length
        logprob_start_len = max(prompt_length - 1, 0)

        preexpanded_image_data = await build_student_preexpanded_image_data(sample)
        if preexpanded_image_data is not None:
            student_input_ids = sample.tokens
            mm_fields = {"image_data": preexpanded_image_data}
        else:
            student_input_ids = sample.rollout_tokens or sample.tokens
            mm_fields = None
            if encode_mm_fn is not None and sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
                mm_fields, _ = await encode_mm_fn(sample.multimodal_inputs)

        payload = self.topk_worker.build_student_payload(
            input_ids=student_input_ids,
            logprob_start_len=logprob_start_len,
            teacher_topk_ids=teacher_topk_ids,
            response_length=response_length,
            mm_fields=mm_fields,
        )

        student_url = url_override or f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}/generate"
        resp_obj = await self._post_logprob(session, student_url, payload, sample, "student-at-teacher-topk")
        if resp_obj is None:
            return
        sample.student_at_teacher_topk_log_probs = self.topk_worker.parse_prefill_other_topk(resp_obj, response_length)

    def _assemble_transfer(self, samples: list[Sample]) -> None:
        if self.topk_worker is None:
            return
        for sample in samples:
            student_self = (
                (sample.student_topk_token_ids, sample.student_topk_log_probs)
                if sample.student_topk_token_ids is not None
                else None
            )
            teacher_self = (
                (sample.teacher_topk_token_ids, sample.teacher_topk_log_probs)
                if sample.teacher_topk_token_ids is not None
                else None
            )
            channels = self.topk_worker.build_transfer_channels(
                student_self_topk=student_self,
                teacher_self_topk=teacher_self,
                teacher_at_student_lp=sample.teacher_at_student_topk_log_probs,
                student_at_teacher_lp=sample.student_at_teacher_topk_log_probs,
            )
            sample.opd_topk_token_ids = channels.get(opd_main_worker.TopkWorker.TRANSFER_TOKEN_IDS)
            sample.opd_topk_student_log_probs = channels.get(opd_main_worker.TopkWorker.TRANSFER_STUDENT_LOG_PROBS)
            sample.opd_topk_teacher_log_probs = channels.get(opd_main_worker.TopkWorker.TRANSFER_TEACHER_LOG_PROBS)
            sample.opd_topk_ksz = channels.get(opd_main_worker.TopkWorker.TRANSFER_K_LENGTHS)

    @staticmethod
    def _raise_if_all_failed(samples: list[Sample], fetch_results: list[bool]) -> None:
        eligible = [(s, ok) for s, ok in zip(samples, fetch_results) if int(getattr(s, "response_length", 0) or 0) > 0]
        if not eligible or any(ok for _, ok in eligible):
            return
        raise RuntimeError(
            f"All OPD teacher fetches failed for {len(eligible)} non-empty samples "
            f"(total={len(samples)}); url={getattr(samples[0], 'opd_teacher_url', None) if samples else None}"
        )


def produce_opd_transfer_data(args, samples: list[Sample], train_data: dict) -> None:
    if not is_opd_enabled(args):
        return
    OpdManager(args).produce_opd_transfer_data(samples, train_data)
