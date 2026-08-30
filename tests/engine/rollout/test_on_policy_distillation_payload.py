# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""OpdManager / TopkWorker payload construction + logprob parsing.

Refactored: the old module-level helpers (``_extract_teacher_topk_pair``,
``_post_teacher_request_with_diagnostics``, ``fetch_teacher_log_probs``,
``_compute_image_id``) were folded into ``LogprobResponse`` /
``TopkWorker`` / ``OpdManager``. Teacher/student top-k logprobs are now
carried as sglang base64 fields, decoded into numpy arrays.
"""

import asyncio
import base64 as pybase64
from types import SimpleNamespace

import numpy as np
import pytest

from relax.engine.rollout.on_policy_distillation import OpdManager
from relax.utils.opd.feedback import OPDFeedback
from relax.utils.opd.opd_main_worker import LogprobResponse, TopkWorker
from relax.utils.opd.opd_opsd_worker import OpsdWorker
from relax.utils.opd.sdpo.feedback import SDPOFeedback, _clear_teacher_payload
from relax.utils.types import Sample


def _b64(arr: np.ndarray) -> str:
    return pybase64.b64encode(arr.tobytes()).decode("utf-8")


def test_teacher_prefill_self_topk_keeps_token_id_zero_and_takes_tail() -> None:
    # Total n=2 rows; response_length=1 -> keep the last row. token_id 0 kept.
    vals = np.array([-1.0, -2.0, -0.1, -0.2], dtype=np.float32)
    ids = np.array([100, 200, 0, 5], dtype=np.int32)
    resp = {
        "meta_info": {
            "input_top_logprobs_val_b64": _b64(vals),
            "input_top_logprobs_idx_b64": _b64(ids),
        }
    }

    pair = LogprobResponse(resp).self_topk("prefill", top_k=2, response_length=1)

    assert pair is not None
    out_ids, out_lps = pair
    np.testing.assert_array_equal(out_ids, np.array([[0, 5]], dtype=np.int32))
    np.testing.assert_allclose(out_lps, np.array([[-0.1, -0.2]], dtype=np.float32))


def test_base_logprobs_1d_from_b64_and_legacy() -> None:
    val = np.array([-0.5, -0.7, -0.9], dtype=np.float32)
    r1 = {"meta_info": {"input_token_logprobs_val_b64": _b64(val)}}
    np.testing.assert_allclose(LogprobResponse(r1).base_logprobs_1d(), val)

    # Legacy plain-list fallback: list of [logprob, token_id] pairs.
    r2 = {"meta_info": {"input_token_logprobs": [[-0.5, 1], [-0.7, 2]]}}
    np.testing.assert_allclose(LogprobResponse(r2).base_logprobs_1d(), np.array([-0.5, -0.7], dtype=np.float32))


def test_plain_sglang_topk_fields_are_decoded() -> None:
    response = {
        "meta_info": {
            "output_top_logprobs": [
                [(-0.1, 0, None), (-0.2, 5, None)],
                [(-0.3, 7, None), (-0.4, 9, None)],
            ],
            "input_top_logprobs": [
                [(-0.5, 1, None), (-0.6, 2, None)],
                [(-0.7, 3, None), (-0.8, 4, None)],
                [(-0.9, 6, None), (-1.0, 8, None)],
            ],
            "input_token_ids_logprobs": [
                [(-0.5, 1, None), (-0.6, 2, None)],
                [(-0.7, 3, None), (-0.8, 4, None)],
                [(-0.9, 6, None), (-1.0, 8, None)],
            ],
        }
    }

    output_ids, output_lps = LogprobResponse(response).self_topk("rollout", top_k=2)
    input_ids, input_lps = LogprobResponse(response).self_topk("prefill", top_k=2, response_length=1)
    input_query_lps = LogprobResponse(response).other_topk(response_length=1, top_k=2)

    np.testing.assert_array_equal(output_ids, np.array([[0, 5], [7, 9]], dtype=np.int32))
    np.testing.assert_allclose(output_lps, [[-0.1, -0.2], [-0.3, -0.4]])
    np.testing.assert_array_equal(input_ids, np.array([[6, 8]], dtype=np.int32))
    np.testing.assert_allclose(input_lps, [[-0.9, -1.0]])
    np.testing.assert_allclose(input_query_lps, [[-0.9, -1.0]])


def test_build_teacher_payload_adds_student_topk_query_ids() -> None:
    # student_topk + kl_coef != 0 => teacher_at_student=True => token_ids_logprob present.
    w = TopkWorker("student_topk", top_k=2, opd_kl_coef=1.0, opd_loss_coef=0.0)
    student_topk_ids = np.array([[3, 5], [7, 9]], dtype=np.int64)  # R=2, K=2

    payload = w.build_teacher_payload(
        input_ids=[1, 2, 3, 4],
        logprob_start_len=1,
        student_topk_ids=student_topk_ids,
        response_length=2,
    )

    assert payload["input_ids"] == [1, 2, 3, 4]
    assert payload["logprob_start_len"] == 1
    assert payload["return_logprob"] is True
    # _flatten_other_topk_ids: [[3,5],[7,9]] flattened + trailing [0]*top_k
    assert payload["token_ids_logprob"] == [3, 5, 7, 9, 0, 0]


def test_rollout_topk_payload_parsing_keeps_ordinary_opd_contract() -> None:
    manager = object.__new__(OpdManager)
    manager.topk_worker = TopkWorker("student_topk", top_k=2, opd_loss_coef=0.0)
    meta_info = {
        "output_token_logprobs_val_b64": _b64(np.array([-0.1, -0.2], dtype=np.float32)),
        "output_token_logprobs_idx_b64": _b64(np.array([0, 5], dtype=np.int32)),
    }

    tokens, log_probs = manager.parse_rollout_logprobs(meta_info, [9, 9], [-1.0, -1.0])

    assert tokens == [0, 5]
    np.testing.assert_allclose(log_probs, [-0.1, -0.2])


def test_sdpo_teacher_prefill_uses_privileged_prompt_offset(monkeypatch) -> None:
    manager = object.__new__(OpdManager)
    manager.args = type("Args", (), {"opd_teacher_url": "http://teacher"})()
    manager.feedback = SDPOFeedback()
    manager.topk_worker = TopkWorker("student_topk", top_k=2, opd_loss_coef=1.0)
    manager.sampled_worker = None
    manager.opsd_worker = OpsdWorker(is_opsd=True)

    sample = Sample(
        tokens=[10, 11, 20, 21],
        response_length=2,
        teacher_prompt=[{"role": "user", "content": "privileged context"}],
        teacher_tokens=[100, 101, 102, 20, 21],
        teacher_prompt_length=3,
        student_topk_token_ids=np.array([[7, 8], [9, 10]], dtype=np.int32),
    )
    captured = {}

    async def fake_post_logprob(session, url, payload, requested_sample, err_tag):
        captured.update(url=url, payload=payload, sample=requested_sample, err_tag=err_tag)
        return LogprobResponse(
            {
                "meta_info": {
                    "input_token_logprobs_val_b64": _b64(np.array([-0.1, -0.2, -0.3], dtype=np.float32)),
                    "input_token_ids_logprobs_val_b64": _b64(np.array([-0.4, -0.5, -0.6, -0.7], dtype=np.float32)),
                }
            }
        )

    monkeypatch.setattr(manager, "_post_logprob", fake_post_logprob)

    assert asyncio.run(manager._teacher_prefill(sample, object())) is True
    assert captured["url"] == "http://teacher"
    assert captured["payload"]["input_ids"] == sample.teacher_tokens
    assert captured["payload"]["logprob_start_len"] == 2
    assert captured["payload"]["token_ids_logprob"] == [7, 8, 9, 10, 0, 0]
    assert sample.teacher_at_student_topk_log_probs.shape == (2, 2)


def test_ordinary_teacher_prefill_keeps_rollout_input_and_original_offset(monkeypatch) -> None:
    manager = object.__new__(OpdManager)
    manager.args = type("Args", (), {"opd_teacher_url": "http://teacher"})()
    manager.feedback = OPDFeedback()
    manager.topk_worker = TopkWorker("student_topk", top_k=2, opd_loss_coef=1.0)
    manager.sampled_worker = None
    manager.opsd_worker = OpsdWorker(is_opsd=True)

    sample = Sample(
        tokens=[10, 11, 20, 21],
        rollout_tokens=[1, 2, 3, 4, 20, 21],
        response_length=2,
        student_topk_token_ids=np.array([[7, 8], [9, 10]], dtype=np.int32),
    )
    captured = {}

    async def fake_post_logprob(session, url, payload, requested_sample, err_tag):
        captured.update(url=url, payload=payload, sample=requested_sample, err_tag=err_tag)
        return LogprobResponse(
            {
                "meta_info": {
                    "input_token_logprobs_val_b64": _b64(np.array([-0.1, -0.2, -0.3], dtype=np.float32)),
                    "input_token_ids_logprobs_val_b64": _b64(np.array([-0.4, -0.5, -0.6, -0.7], dtype=np.float32)),
                }
            }
        )

    monkeypatch.setattr(manager, "_post_logprob", fake_post_logprob)

    assert asyncio.run(manager._teacher_prefill(sample, object())) is True
    assert captured["payload"]["input_ids"] == sample.rollout_tokens
    assert captured["payload"]["logprob_start_len"] == 1
    assert captured["payload"]["token_ids_logprob"] == [7, 8, 9, 10, 0, 0]
    assert "image_data" not in captured["payload"]


@pytest.mark.parametrize(
    "student_topk_token_ids",
    [
        None,
        np.array([[7, 8]], dtype=np.int32),
        np.array([[7, -1], [9, 10]], dtype=np.int32),
    ],
)
def test_sdpo_rejects_invalid_student_topk_before_teacher_request(
    monkeypatch,
    student_topk_token_ids,
) -> None:
    manager = object.__new__(OpdManager)
    manager.args = type("Args", (), {"opd_teacher_url": "http://teacher"})()
    manager.feedback = SDPOFeedback()
    manager.topk_worker = TopkWorker("student_topk", top_k=2, opd_loss_coef=1.0)
    manager.sampled_worker = None
    manager.opsd_worker = OpsdWorker(is_opsd=True)
    sample = Sample(
        index=8,
        tokens=[10, 11, 20, 21],
        response_length=2,
        teacher_prompt="privileged context",
        student_topk_token_ids=student_topk_token_ids,
    )

    async def fail_if_requested(*args, **kwargs):
        raise AssertionError("SDPO must validate rollout Top-K ids before the teacher request")

    monkeypatch.setattr(manager, "_post_logprob", fail_if_requested)

    with pytest.raises(ValueError, match="student Top-K"):
        asyncio.run(manager._teacher_prefill(sample, object()))


def test_sdpo_manager_constructs_opsd_worker_and_feedback() -> None:
    args = type(
        "Args",
        (),
        {
            "opd_token_selection": "student_topk",
            "opd_log_prob_top_k": 2,
            "opd_kl_coef": 0.0,
            "opd_loss_coef": 1.0,
            "opd_feedback_class": "relax.utils.opd.sdpo.feedback.GoldenAnswerSDPOFeedback",
            "opd_feedback_kwargs": None,
            "opd_teacher_image_key": None,
        },
    )()

    manager = OpdManager(args)

    assert manager.opsd_worker is not None
    assert manager.opsd_worker.is_opsd
    assert isinstance(manager.feedback, SDPOFeedback)


def test_sdpo_transfer_schema_contains_sample_mask_and_topk_payload() -> None:
    manager = object.__new__(OpdManager)
    manager.feedback = SDPOFeedback()
    manager.topk_worker = TopkWorker("student_topk", top_k=2, opd_loss_coef=1.0)
    manager.sampled_worker = None

    assert manager.schema_opd_transfer_data() == [
        "opd_sample_mask",
        TopkWorker.TRANSFER_TOKEN_IDS,
        TopkWorker.TRANSFER_TEACHER_LOG_PROBS,
    ]


def test_sdpo_transfer_preserves_sample_mask_order() -> None:
    manager = object.__new__(OpdManager)
    manager.feedback = SDPOFeedback()
    manager.topk_worker = TopkWorker("student_topk", top_k=2, opd_loss_coef=1.0)
    manager.sampled_worker = None
    samples = [Sample(index=0, opd_sample_mask=True), Sample(index=1, opd_sample_mask=False)]
    train_data = {}

    manager.produce_opd_transfer_data(samples, train_data)

    assert train_data["opd_sample_mask"] == [True, False]


@pytest.mark.parametrize(
    ("mode", "selection", "kl_coef", "loss_coef", "expected"),
    [
        (
            "opd",
            "student_sampled",
            0.0,
            1.0,
            ["teacher_log_probs", "rollout_log_probs"],
        ),
        ("opd", "student_topk", 0.0, 1.0, ["opd_topk_token_ids", "opd_topk_teacher_log_probs"]),
        (
            "opd",
            "student_topk",
            1.0,
            0.0,
            ["opd_topk_token_ids", "opd_topk_teacher_log_probs", "opd_topk_student_log_probs"],
        ),
        ("opd", "teacher_topk", 0.0, 1.0, ["opd_topk_token_ids", "opd_topk_teacher_log_probs"]),
        ("opd", "union", 0.0, 1.0, ["opd_topk_token_ids", "opd_topk_teacher_log_probs", "opd_topk_ksz"]),
        ("opsd", "student_topk", 0.0, 1.0, ["opd_topk_token_ids", "opd_topk_teacher_log_probs"]),
        (
            "sdpo",
            "student_topk",
            0.0,
            1.0,
            ["opd_sample_mask", "opd_topk_token_ids", "opd_topk_teacher_log_probs"],
        ),
    ],
)
def test_teacher_transfer_schema_matrix(mode, selection, kl_coef, loss_coef, expected) -> None:
    args = SimpleNamespace(
        use_opd=True,
        opd_type="sglang",
        group_rm=mode == "sdpo",
        opd_feedback_class="relax.utils.opd.sdpo.feedback.GoldenAnswerSDPOFeedback" if mode == "sdpo" else None,
        opd_feedback_kwargs={"teacher_prompt_key": "teacher_prompt"} if mode == "opsd" else None,
        opd_token_selection=selection,
        opd_log_prob_top_k=2,
        opd_kl_coef=kl_coef,
        opd_loss_coef=loss_coef,
        opd_teacher_image_key=None,
    )

    schema = OpdManager(args).schema_opd_transfer_data()

    assert schema == expected


def test_sdpo_assembly_rejects_missing_teacher_topk_payload() -> None:
    manager = object.__new__(OpdManager)
    manager.feedback = SDPOFeedback()
    manager.topk_worker = TopkWorker("student_topk", top_k=2, opd_loss_coef=1.0)
    sample = Sample(
        index=4,
        response_length=1,
        student_topk_token_ids=np.array([[1, 2]], dtype=np.int32),
        student_topk_log_probs=np.array([[-0.1, -0.2]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="complete teacher Top-K payload"):
        manager._assemble_transfer([sample])


def test_teacher_payload_reset_preserves_student_rollout_topk() -> None:
    sample = Sample(
        student_topk_token_ids=np.array([[1, 2]], dtype=np.int32),
        student_topk_log_probs=np.array([[-0.1, -0.2]], dtype=np.float32),
        teacher_log_probs=[-0.3],
        teacher_topk_token_ids=np.array([[3, 4]], dtype=np.int32),
        teacher_topk_log_probs=np.array([[-0.3, -0.4]], dtype=np.float32),
        teacher_at_student_topk_log_probs=np.array([[-0.5, -0.6]], dtype=np.float32),
        student_at_teacher_topk_log_probs=np.array([[-0.7, -0.8]], dtype=np.float32),
        opd_topk_token_ids=[np.array([[3, 4]], dtype=np.int32)],
        opd_topk_teacher_log_probs=[np.array([[-0.3, -0.4]], dtype=np.float32)],
        teacher_tokens=[9, 10],
        teacher_prompt_length=1,
    )

    _clear_teacher_payload(sample)

    assert sample.teacher_log_probs is None
    assert sample.teacher_topk_token_ids is None
    assert sample.teacher_topk_log_probs is None
    assert sample.teacher_at_student_topk_log_probs is None
    assert sample.student_at_teacher_topk_log_probs is None
    assert sample.opd_topk_token_ids is None
    assert sample.opd_topk_teacher_log_probs is None
    assert sample.teacher_tokens is None
    assert sample.teacher_prompt_length is None
    np.testing.assert_array_equal(sample.student_topk_token_ids, np.array([[1, 2]], dtype=np.int32))
    np.testing.assert_array_equal(sample.student_topk_log_probs, np.array([[-0.1, -0.2]], dtype=np.float32))


def test_ordinary_prefill_does_not_clear_existing_teacher_payload(monkeypatch) -> None:
    manager = object.__new__(OpdManager)
    manager.args = type(
        "Args",
        (),
        {"opd_teacher_connector_limit": 1, "opd_teacher_timeout_s": 1.0},
    )()
    manager.feedback = OPDFeedback()
    manager.topk_worker = None
    manager.sampled_worker = None
    manager.opsd_worker = OpsdWorker(is_opsd=True)
    sample = Sample(response_length=1, teacher_log_probs=[-0.25])

    async def fake_teacher_prefill(requested_sample, session):
        return True

    monkeypatch.setattr(manager, "_teacher_prefill", fake_teacher_prefill)
    monkeypatch.setattr(manager, "_assemble_transfer", lambda samples: None)
    monkeypatch.setattr(manager, "_raise_if_all_failed", lambda samples, results: None)

    asyncio.run(manager.prefill(sample))

    assert sample.teacher_log_probs == [-0.25]


def test_ordinary_prefill_keeps_zero_response_teacher_dispatch(monkeypatch) -> None:
    manager = object.__new__(OpdManager)
    manager.args = type(
        "Args",
        (),
        {"opd_teacher_connector_limit": 1, "opd_teacher_timeout_s": 1.0},
    )()
    manager.feedback = OPDFeedback()
    manager.topk_worker = None
    manager.sampled_worker = None
    manager.opsd_worker = OpsdWorker(is_opsd=True)
    sample = Sample(response_length=0)
    requested = []

    async def fake_teacher_prefill(requested_sample, session):
        requested.append(requested_sample)
        return True

    monkeypatch.setattr(manager, "_teacher_prefill", fake_teacher_prefill)
    monkeypatch.setattr(manager, "_assemble_transfer", lambda samples: None)
    monkeypatch.setattr(manager, "_raise_if_all_failed", lambda samples, results: None)

    asyncio.run(manager.prefill(sample))

    assert requested == [sample]


def test_build_transfer_channels_union_merges_and_pads() -> None:
    # union + kl_coef != 0 (is_advantage=True): merge student/teacher self-topk
    # ids per position, keep first-seen logprobs, pad ragged rows.
    w = TopkWorker("union", top_k=2, opd_kl_coef=1.0, opd_loss_coef=0.0)

    s_ids = np.array([[3, 5], [1, 2]], dtype=np.int32)
    s_student_lp = np.array([[-0.1, -0.2], [-0.5, -0.6]], dtype=np.float32)
    t_ids = np.array([[5, 7], [1, 2]], dtype=np.int32)
    t_teacher_lp = np.array([[-0.3, -0.4], [-0.7, -0.8]], dtype=np.float32)
    teacher_at_student_lp = np.array([[-1.0, -1.1], [-1.5, -1.6]], dtype=np.float32)
    student_at_teacher_lp = np.array([[-2.0, -2.1], [-2.5, -2.6]], dtype=np.float32)

    channels = w.build_transfer_channels(
        student_self_topk=(s_ids, s_student_lp),
        teacher_self_topk=(t_ids, t_teacher_lp),
        teacher_at_student_lp=teacher_at_student_lp,
        student_at_teacher_lp=student_at_teacher_lp,
    )

    # row0: union({3,5},{5,7}) = [3,5,7] (kp=3); row1: union({1,2},{1,2}) = [1,2] (kp=2, padded to 3 with -1)
    np.testing.assert_array_equal(
        channels[TopkWorker.TRANSFER_TOKEN_IDS],
        np.array([[3, 5, 7], [1, 2, -1]], dtype=np.int32),
    )
    # teacher logprob per unique id (first-seen): row0 keeps s_teacher for 3,5 and t_teacher for 7
    np.testing.assert_allclose(
        channels[TopkWorker.TRANSFER_TEACHER_LOG_PROBS],
        np.array([[-1.0, -1.1, -0.4], [-1.5, -1.6, 0.0]], dtype=np.float32),
    )
    # student logprob per unique id (first-seen)
    np.testing.assert_allclose(
        channels[TopkWorker.TRANSFER_STUDENT_LOG_PROBS],
        np.array([[-0.1, -0.2, -2.1], [-0.5, -0.6, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(channels[TopkWorker.TRANSFER_K_LENGTHS], np.array([3, 2], dtype=np.int32))
