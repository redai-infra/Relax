# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Coverage for the OPD feedback strategy hierarchy (OPD / OPSD / SDPO)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from relax.utils.types import Sample


def _sample(group: int | None, index: int, response: str, reward: object, **metadata: object) -> Sample:
    return Sample(
        group_index=group,
        index=index,
        prompt=f"question-{group}",
        response=response,
        response_length=len(response),
        reward=reward,
        metadata=dict(metadata),
    )


_SDPO_FEEDBACK_CLASSES = ["GoldenAnswerSDPOFeedback"]


def _sdpo_feedback_class(name: str) -> type:
    import relax.utils.opd.sdpo.feedback as sdpo_feedback_module

    return getattr(sdpo_feedback_module, name)


# --- record_sample_feedback -------------------------------------------------


def test_record_appends_text_only_to_originating_sample() -> None:
    from relax.utils.opd.feedback import EnvironmentFeedback

    sample = _sample(1, 0, "bad", 0.0)
    other = _sample(1, 1, "good", 1.0)
    EnvironmentFeedback.record(sample, "failed test case")
    EnvironmentFeedback.record(sample, "retry with a shorter answer")

    assert sample.metadata["env_feedback"] == ["failed test case", "retry with a shorter answer"]
    assert "env_feedback" not in other.metadata


def test_record_sample_feedback_defaults_to_env_feedback_recording() -> None:
    from relax.utils.opd.feedback import OPDFeedback
    from relax.utils.opd.opsd.feedback import OPSDFeedback
    from relax.utils.opd.sdpo.feedback import GoldenAnswerSDPOFeedback

    opd_sample = _sample(1, 0, "answer", {"score": 0.0, "feedback": "opd said no"})
    opsd_sample = _sample(1, 1, "answer", {"score": 0.0, "feedback": "opsd said no"})
    sdpo_sample = _sample(1, 2, "answer", {"score": 0.0, "feedback": "fix the second step"})

    OPDFeedback().record_sample_feedback(opd_sample, opd_sample.reward)
    OPSDFeedback(teacher_prompt_key="teacher_prompt").record_sample_feedback(opsd_sample, opsd_sample.reward)
    GoldenAnswerSDPOFeedback().record_sample_feedback(sdpo_sample, sdpo_sample.reward)

    assert opd_sample.metadata["env_feedback"] == ["opd said no"]
    assert opsd_sample.metadata["env_feedback"] == ["opsd said no"]
    assert sdpo_sample.metadata["env_feedback"] == ["fix the second step"]


# --- teacher prompt construction --------------------------------------------


@pytest.mark.parametrize("feedback_class_name", _SDPO_FEEDBACK_CLASSES)
def test_sdpo_peer_solution_shared_only_inside_group_and_normal_error_dropped(feedback_class_name: str) -> None:
    target = _sample(7, 0, "wrong", {"score": 0.0}, env_feedback=["fix arithmetic"])
    success = _sample(7, 1, "correct solution", {"score": 1.0}, env_feedback=["success details"])
    unrelated = _sample(8, 2, "other", {"score": 0.0}, env_feedback=["unrelated feedback"])

    _sdpo_feedback_class(feedback_class_name)().prepare_teacher_prompts(
        [target, success, unrelated], [target.reward, success.reward, unrelated.reward]
    )

    assert target.teacher_prompt is not None
    target_text = (
        target.teacher_prompt[-1]["content"] if isinstance(target.teacher_prompt, list) else target.teacher_prompt
    )
    assert "correct solution" in target_text
    assert "fix arithmetic" not in target_text
    assert "success details" not in target_text
    assert unrelated.teacher_prompt == unrelated.prompt
    assert unrelated.opd_sample_mask is False
    assert "unrelated feedback" not in str(unrelated.teacher_prompt)


@pytest.mark.parametrize("feedback_class_name", _SDPO_FEEDBACK_CLASSES)
def test_sdpo_prefers_peer_solution_and_falls_back_to_self_success(feedback_class_name: str) -> None:
    self_success = _sample(9, 0, "self solution", {"score": 1.0})
    peer_success = _sample(9, 1, "peer solution", {"score": 1.0})

    singleton_success = _sample(10, 0, "only solution", {"score": 1.0})
    _sdpo_feedback_class(feedback_class_name)().prepare_teacher_prompts(
        [self_success, peer_success, singleton_success],
        [self_success.reward, peer_success.reward, singleton_success.reward],
    )

    assert "peer solution" in str(self_success.teacher_prompt)
    assert "self solution" not in str(self_success.teacher_prompt)
    assert "self solution" in str(peer_success.teacher_prompt)
    assert "peer solution" not in str(peer_success.teacher_prompt)
    assert "only solution" in str(singleton_success.teacher_prompt)
    assert singleton_success.opd_sample_mask is True


def test_sdpo_falls_back_to_original_prompt_without_solution_or_feedback() -> None:
    from relax.utils.opd.sdpo.feedback import GoldenAnswerSDPOFeedback

    sample = _sample(3, 0, "an answer", {"score": 0.0})
    GoldenAnswerSDPOFeedback().prepare_teacher_prompts([sample], [sample.reward])

    assert sample.teacher_prompt == sample.prompt
    assert sample.opd_sample_mask is False


def test_sdpo_fallback_copies_original_message_prompt() -> None:
    from relax.utils.opd.sdpo.feedback import GoldenAnswerSDPOFeedback

    prompt = [{"role": "user", "content": "question"}]
    sample = Sample(
        group_index=3,
        prompt=prompt,
        response="answer",
        response_length=1,
        reward={"score": 0.0},
    )

    GoldenAnswerSDPOFeedback().prepare_teacher_prompts([sample], [sample.reward])

    assert sample.teacher_prompt == prompt
    assert sample.teacher_prompt is not prompt
    assert sample.opd_sample_mask is False


@pytest.mark.parametrize("feedback_class_name", _SDPO_FEEDBACK_CLASSES)
def test_sdpo_format_error_without_peer_injects_feedback(feedback_class_name: str) -> None:
    sample = _sample(12, 0, "wrong", {"score": 0.0, "format_error": 1, "feedback": "wrong format message"})
    _sdpo_feedback_class(feedback_class_name)().prepare_teacher_prompts([sample], [sample.reward])

    assert "wrong format message" in str(sample.teacher_prompt)
    assert sample.opd_sample_mask is True


@pytest.mark.parametrize("feedback_class_name", _SDPO_FEEDBACK_CLASSES)
def test_sdpo_truncation_without_peer_injects_feedback(feedback_class_name: str) -> None:
    sample = _sample(13, 0, "truncated", {"score": 0.0, "feedback": "truncation message"})
    sample.status = Sample.Status.TRUNCATED
    _sdpo_feedback_class(feedback_class_name)().prepare_teacher_prompts([sample], [sample.reward])

    assert "truncation message" in str(sample.teacher_prompt)
    assert sample.opd_sample_mask is True


@pytest.mark.parametrize("feedback_class_name", _SDPO_FEEDBACK_CLASSES)
def test_sdpo_normal_wrong_without_peer_drops_feedback_and_skips_distillation(feedback_class_name: str) -> None:
    sample = _sample(14, 0, "wrong", {"score": 0.0, "feedback": "generic wrong-answer feedback"})
    _sdpo_feedback_class(feedback_class_name)().prepare_teacher_prompts([sample], [sample.reward])

    assert sample.teacher_prompt == sample.prompt
    assert sample.opd_sample_mask is False


@pytest.mark.parametrize("feedback_class_name", _SDPO_FEEDBACK_CLASSES)
def test_sdpo_uses_same_group_successful_rollout(feedback_class_name: str) -> None:
    failed = _sample(4, 0, "wrong", {"score": 0.0})
    solved = _sample(4, 1, "worked solution", {"score": 1.0})
    _sdpo_feedback_class(feedback_class_name)().prepare_teacher_prompts(
        [failed, solved], [failed.reward, solved.reward]
    )

    assert "worked solution" in str(failed.teacher_prompt)
    assert failed.opd_sample_mask is True


@pytest.mark.parametrize("feedback_class_name", ["GoldenAnswerSDPOFeedback"])
def test_tool_use_uid_when_group_index_is_missing(feedback_class_name: str) -> None:
    failed = _sample(None, 0, "failed attempt", {"score": 0.0}, uid="uid-a")
    solved = _sample(None, 1, "same uid solution", {"score": 1.0}, uid="uid-a")
    unrelated = _sample(None, 2, "other uid solution", {"score": 1.0}, uid="uid-b")

    _sdpo_feedback_class(feedback_class_name)().prepare_teacher_prompts(
        [failed, solved, unrelated], [failed.reward, solved.reward, unrelated.reward]
    )

    assert "same uid solution" in str(failed.teacher_prompt)
    assert "other uid solution" not in str(failed.teacher_prompt)
    assert failed.opd_sample_mask is True


def test_sdpo_success_reward_threshold_configures_success_boundary() -> None:
    from relax.utils.opd.sdpo.feedback import GoldenAnswerSDPOFeedback

    low = _sample(5, 0, "partial solution", {"score": 0.9})
    strict = GoldenAnswerSDPOFeedback(success_reward_threshold=1.0)
    strict.prepare_teacher_prompts([low], [low.reward])
    assert low.teacher_prompt == low.prompt
    assert low.opd_sample_mask is False

    lenient = GoldenAnswerSDPOFeedback(success_reward_threshold=0.8)
    lenient.prepare_teacher_prompts([low], [low.reward])
    assert "partial solution" in str(low.teacher_prompt)
    assert low.opd_sample_mask is True


def test_opsd_assigns_dataset_privilege_while_opd_stays_empty() -> None:
    from relax.utils.opd.feedback import OPDFeedback
    from relax.utils.opd.opsd.feedback import OPSDFeedback

    opd_sample = _sample(6, 0, "answer", 1.0)
    privileged = _sample(6, 1, "answer", 1.0)
    privileged.metadata["opd_teacher_prompt"] = "dataset teacher prompt"
    unprivileged = _sample(6, 2, "answer", 1.0)

    OPDFeedback().prepare_teacher_prompts([opd_sample], [opd_sample.reward])
    OPSDFeedback(teacher_prompt_key="teacher_prompt").prepare_teacher_prompts(
        [privileged, unprivileged], [privileged.reward, unprivileged.reward]
    )

    assert opd_sample.teacher_prompt is None
    assert opd_sample.opd_sample_mask is None
    assert privileged.teacher_prompt == "dataset teacher prompt"
    assert privileged.opd_sample_mask is None
    assert unprivileged.teacher_prompt is None
    assert unprivileged.opd_sample_mask is None


@pytest.mark.parametrize("feedback_class_name", ["GoldenAnswerSDPOFeedback"])
def test_tool_use_feedback_share_peer_solutions(feedback_class_name: str) -> None:
    failed = _sample(11, 0, "failed attempt", {"score": 0.0, "feedback": "fix the tool call"})
    solved = _sample(11, 1, "successful attempt", {"score": 1.0})
    _sdpo_feedback_class(feedback_class_name)().prepare_teacher_prompts(
        [failed, solved], [failed.reward, solved.reward]
    )

    teacher_text = str(failed.teacher_prompt)
    assert "successful attempt" in teacher_text
    assert "fix the tool call" not in teacher_text
    assert failed.opd_sample_mask is True


# --- strategy hook contract ---------------------------------------------------


def test_load_feedback_class_defaults_to_opd() -> None:
    from relax.utils.opd.feedback import OPDFeedback, load_feedback_class

    assert load_feedback_class(None) is OPDFeedback
    assert load_feedback_class("") is OPDFeedback


def test_load_feedback_class_rejects_non_feedback_path() -> None:
    from relax.utils.opd.feedback import load_feedback_class

    with pytest.raises(TypeError, match="EnvironmentFeedback subclass"):
        load_feedback_class("relax.utils.opd.opd_utils.OPD_SAMPLE_MASK")

    with pytest.raises(ValueError, match="Invalid feedback class path"):
        load_feedback_class("not-a-dotted-path")


def test_load_feedback_defaults_to_opd_with_empty_kwargs() -> None:
    from relax.utils.opd.feedback import OPDFeedback, load_feedback

    feedback = load_feedback(None, None)
    assert isinstance(feedback, OPDFeedback)
    assert feedback.teacher_prompt_key is None
    assert feedback.success_reward_threshold == 1.0


def test_load_feedback_binds_kwargs_to_selected_class() -> None:
    from relax.utils.opd.feedback import load_feedback
    from relax.utils.opd.opsd.feedback import OPSDFeedback
    from relax.utils.opd.sdpo.feedback import SDPOFeedback

    sdpo = load_feedback(
        "relax.utils.opd.sdpo.feedback.SDPOFeedback",
        {"success_reward_threshold": 0.8},
    )
    assert isinstance(sdpo, SDPOFeedback)
    assert sdpo.success_reward_threshold == 0.8

    opsd = load_feedback(
        "relax.utils.opd.opsd.feedback.OPSDFeedback",
        {"teacher_prompt_key": "teacher_prompt"},
    )
    assert isinstance(opsd, OPSDFeedback)
    assert opsd.teacher_prompt_key == "teacher_prompt"


def test_load_feedback_rejects_unknown_kwargs_and_missing_opsd_key() -> None:
    from relax.utils.opd.feedback import load_feedback

    with pytest.raises(TypeError, match="Invalid --opd-feedback-kwargs for SDPOFeedback"):
        load_feedback("relax.utils.opd.sdpo.feedback.SDPOFeedback", {"unknown_param": 1})

    with pytest.raises(ValueError, match="requires"):
        load_feedback("relax.utils.opd.opsd.feedback.OPSDFeedback", {})


def test_code_sdpo_feedback_is_a_placeholder() -> None:
    from relax.utils.opd.sdpo.feedback import CodeSDPOFeedback

    feedback = CodeSDPOFeedback()
    with pytest.raises(NotImplementedError):
        feedback.record_sample_feedback(Sample(), 0.0)
    with pytest.raises(NotImplementedError):
        feedback.prepare_teacher_prompts([Sample()], [0.0])


def test_opd_feedback_hooks_are_no_ops() -> None:
    from relax.utils.opd.feedback import OPDFeedback

    feedback = OPDFeedback()
    assert feedback.extra_transfer_schema() == []
    feedback.produce_extra_transfer([Sample()], {})
    feedback.check_student_topk_ids(Sample(), 8)
    feedback.check_transfer_channels(Sample(), {}, 8)
    assert OPDFeedback.validate_launch_args(SimpleNamespace()) is None


def test_opd_create_opsd_worker_follows_dataset_keys() -> None:
    from relax.utils.opd.feedback import OPDFeedback

    args = SimpleNamespace(opd_teacher_image_key=None)
    assert OPDFeedback.create_opsd_worker(args).is_opsd is False

    args.opd_teacher_image_key = "images"
    assert OPDFeedback.create_opsd_worker(args).is_opsd is True


def test_sdpo_extra_transfer_schema_and_mask_column() -> None:
    from relax.utils.opd.sdpo.feedback import SDPOFeedback

    feedback = SDPOFeedback()
    assert feedback.extra_transfer_schema() == ["opd_sample_mask"]

    samples = [Sample(index=0, opd_sample_mask=True), Sample(index=1, opd_sample_mask=False)]
    train_data: dict = {}
    feedback.produce_extra_transfer(samples, train_data)

    assert train_data["opd_sample_mask"] == [True, False]


def test_sdpo_create_opsd_worker_is_always_active() -> None:
    from relax.utils.opd.sdpo.feedback import SDPOFeedback

    assert SDPOFeedback.create_opsd_worker(SimpleNamespace()).is_opsd is True


def _sdpo_launch_args(**overrides: object) -> SimpleNamespace:
    args = SimpleNamespace(
        opd_type="sglang",
        pipeline_model_parallel_size=1,
        enable_mtp_training=False,
        opd_token_selection="student_topk",
        opd_kl_type="jsd",
        opd_norm_mode="tail",
        calculate_per_token_loss=True,
        multimodal_keys=None,
        opd_teacher_image_key=None,
        opd_teacher_video_key=None,
        opd_teacher_audio_key=None,
        group_rm=True,
        opd_kl_coef=0.0,
        opd_loss_coef=1.0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_sdpo_validate_launch_args_accepts_valid_configuration() -> None:
    from relax.utils.opd.sdpo.feedback import SDPOFeedback

    assert SDPOFeedback.validate_launch_args(_sdpo_launch_args()) is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"group_rm": False}, "group-rm"),
        ({"opd_token_selection": "teacher_topk"}, "student_topk"),
        ({"opd_kl_coef": 1.0}, "opd-loss-coef"),
        ({"opd_type": "megatron"}, "opd-type=sglang"),
        ({"pipeline_model_parallel_size": 2}, "pipeline parallelism"),
        ({"calculate_per_token_loss": False}, "calculate-per-token-loss"),
    ],
)
def test_sdpo_validate_launch_args_rejects_invalid_configuration(overrides: dict, message: str) -> None:
    from relax.utils.opd.sdpo.feedback import SDPOFeedback

    with pytest.raises(ValueError, match=message):
        SDPOFeedback.validate_launch_args(_sdpo_launch_args(**overrides))


def test_sdpo_check_student_topk_ids_delegates_to_validation() -> None:
    from relax.utils.opd.sdpo.feedback import SDPOFeedback

    feedback = SDPOFeedback()
    valid = Sample(index=0, response_length=2, student_topk_token_ids=np.array([[1, 2], [3, 4]], dtype=np.int32))
    feedback.check_student_topk_ids(valid, 2)

    bad_shape = Sample(index=0, response_length=2, student_topk_token_ids=np.array([[1, 2]], dtype=np.int32))
    with pytest.raises(ValueError, match="student Top-K"):
        feedback.check_student_topk_ids(bad_shape, 2)

    with pytest.raises(ValueError, match="student Top-K"):
        feedback.check_student_topk_ids(Sample(index=0, response_length=2), 2)


def test_sdpo_check_transfer_channels_delegates_to_validation() -> None:
    from relax.utils.opd.opd_main_worker import TopkWorker
    from relax.utils.opd.sdpo.feedback import SDPOFeedback

    feedback = SDPOFeedback()
    feedback.check_transfer_channels(Sample(index=0, response_length=0), {}, 2)

    non_empty = Sample(index=0, response_length=1)
    with pytest.raises(ValueError, match="complete teacher Top-K payload"):
        feedback.check_transfer_channels(non_empty, {}, 2)

    complete = {
        TopkWorker.TRANSFER_TOKEN_IDS: np.array([[1, 2]], dtype=np.int32),
        TopkWorker.TRANSFER_TEACHER_LOG_PROBS: np.array([[-0.1, -0.2]], dtype=np.float32),
    }
    feedback.check_transfer_channels(non_empty, complete, 2)


def test_sdpo_prepare_teacher_prompts_rejects_multimodal() -> None:
    from relax.utils.opd.sdpo.feedback import SDPOFeedback

    sample = Sample(
        prompt="question",
        response="answer",
        response_length=1,
        multimodal_inputs={"images": [b"image"]},
    )

    with pytest.raises(ValueError, match="SDPO only supports text inputs"):
        SDPOFeedback().prepare_teacher_prompts([sample], [{"score": 0.0}])


def test_sdpo_prepare_teacher_prompts_requires_teacher_prompt_for_nonempty_response() -> None:
    from relax.utils.opd.sdpo.feedback import SDPOFeedback

    empty_prompt = Sample(prompt="", response="answer", response_length=6, reward={"score": 0.0})
    with pytest.raises(ValueError, match="SDPO requires a teacher prompt"):
        SDPOFeedback().prepare_teacher_prompts([empty_prompt], [empty_prompt.reward])

    empty_response = Sample(prompt="", response="", response_length=0, reward={"score": 0.0})
    SDPOFeedback().prepare_teacher_prompts([empty_response], [empty_response.reward])


def test_sdpo_prepare_teacher_prompts_clears_stale_teacher_payload() -> None:
    from relax.utils.opd.sdpo.feedback import SDPOFeedback

    sample = Sample(
        group_index=1,
        prompt="question",
        response="answer",
        response_length=1,
        reward={"score": 0.0},
        student_topk_token_ids=np.array([[1, 2]], dtype=np.int32),
        student_topk_log_probs=np.array([[-0.1, -0.2]], dtype=np.float32),
        teacher_log_probs=[-0.3],
        teacher_topk_token_ids=np.array([[3, 4]], dtype=np.int32),
        opd_topk_token_ids=np.array([[3, 4]], dtype=np.int32),
        teacher_tokens=[9, 10],
        teacher_prompt_length=1,
    )

    SDPOFeedback().prepare_teacher_prompts([sample], [sample.reward])

    assert sample.teacher_log_probs is None
    assert sample.teacher_topk_token_ids is None
    assert sample.opd_topk_token_ids is None
    assert sample.teacher_tokens is None
    assert sample.teacher_prompt_length is None
    np.testing.assert_array_equal(sample.student_topk_token_ids, np.array([[1, 2]], dtype=np.int32))
    np.testing.assert_array_equal(sample.student_topk_log_probs, np.array([[-0.1, -0.2]], dtype=np.float32))
