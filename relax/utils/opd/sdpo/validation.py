# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from typing import Any

import torch


def validate_sdpo_text_only(sample: Any) -> None:
    """Reject multimodal or structured-content samples at the SDPO boundary."""

    def present(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, dict):
            return any(present(item) for item in value.values())
        if isinstance(value, (list, tuple, set)):
            return any(present(item) for item in value)
        if isinstance(value, str):
            return bool(value)
        if hasattr(value, "numel"):
            return bool(value.numel())
        return True

    for field_name in (
        "multimodal_inputs",
        "multimodal_train_inputs",
        "teacher_multimodal_inputs",
        "teacher_image_data",
        "teacher_image_b64_list",
        "teacher_image_grid_thw",
    ):
        value = getattr(sample, field_name, None)
        if present(value):
            raise ValueError(f"SDPO only supports text inputs; sample contains {field_name}")

    for field_name in ("prompt", "teacher_prompt"):
        prompt = getattr(sample, field_name, None)
        if prompt is None or isinstance(prompt, str):
            continue
        if not isinstance(prompt, list):
            raise ValueError(f"SDPO only supports text {field_name}; expected a string or text messages")
        for index, message in enumerate(prompt):
            if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                raise ValueError(f"SDPO text message {index} in {field_name} must contain a string role")
            if "content" in message and not isinstance(message["content"], str):
                raise ValueError("Relax-SDPO only supports text chat message content")


def validate_sdpo_topk_payload(
    *, token_ids: Any, teacher_log_probs: Any, response_rows: int, top_k: int, sample_index: int
) -> None:
    if response_rows == 0:
        return
    if token_ids is None or teacher_log_probs is None:
        raise ValueError(f"SDPO sample {sample_index} is missing its complete teacher Top-K payload")
    expected_shape = (response_rows, top_k)
    token_ids_tensor = torch.as_tensor(token_ids)
    teacher_tensor = torch.as_tensor(teacher_log_probs)
    if tuple(token_ids_tensor.shape) != expected_shape:
        raise ValueError(
            f"SDPO sample {sample_index} token-id payload shape {tuple(token_ids_tensor.shape)} "
            f"does not match {expected_shape}"
        )
    if tuple(teacher_tensor.shape) != expected_shape:
        raise ValueError(
            f"SDPO sample {sample_index} teacher payload shape {tuple(teacher_tensor.shape)} "
            f"does not match {expected_shape}"
        )
    if torch.isnan(teacher_tensor).any() or torch.isposinf(teacher_tensor).any():
        raise ValueError(f"SDPO sample {sample_index} teacher Top-K payload contains NaN or +inf")


def validate_sdpo_student_topk_ids(*, token_ids: Any, response_rows: int, top_k: int, sample_index: int) -> None:
    if token_ids is None:
        raise ValueError(f"SDPO sample {sample_index} is missing student Top-K token ids")
    if response_rows <= 0 or top_k <= 0:
        raise ValueError(
            f"SDPO sample {sample_index} has invalid Top-K dimensions: rows={response_rows}, top_k={top_k}"
        )
    token_ids_tensor = torch.as_tensor(token_ids)
    expected_shape = (response_rows, top_k)
    if tuple(token_ids_tensor.shape) != expected_shape:
        raise ValueError(
            f"SDPO sample {sample_index} student Top-K id shape {tuple(token_ids_tensor.shape)} "
            f"does not match {expected_shape}"
        )
    if token_ids_tensor.dtype == torch.bool or token_ids_tensor.dtype.is_floating_point:
        raise ValueError(f"SDPO sample {sample_index} student Top-K ids must be integer token ids")
    if token_ids_tensor.numel() and bool((token_ids_tensor < 0).any()):
        raise ValueError(f"SDPO sample {sample_index} student Top-K ids contain a negative token id")
