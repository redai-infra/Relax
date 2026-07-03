# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import base64
import json
import logging
import re
from copy import deepcopy
from io import BytesIO
from math import ceil, floor
from typing import Any

from PIL import Image


logger = logging.getLogger(__name__)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
SUPPORTED_TOOL_NAMES = {"image_zoom_in_tool", "image_rotate_tool"}


def load_initial_image(messages: list[dict[str, Any]]) -> Image.Image:
    item = messages[-1]["content"][0]
    _, _, encoded = item["image_url"]["url"].partition(",")
    with BytesIO(base64.b64decode(encoded)) as fh:
        image = Image.open(fh)
        image.load()
        return image


def encode_image_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    rgb_image = image.convert("RGB") if image.mode != "RGB" else image
    rgb_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _extract_tool_call(text: str) -> dict[str, Any] | None:
    matches = list(TOOL_CALL_RE.finditer(text))
    if not matches:
        return None
    raw_payload = matches[-1].group(1).strip()
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError:
        return None
    name = payload.get("name") or payload.get("function", {}).get("name")
    arguments = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not name:
        return None
    return {"name": name, "arguments": arguments}


def _validate_bbox(left: float, top: float, right: float, bottom: float) -> bool:
    try:
        if not (left < right and bottom > top):
            raise ValueError(f"invalid shape for {left=}, {top=}, {right=}, {bottom=}")
        height = bottom - top
        width = right - left
        if max(height, width) / min(height, width) > 100:
            raise ValueError(f"aspect ratio error: {left=}, {top=}, {right=}, {bottom=}")
        if min(height, width) <= 30:
            raise ValueError(f"{height=}, {width=} is too small")
        return True
    except Exception as exc:
        logger.warning("BBox validation failed: %s", exc)
        return False


def _maybe_resize_bbox(
    image: Image.Image,
    bbox_2d: list[float],
    *,
    min_dimension: int,
    normalize_bbox: bool = True,
) -> list[float] | None:
    image_width = image.width
    image_height = image.height
    left, top, right, bottom = bbox_2d

    if normalize_bbox:
        left = left / 1000.0 * image_width
        top = top / 1000.0 * image_height
        right = right / 1000.0 * image_width
        bottom = bottom / 1000.0 * image_height

    left = max(0.0, float(left))
    top = max(0.0, float(top))
    right = min(float(image_width), float(right))
    bottom = min(float(image_height), float(bottom))
    if not _validate_bbox(left, top, right, bottom):
        return None

    current_bbox = [left, top, right, bottom]
    height = bottom - top
    width = right - left
    if height < min_dimension or width < min_dimension:
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        min_dim = min(height, width)
        if min_dim == 0:
            return None
        ratio = min_dimension / min_dim
        target_width = width * ratio
        target_height = height * ratio
        if target_width > image_width:
            scale_down = image_width / target_width
            target_width = image_width
            target_height *= scale_down
        if target_height > image_height:
            scale_down = image_height / target_height
            target_height = image_height
            target_width *= scale_down
        new_half_width = target_width / 2.0
        new_half_height = target_height / 2.0
        new_left = max(0.0, center_x - new_half_width)
        new_top = max(0.0, center_y - new_half_height)
        if new_left + target_width > image_width:
            new_left = image_width - target_width
        if new_top + target_height > image_height:
            new_top = image_height - target_height
        current_bbox = [
            floor(new_left),
            floor(new_top),
            ceil(new_left + target_width),
            ceil(new_top + target_height),
        ]

    final_left, final_top, final_right, final_bottom = current_bbox
    if not _validate_bbox(final_left, final_top, final_right, final_bottom):
        logger.warning("Final bbox is invalid after processing: %s", current_bbox)
        return None
    final_height = floor(final_bottom) - floor(final_top)
    final_width = floor(final_right) - floor(final_left)
    if final_height < min_dimension or final_width < min_dimension:
        logger.warning(
            "Final bbox size (%sx%s) are still smaller than minimum (%s)."
            "Original bbox: %s, original image size: %sx%s",
            final_width,
            final_height,
            min_dimension,
            bbox_2d,
            image_width,
            image_height,
        )
        return None
    return current_bbox


class DeepeyesToolEnv:
    MIN_DIMENSION = 28

    def __init__(
        self,
        *,
        max_turns: int,
        current_image: Image.Image,
        normalize_bbox: bool = True,
    ) -> None:
        self.max_turns = max_turns
        self.turn = 0
        self.current_image = current_image
        self.normalize_bbox = normalize_bbox

    def _build_observation_message(
        self, *, text: str | None = None, image: Image.Image | None = None
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image_url", "image_url": {"url": encode_image_data_uri(image)}})
        if isinstance(text, str) and text:
            content.append({"type": "text", "text": text})
        return {"role": "tool", "content": content}

    def _apply_tool(self, tool_call: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        info: dict[str, Any] = {"tool_call": deepcopy(tool_call), "tool_executed": True}
        current_image = self.current_image
        name = tool_call["name"]
        arguments = tool_call["arguments"]
        if name not in SUPPORTED_TOOL_NAMES:
            raise ValueError(f"Unknown tool name: {name}")
        if name == "image_zoom_in_tool":
            bbox = arguments["bbox_2d"]
            resized_bbox = _maybe_resize_bbox(
                current_image,
                bbox,
                min_dimension=self.MIN_DIMENSION,
                normalize_bbox=self.normalize_bbox,
            )
            if resized_bbox is None:
                raise ValueError("ZOOM IN ARGUMENTS ARE INVALID")
            self.current_image = current_image.crop(resized_bbox)
        else:
            angle = arguments["angle"]
            self.current_image = current_image.rotate(angle)
        info["tool_used"] = name
        info["status"] = "success"
        return self._build_observation_message(image=self.current_image), info

    def step(self, response_text: str) -> tuple[dict[str, Any] | None, bool, dict[str, Any]]:
        self.turn += 1
        if ANSWER_RE.search(response_text):
            return None, True, {"final_answer": True}

        tool_call = _extract_tool_call(response_text)
        if tool_call is None:
            return None, True, {"tool_executed": False, "stop_reason": "no_tool_call"}

        try:
            observation_message, info = self._apply_tool(tool_call)
        except Exception as exc:
            info = {
                "tool_call": deepcopy(tool_call),
                "tool_executed": False,
                "status": "failed",
                "error": str(exc),
            }
            observation_message = self._build_observation_message(text=f"Error: {exc}")
        done = self.turn >= self.max_turns
        if done:
            info["stop_reason"] = "max_turns"
        return observation_message, done, info
