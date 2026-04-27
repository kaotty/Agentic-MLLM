"""Re-bakes cropped images using the *exact* same logic as ``ImageZoomInTool``.

Why a re-bake instead of trusting the image embedded in the source dataset?
DeepEyesV2 trajectories visualise crops via ``matplotlib`` (with axes, titles
and figure padding) — those rendered PNGs would never match what
``ImageZoomInTool.execute`` produces during RL rollouts. To keep SFT and RL
input distributions aligned we redo the crop with raw PIL using the same
``MIN_DIMENSION`` clamp + bbox resize that lives in
``verl/tools/image_zoom_in_tool.py``.
"""

from __future__ import annotations

from math import ceil, floor
from typing import Optional, Tuple

from PIL import Image


# Mirrors ``ImageZoomInTool.MIN_DIMENSION``. Keep in sync.
MIN_DIMENSION = 28


def _validate_bbox(left: float, top: float, right: float, bottom: float) -> bool:
    if not (left < right and top < bottom):
        return False
    height = bottom - top
    width = right - left
    if min(height, width) == 0:
        return False
    if max(height, width) / min(height, width) > 100:
        return False
    return True


def maybe_resize_bbox(
    bbox_2d: list[float],
    image_width: int,
    image_height: int,
) -> Optional[list[int]]:
    """Replica of ``ImageZoomInTool._maybe_resize_bbox`` (ints out)."""

    left, top, right, bottom = bbox_2d
    left = max(0.0, float(left))
    top = max(0.0, float(top))
    right = min(float(image_width), float(right))
    bottom = min(float(image_height), float(bottom))

    if not _validate_bbox(left, top, right, bottom):
        return None

    current = [left, top, right, bottom]
    height = bottom - top
    width = right - left

    if height < MIN_DIMENSION or width < MIN_DIMENSION:
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        min_dim = min(height, width)
        if min_dim == 0:
            return None
        ratio = MIN_DIMENSION / min_dim
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
        new_left = center_x - new_half_width
        new_top = center_y - new_half_height
        if new_left < 0:
            new_left = 0
        if new_top < 0:
            new_top = 0
        if new_left + target_width > image_width:
            new_left = image_width - target_width
        if new_top + target_height > image_height:
            new_top = image_height - target_height
        new_right = new_left + target_width
        new_bottom = new_top + target_height
        current = [floor(new_left), floor(new_top), ceil(new_right), ceil(new_bottom)]

    final_left, final_top, final_right, final_bottom = current
    if not _validate_bbox(final_left, final_top, final_right, final_bottom):
        return None
    final_h = floor(final_bottom) - floor(final_top)
    final_w = floor(final_right) - floor(final_left)
    if final_h < MIN_DIMENSION or final_w < MIN_DIMENSION:
        return None
    return [int(round(c)) for c in current]


def bake_zoom(
    image: Image.Image, bbox_2d: list[float]
) -> Optional[Tuple[Image.Image, list[int]]]:
    """Apply the same clamp/resize/crop the runtime tool uses.

    Returns the cropped PIL image plus the (possibly resized) bbox actually
    cropped, or ``None`` when the bbox is unrecoverable.
    """

    width, height = image.size
    resized = maybe_resize_bbox(bbox_2d, image_width=width, image_height=height)
    if resized is None:
        return None
    try:
        cropped = image.crop(resized)
    except Exception:
        return None
    if cropped.size[0] < 1 or cropped.size[1] < 1:
        return None
    return cropped, resized


def zoom_response_text(bbox_2d: list[int], label: Optional[str] = None) -> str:
    """Match ``ImageZoomInTool.execute``'s success message verbatim."""

    if label:
        return f"Zoomed in on the image to the region {bbox_2d} with label {label}."
    return f"Zoomed in on the image to the region {bbox_2d}."
