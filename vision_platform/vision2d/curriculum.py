from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from vision_platform.errors import VisionPlatformError
from vision_platform.vision2d.models import (
    PixelScale,
    Vision2DConfig,
)


_CONFIG_FIELDS = frozenset(
    {
        "profile_id",
        "roi_px",
        "min_area_ratio",
        "max_area_ratio",
        "saturation_min",
        "value_min",
        "pixel_scale_mm",
    }
)
_STANDARD_IMAGE_SIZE = (512, 512)


class Vision2DConfigError(VisionPlatformError, ValueError):
    def __init__(self, message: str) -> None:
        super().__init__("VISION2D_CONFIG_INVALID", message)


@dataclass(frozen=True)
class CurriculumVision2DConfig:
    profile_id: str
    image_size: tuple[int, int]
    roi_px: tuple[int, int, int, int]
    vision2d_config: Vision2DConfig

    def to_public_dict(self) -> dict[str, Any]:
        scale = self.vision2d_config.pixel_scale
        return {
            "profile_id": self.profile_id,
            "roi_px": list(self.roi_px),
            "min_area_ratio": self.vision2d_config.min_area_ratio,
            "max_area_ratio": self.vision2d_config.max_area_ratio,
            "saturation_min": self.vision2d_config.saturation_min,
            "value_min": self.vision2d_config.value_min,
            "pixel_scale_mm": (
                None
                if scale is None
                else [scale.mm_per_pixel_x, scale.mm_per_pixel_y]
            ),
        }


def _invalid(message: str) -> Vision2DConfigError:
    return Vision2DConfigError(message)


def _finite_number(value: Any, *, field: str) -> float:
    if type(value) not in {int, float}:
        raise _invalid(f"vision2d.{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise _invalid(f"vision2d.{field} must be a finite number")
    return result


def _threshold(value: Any, *, field: str) -> int:
    if type(value) is not int or not 0 <= value <= 255:
        raise _invalid(f"vision2d.{field} must be an integer from 0 to 255")
    return value


def _published_image_size(value: Any) -> tuple[int, int]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(component) is not int for component in value)
        or value != _STANDARD_IMAGE_SIZE
    ):
        raise _invalid("image_size must be the standard 512 x 512 profile")
    return value


def _roi(value: Any, *, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    if (
        type(value) not in {list, tuple}
        or len(value) != 4
        or any(type(component) is not int for component in value)
    ):
        raise _invalid("vision2d.roi_px must contain four exact integers")
    x_min, y_min, x_max, y_max = value
    width, height = image_size
    if not (
        0 <= x_min < x_max <= width
        and 0 <= y_min < y_max <= height
    ):
        raise _invalid("vision2d.roi_px must stay inside the published image")
    return x_min, y_min, x_max, y_max


def _pixel_scale(value: Any) -> PixelScale | None:
    if value is None:
        return None
    if type(value) not in {list, tuple} or len(value) != 2:
        raise _invalid("vision2d.pixel_scale_mm must contain two values or null")
    try:
        x_value = _finite_number(value[0], field="pixel_scale_mm[0]")
        y_value = _finite_number(value[1], field="pixel_scale_mm[1]")
        return PixelScale(x_value, y_value)
    except Vision2DConfigError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise _invalid(
            "vision2d.pixel_scale_mm must contain two finite positive values"
        ) from error


def parse_curriculum_config(
    public_parameters: Mapping[str, Any],
    *,
    image_size: tuple[int, int],
) -> CurriculumVision2DConfig:
    try:
        if not isinstance(public_parameters, Mapping):
            raise _invalid("public_parameters.vision2d is required")
        payload = public_parameters.get("vision2d")
        if not isinstance(payload, Mapping):
            raise _invalid("public_parameters.vision2d must be an object")
        if set(payload) != _CONFIG_FIELDS:
            raise _invalid("vision2d fields do not match the published contract")
        selected_size = _published_image_size(image_size)
        profile_id = payload["profile_id"]
        if type(profile_id) is not str or profile_id != "standard":
            raise _invalid("vision2d.profile_id must equal standard")
        roi_px = _roi(payload["roi_px"], image_size=selected_size)
        min_area_ratio = _finite_number(
            payload["min_area_ratio"], field="min_area_ratio"
        )
        max_area_ratio = _finite_number(
            payload["max_area_ratio"], field="max_area_ratio"
        )
        if not 0.0 < min_area_ratio < max_area_ratio <= 1.0:
            raise _invalid(
                "vision2d area ratios must satisfy 0 < min < max <= 1"
            )
        config = Vision2DConfig(
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            saturation_min=_threshold(
                payload["saturation_min"], field="saturation_min"
            ),
            value_min=_threshold(payload["value_min"], field="value_min"),
            pixel_scale=_pixel_scale(payload["pixel_scale_mm"]),
        )
        return CurriculumVision2DConfig(
            profile_id=profile_id,
            image_size=selected_size,
            roi_px=roi_px,
            vision2d_config=config,
        )
    except Vision2DConfigError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise _invalid("vision2d configuration is invalid") from error


def mask_to_curriculum_roi(
    image_bgr: np.ndarray,
    config: CurriculumVision2DConfig,
) -> np.ndarray:
    try:
        if type(config) is not CurriculumVision2DConfig:
            raise _invalid("curriculum config is invalid")
        width, height = config.image_size
        if (
            type(image_bgr) is not np.ndarray
            or image_bgr.dtype != np.uint8
            or image_bgr.ndim != 3
            or image_bgr.shape != (height, width, 3)
        ):
            raise _invalid("image must match the published uint8 BGR profile")
        x_min, y_min, x_max, y_max = config.roi_px
        masked = np.zeros_like(image_bgr)
        masked[y_min:y_max, x_min:x_max] = image_bgr[
            y_min:y_max, x_min:x_max
        ]
        return masked
    except Vision2DConfigError:
        raise
    except (TypeError, ValueError, IndexError) as error:
        raise _invalid("image cannot be masked with the published ROI") from error
