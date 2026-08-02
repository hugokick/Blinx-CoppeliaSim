from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


VISION_TEMPLATE_INPUT_INVALID = "VISION_TEMPLATE_INPUT_INVALID"
VISION_TEMPLATE_MATCH_FAILED = "VISION_TEMPLATE_MATCH_FAILED"
VISION_TEMPLATE_ASSET_INVALID = "VISION_TEMPLATE_ASSET_INVALID"


class TemplateMatchError(ValueError):
    """A safe, stable error raised for invalid template-match inputs."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class TemplateMatchConfig:
    template_id: str = "v1_06_red_rectangle"
    template_version: str = "1.0.0"
    threshold: float = 0.72
    search_roi_px: tuple[int, int, int, int] = (0, 0, 256, 256)
    method: str = "TM_CCOEFF_NORMED"

    def __post_init__(self) -> None:
        if not isinstance(self.template_id, str) or not self.template_id.strip():
            raise ValueError("template_id must be a non-empty string")
        if not isinstance(self.template_version, str) or not self.template_version.strip():
            raise ValueError("template_version must be a non-empty string")
        threshold = float(self.threshold)
        if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be finite and between -1 and 1")
        if len(self.search_roi_px) != 4:
            raise ValueError("search_roi_px must contain x, y, width and height")
        x, y, width, height = (int(value) for value in self.search_roi_px)
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("search_roi_px must describe a positive ROI")
        if self.method != "TM_CCOEFF_NORMED":
            raise ValueError("method must be TM_CCOEFF_NORMED")
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "search_roi_px", (x, y, width, height))


@dataclass(frozen=True)
class TemplateMatchResult:
    template_id: str
    template_version: str
    status: str
    matched: bool
    score: float
    threshold: float
    bbox_px: tuple[int, int, int, int] | None
    center_px: tuple[float, float] | None
    image_size: tuple[int, int]
    search_roi_px: tuple[int, int, int, int]
    method: str = "TM_CCOEFF_NORMED"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.status not in {"MATCHED", "NOT_MATCHED"}:
            raise ValueError("status must be MATCHED or NOT_MATCHED")
        if not math.isfinite(float(self.score)):
            raise ValueError("score must be finite")
        if len(self.image_size) != 2 or any(int(value) <= 0 for value in self.image_size):
            raise ValueError("image_size must contain positive width and height")
        if len(self.search_roi_px) != 4:
            raise ValueError("search_roi_px must contain x, y, width and height")
        if self.bbox_px is not None:
            if len(self.bbox_px) != 4 or any(int(value) <= 0 for value in self.bbox_px[2:]):
                raise ValueError("bbox_px must contain x, y, width and height")
        if self.center_px is not None and len(self.center_px) != 2:
            raise ValueError("center_px must contain x and y")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_image(value: object, *, code: str, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise TemplateMatchError(code, f"{name} must be a uint8 numpy array")
    if value.ndim not in {2, 3}:
        raise TemplateMatchError(code, f"{name} must be 2D or 3D")
    if value.ndim == 3 and value.shape[2] not in {1, 3}:
        raise TemplateMatchError(code, f"{name} must have one or three channels")
    if value.size == 0 or value.shape[0] <= 0 or value.shape[1] <= 0:
        raise TemplateMatchError(code, f"{name} must not be empty")
    return value


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2 or image.shape[2] == 1:
        return image if image.ndim == 2 else image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _best_location(response: np.ndarray) -> tuple[int, int, float]:
    if response.ndim != 2 or response.size == 0:
        raise TemplateMatchError(VISION_TEMPLATE_MATCH_FAILED, "empty match response")
    if not np.isfinite(response).all():
        raise TemplateMatchError(VISION_TEMPLATE_MATCH_FAILED, "match response is not finite")
    best = float(np.max(response))
    candidates = np.argwhere(np.abs(response - best) <= 1e-12)
    if candidates.size == 0:
        raise TemplateMatchError(VISION_TEMPLATE_MATCH_FAILED, "no best match")
    y, x = min((int(point[0]), int(point[1])) for point in candidates)
    return x, y, best


def match_template(
    image_bgr: np.ndarray,
    template_bgr: np.ndarray,
    config: TemplateMatchConfig | None = None,
) -> TemplateMatchResult:
    """Match one verified template in a fixed image ROI."""

    config = config or TemplateMatchConfig()
    image = _validate_image(
        image_bgr, code=VISION_TEMPLATE_INPUT_INVALID, name="image_bgr"
    )
    template = _validate_image(
        template_bgr, code=VISION_TEMPLATE_ASSET_INVALID, name="template_bgr"
    )
    image_height, image_width = image.shape[:2]
    roi_x, roi_y, roi_width, roi_height = config.search_roi_px
    if roi_x + roi_width > image_width or roi_y + roi_height > image_height:
        raise TemplateMatchError(
            VISION_TEMPLATE_INPUT_INVALID, "search ROI is outside the image"
        )
    template_height, template_width = template.shape[:2]
    if template_width > roi_width or template_height > roi_height:
        raise TemplateMatchError(
            VISION_TEMPLATE_ASSET_INVALID,
            "template must fit inside the search ROI",
        )
    if image.ndim != template.ndim or (
        image.ndim == 3 and image.shape[2] != template.shape[2]
    ):
        raise TemplateMatchError(
            VISION_TEMPLATE_INPUT_INVALID,
            "image and template channel layouts must match",
        )
    roi = image[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
    response = cv2.matchTemplate(
        _as_gray(roi), _as_gray(template), cv2.TM_CCOEFF_NORMED
    )
    local_x, local_y, score = _best_location(response)
    x = roi_x + local_x
    y = roi_y + local_y
    bbox = (x, y, int(template_width), int(template_height))
    center = (
        round(x + template_width / 2.0, 6),
        round(y + template_height / 2.0, 6),
    )
    matched = score >= config.threshold
    return TemplateMatchResult(
        template_id=config.template_id,
        template_version=config.template_version,
        status="MATCHED" if matched else "NOT_MATCHED",
        matched=matched,
        score=score,
        threshold=config.threshold,
        bbox_px=bbox,
        center_px=center,
        image_size=(int(image_width), int(image_height)),
        search_roi_px=config.search_roi_px,
        method=config.method,
    )


def annotate_template_match(
    image_bgr: np.ndarray, result: TemplateMatchResult
) -> np.ndarray:
    """Return a copy with the fixed search ROI and best match overlaid."""

    image = _validate_image(
        image_bgr, code=VISION_TEMPLATE_INPUT_INVALID, name="image_bgr"
    )
    annotated = image.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    elif annotated.shape[2] == 1:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    roi_x, roi_y, roi_width, roi_height = result.search_roi_px
    cv2.rectangle(
        annotated,
        (roi_x, roi_y),
        (roi_x + roi_width - 1, roi_y + roi_height - 1),
        (255, 180, 0),
        1,
    )
    if result.bbox_px is not None:
        x, y, width, height = result.bbox_px
        color = (0, 220, 0) if result.matched else (0, 165, 255)
        cv2.rectangle(annotated, (x, y), (x + width - 1, y + height - 1), color, 2)
        label = f"{result.template_id} {result.score:.3f}"
        cv2.putText(
            annotated,
            label,
            (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def template_match_result_to_dict(result: TemplateMatchResult) -> dict[str, Any]:
    return result.to_dict()


__all__ = [
    "TemplateMatchConfig",
    "TemplateMatchError",
    "TemplateMatchResult",
    "VISION_TEMPLATE_ASSET_INVALID",
    "VISION_TEMPLATE_INPUT_INVALID",
    "VISION_TEMPLATE_MATCH_FAILED",
    "annotate_template_match",
    "match_template",
    "template_match_result_to_dict",
]
