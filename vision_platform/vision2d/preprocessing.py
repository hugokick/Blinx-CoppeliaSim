from __future__ import annotations

import cv2
import numpy as np

from vision_platform.vision2d.models import Vision2DConfig


def validate_bgr_image(image_bgr: np.ndarray) -> None:
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a numpy BGR array")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have non-empty HxWx3 BGR shape")
    if image_bgr.shape[0] == 0 or image_bgr.shape[1] == 0:
        raise ValueError("image_bgr must be non-empty")
    if image_bgr.dtype != np.uint8:
        raise TypeError("image_bgr must use uint8 BGR values")


def preprocess_image(
    image_bgr: np.ndarray,
    config: Vision2DConfig,
) -> dict[str, np.ndarray]:
    validate_bgr_image(image_bgr)
    blurred = cv2.GaussianBlur(
        image_bgr,
        (config.gaussian_kernel_size, config.gaussian_kernel_size),
        0,
    )
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    segmentation_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    foreground = cv2.inRange(
        segmentation_hsv,
        np.array([0, config.saturation_min, config.value_min], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.morphology_kernel_size, config.morphology_kernel_size),
    )
    cleaned = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return {
        "gray": gray,
        "hsv": hsv,
        "foreground_mask": foreground,
        "cleaned_mask": cleaned,
    }
