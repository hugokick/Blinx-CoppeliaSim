from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterable

import cv2
import numpy as np

from vision_platform.models import Detection


class ColorShapeRecognizer:
    def __init__(
        self,
        *,
        min_area_ratio: float = 0.002,
        max_area_ratio: float = 0.40,
        saturation_min: int = 60,
        value_min: int = 40,
    ) -> None:
        if not 0 < min_area_ratio < max_area_ratio <= 1:
            raise ValueError("area ratios must satisfy 0 < min < max <= 1")
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        self.saturation_min = int(saturation_min)
        self.value_min = int(value_min)

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        self._validate_image(image_bgr)
        image_area = float(image_bgr.shape[0] * image_bgr.shape[1])
        min_area = image_area * self.min_area_ratio
        max_area = image_area * self.max_area_ratio

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0, self.saturation_min, self.value_min], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        kernel_size = max(3, int(round(min(image_bgr.shape[:2]) / 160)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        unsorted: list[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue
            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-9:
                continue
            center = (
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            )
            color, color_confidence = self._classify_color(hsv, contour)
            shape, angle_deg, shape_confidence = self._classify_shape(contour)
            confidence = max(
                0.0,
                min(1.0, (color_confidence + shape_confidence) / 2.0),
            )
            unsorted.append(
                Detection(
                    detection_id="pending",
                    center_px=(
                        round(center[0], 3),
                        round(center[1], 3),
                    ),
                    color=color,
                    shape=shape,
                    angle_deg=round(angle_deg, 3),
                    area_px=area,
                    confidence=confidence,
                    contour=contour.copy(),
                )
            )

        unsorted.sort(key=lambda item: (item.center_px[1], item.center_px[0]))
        return [
            replace(item, detection_id=f"det-{index:03d}")
            for index, item in enumerate(unsorted, start=1)
        ]

    def annotate(
        self,
        image_bgr: np.ndarray,
        detections: Iterable[Detection],
    ) -> np.ndarray:
        self._validate_image(image_bgr)
        output = image_bgr.copy()
        for detection in detections:
            if detection.contour is not None:
                cv2.drawContours(output, [detection.contour], -1, (255, 255, 255), 2)
            center = (
                int(round(detection.center_px[0])),
                int(round(detection.center_px[1])),
            )
            cv2.circle(output, center, 4, (255, 255, 255), -1)
            label = (
                f"{detection.detection_id} "
                f"{detection.color}/{detection.shape} "
                f"{detection.confidence:.2f}"
            )
            origin = (max(0, center[0] - 60), max(16, center[1] - 12))
            cv2.putText(
                output,
                label,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return output

    @staticmethod
    def _validate_image(image_bgr: np.ndarray) -> None:
        if (
            not isinstance(image_bgr, np.ndarray)
            or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3
            or image_bgr.size == 0
        ):
            raise ValueError("image_bgr must be a non-empty HxWx3 array")

    @staticmethod
    def _classify_color(
        hsv: np.ndarray,
        contour: np.ndarray,
    ) -> tuple[str, float]:
        contour_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        pixels = hsv[contour_mask > 0]
        if pixels.size == 0:
            return "unknown", 0.0
        hue = float(np.median(pixels[:, 0]))
        saturation = float(np.median(pixels[:, 1]))
        if hue <= 10 or hue >= 170:
            color = "red"
        elif 15 <= hue <= 40:
            color = "yellow"
        elif 40 < hue <= 90:
            color = "green"
        elif 90 < hue <= 140:
            color = "blue"
        else:
            color = "unknown"
        confidence = min(1.0, saturation / 180.0)
        if color == "unknown":
            confidence *= 0.5
        return color, confidence

    @staticmethod
    def _classify_shape(
        contour: np.ndarray,
    ) -> tuple[str, float, float]:
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            return "unknown", 0.0, 0.0
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        vertex_count = len(approx)
        rectangle = cv2.minAreaRect(contour)
        width, height = rectangle[1]
        angle = float(rectangle[2])
        if width > 0 and height > 0 and width < height:
            angle += 90.0

        if vertex_count == 3:
            return "triangle", angle, 0.95
        if vertex_count == 4:
            shorter = min(width, height)
            longer = max(width, height)
            aspect = shorter / longer if longer > 0 else 0.0
            if aspect >= 0.85:
                return "square", angle, min(1.0, aspect)
            return "rectangle", angle, min(1.0, 1.0 - abs(aspect - 0.5))

        area = float(cv2.contourArea(contour))
        circularity = (
            4.0 * math.pi * area / (perimeter * perimeter)
            if perimeter > 0
            else 0.0
        )
        if vertex_count >= 5 and circularity >= 0.70:
            return "circle", 0.0, min(1.0, circularity)
        return "polygon", angle, max(0.2, min(0.8, circularity))
