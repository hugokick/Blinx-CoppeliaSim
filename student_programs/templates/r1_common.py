from __future__ import annotations

from dataclasses import dataclass
from math import pi
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VisualObject:
    center_px: tuple[float, float]
    world_xy_mm: tuple[float, float]
    color: str
    shape: str
    confidence: float


def pixel_to_world(matrix, center_px):
    u, v = (float(center_px[0]), float(center_px[1]))
    return (
        float(matrix[0][0]) * u
        + float(matrix[0][1]) * v
        + float(matrix[0][2]),
        float(matrix[1][0]) * u
        + float(matrix[1][1]) * v
        + float(matrix[1][2]),
    )


def _color_name(hue):
    if hue < 10 or hue >= 170:
        return "red"
    if hue < 35:
        return "yellow"
    if hue < 85:
        return "green"
    if hue < 135:
        return "blue"
    return "unknown"


def detect_colored_objects(
    image_bgr,
    *,
    calibration_matrix,
    pick_x_max_mm,
    minimum_area_px=180,
):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([0, 70, 60], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    output = []
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(minimum_area_px):
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        center = (
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        )
        world_xy = pixel_to_world(calibration_matrix, center)
        if world_xy[0] > float(pick_x_max_mm):
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = (
            4.0 * pi * area / (perimeter * perimeter)
            if perimeter > 0
            else 0.0
        )
        sample = hsv[int(round(center[1])), int(round(center[0]))]
        output.append(
            VisualObject(
                center_px=center,
                world_xy_mm=world_xy,
                color=_color_name(int(sample[0])),
                shape="cylinder" if circularity >= 0.78 else "block",
                confidence=min(1.0, area / 600.0),
            )
        )
    return sorted(output, key=lambda item: item.world_xy_mm)


def detect_digit_objects(
    image_bgr,
    *,
    calibration_matrix,
    pick_x_max_mm,
    reference_dir,
):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    bright = cv2.inRange(gray, 180, 255)
    contours, _ = cv2.findContours(
        bright,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    references = {
        digit: cv2.imread(
            str(Path(reference_dir) / f"{digit}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        for digit in (1, 2, 3)
    }
    if any(image is None for image in references.values()):
        raise RuntimeError("数字参考图读取失败")
    output = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width * height < 300:
            continue
        center = (x + width / 2.0, y + height / 2.0)
        world_xy = pixel_to_world(calibration_matrix, center)
        if world_xy[0] > float(pick_x_max_mm):
            continue
        crop = gray[y : y + height, x : x + width]
        normalized = cv2.resize(crop, (64, 96))
        scores = {
            digit: float(
                cv2.matchTemplate(
                    normalized,
                    reference,
                    cv2.TM_CCOEFF_NORMED,
                )[0, 0]
            )
            for digit, reference in references.items()
        }
        digit, confidence = max(scores.items(), key=lambda item: item[1])
        output.append((digit, world_xy, confidence))
    return sorted(output, key=lambda item: item[0])


def pick_and_place(
    ctx,
    *,
    pick_xy,
    drop_xyz,
    pick_z_mm,
    safe_z_mm,
    speed,
):
    x_mm, y_mm = (float(pick_xy[0]), float(pick_xy[1]))
    drop_x, drop_y, drop_z = (
        float(drop_xyz[0]),
        float(drop_xyz[1]),
        float(drop_xyz[2]),
    )
    ctx.robot.move_world(x_mm, y_mm, safe_z_mm, speed=speed)
    ctx.robot.move_world(x_mm, y_mm, pick_z_mm, speed=8.0)
    ctx.tool.on()
    ctx.robot.move_world(x_mm, y_mm, safe_z_mm, speed=speed)
    ctx.robot.move_world(drop_x, drop_y, safe_z_mm, speed=speed)
    ctx.robot.move_world(drop_x, drop_y, drop_z, speed=8.0)
    ctx.tool.off()
    ctx.robot.move_world(drop_x, drop_y, safe_z_mm, speed=speed)
