from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from pathlib import Path

import cv2
import numpy as np


MIN_VISUAL_CONFIDENCE = 0.5
_HUE_FAMILIES = (
    ("red", ((0, 9), (170, 179))),
    ("yellow", ((10, 34),)),
    ("green", ((35, 84),)),
    ("blue", ((85, 134),)),
    ("unknown", ((135, 169),)),
)


@dataclass(frozen=True)
class VisualObject:
    center_px: tuple[float, float]
    world_xy_mm: tuple[float, float]
    color: str
    shape: str
    confidence: float


def _finite_float(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} 必须是有限数值")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} 必须是有限数值") from error
    if not isfinite(result):
        raise ValueError(f"{label} 必须是有限数值")
    return result


def _finite_vector(value, length, label):
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} 必须包含 {length} 个有限数值")
    try:
        if len(value) != length:
            raise ValueError
        return tuple(
            _finite_float(value[index], f"{label}[{index}]")
            for index in range(length)
        )
    except (TypeError, ValueError, IndexError) as error:
        if isinstance(error, ValueError) and "必须是有限数值" in str(error):
            raise
        raise ValueError(
            f"{label} 必须包含 {length} 个有限数值"
        ) from error


def _affine_matrix(matrix):
    if isinstance(matrix, (str, bytes, bytearray)):
        raise ValueError("calibration_matrix 必须是 2×3 有限矩阵")
    try:
        if len(matrix) != 2:
            raise ValueError
        return tuple(
            _finite_vector(row, 3, f"calibration_matrix[{index}]")
            for index, row in enumerate(matrix)
        )
    except (TypeError, ValueError, IndexError) as error:
        if isinstance(error, ValueError) and "必须是有限数值" in str(error):
            raise
        raise ValueError("calibration_matrix 必须是 2×3 有限矩阵") from error


def _bgr_image(image_bgr):
    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.dtype != np.uint8
        or image_bgr.ndim != 3
        or image_bgr.shape[0] <= 0
        or image_bgr.shape[1] <= 0
        or image_bgr.shape[2] != 3
    ):
        raise ValueError("image_bgr 必须是非空 uint8 BGR 图像")
    return image_bgr


def pixel_to_world(matrix, center_px):
    affine = _affine_matrix(matrix)
    u, v = _finite_vector(center_px, 2, "center_px")
    result = (
        affine[0][0] * u + affine[0][1] * v + affine[0][2],
        affine[1][0] * u + affine[1][1] * v + affine[1][2],
    )
    if not all(isfinite(component) for component in result):
        raise ValueError("像素到世界坐标的结果必须是有限数值")
    return result


def _snap_world_xy_to_robot_grid(world_xy):
    return tuple(
        round(component * 2.0) / 2.0
        for component in _finite_vector(world_xy, 2, "world_xy")
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
    image_bgr = _bgr_image(image_bgr)
    calibration_matrix = _affine_matrix(calibration_matrix)
    pick_x_max_mm = _finite_float(pick_x_max_mm, "pick_x_max_mm")
    minimum_area_px = _finite_float(minimum_area_px, "minimum_area_px")
    if minimum_area_px <= 0:
        raise ValueError("minimum_area_px 必须大于 0")
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    output = []
    kernel = np.ones((3, 3), dtype=np.uint8)
    for color_name, hue_ranges in _HUE_FAMILIES:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for hue_min, hue_max in hue_ranges:
            mask = cv2.bitwise_or(
                mask,
                cv2.inRange(
                    hsv,
                    np.array([hue_min, 70, 60], dtype=np.uint8),
                    np.array([hue_max, 255, 255], dtype=np.uint8),
                ),
            )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < minimum_area_px:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            center = (
                moments["m10"] / moments["m00"],
                moments["m01"] / moments["m00"],
            )
            raw_world_xy = pixel_to_world(calibration_matrix, center)
            if raw_world_xy[0] > pick_x_max_mm:
                continue
            world_xy = _snap_world_xy_to_robot_grid(raw_world_xy)
            perimeter = float(cv2.arcLength(contour, True))
            circularity = (
                4.0 * pi * area / (perimeter * perimeter)
                if perimeter > 0
                else 0.0
            )
            output.append(
                VisualObject(
                    center_px=center,
                    world_xy_mm=world_xy,
                    color=color_name,
                    shape="cylinder" if circularity >= 0.86 else "block",
                    confidence=min(1.0, area / 600.0),
                )
            )
    return sorted(output, key=lambda item: item.world_xy_mm)


def _canonical_digit_topology(image):
    _, foreground_mask = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    canvas = np.zeros((96, 64), dtype=np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground_mask,
        connectivity=8,
    )
    if component_count <= 1:
        return canvas
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    minimum_area = max(2.0, float(component_areas.max()) * 0.02)
    filtered_mask = np.zeros_like(foreground_mask)
    for component_id, area in enumerate(component_areas, start=1):
        if float(area) >= minimum_area:
            filtered_mask[labels == component_id] = 255

    points = cv2.findNonZero(filtered_mask)
    if points is None:
        return canvas
    x, y, width, height = cv2.boundingRect(points)
    foreground = filtered_mask[y : y + height, x : x + width]
    target_width = 40
    target_height = 72
    resized = cv2.resize(
        foreground,
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )
    left = (64 - target_width) // 2
    top = (96 - target_height) // 2
    canvas[top : top + target_height, left : left + target_width] = resized
    return canvas


def detect_digit_objects(
    image_bgr,
    *,
    calibration_matrix,
    pick_x_max_mm,
    reference_dir,
):
    image_bgr = _bgr_image(image_bgr)
    calibration_matrix = _affine_matrix(calibration_matrix)
    pick_x_max_mm = _finite_float(pick_x_max_mm, "pick_x_max_mm")
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
    if any(
        image.dtype != np.uint8 or image.shape != (96, 64)
        for image in references.values()
    ):
        raise RuntimeError("数字参考图尺寸必须为 64×96")
    canonical_references = {
        digit: _canonical_digit_topology(image)
        for digit, image in references.items()
    }
    output = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width * height < 300:
            continue
        center = (x + width / 2.0, y + height / 2.0)
        raw_world_xy = pixel_to_world(calibration_matrix, center)
        if raw_world_xy[0] > pick_x_max_mm:
            continue
        world_xy = _snap_world_xy_to_robot_grid(raw_world_xy)
        crop = gray[y : y + height, x : x + width]
        normalized = _canonical_digit_topology(crop)
        scores = {}
        for digit, reference in canonical_references.items():
            orientation_scores = []
            for orientation in (normalized, cv2.flip(normalized, 1)):
                try:
                    score = float(
                        cv2.matchTemplate(
                            orientation,
                            reference,
                            cv2.TM_CCOEFF_NORMED,
                        )[0, 0]
                    )
                except (cv2.error, TypeError, ValueError, IndexError) as error:
                    raise RuntimeError("数字模板匹配失败") from error
                if not isfinite(score):
                    raise RuntimeError("数字模板匹配置信度必须是有限数值")
                orientation_scores.append(score)
            scores[digit] = max(orientation_scores)
        digit, confidence = max(scores.items(), key=lambda item: item[1])
        if confidence < MIN_VISUAL_CONFIDENCE:
            raise RuntimeError(
                "数字模板匹配置信度低于 "
                f"{MIN_VISUAL_CONFIDENCE:.1f}，停止运动"
            )
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
    use_command_xy=False,
):
    x_mm, y_mm = _finite_vector(pick_xy, 2, "pick_xy")
    drop_x, drop_y, drop_z = _finite_vector(drop_xyz, 3, "drop_xyz")
    pick_z_mm = _finite_float(pick_z_mm, "pick_z_mm")
    safe_z_mm = _finite_float(safe_z_mm, "safe_z_mm")
    hover_z_mm = _finite_float(safe_z_mm + 1.0, "hover_z_mm")
    speed = _finite_float(speed, "speed")
    if speed <= 0:
        raise ValueError("speed 必须大于 0")
    if safe_z_mm <= max(pick_z_mm, drop_z):
        raise ValueError("safe_z_mm 必须高于抓取和放置高度")

    try:
        current_x, current_y, current_z = _finite_vector(
            ctx.robot.pose(),
            3,
            "robot.pose",
        )
        if current_z < hover_z_mm:
            ctx.robot.move_world(
                current_x,
                current_y,
                hover_z_mm,
                speed=speed,
            )
        ctx.robot.move_world(x_mm, y_mm, hover_z_mm, speed=speed)
        if use_command_xy:
            pick_hover_x, pick_hover_y = x_mm, y_mm
        else:
            pick_hover_x, pick_hover_y, _ = _finite_vector(
                ctx.robot.pose(),
                3,
                "robot.pose",
            )
        ctx.robot.move_world(
            pick_hover_x,
            pick_hover_y,
            pick_z_mm,
            speed=8.0,
        )
        ctx.tool.on()
        if use_command_xy:
            pick_low_x, pick_low_y = x_mm, y_mm
        else:
            pick_low_x, pick_low_y, _ = _finite_vector(
                ctx.robot.pose(),
                3,
                "robot.pose",
            )
        ctx.robot.move_world(
            pick_low_x,
            pick_low_y,
            hover_z_mm,
            speed=speed,
        )
        ctx.robot.move_world(drop_x, drop_y, hover_z_mm, speed=speed)
        if use_command_xy:
            drop_hover_x, drop_hover_y = drop_x, drop_y
        else:
            drop_hover_x, drop_hover_y, _ = _finite_vector(
                ctx.robot.pose(),
                3,
                "robot.pose",
            )
        ctx.robot.move_world(
            drop_hover_x,
            drop_hover_y,
            drop_z,
            speed=8.0,
        )
        ctx.tool.off()
        if use_command_xy:
            drop_low_x, drop_low_y = drop_x, drop_y
        else:
            drop_low_x, drop_low_y, _ = _finite_vector(
                ctx.robot.pose(),
                3,
                "robot.pose",
            )
        ctx.robot.move_world(
            drop_low_x,
            drop_low_y,
            hover_z_mm,
            speed=speed,
        )
    except BaseException as error:
        try:
            ctx.tool.off()
        except BaseException as cleanup_error:
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note(
                    "吸盘安全释放失败："
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        raise
