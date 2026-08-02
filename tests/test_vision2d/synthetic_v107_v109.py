"""Original, deterministic in-memory fixtures for V1-07 through V1-09."""

from __future__ import annotations

import cv2
import numpy as np


def _transform_gray(
    image: np.ndarray,
    *,
    rotate_deg: float = 0.0,
    scale: float = 1.0,
    noise_sigma: float = 0.0,
    brightness: float = 0.0,
    seed: int = 20260802,
) -> np.ndarray:
    """Apply bounded deterministic image perturbations without file I/O."""
    output = image
    if scale != 1.0:
        interpolation = cv2.INTER_NEAREST if scale >= 1.0 else cv2.INTER_AREA
        output = cv2.resize(output, None, fx=scale, fy=scale, interpolation=interpolation)
    if rotate_deg:
        height, width = output.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), rotate_deg, 1.0)
        output = cv2.warpAffine(
            output,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
    if brightness:
        output = np.clip(output.astype(np.float32) + brightness, 0.0, 255.0).astype(
            np.uint8
        )
    if noise_sigma:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, noise_sigma, output.shape)
        output = np.clip(output.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
    return output


def make_qr(
    payload: str,
    *,
    scale: int = 4,
    rotate_deg: float = 0.0,
    noise_sigma: float = 0.0,
    brightness: float = 0.0,
) -> np.ndarray:
    """Create an OpenCV-encoded QR image with a deterministic white quiet zone."""
    encoded = cv2.QRCodeEncoder_create().encode(payload)
    scaled = cv2.resize(encoded, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    quiet = max(12, 4 * scale)
    canvas = cv2.copyMakeBorder(
        scaled,
        quiet,
        quiet,
        quiet,
        quiet,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    transformed = _transform_gray(
        canvas,
        rotate_deg=rotate_deg,
        noise_sigma=noise_sigma,
        brightness=brightness,
        seed=11,
    )
    return cv2.cvtColor(transformed, cv2.COLOR_GRAY2BGR)


def _rs1d_bits(payload: str) -> list[int]:
    raw = payload.encode("utf-8")
    if len(raw) > 64:
        raise ValueError("fixture payload is limited to 64 bytes")
    frame = bytes([0xA7, len(raw)]) + raw + bytes([sum(raw) & 0xFF])
    bits: list[int] = []
    for value in frame:
        bits.extend((value >> shift) & 1 for shift in range(7, -1, -1))
    return bits


def make_rs1d(
    payload: str,
    *,
    module: int = 3,
    bar_height: int = 72,
    rotate_deg: float = 0.0,
    scale: float = 1.0,
    noise_sigma: float = 0.0,
    brightness: float = 0.0,
    damaged: bool = False,
) -> np.ndarray:
    """Create the project's original RS1D teaching barcode.

    A bit is a black bar of one module for 0 or two modules for 1, followed by
    one white module. Four-module black guards delimit the frame.
    """
    if module < 2 or bar_height < 20:
        raise ValueError("module and bar_height are too small")
    bits = _rs1d_bits(payload)
    run_widths = [4 * module] + [module * (1 + bit) for bit in bits] + [4 * module]
    quiet = 12 * module
    width = 2 * quiet + sum(run_widths) + module * len(run_widths)
    height = bar_height + 2 * quiet
    image = np.full((height, width), 255, dtype=np.uint8)
    x = quiet
    for index, bar_width in enumerate(run_widths):
        image[quiet : quiet + bar_height, x : x + bar_width] = 0
        if damaged and index == len(run_widths) // 2:
            image[quiet : quiet + bar_height, x + bar_width // 3 : x + (2 * bar_width) // 3] = 255
        x += bar_width + module
    transformed = _transform_gray(
        image,
        rotate_deg=rotate_deg,
        scale=scale,
        noise_sigma=noise_sigma,
        brightness=brightness,
        seed=17,
    )
    return cv2.cvtColor(transformed, cv2.COLOR_GRAY2BGR)


def compose_side_by_side(*images: np.ndarray, gap: int = 36) -> np.ndarray:
    """Place same-height BGR fixtures on a white canvas."""
    if not images:
        raise ValueError("at least one image is required")
    height = max(image.shape[0] for image in images)
    width = sum(image.shape[1] for image in images) + gap * (len(images) - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    x = 0
    for image in images:
        y = (height - image.shape[0]) // 2
        canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
        x += image.shape[1] + gap
    return canvas
