"""Deterministic QR, EAN-13 and project-original RS1D code recognition.

The RS1D codec is intentionally a small teaching/test symbology.  It is not a
commercial barcode decoder; the standard EAN-13 path is implemented separately.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


_RS_MAGIC = 0xA7
_RS_MAX_PAYLOAD = 64
_RS_MAX_CODES = 16
_RS_SCAN_ANGLES = tuple(range(-30, 31, 3)) + (-90, 90)
_EAN_SCAN_ANGLES = tuple(range(-30, 31, 3)) + (-90, 90)

_EAN_L_PATTERNS = (
    "0001101", "0011001", "0010011", "0111101", "0100011",
    "0110001", "0101111", "0111011", "0110111", "0001011",
)
_EAN_G_PATTERNS = (
    "0100111", "0110011", "0011011", "0100001", "0011101",
    "0111001", "0000101", "0010001", "0001001", "0010111",
)
_EAN_R_PATTERNS = (
    "1110010", "1100110", "1101100", "1000010", "1011100",
    "1001110", "1010000", "1000100", "1001000", "1110100",
)
_EAN_PARITY = (
    "AAAAAA", "AABABB", "AABBAB", "AABBBA", "ABAABB",
    "ABBAAB", "ABBBAA", "ABABAB", "ABABBA", "ABBABA",
)


@dataclass(frozen=True)
class CodeReading:
    """One decoded or located code in original image coordinates."""

    code_type: str
    data: str | None
    polygon_px: tuple[tuple[float, float], ...]
    bbox_px: tuple[int, int, int, int]
    center_px: tuple[float, float]
    confidence: float
    decoded: bool
    failure_code: str | None = None

    def __post_init__(self) -> None:
        if self.code_type not in {"qr", "ean13", "rs1d"}:
            raise ValueError("unsupported code_type")
        if len(self.polygon_px) < 4:
            raise ValueError("polygon_px must contain at least four points")
        if len(self.bbox_px) != 4 or self.bbox_px[2] < 0 or self.bbox_px[3] < 0:
            raise ValueError("bbox_px is invalid")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite in [0, 1]")
        if self.decoded and not self.data:
            raise ValueError("decoded readings require data")


@dataclass(frozen=True)
class CodeRecognitionResult:
    """Stable result contract for QR/RS1D recognition."""

    status: str
    readings: tuple[CodeReading, ...]
    image_size: tuple[int, int]
    failure_code: str | None
    detector_order: tuple[str, ...]
    processing_ms: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "PARTIAL", "NO_TARGETS", "REJECTED"}:
            raise ValueError("unsupported code recognition status")
        if len(self.image_size) != 2 or any(int(value) <= 0 for value in self.image_size):
            raise ValueError("image_size must be positive width/height")
        elapsed = float(self.processing_ms)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("processing_ms must be finite and non-negative")


def _invalid_result(code: str, image_size: tuple[int, int] = (1, 1)) -> CodeRecognitionResult:
    return CodeRecognitionResult(
        status="REJECTED",
        readings=(),
        image_size=image_size,
        failure_code=code,
        detector_order=(),
        processing_ms=0.0,
    )


def _validate_bgr(image_bgr: object) -> tuple[np.ndarray | None, tuple[int, int]]:
    if not isinstance(image_bgr, np.ndarray):
        return None, (1, 1)
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        return None, (1, 1)
    if image_bgr.shape[0] <= 0 or image_bgr.shape[1] <= 0:
        return None, (1, 1)
    return image_bgr, (int(image_bgr.shape[1]), int(image_bgr.shape[0]))


def _points_to_reading(
    code_type: str,
    data: str | None,
    points: np.ndarray,
    *,
    confidence: float,
    failure_code: str | None,
) -> CodeReading:
    normalized = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if normalized.shape[0] < 4:
        raise ValueError("detector returned fewer than four points")
    polygon = tuple((float(x), float(y)) for x, y in normalized[:4])
    min_x = max(0, int(math.floor(float(np.min(normalized[:, 0])))))
    min_y = max(0, int(math.floor(float(np.min(normalized[:, 1])))))
    max_x = int(math.ceil(float(np.max(normalized[:, 0]))))
    max_y = int(math.ceil(float(np.max(normalized[:, 1]))))
    center = (
        float(np.mean(normalized[:, 0])),
        float(np.mean(normalized[:, 1])),
    )
    return CodeReading(
        code_type=code_type,
        data=data,
        polygon_px=polygon,
        bbox_px=(min_x, min_y, max(0, max_x - min_x), max(0, max_y - min_y)),
        center_px=center,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        decoded=bool(data),
        failure_code=failure_code,
    )


def _detect_qr(gray: np.ndarray) -> list[CodeReading]:
    detector = cv2.QRCodeDetector()
    readings: list[CodeReading] = []
    try:
        outcome = detector.detectAndDecodeMulti(gray)
    except cv2.error:
        outcome = (False, (), None, ())
    if isinstance(outcome, tuple) and len(outcome) >= 3:
        ok, decoded_info, points = outcome[:3]
        if points is not None and len(points):
            texts = tuple(str(item) for item in (decoded_info or ()))
            for index, raw_points in enumerate(np.asarray(points)):
                text = texts[index] if index < len(texts) and texts[index] else None
                readings.append(
                    _points_to_reading(
                        "qr",
                        text,
                        raw_points,
                        confidence=0.98 if text else 0.30,
                        failure_code=None if text else "QR_DECODE_FAILED",
                    )
                )
            if readings:
                return readings

    try:
        text, points, _straight = detector.detectAndDecode(gray)
    except cv2.error:
        return []
    if points is None:
        return []
    try:
        return [
            _points_to_reading(
                "qr",
                str(text) if text else None,
                points,
                confidence=0.98 if text else 0.30,
                failure_code=None if text else "QR_DECODE_FAILED",
            )
        ]
    except (TypeError, ValueError):
        return []


def _bits_from_bytes(payload: bytes) -> list[int]:
    frame = bytes([_RS_MAGIC, len(payload)]) + payload + bytes([sum(payload) & 0xFF])
    bits: list[int] = []
    for value in frame:
        bits.extend((value >> shift) & 1 for shift in range(7, -1, -1))
    return bits


def encode_rs1d_payload(
    payload: str,
    *,
    module_px: int = 3,
    bar_height_px: int = 72,
    quiet_modules: int = 12,
) -> np.ndarray:
    """Encode the project-original RS1D test symbology as a BGR image."""
    if not isinstance(payload, str):
        raise TypeError("payload must be str")
    raw = payload.encode("utf-8")
    if len(raw) > _RS_MAX_PAYLOAD:
        raise ValueError("payload exceeds RS1D maximum length")
    if int(module_px) != module_px or module_px < 2:
        raise ValueError("module_px must be an integer >= 2")
    if int(bar_height_px) != bar_height_px or bar_height_px < 20:
        raise ValueError("bar_height_px must be an integer >= 20")
    if int(quiet_modules) != quiet_modules or quiet_modules < 4:
        raise ValueError("quiet_modules must be an integer >= 4")
    module_px = int(module_px)
    bar_height_px = int(bar_height_px)
    quiet_px = int(quiet_modules) * module_px
    bits = _bits_from_bytes(raw)
    widths = [4 * module_px] + [module_px * (1 + bit) for bit in bits] + [4 * module_px]
    width = 2 * quiet_px + sum(widths) + module_px * len(widths)
    height = bar_height_px + 2 * quiet_px
    image = np.full((height, width), 255, dtype=np.uint8)
    x = quiet_px
    for bar_width in widths:
        image[quiet_px : quiet_px + bar_height_px, x : x + bar_width] = 0
        x += bar_width + module_px
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _ean13_checksum(first_twelve: str) -> str:
    total = sum((3 if index % 2 else 1) * int(value) for index, value in enumerate(first_twelve))
    return str((-total) % 10)


def encode_ean13_payload(
    digits: str,
    *,
    module_px: int = 3,
    bar_height_px: int = 72,
    quiet_modules: int = 10,
) -> np.ndarray:
    """Encode a standards-shaped EAN-13 symbol without external dependencies."""
    if not isinstance(digits, str) or not digits.isdigit() or len(digits) not in {12, 13}:
        raise ValueError("EAN-13 requires 12 digits or 13 digits including checksum")
    if len(digits) == 12:
        digits += _ean13_checksum(digits)
    if digits[-1] != _ean13_checksum(digits[:12]):
        raise ValueError("EAN-13 checksum is invalid")
    if isinstance(module_px, bool) or not isinstance(module_px, (int, np.integer)) or module_px < 2:
        raise ValueError("module_px must be an integer >= 2")
    if isinstance(bar_height_px, bool) or not isinstance(bar_height_px, (int, np.integer)) or bar_height_px < 20:
        raise ValueError("bar_height_px must be an integer >= 20")
    if isinstance(quiet_modules, bool) or not isinstance(quiet_modules, (int, np.integer)) or quiet_modules < 9:
        raise ValueError("quiet_modules must be an integer >= 9")
    module_px = int(module_px)
    bar_height_px = int(bar_height_px)
    quiet_px = int(quiet_modules) * module_px
    bits = "101"
    parity = _EAN_PARITY[int(digits[0])]
    for digit, side in zip(digits[1:7], parity):
        bits += (_EAN_L_PATTERNS if side == "A" else _EAN_G_PATTERNS)[int(digit)]
    bits += "01010"
    for digit in digits[7:]:
        bits += _EAN_R_PATTERNS[int(digit)]
    bits += "101"
    image = np.full(
        (bar_height_px + 2 * quiet_px, len(bits) * module_px + 2 * quiet_px),
        255,
        dtype=np.uint8,
    )
    for index, bit in enumerate(bits):
        if bit == "1":
            left = quiet_px + index * module_px
            image[quiet_px : quiet_px + bar_height_px, left : left + module_px] = 0
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _dark_runs(row: np.ndarray) -> list[tuple[int, int]]:
    dark = np.asarray(row < 180, dtype=np.uint8)
    if not np.any(dark):
        return []
    padded = np.pad(dark, (1, 1), mode="constant")
    starts = np.flatnonzero((padded[1:-1] == 1) & (padded[:-2] == 0))
    ends = np.flatnonzero((padded[1:-1] == 1) & (padded[2:] == 0))
    return [(int(start), int(end - start + 1)) for start, end in zip(starts, ends)]


def _decode_run_sequence(
    runs: list[tuple[int, int]],
) -> tuple[str, int, int, float] | None:
    if len(runs) < 26:
        return None
    widths = np.asarray([width for _start, width in runs], dtype=np.float64)
    # Ignore isolated noise pixels while estimating the smallest module.
    valid_widths = widths[widths >= 2.0]
    if valid_widths.size < 8:
        return None
    module = float(np.percentile(valid_widths, 12.5))
    if module < 1.25:
        return None
    for start_index, start_width in enumerate(widths):
        if start_width < 3.0 * module:
            continue
        for stop_index in range(start_index + 9, min(len(runs), start_index + 8 * (3 + _RS_MAX_PAYLOAD) + 2)):
            stop_width = widths[stop_index]
            if stop_width < 3.0 * module:
                continue
            bit_widths = widths[start_index + 1 : stop_index]
            if len(bit_widths) < 24 or len(bit_widths) % 8:
                continue
            ratios = bit_widths / module
            if np.any(ratios < 0.55) or np.any(ratios > 2.9):
                continue
            bits = [1 if ratio >= 1.45 else 0 for ratio in ratios]
            values = []
            for offset in range(0, len(bits), 8):
                value = 0
                for bit in bits[offset : offset + 8]:
                    value = (value << 1) | bit
                values.append(value)
            if len(values) < 3 or values[0] != _RS_MAGIC:
                continue
            payload_length = values[1]
            expected_values = 3 + payload_length
            if payload_length > _RS_MAX_PAYLOAD or len(values) != expected_values:
                continue
            payload_bytes = bytes(values[2 : 2 + payload_length])
            if values[-1] != sum(payload_bytes) & 0xFF:
                continue
            try:
                payload = payload_bytes.decode("utf-8")
            except UnicodeDecodeError:
                continue
            quality = float(
                np.mean(np.minimum(np.abs(ratios - 1.0), np.abs(ratios - 2.0)) < 0.45)
            )
            return payload, int(runs[start_index][0]), int(runs[stop_index][0] + runs[stop_index][1]), quality
    return None


def _decode_ean13_row(
    row: np.ndarray,
    runs: list[tuple[int, int]],
) -> tuple[str, int, int, float] | None:
    """Decode one horizontal row using the fixed 95-module EAN-13 grammar."""
    if len(runs) < 8:
        return None
    widths = np.asarray([width for _start, width in runs], dtype=np.float64)
    module_candidates = widths[widths >= 2.0]
    if module_candidates.size == 0:
        return None
    module_estimate = float(np.percentile(module_candidates, 20))
    if module_estimate < 1.5:
        return None
    # The smallest bars are one module; a narrow bounded band handles
    # fractional scaling after a resize without inventing a new symbology.
    scales = tuple(module_estimate * factor for factor in (0.78, 0.86, 0.94, 1.0, 1.08, 1.16, 1.26, 1.38, 1.5))
    for run_index, (start_x, start_width) in enumerate(runs):
        if start_width < 0.55 * module_estimate or start_width > 1.8 * module_estimate:
            continue
        for module in scales:
            end_x = float(start_x) + 95.0 * module
            if end_x > row.shape[0] - 1:
                continue
            bits = "".join(
                "1" if row[min(row.shape[0] - 1, max(0, int(round(start_x + (index + 0.5) * module))))] < 180 else "0"
                for index in range(95)
            )
            if bits[:3] != "101" or bits[45:50] != "01010" or bits[92:] != "101":
                continue
            parity = []
            left_digits: list[str] = []
            valid = True
            for offset in range(3, 45, 7):
                pattern = bits[offset : offset + 7]
                if pattern in _EAN_L_PATTERNS:
                    parity.append("A")
                    left_digits.append(str(_EAN_L_PATTERNS.index(pattern)))
                elif pattern in _EAN_G_PATTERNS:
                    parity.append("B")
                    left_digits.append(str(_EAN_G_PATTERNS.index(pattern)))
                else:
                    valid = False
                    break
            if not valid or len(parity) != 6 or "".join(parity) not in _EAN_PARITY:
                continue
            right_digits: list[str] = []
            for offset in range(50, 92, 7):
                pattern = bits[offset : offset + 7]
                if pattern not in _EAN_R_PATTERNS:
                    valid = False
                    break
                right_digits.append(str(_EAN_R_PATTERNS.index(pattern)))
            if not valid or len(right_digits) != 6:
                continue
            digits = str(_EAN_PARITY.index("".join(parity))) + "".join(left_digits + right_digits)
            if digits[-1] != _ean13_checksum(digits[:12]):
                continue
            # A checksum-valid symbol is sufficiently specific to reject
            # arbitrary stripe textures; quality reflects module sampling.
            quality = float(
                np.clip(1.0 - abs(module - module_estimate) / max(module_estimate, 1e-6), 0.0, 1.0)
            )
            return digits, int(start_x), int(round(end_x)), quality
    return None


def _detect_ean13(gray: np.ndarray) -> list[CodeReading]:
    found: list[CodeReading] = []
    smoothed = cv2.medianBlur(gray, 3)
    for angle in _EAN_SCAN_ANGLES:
        rotated, matrix = _rotate_gray(smoothed, float(angle))
        mask = rotated < 180
        row_candidates = [
            (row_index, _dark_runs(rotated[row_index]))
            for row_index in range(0, rotated.shape[0], 2)
        ]
        row_candidates = [item for item in row_candidates if len(item[1]) >= 8]
        row_candidates.sort(key=lambda item: len(item[1]), reverse=True)
        for row_index, runs in row_candidates[:8]:
            decoded = _decode_ean13_row(rotated[row_index], runs)
            if decoded is None:
                continue
            digits, start_x, stop_x, quality = decoded
            x0 = max(0, start_x - 2)
            x1 = min(rotated.shape[1], stop_x + 2)
            active_rows = np.flatnonzero(np.any(mask[:, x0:x1], axis=1))
            if active_rows.size == 0:
                continue
            bands: list[tuple[int, int]] = []
            band_start = previous = int(active_rows[0])
            for value in active_rows[1:]:
                current = int(value)
                if current > previous + 1:
                    bands.append((band_start, previous))
                    band_start = current
                previous = current
            bands.append((band_start, previous))
            y0, y1 = min(bands, key=lambda band: abs((band[0] + band[1]) / 2 - row_index))
            polygon = _map_points(
                ((start_x, y0), (stop_x, y0), (stop_x, y1), (start_x, y1)),
                matrix,
            )
            candidate = CodeReading(
                code_type="ean13",
                data=digits,
                polygon_px=polygon,
                bbox_px=_bbox_from_polygon(polygon),
                center_px=(
                    float(np.mean([point[0] for point in polygon])),
                    float(np.mean([point[1] for point in polygon])),
                ),
                confidence=float(np.clip(0.82 + 0.16 * quality, 0.0, 1.0)),
                decoded=True,
            )
            if not any(
                item.code_type == "ean13"
                and item.data == candidate.data
                and math.dist(item.center_px, candidate.center_px) < 12.0
                for item in found
            ):
                found.append(candidate)
            if len(found) >= _RS_MAX_CODES:
                return found
    return found


def _rotate_gray(gray: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = gray.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return rotated, matrix


def _map_points(points: Iterable[tuple[float, float]], matrix: np.ndarray) -> tuple[tuple[float, float], ...]:
    inverse = cv2.invertAffineTransform(matrix)
    values = np.asarray(list(points), dtype=np.float64).reshape(-1, 1, 2)
    mapped = cv2.transform(values, inverse).reshape(-1, 2)
    return tuple((float(x), float(y)) for x, y in mapped)


def _detect_rs1d(gray: np.ndarray) -> list[CodeReading]:
    found: list[CodeReading] = []
    smoothed = cv2.medianBlur(gray, 3)
    for angle in _RS_SCAN_ANGLES:
        rotated, matrix = _rotate_gray(smoothed, float(angle))
        mask = rotated < 180
        # The middle of a bar has the densest dark runs; scanning every second
        # row keeps the search bounded while covering short bars.
        for row_index in range(0, rotated.shape[0], 2):
            row = rotated[row_index]
            runs = _dark_runs(row)
            if len(runs) < 26:
                continue
            decoded = _decode_run_sequence(runs)
            if decoded is None:
                continue
            payload, start_x, stop_x, quality = decoded
            x0 = max(0, start_x - 2)
            x1 = min(rotated.shape[1], stop_x + 2)
            column_mask = mask[:, x0:x1]
            active_rows = np.flatnonzero(np.any(column_mask, axis=1))
            if active_rows.size == 0:
                continue
            # Pick the contiguous active band nearest the scanned row.
            bands: list[tuple[int, int]] = []
            band_start = previous = int(active_rows[0])
            for value in active_rows[1:]:
                current = int(value)
                if current > previous + 1:
                    bands.append((band_start, previous))
                    band_start = current
                previous = current
            bands.append((band_start, previous))
            y0, y1 = min(bands, key=lambda band: abs((band[0] + band[1]) / 2 - row_index))
            polygon_rotated = ((start_x, y0), (stop_x, y0), (stop_x, y1), (start_x, y1))
            polygon = _map_points(polygon_rotated, matrix)
            candidate = CodeReading(
                code_type="rs1d",
                data=payload,
                polygon_px=polygon,
                bbox_px=_bbox_from_polygon(polygon),
                center_px=(
                    float(np.mean([point[0] for point in polygon])),
                    float(np.mean([point[1] for point in polygon])),
                ),
                confidence=float(np.clip(0.80 + 0.18 * quality, 0.0, 1.0)),
                decoded=True,
            )
            if not any(
                item.code_type == "rs1d"
                and item.data == candidate.data
                and math.dist(item.center_px, candidate.center_px) < 12.0
                for item in found
            ):
                found.append(candidate)
            if len(found) >= _RS_MAX_CODES:
                return found
    return found


def _bbox_from_polygon(polygon: tuple[tuple[float, float], ...]) -> tuple[int, int, int, int]:
    points = np.asarray(polygon, dtype=np.float64)
    min_x = max(0, int(math.floor(float(np.min(points[:, 0])))))
    min_y = max(0, int(math.floor(float(np.min(points[:, 1])))))
    max_x = int(math.ceil(float(np.max(points[:, 0]))))
    max_y = int(math.ceil(float(np.max(points[:, 1]))))
    return min_x, min_y, max(0, max_x - min_x), max(0, max_y - min_y)


def _deduplicate(readings: Iterable[CodeReading], max_codes: int) -> tuple[CodeReading, ...]:
    output: list[CodeReading] = []
    for reading in readings:
        duplicate = any(
            existing.code_type == reading.code_type
            and existing.data == reading.data
            and math.dist(existing.center_px, reading.center_px) < 12.0
            for existing in output
        )
        if not duplicate:
            output.append(reading)
        if len(output) >= max_codes:
            break
    return tuple(output)


def recognize_codes(image_bgr: object, *, max_codes: int = 16) -> CodeRecognitionResult:
    """Detect QR and RS1D codes in a BGR uint8 image."""
    started = time.perf_counter()
    image, image_size = _validate_bgr(image_bgr)
    if image is None:
        return _invalid_result("INPUT_INVALID", image_size)
    if isinstance(max_codes, bool) or not isinstance(max_codes, int) or not 1 <= max_codes <= _RS_MAX_CODES:
        return _invalid_result("MAX_CODES_INVALID", image_size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    readings = _deduplicate([*_detect_qr(gray), *_detect_ean13(gray), *_detect_rs1d(gray)], max_codes)
    decoded = tuple(item for item in readings if item.decoded)
    if decoded:
        status = "PASS"
        failure_code = None
    elif readings:
        status = "PARTIAL"
        failure_code = "DECODE_FAILED"
    else:
        status = "NO_TARGETS"
        failure_code = "NO_CODES"
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return CodeRecognitionResult(
        status=status,
        readings=readings,
        image_size=image_size,
        failure_code=failure_code,
        detector_order=tuple(item.code_type for item in readings),
        processing_ms=elapsed_ms,
    )


__all__ = [
    "CodeReading",
    "CodeRecognitionResult",
    "encode_ean13_payload",
    "encode_rs1d_payload",
    "recognize_codes",
]
