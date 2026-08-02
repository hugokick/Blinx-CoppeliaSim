from __future__ import annotations

import math
import time

import cv2
import numpy as np
import pytest

from vision_platform.vision2d.code_recognition import recognize_codes

from .synthetic_v107_v109 import compose_side_by_side, make_ean13, make_qr, make_rs1d


def _all_finite(reading) -> bool:
    values = [*reading.center_px, *reading.bbox_px, reading.confidence]
    values.extend(value for point in reading.polygon_px for value in point)
    return all(math.isfinite(float(value)) for value in values)


def test_qr_round_trip_returns_payload_and_location() -> None:
    image = make_qr("RS|V1-07|QR-A", scale=4)

    started = time.perf_counter()
    result = recognize_codes(image)
    elapsed = time.perf_counter() - started

    assert result.status == "PASS"
    reading = next(item for item in result.readings if item.code_type == "qr")
    assert reading.data == "RS|V1-07|QR-A"
    assert reading.decoded is True
    assert len(reading.polygon_px) == 4
    assert _all_finite(reading)
    assert 0.0 <= reading.confidence <= 1.0
    assert abs(reading.center_px[0] - image.shape[1] / 2) < image.shape[1] * 0.12
    assert abs(reading.center_px[1] - image.shape[0] / 2) < image.shape[0] * 0.12
    assert math.isfinite(result.processing_ms)
    assert elapsed < 1.0


def test_qr_handles_rotation_scale_brightness_and_noise() -> None:
    image = make_qr(
        "RS|V1-07|ROBUST",
        scale=3,
        rotate_deg=23,
        noise_sigma=3.0,
        brightness=-12,
    )

    result = recognize_codes(image)

    assert result.status == "PASS"
    assert any(item.data == "RS|V1-07|ROBUST" for item in result.readings)


def test_multiple_qr_codes_are_returned_in_detector_order() -> None:
    left = make_qr("LEFT", scale=3)
    right = make_qr("RIGHT", scale=3)
    image = compose_side_by_side(left, right, gap=48)

    result = recognize_codes(image, max_codes=4)

    payloads = [item.data for item in result.readings if item.code_type == "qr"]
    assert result.status == "PASS"
    assert set(payloads) == {"LEFT", "RIGHT"}
    assert len(payloads) == len(set(payloads))


def test_damaged_qr_is_not_silently_reported_as_success() -> None:
    image = make_qr("DAMAGED", scale=4)
    height, width = image.shape[:2]
    cv2.rectangle(image, (width // 3, height // 3), (width // 2, height // 2), (255, 255, 255), -1)

    result = recognize_codes(image)

    assert result.status in {"PARTIAL", "NO_TARGETS"}
    assert not any(item.data == "DAMAGED" and item.decoded for item in result.readings)
    assert result.failure_code is not None or any(item.failure_code for item in result.readings)


def test_rs1d_round_trip_returns_original_payload_and_location() -> None:
    image = make_rs1d("RS1D-042", module=3)

    result = recognize_codes(image)

    assert result.status == "PASS"
    reading = next(item for item in result.readings if item.code_type == "rs1d")
    assert reading.data == "RS1D-042"
    assert reading.decoded is True
    assert len(reading.polygon_px) == 4
    assert _all_finite(reading)


def test_standard_ean13_round_trip_returns_standard_payload() -> None:
    image = make_ean13("590123412345")

    result = recognize_codes(image)

    assert result.status == "PASS"
    reading = next(item for item in result.readings if item.code_type == "ean13")
    assert reading.data == "5901234123457"
    assert reading.decoded is True
    assert len(reading.polygon_px) == 4
    assert _all_finite(reading)


def test_standard_ean13_handles_rotation_scale_and_noise() -> None:
    image = make_ean13(
        "400638133393",
        rotate_deg=11,
        scale=1.15,
        noise_sigma=2.0,
        brightness=-8,
    )

    result = recognize_codes(image)

    assert result.status == "PASS"
    assert any(item.code_type == "ean13" and item.data == "4006381333931" for item in result.readings)


def test_standard_ean13_invalid_checksum_is_not_decoded() -> None:
    image = make_ean13("590123412345")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[gray.shape[0] // 2 :, gray.shape[1] // 2 : gray.shape[1] // 2 + 3] = 255
    result = recognize_codes(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    assert not any(item.code_type == "ean13" and item.decoded for item in result.readings)
    assert result.status in {"NO_TARGETS", "PARTIAL"}


def test_rs1d_handles_rotation_scale_and_noise() -> None:
    image = make_rs1d(
        "BATCH-7",
        module=3,
        rotate_deg=12,
        scale=1.15,
        noise_sigma=2.0,
        brightness=8,
    )

    result = recognize_codes(image)

    assert result.status == "PASS"
    assert any(item.code_type == "rs1d" and item.data == "BATCH-7" for item in result.readings)


def test_rs1d_checksum_damage_is_rejected() -> None:
    damaged = make_rs1d("CHECKSUM", module=3, damaged=True)

    result = recognize_codes(damaged)

    assert result.status in {"PARTIAL", "NO_TARGETS"}
    assert not any(item.data == "CHECKSUM" and item.decoded for item in result.readings)


def test_multiple_rs1d_codes_are_deduplicated_and_bounded() -> None:
    image = compose_side_by_side(make_rs1d("A1"), make_rs1d("B2"), gap=40)

    result = recognize_codes(image, max_codes=1)

    decoded = [item for item in result.readings if item.decoded]
    assert len(decoded) <= 1
    assert len({(item.code_type, item.data, item.center_px) for item in result.readings}) == len(
        result.readings
    )


def test_blank_and_invalid_inputs_have_explicit_status() -> None:
    blank = np.full((240, 320, 3), 255, dtype=np.uint8)
    empty = recognize_codes(blank)
    invalid = recognize_codes(np.zeros((20, 20), dtype=np.uint8))

    assert empty.status == "NO_TARGETS"
    assert empty.failure_code in {"NO_CODES", "DECODE_FAILED"}
    assert invalid.status == "REJECTED"
    assert invalid.failure_code == "INPUT_INVALID"


def test_unsupported_commercial_barcode_boundary_is_explicit() -> None:
    # This is a Code-128-like stripe texture, not the project's RS1D format.
    image = np.full((100, 360, 3), 255, dtype=np.uint8)
    x = 30
    for width in (2, 1, 3, 1, 2, 2, 1, 3, 2, 1, 2, 3, 1, 2, 2):
        image[20:80, x : x + width] = 0
        x += width + 2

    result = recognize_codes(image)

    assert not any(item.code_type == "rs1d" and item.decoded for item in result.readings)
    assert result.status in {"NO_TARGETS", "PARTIAL"}
