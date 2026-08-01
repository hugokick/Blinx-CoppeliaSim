from pathlib import Path

import pytest

from tests.test_vision2d.synthetic_factory import (
    SyntheticObject,
    add_gaussian_noise,
    adjust_brightness,
    make_scene,
)
from vision_platform.recognition.color_shape import ColorShapeRecognizer
from vision_platform.vision2d import analyze_image


def _axis_error(actual: float, expected: float) -> float:
    direct = abs(actual - expected) % 180.0
    return min(direct, 180.0 - direct)


def test_clean_rotated_rectangle_meets_precision_budget():
    image, _ = make_scene(
        (400, 300),
        (SyntheticObject("rectangle", "red", (180, 140), (100, 50), 30.0),),
    )

    target = analyze_image(image).result.targets[0]

    assert target.center_px == pytest.approx((180.0, 140.0), abs=1.5)
    assert target.long_side_px == pytest.approx(100.0, abs=2.0)
    assert target.short_side_px == pytest.approx(50.0, abs=2.0)
    assert _axis_error(target.angle_deg, 30.0) <= 2.0
    assert target.area_px2 == pytest.approx(5000.0, rel=0.05)
    assert target.perimeter_px == pytest.approx(300.0, rel=0.05)


def test_light_noise_preserves_target_count_and_labels():
    image, _ = make_scene(
        (360, 260),
        (
            SyntheticObject("rectangle", "yellow", (90, 80), (70, 35), 15.0),
            SyntheticObject("circle", "blue", (250, 180), (50, 50), 0.0),
        ),
    )
    noisy = add_gaussian_noise(image, sigma=4.0, seed=20260731)
    noisy = adjust_brightness(noisy, factor=0.75)

    result = analyze_image(noisy).result

    assert result.status == "PASS"
    assert [(item.color, item.shape) for item in result.targets] == [
        ("yellow", "rectangle"),
        ("blue", "circle"),
    ]


def test_new_and_existing_recognizers_share_core_label_semantics():
    image, _ = make_scene(
        (360, 260),
        (
            SyntheticObject("square", "green", (90, 80), (50, 50), 0.0),
            SyntheticObject("circle", "blue", (250, 180), (50, 50), 0.0),
        ),
    )

    legacy = ColorShapeRecognizer().detect(image)
    modern = analyze_image(image).result.targets

    assert [(item.detection_id, item.color, item.shape) for item in modern] == [
        (item.detection_id, item.color, item.shape) for item in legacy
    ]


def test_runtime_package_does_not_import_legacy_private_implementation():
    package = Path(__file__).resolve().parents[2] / "vision_platform" / "vision2d"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in package.glob("*.py")
    )

    assert "ColorShapeRecognizer" not in source
    assert "vision_platform.recognition.color_shape" not in source
