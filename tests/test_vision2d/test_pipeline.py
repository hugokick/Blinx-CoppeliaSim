import numpy as np

from tests.test_vision2d.synthetic_factory import SyntheticObject, make_scene
from vision_platform.vision2d import Vision2DConfig, analyze_image


def test_pipeline_returns_spatially_sorted_multi_object_results():
    image, _ = make_scene(
        (360, 260),
        (
            SyntheticObject("circle", "blue", (260, 190), (50, 50), 0.0),
            SyntheticObject("rectangle", "red", (90, 70), (70, 35), 30.0),
        ),
    )
    before = image.copy()

    analysis = analyze_image(image, Vision2DConfig())

    assert analysis.result.status == "PASS"
    assert [item.detection_id for item in analysis.result.targets] == [
        "det-001",
        "det-002",
    ]
    assert [(item.color, item.shape) for item in analysis.result.targets] == [
        ("red", "rectangle"),
        ("blue", "circle"),
    ]
    assert analysis.result.targets[1].angle_deg is None
    assert "ANGLE_UNDEFINED_FOR_CIRCLE" in analysis.result.targets[1].quality_flags
    assert set(analysis.intermediate_images) == {
        "gray",
        "hsv",
        "foreground_mask",
        "cleaned_mask",
        "annotated",
    }
    assert np.array_equal(image, before)


def test_pipeline_distinguishes_no_targets_and_rejected_candidates():
    empty = np.zeros((200, 300, 3), dtype=np.uint8)
    assert analyze_image(empty).result.status == "NO_TARGETS"

    border, _ = make_scene(
        (300, 200),
        (SyntheticObject("rectangle", "red", (20, 70), (40, 50), 0.0),),
    )
    assert analyze_image(border).result.status == "REJECTED"


def test_pipeline_reports_partial_for_valid_and_rejected_candidates():
    image, _ = make_scene(
        (300, 200),
        (
            SyntheticObject("rectangle", "red", (170, 100), (70, 40), 0.0),
            SyntheticObject("circle", "blue", (60, 60), (8, 8), 0.0),
        ),
    )

    result = analyze_image(
        image,
        Vision2DConfig(
            min_area_ratio=0.002,
            gaussian_kernel_size=1,
            morphology_kernel_size=1,
        ),
    ).result

    assert result.status == "PARTIAL"
    assert len(result.targets) == 1
    assert {item.code for item in result.rejected_targets} == {"AREA_TOO_SMALL"}
