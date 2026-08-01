import cv2
import numpy as np

from tests.test_vision2d.synthetic_factory import SyntheticObject, make_scene
from vision_platform.vision2d.models import Vision2DConfig
from vision_platform.vision2d.preprocessing import preprocess_image
from vision_platform.vision2d.segmentation import segment_contours


def test_segmentation_returns_two_independent_contours():
    image, _ = make_scene(
        (320, 240),
        (
            SyntheticObject("rectangle", "red", (80, 80), (60, 30), 0.0),
            SyntheticObject("circle", "blue", (240, 160), (50, 50), 0.0),
        ),
    )
    processed = preprocess_image(image, Vision2DConfig())

    accepted, rejected = segment_contours(
        processed["cleaned_mask"], Vision2DConfig()
    )

    assert len(accepted) == 2
    assert rejected == ()


def test_segmentation_rejects_small_and_border_touching_candidates():
    mask = np.zeros((200, 300), dtype=np.uint8)
    cv2.rectangle(mask, (0, 30), (40, 90), 255, -1)
    cv2.circle(mask, (150, 100), 3, 255, -1)
    config = Vision2DConfig(min_area_ratio=0.001, border_margin_px=1)

    accepted, rejected = segment_contours(mask, config)

    assert accepted == ()
    assert {item.code for item in rejected} == {
        "AREA_TOO_SMALL",
        "TOUCHES_IMAGE_BORDER",
    }


def test_segmentation_rejects_degenerate_line_contour():
    mask = np.zeros((200, 300), dtype=np.uint8)
    cv2.line(mask, (150, 30), (150, 170), 255, 1)

    accepted, rejected = segment_contours(
        mask,
        Vision2DConfig(min_area_ratio=0.000001),
    )

    assert accepted == ()
    assert [item.code for item in rejected] == [
        "INSUFFICIENT_CONTOUR_POINTS"
    ]
