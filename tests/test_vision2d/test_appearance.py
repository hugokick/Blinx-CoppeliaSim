import cv2
import numpy as np

from tests.test_vision2d.synthetic_factory import SyntheticObject, make_scene
from vision_platform.vision2d.appearance import measure_appearance
from vision_platform.vision2d.models import Vision2DConfig


def _single_contour(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 60, 40]), np.array([179, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return hsv, max(contours, key=cv2.contourArea)


def test_appearance_labels_supported_colors_and_shapes():
    cases = (
        ("red", "rectangle"),
        ("yellow", "square"),
        ("green", "triangle"),
        ("blue", "circle"),
    )
    for color, shape in cases:
        size = (60, 35) if shape == "rectangle" else (50, 50)
        image, _ = make_scene(
            (180, 160),
            (SyntheticObject(shape, color, (90, 80), size, 0.0),),
        )
        hsv, contour = _single_contour(image)

        result = measure_appearance(hsv, contour, Vision2DConfig())

        assert result.color == color
        assert result.shape == shape


def test_low_saturation_target_is_unknown_color():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(image, (40, 30), (120, 90), (160, 160, 160), -1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    contour = np.array([[[40, 30]], [[120, 30]], [[120, 90]], [[40, 90]]])

    result = measure_appearance(hsv, contour, Vision2DConfig())

    assert result.color == "unknown"
    assert result.shape == "rectangle"


def test_irregular_concave_contour_is_polygon():
    image = np.zeros((140, 180, 3), dtype=np.uint8)
    points = np.array(
        [[20, 50], [80, 20], [160, 30], [105, 65], [155, 115], [65, 100]],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [points], (0, 255, 0))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    contour = points.reshape(-1, 1, 2)

    result = measure_appearance(hsv, contour, Vision2DConfig())

    assert result.color == "green"
    assert result.shape == "polygon"
