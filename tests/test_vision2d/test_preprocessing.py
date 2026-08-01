import numpy as np
import pytest

from tests.test_vision2d.synthetic_factory import SyntheticObject, make_scene
from vision_platform.vision2d.models import Vision2DConfig
from vision_platform.vision2d.preprocessing import preprocess_image


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((20, 30), dtype=np.uint8),
        np.zeros((20, 30, 4), dtype=np.uint8),
        np.zeros((20, 30, 3), dtype=np.float32),
        np.zeros((0, 30, 3), dtype=np.uint8),
    ],
)
def test_preprocess_rejects_non_uint8_bgr_images(image):
    with pytest.raises((TypeError, ValueError), match="BGR|uint8|non-empty"):
        preprocess_image(image, Vision2DConfig())


def test_preprocess_returns_named_images_without_mutating_input():
    image, _ = make_scene(
        (240, 180),
        (SyntheticObject("rectangle", "red", (120, 90), (80, 40), 0.0),),
    )
    before = image.copy()

    output = preprocess_image(image, Vision2DConfig())

    assert set(output) == {"gray", "hsv", "foreground_mask", "cleaned_mask"}
    assert output["gray"].shape == image.shape[:2]
    assert output["hsv"].shape == image.shape
    assert output["cleaned_mask"].dtype == np.uint8
    assert output["cleaned_mask"][90, 120] == 255
    assert np.array_equal(image, before)
