import numpy as np

from tests.test_vision2d.synthetic_factory import (
    SyntheticObject,
    add_gaussian_noise,
    adjust_brightness,
    make_scene,
)


def test_scene_factory_returns_original_image_and_truth():
    objects = (
        SyntheticObject("rectangle", "red", (90, 80), (60, 30), 30.0),
        SyntheticObject("circle", "blue", (220, 150), (40, 40), 0.0),
    )

    image, truth = make_scene((320, 240), objects)

    assert image.shape == (240, 320, 3)
    assert image.dtype == np.uint8
    assert [item.shape for item in truth] == ["rectangle", "circle"]
    assert [item.color for item in truth] == ["red", "blue"]


def test_noise_is_reproducible_for_fixed_seed():
    image, _ = make_scene(
        (160, 120),
        (SyntheticObject("square", "green", (80, 60), (40, 40), 15.0),),
    )

    first = add_gaussian_noise(image, sigma=4.0, seed=20260731)
    second = add_gaussian_noise(image, sigma=4.0, seed=20260731)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, image)


def test_brightness_adjustment_is_explicit_and_deterministic():
    image, _ = make_scene(
        (160, 120),
        (SyntheticObject("rectangle", "yellow", (80, 60), (50, 30), 0.0),),
    )

    adjusted = adjust_brightness(image, factor=0.65)

    assert adjusted.dtype == np.uint8
    assert adjusted.shape == image.shape
    assert np.array_equal(adjusted, adjust_brightness(image, factor=0.65))
    assert not np.array_equal(adjusted, image)
