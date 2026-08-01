from __future__ import annotations

import math

import numpy as np
import pytest

from vision_platform.vision2d.curriculum import (
    CurriculumVision2DConfig,
    Vision2DConfigError,
    mask_to_curriculum_roi,
    parse_curriculum_config,
)


def _parameters(*, pixel_scale_mm=(1.4, 1.5)):
    return {
        "vision2d": {
            "profile_id": "standard",
            "roi_px": [100, 120, 400, 420],
            "min_area_ratio": 0.002,
            "max_area_ratio": 0.05,
            "saturation_min": 60,
            "value_min": 40,
            "pixel_scale_mm": (
                None
                if pixel_scale_mm is None
                else list(pixel_scale_mm)
            ),
        },
        "analysis_focus": "size",
    }


def test_parse_curriculum_config_normalizes_exact_published_contract():
    source = _parameters()

    parsed = parse_curriculum_config(source, image_size=(512, 512))

    assert isinstance(parsed, CurriculumVision2DConfig)
    assert parsed.profile_id == "standard"
    assert parsed.image_size == (512, 512)
    assert parsed.roi_px == (100, 120, 400, 420)
    assert parsed.vision2d_config.min_area_ratio == pytest.approx(0.002)
    assert parsed.vision2d_config.max_area_ratio == pytest.approx(0.05)
    assert parsed.vision2d_config.saturation_min == 60
    assert parsed.vision2d_config.value_min == 40
    assert parsed.vision2d_config.pixel_scale is not None
    assert parsed.vision2d_config.pixel_scale.mm_per_pixel_x == pytest.approx(
        1.4
    )
    assert parsed.vision2d_config.pixel_scale.mm_per_pixel_y == pytest.approx(
        1.5
    )
    assert parsed.to_public_dict() == source["vision2d"]

    source["vision2d"]["roi_px"][0] = 0
    source["vision2d"]["pixel_scale_mm"][0] = 99.0
    assert parsed.roi_px == (100, 120, 400, 420)
    assert parsed.to_public_dict()["pixel_scale_mm"] == [1.4, 1.5]


def test_parse_curriculum_config_accepts_explicit_no_scale():
    parsed = parse_curriculum_config(
        _parameters(pixel_scale_mm=None),
        image_size=(512, 512),
    )

    assert parsed.vision2d_config.pixel_scale is None
    assert parsed.to_public_dict()["pixel_scale_mm"] is None


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value.pop("vision2d"), "vision2d"),
        (
            lambda value: value.__setitem__("vision2d", []),
            "vision2d",
        ),
        (
            lambda value: value["vision2d"].__setitem__("extra", 1),
            "fields",
        ),
        (
            lambda value: value["vision2d"].pop("value_min"),
            "fields",
        ),
        (
            lambda value: value["vision2d"].__setitem__(
                "profile_id", "wide_dim"
            ),
            "profile_id",
        ),
        (
            lambda value: value["vision2d"].__setitem__(
                "roi_px", [100, 120, 100, 420]
            ),
            "roi_px",
        ),
        (
            lambda value: value["vision2d"].__setitem__(
                "roi_px", [100, 120, 513, 420]
            ),
            "roi_px",
        ),
        (
            lambda value: value["vision2d"].__setitem__(
                "roi_px", [True, 120, 400, 420]
            ),
            "roi_px",
        ),
        (
            lambda value: value["vision2d"].__setitem__(
                "min_area_ratio", True
            ),
            "min_area_ratio",
        ),
        (
            lambda value: value["vision2d"].__setitem__(
                "max_area_ratio", math.nan
            ),
            "max_area_ratio",
        ),
        (
            lambda value: value["vision2d"].__setitem__(
                "saturation_min", 256
            ),
            "saturation_min",
        ),
        (
            lambda value: value["vision2d"].__setitem__(
                "value_min", 40.0
            ),
            "value_min",
        ),
        (
            lambda value: value["vision2d"].__setitem__(
                "pixel_scale_mm", [1.0]
            ),
            "pixel_scale_mm",
        ),
        (
            lambda value: value["vision2d"].__setitem__(
                "pixel_scale_mm", [1.0, math.inf]
            ),
            "pixel_scale_mm",
        ),
    ],
)
def test_parse_curriculum_config_rejects_invalid_published_values(
    mutate,
    match,
):
    parameters = _parameters()
    mutate(parameters)

    with pytest.raises(Vision2DConfigError, match=match) as caught:
        parse_curriculum_config(parameters, image_size=(512, 512))

    assert caught.value.code == "VISION2D_CONFIG_INVALID"


@pytest.mark.parametrize(
    "image_size",
    [
        (256, 256),
        (512, 511),
        (True, 512),
        [512, 512],
        (512,),
    ],
)
def test_parse_curriculum_config_requires_standard_image_size(image_size):
    with pytest.raises(Vision2DConfigError, match="image_size"):
        parse_curriculum_config(_parameters(), image_size=image_size)


def test_mask_to_curriculum_roi_preserves_coordinates_and_source():
    parsed = parse_curriculum_config(
        {
            "vision2d": {
                **_parameters()["vision2d"],
                "roi_px": [1, 1, 4, 3],
            }
        },
        image_size=(512, 512),
    )
    image = np.full((512, 512, 3), 17, dtype=np.uint8)
    before = image.copy()

    masked = mask_to_curriculum_roi(image, parsed)

    assert masked.shape == image.shape
    assert masked.dtype == np.uint8
    assert not np.shares_memory(masked, image)
    assert np.array_equal(image, before)
    assert np.all(masked[1:3, 1:4] == 17)
    outside = masked.copy()
    outside[1:3, 1:4] = 0
    assert np.count_nonzero(outside) == 0


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((512, 512), dtype=np.uint8),
        np.zeros((512, 512, 3), dtype=np.float32),
        np.zeros((256, 256, 3), dtype=np.uint8),
        [[0]],
    ],
)
def test_mask_to_curriculum_roi_rejects_invalid_image(image):
    parsed = parse_curriculum_config(
        _parameters(),
        image_size=(512, 512),
    )

    with pytest.raises(Vision2DConfigError, match="image"):
        mask_to_curriculum_roi(image, parsed)
