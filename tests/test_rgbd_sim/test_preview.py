from __future__ import annotations

import json

import numpy as np
import pytest

from vision_platform.rgbd_sim.errors import RgbdSimContractError
from vision_platform.rgbd_sim.preview import (
    colorize_depth,
    copy_bgr_preview,
    depth_summary,
)


def test_bgr_preview_is_copied_and_has_contract_shape() -> None:
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    preview = copy_bgr_preview(image)

    assert preview.dtype == np.uint8
    assert preview.shape == (2, 3, 3)
    assert preview.flags.c_contiguous
    np.testing.assert_array_equal(preview, image)
    assert preview is not image
    image[0, 0, 0] = 255
    assert preview[0, 0, 0] != 255


def test_zero_depth_has_explicit_invalid_colour_and_input_is_unchanged() -> None:
    depth = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    before = depth.copy()

    preview = colorize_depth(depth, minimum_m=0.2, maximum_m=1.2)

    assert tuple(preview[0, 0]) == (255, 0, 255)
    assert preview.dtype == np.uint8
    assert preview.shape == (1, 3, 3)
    np.testing.assert_array_equal(depth, before)


def test_colorization_is_deterministic_for_one_narrow_and_extreme_values() -> None:
    cases = [
        np.array([[0.0, 0.75, 0.0]], dtype=np.float32),
        np.array([[0.0, 1.0, 1.000001]], dtype=np.float32),
        np.array([[0.0, 1.0e-20, 1.0e20]], dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    ]
    for depth in cases:
        first = colorize_depth(depth)
        second = colorize_depth(depth)
        np.testing.assert_array_equal(first, second)
        assert first.dtype == np.uint8
        assert first.shape == (*depth.shape, 3)
        assert np.all(np.isfinite(first))
        if np.all(depth == 0):
            assert np.all(first == np.array([255, 0, 255], dtype=np.uint8))


@pytest.mark.parametrize(
    "depth",
    [
        np.array([[np.nan]], dtype=np.float32),
        np.array([[np.inf]], dtype=np.float32),
        np.array([[-0.1]], dtype=np.float32),
        np.array([[1]], dtype=np.int32),
        np.array([1.0], dtype=np.float32),
    ],
)
def test_preview_rejects_invalid_depth_contract(depth: np.ndarray) -> None:
    with pytest.raises(RgbdSimContractError) as exc_info:
        colorize_depth(depth)
    assert exc_info.value.code == "RGBD_SIM_PREVIEW_INVALID"


def test_invalid_preview_arguments_and_bgr_contract_are_structured() -> None:
    with pytest.raises(RgbdSimContractError):
        colorize_depth(np.ones((1, 1), dtype=np.float32), minimum_m=1.0, maximum_m=1.0)
    with pytest.raises(RgbdSimContractError):
        copy_bgr_preview(np.zeros((1, 1), dtype=np.uint8))


def test_depth_summary_is_json_native_and_hides_raw_array() -> None:
    depth = np.array([[0.0, 0.25], [0.75, 1.0]], dtype=np.float32)
    summary = depth_summary(depth)

    assert summary == {
        "valid_count": 3,
        "invalid_count": 1,
        "min_depth_m": 0.25,
        "median_depth_m": 0.75,
        "max_depth_m": 1.0,
    }
    encoded = json.dumps(summary, allow_nan=False, sort_keys=True)
    assert "depth" not in encoded.lower() or "depth_m" in encoded
    assert "source_depth" not in encoded
