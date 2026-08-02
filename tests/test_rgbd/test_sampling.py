from __future__ import annotations

import numpy as np
import pytest

from vision_platform.rgbd.errors import RgbdContractError
from vision_platform.rgbd.models import RgbdFrame
from vision_platform.rgbd.sampling import sample_depth


def _frame(depth: np.ndarray) -> RgbdFrame:
    image = np.zeros((*depth.shape, 3), dtype=np.uint8)
    return RgbdFrame(image, np.asarray(depth, dtype=np.float32))


def test_single_pixel_sample_is_exact() -> None:
    sample = sample_depth(_frame(np.array([[0.5]], dtype=np.float32)), 0, 0)
    assert sample.status == "PASS"
    assert sample.depth_m == pytest.approx(0.5)
    assert sample.valid_count == 1
    assert sample.failure_code is None


@pytest.mark.parametrize("window_size", [3, 5, 7, 9])
def test_odd_window_uses_valid_median(window_size: int) -> None:
    depth = np.zeros((9, 9), dtype=np.float32)
    depth[3:6, 3:6] = np.array(
        [[0.0, 0.3, 0.0], [0.1, 0.2, 0.4], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    sample = sample_depth(_frame(depth), 4, 4, window_size=window_size)
    assert sample.depth_m == pytest.approx(0.25)
    assert sample.valid_count == 4


def test_boundary_window_clips_without_wrapping() -> None:
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[0, 0] = 0.2
    depth[0, 1] = 0.4
    depth[-1, -1] = 9.0
    sample = sample_depth(_frame(depth), 0, 0, window_size=3)
    assert sample.depth_m == pytest.approx(0.3)
    assert sample.valid_count == 2


def test_no_valid_depth_is_structured() -> None:
    sample = sample_depth(_frame(np.zeros((3, 3), np.float32)), 1, 1, window_size=3)
    assert sample.status == "NO_VALID_DEPTH"
    assert sample.depth_m is None
    assert sample.failure_code == "NO_VALID_DEPTH"


@pytest.mark.parametrize(
    "arguments",
    [
        {"u_px": -1, "v_px": 0},
        {"u_px": 0, "v_px": 3},
        {"u_px": True, "v_px": 0},
        {"u_px": 0, "v_px": 0, "window_size": 2},
        {"u_px": 0, "v_px": 0, "window_size": 11},
        {"u_px": 0, "v_px": 0, "min_valid_count": 0},
    ],
)
def test_sample_rejects_invalid_arguments(arguments: dict[str, object]) -> None:
    with pytest.raises(RgbdContractError) as captured:
        sample_depth(_frame(np.ones((3, 3), np.float32)), **arguments)
    assert captured.value.code == "RGBD_SAMPLE_INVALID"
