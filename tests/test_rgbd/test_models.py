from __future__ import annotations

import numpy as np
import pytest

from vision_platform.rgbd.errors import RgbdContractError
from vision_platform.rgbd.models import CameraIntrinsics, RgbdFrame


def test_intrinsics_normalize_finite_builtin_values() -> None:
    value = CameraIntrinsics(4, 3, 100.0, 110.0, 1.5, 1.0)
    assert value.width_px == 4
    assert value.height_px == 3
    assert value.fx_px == 100.0
    assert value.pixel_center_convention == "integer_center_top_left_zero"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("width_px", 0),
        ("width_px", 4.0),
        ("height_px", -1),
        ("height_px", False),
        ("width_px", True),
        ("fx_px", 0.0),
        ("fy_px", -1.0),
        ("fy_px", float("inf")),
        ("cx_px", float("nan")),
        ("cx_px", float("inf")),
        ("cy_px", True),
    ],
)
def test_intrinsics_reject_invalid_values(field: str, value: object) -> None:
    arguments: dict[str, object] = {
        "width_px": 4,
        "height_px": 3,
        "fx_px": 100.0,
        "fy_px": 110.0,
        "cx_px": 1.5,
        "cy_px": 1.0,
    }
    arguments[field] = value
    with pytest.raises(RgbdContractError) as captured:
        CameraIntrinsics(**arguments)  # type: ignore[arg-type]
    assert captured.value.code == "RGBD_INTRINSICS_INVALID"


def test_frame_copies_and_seals_exact_arrays() -> None:
    image = np.full((3, 4, 3), 17, dtype=np.uint8)
    depth = np.full((3, 4), 0.75, dtype=np.float32)
    frame = RgbdFrame(image, depth)
    image[:] = 99
    depth[:] = 1.5
    assert int(frame.image_bgr[0, 0, 0]) == 17
    assert float(frame.depth_m[0, 0]) == pytest.approx(0.75)
    assert frame.image_bgr.flags.c_contiguous
    assert frame.depth_m.flags.c_contiguous
    assert not frame.image_bgr.flags.writeable
    assert not frame.depth_m.flags.writeable


@pytest.mark.parametrize(
    "depth",
    [
        np.ones((3, 4), dtype=np.float64),
        np.full((3, 4), -0.1, dtype=np.float32),
        np.full((3, 4), np.nan, dtype=np.float32),
        np.full((3, 4), np.inf, dtype=np.float32),
    ],
)
def test_frame_rejects_invalid_depth(depth: np.ndarray) -> None:
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    with pytest.raises(RgbdContractError) as captured:
        RgbdFrame(image, depth)
    assert captured.value.code == "RGBD_FRAME_INVALID"


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((3, 4), dtype=np.uint8),
        np.zeros((3, 4, 4), dtype=np.uint8),
        np.zeros((3, 4, 3), dtype=np.float32),
        np.zeros((0, 4, 3), dtype=np.uint8),
    ],
)
def test_frame_rejects_invalid_image(image: np.ndarray) -> None:
    with pytest.raises(RgbdContractError) as captured:
        RgbdFrame(image, np.ones((3, 4), dtype=np.float32))
    assert captured.value.code == "RGBD_FRAME_INVALID"


def test_frame_rejects_spatial_size_mismatch() -> None:
    with pytest.raises(RgbdContractError) as captured:
        RgbdFrame(
            np.zeros((3, 4, 3), dtype=np.uint8),
            np.ones((4, 3), dtype=np.float32),
        )
    assert captured.value.code == "RGBD_FRAME_INVALID"


def test_frame_accepts_zero_as_missing_depth() -> None:
    frame = RgbdFrame(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.float32),
    )
    assert np.count_nonzero(frame.depth_m) == 0
