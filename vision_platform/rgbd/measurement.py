"""Composition of depth sampling, deprojection, and optional transforms."""

from __future__ import annotations

from .errors import RgbdContractError
from .geometry import deproject_pixel
from .models import CameraIntrinsics, RgbdFrame, RgbdMeasurement
from .sampling import sample_depth
from .transforms import RigidTransform, transform_point


def measure_pixel(
    frame: RgbdFrame,
    intrinsics: CameraIntrinsics,
    *,
    u_px: int,
    v_px: int,
    window_size: int = 1,
    min_valid_count: int = 1,
    camera_frame_id: str = "camera",
    target_frame_id: str | None = None,
    camera_to_target: RigidTransform | None = None,
) -> RgbdMeasurement:
    if (target_frame_id is None) != (camera_to_target is None):
        raise RgbdContractError(
            "RGBD_MEASUREMENT_INVALID",
            "target frame and transform must be supplied together",
        )
    if camera_to_target is not None and not isinstance(camera_to_target, RigidTransform):
        raise RgbdContractError(
            "RGBD_MEASUREMENT_INVALID", "camera_to_target is invalid"
        )
    if (
        not isinstance(frame, RgbdFrame)
        or not isinstance(intrinsics, CameraIntrinsics)
        or frame.depth_m.shape != (intrinsics.height_px, intrinsics.width_px)
    ):
        raise RgbdContractError(
            "RGBD_MEASUREMENT_INVALID", "frame and intrinsics size do not match"
        )
    if type(camera_frame_id) is not str or not camera_frame_id:
        raise RgbdContractError(
            "RGBD_MEASUREMENT_INVALID", "camera frame ID is invalid"
        )
    sample = sample_depth(
        frame,
        u_px,
        v_px,
        window_size=window_size,
        min_valid_count=min_valid_count,
    )
    if sample.status == "NO_VALID_DEPTH":
        return RgbdMeasurement(
            "NO_VALID_DEPTH",
            sample,
            camera_frame_id,
            target_frame_id,
            None,
            None,
            "NO_VALID_DEPTH",
        )
    assert sample.depth_m is not None
    camera_point = deproject_pixel(
        intrinsics,
        u_px=float(u_px),
        v_px=float(v_px),
        depth_m=sample.depth_m,
    )
    target_point = (
        None
        if camera_to_target is None
        else transform_point(camera_point, camera_to_target)
    )
    return RgbdMeasurement(
        "PASS",
        sample,
        camera_frame_id,
        target_frame_id,
        camera_point,
        target_point,
        None,
    )


__all__ = ["measure_pixel"]
