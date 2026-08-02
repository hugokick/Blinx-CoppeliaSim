from __future__ import annotations

from time import perf_counter

from tests.test_rgbd.synthetic_factory import make_plane
from vision_platform.rgbd.measurement import measure_pixel
from vision_platform.rgbd.models import CameraIntrinsics, RgbdFrame
from vision_platform.rgbd.serialization import measurement_to_dict


def test_512_frame_and_1024_measurements_fit_cpu_budget() -> None:
    image, depth = make_plane(512, 512, 1.0)
    intrinsics = CameraIntrinsics(512, 512, 400.0, 400.0, 255.5, 255.5)
    warm_frame = RgbdFrame(image, depth)
    measurement_to_dict(
        measure_pixel(warm_frame, intrinsics, u_px=16, v_px=16, window_size=3)
    )
    started = perf_counter()
    frame = RgbdFrame(image, depth)
    for v_px in range(8, 512, 16):
        for u_px in range(8, 512, 16):
            measurement_to_dict(
                measure_pixel(
                    frame, intrinsics, u_px=u_px, v_px=v_px, window_size=3
                )
            )
    elapsed_s = perf_counter() - started
    assert elapsed_s < 2.0, f"RGB-D CPU budget exceeded: {elapsed_s:.3f}s"
