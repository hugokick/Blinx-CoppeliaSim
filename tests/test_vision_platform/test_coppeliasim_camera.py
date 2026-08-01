import struct

import numpy as np
import pytest

from vision_platform.cameras.coppeliasim import CoppeliaSimCamera
from vision_platform.errors import FrameFormatError, FrameTimeoutError


class FakeSim:
    def __init__(self, *, raw, resolution, depth=None, explicit=0):
        self.raw = raw
        self.resolution = list(resolution)
        self.depth = depth
        self.explicit = explicit
        self.get_object_calls = []
        self.image_calls = []
        self.depth_calls = []
        self.render_calls = []

    def getObject(self, path):
        self.get_object_calls.append(path)
        return 42

    def getVisionSensorImg(self, handle):
        self.image_calls.append(handle)
        return self.raw, list(self.resolution)

    def getExplicitHandling(self, handle):
        assert handle == 42
        return self.explicit

    def handleVisionSensor(self, handle):
        self.render_calls.append(handle)

    def getVisionSensorDepth(self, handle, options=0):
        self.depth_calls.append((handle, options))
        return self.depth


def test_coppeliasim_rgb_bytes_are_flipped_and_converted_to_bgr():
    # Coppelia returns RGB rows bottom-to-top: bottom red, then top blue.
    raw = bytes(
        [
            255,
            0,
            0,
            255,
            0,
            0,
            0,
            0,
            255,
            0,
            0,
            255,
        ]
    )
    sim = FakeSim(raw=raw, resolution=[2, 2])
    camera = CoppeliaSimCamera(sim=sim, sensor_path="/VisionLab/Camera")

    frame = camera.read()

    assert frame.image_bgr[0, 0].tolist() == [255, 0, 0]
    assert frame.image_bgr[1, 0].tolist() == [0, 0, 255]
    assert (frame.width, frame.height) == (2, 2)


def test_sensor_path_is_resolved_once_and_handle_is_cached():
    sim = FakeSim(raw=bytes([1, 2, 3]), resolution=[1, 1])
    camera = CoppeliaSimCamera(sim=sim, sensor_path="/VisionLab/Camera")

    camera.read()
    camera.read()

    assert sim.get_object_calls == ["/VisionLab/Camera"]
    assert sim.image_calls == [42, 42]
    assert sim.render_calls == []


def test_explicit_sensor_is_rendered_immediately_before_each_read():
    sim = FakeSim(
        raw=bytes([1, 2, 3]),
        resolution=[1, 1],
        explicit=1,
    )
    camera = CoppeliaSimCamera(sim=sim, sensor_path="/VisionQualityLab/Camera")

    camera.read()
    camera.read()

    assert sim.render_calls == [42, 42]
    assert sim.image_calls == [42, 42]


def test_empty_image_buffer_reports_frame_timeout():
    camera = CoppeliaSimCamera(
        sim=FakeSim(raw=b"", resolution=[2, 2]),
        sensor_path="/VisionLab/Camera",
    )

    with pytest.raises(FrameTimeoutError) as exc:
        camera.read(timeout_s=0.05)

    assert exc.value.code == "FRAME_TIMEOUT"


@pytest.mark.parametrize(
    ("raw", "resolution"),
    [
        (bytes([1, 2, 3]), [2, 2]),
        (bytes([1, 2, 3]), [0, 1]),
        (bytes([1, 2, 3]), [1]),
    ],
)
def test_invalid_resolution_or_byte_count_is_explicit(raw, resolution):
    camera = CoppeliaSimCamera(
        sim=FakeSim(raw=raw, resolution=resolution),
        sensor_path="/VisionLab/Camera",
    )

    with pytest.raises(FrameFormatError) as exc:
        camera.read()

    assert exc.value.code == "FRAME_INVALID"


def test_resolution_changes_are_reflected_in_each_frame():
    sim = FakeSim(raw=bytes([1, 2, 3]), resolution=[1, 1])
    camera = CoppeliaSimCamera(sim=sim, sensor_path="/VisionLab/Camera")
    first = camera.read()
    sim.raw = bytes(range(12))
    sim.resolution = [2, 2]

    second = camera.read()

    assert (first.width, first.height) == (1, 1)
    assert (second.width, second.height) == (2, 2)
    assert second.sequence_id == first.sequence_id + 1


def test_metric_depth_is_decoded_and_flipped_with_the_color_frame():
    rgb = bytes([0] * 12)
    # Bottom depth row 1m, top depth row 2m.
    depth = struct.pack("<ffff", 1.0, 1.0, 2.0, 2.0)
    sim = FakeSim(raw=rgb, resolution=[2, 2], depth=depth)
    camera = CoppeliaSimCamera(
        sim=sim,
        sensor_path="/VisionLab/Camera",
        include_depth=True,
    )

    frame = camera.read()

    np.testing.assert_allclose(frame.depth_m, [[2.0, 2.0], [1.0, 1.0]])
    assert sim.depth_calls == [(42, 1)]


def test_close_is_idempotent_and_read_can_reopen():
    sim = FakeSim(raw=bytes([1, 2, 3]), resolution=[1, 1])
    camera = CoppeliaSimCamera(sim=sim, sensor_path="/VisionLab/Camera")
    camera.open()

    camera.close()
    camera.close()
    frame = camera.read()

    assert frame.source == "coppeliasim"
    assert sim.get_object_calls == ["/VisionLab/Camera", "/VisionLab/Camera"]
