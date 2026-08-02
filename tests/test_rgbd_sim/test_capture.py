from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy as np
import pytest

from vision_platform.rgbd_sim.capture import CoppeliaRgbdCapture
from vision_platform.rgbd_sim.errors import RgbdSimContractError


class FakeSim:
    visionintparam_perspective_operation = 1
    visionintparam_rgbignored = 2
    visionintparam_depthignored = 3
    visionfloatparam_perspective_angle = 4
    visionfloatparam_near_clipping = 5
    visionfloatparam_far_clipping = 6

    def __init__(self, *, explicit: int = 1, resolution: tuple[int, int] = (2, 2)):
        self.sensor = 42
        self.explicit = explicit
        self.resolution = resolution
        self.calls: list[tuple[object, ...]] = []
        self.image = bytes(
            [
                255, 0, 0, 0, 255, 0,
                0, 0, 255, 255, 255, 255,
            ]
        )
        self.depth = struct.pack("<ffff", 1.0, 1.1, 1.2, 1.3)

    def getObject(self, path: str) -> int:
        self.calls.append(("getObject", path))
        return self.sensor

    def getExplicitHandling(self, handle: int) -> int:
        self.calls.append(("getExplicitHandling", handle))
        return self.explicit

    def handleVisionSensor(self, handle: int) -> None:
        self.calls.append(("handleVisionSensor", handle))

    def getVisionSensorImg(self, handle: int):
        self.calls.append(("getVisionSensorImg", handle))
        return self.image, list(self.resolution)

    def getVisionSensorDepth(self, handle: int, options: int):
        self.calls.append(("getVisionSensorDepth", handle, options))
        return self.depth, list(self.resolution)

    def getVisionSensorResolution(self, handle: int):
        self.calls.append(("getVisionSensorResolution", handle))
        return list(self.resolution)

    def getObjectInt32Param(self, handle: int, parameter: int) -> int:
        self.calls.append(("getObjectInt32Param", handle, parameter))
        return {1: 1, 2: 0, 3: 0}[parameter]

    def getObjectFloatParam(self, handle: int, parameter: int) -> float:
        self.calls.append(("getObjectFloatParam", handle, parameter))
        return {4: 1.0471975512, 5: 0.01, 6: 5.0}[parameter]


def _binding(**overrides: object) -> SimpleNamespace:
    values = {
        "scene_path": "simulation/rgbd_lab/BL23_rgbd_lab.ttt",
        "scene_sha256": "a" * 64,
        "expected_source_depth_model": "optical_z",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_source_capture_uses_one_explicit_handling_cycle_and_metric_depth_option() -> None:
    sim = FakeSim()
    source = CoppeliaRgbdCapture(
        sim=sim,
        sensor_path="/RgbdLab/CameraRig/RgbdSensor",
        scene_binding=_binding(),
    ).read_source()
    assert sim.calls.count(("handleVisionSensor", sim.sensor)) == 1
    assert ("getVisionSensorDepth", sim.sensor, 1) in sim.calls
    assert source.image_bgr.dtype == np.uint8
    assert source.source_depth_m.dtype == np.float32
    assert source.image_bgr.shape[:2] == source.source_depth_m.shape
    assert source.metadata.explicit_handling is True
    assert source.metadata.vertical_flip == "vertical_flip"
    assert source.metadata.color_conversion == "RGB_to_BGR"


def test_source_capture_call_order_checks_explicit_before_handle_and_reads() -> None:
    sim = FakeSim()
    CoppeliaRgbdCapture(sim=sim, scene_binding=_binding()).read_source()
    names = [call[0] for call in sim.calls]
    assert names.index("getExplicitHandling") < names.index("handleVisionSensor")
    assert names.index("handleVisionSensor") < names.index("getVisionSensorImg")
    assert names.index("getVisionSensorImg") < names.index("getVisionSensorDepth")


@pytest.mark.parametrize("explicit", [0, 2])
def test_non_explicit_sensor_fails_closed_without_handle_or_reads(explicit: int) -> None:
    sim = FakeSim(explicit=explicit)
    with pytest.raises(RgbdSimContractError) as captured:
        CoppeliaRgbdCapture(sim=sim, scene_binding=_binding()).read_source()
    assert captured.value.code == "RGBD_SIM_CAPTURE_INVALID"
    assert not any(call[0] == "handleVisionSensor" for call in sim.calls)
    assert not any(call[0] in {"getVisionSensorImg", "getVisionSensorDepth"} for call in sim.calls)


def test_source_capture_rejects_color_depth_resolution_mismatch() -> None:
    sim = FakeSim()
    sim.depth = struct.pack("<f", 1.0)
    with pytest.raises(RgbdSimContractError) as captured:
        CoppeliaRgbdCapture(sim=sim, scene_binding=_binding()).read_source()
    assert captured.value.code == "RGBD_SIM_CAPTURE_INVALID"
