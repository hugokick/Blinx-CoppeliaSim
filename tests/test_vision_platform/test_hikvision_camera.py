from __future__ import annotations

import numpy as np
import pytest

from vision_platform.cameras.hikvision import (
    HikvisionCamera,
    HikvisionSdkFrame,
    MvsSdkError,
)
from vision_platform.errors import (
    CameraUnavailableError,
    FrameFormatError,
    FrameTimeoutError,
)


class FakeMvsAdapter:
    def __init__(
        self,
        *,
        frame=None,
        devices=None,
        read_error=None,
        open_error=None,
    ):
        self.frame = frame
        self.devices = [{"index": 0, "model": "FAKE-MVS"}] if devices is None else devices
        self.read_error = read_error
        self.open_error = open_error
        self.actions = []

    def enumerate_devices(self):
        self.actions.append("enumerate")
        return list(self.devices)

    def open_device(self, index):
        self.actions.append(("open", index))
        if self.open_error is not None:
            raise self.open_error

    def start_grabbing(self):
        self.actions.append("start")

    def read_frame(self, timeout_ms):
        self.actions.append(("read", timeout_ms))
        if self.read_error is not None:
            raise self.read_error
        return self.frame

    def stop_grabbing(self):
        self.actions.append("stop")

    def close_device(self):
        self.actions.append("close")


def _frame(pixel_format):
    # Two pixels: red, then blue in the declared channel order.
    if pixel_format == "RGB":
        data = bytes([255, 0, 0, 0, 0, 255])
    else:
        data = bytes([0, 0, 255, 255, 0, 0])
    return HikvisionSdkFrame(
        data=data,
        width=2,
        height=1,
        pixel_format=pixel_format,
        frame_id=17,
    )


def test_hikvision_backend_reports_actionable_unavailable_error(monkeypatch):
    monkeypatch.delenv("HIK_MVS_ROOT", raising=False)
    with pytest.raises(CameraUnavailableError) as exc:
        HikvisionCamera(search_roots=[]).open()
    assert exc.value.code == "CAMERA_UNAVAILABLE"
    assert "MVS" in str(exc.value)


@pytest.mark.parametrize("pixel_format", ["RGB", "BGR"])
def test_hikvision_fake_sdk_converts_rgb_or_preserves_bgr(pixel_format):
    adapter = FakeMvsAdapter(frame=_frame(pixel_format))
    camera = HikvisionCamera(sdk_adapter=adapter, timeout_ms=250)

    camera.open()
    frame = camera.read()

    assert frame.image_bgr.tolist() == [[[0, 0, 255], [255, 0, 0]]]
    assert frame.source == "hikvision"
    assert frame.metadata["sdk_frame_id"] == 17
    assert adapter.actions[:4] == [
        "enumerate",
        ("open", 0),
        "start",
        ("read", 250),
    ]


def test_hikvision_timeout_has_stable_error_code():
    adapter = FakeMvsAdapter(
        frame=None,
        read_error=TimeoutError("no frame"),
    )
    camera = HikvisionCamera(sdk_adapter=adapter)

    with pytest.raises(FrameTimeoutError) as exc:
        camera.read(timeout_s=0.125)

    assert exc.value.code == "FRAME_TIMEOUT"
    assert ("read", 125) in adapter.actions


def test_hikvision_sdk_error_is_translated_and_cleanup_is_idempotent():
    adapter = FakeMvsAdapter(
        frame=_frame("BGR"),
        open_error=MvsSdkError("open device", 0x80000003),
    )
    camera = HikvisionCamera(sdk_adapter=adapter)

    with pytest.raises(CameraUnavailableError) as exc:
        camera.open()
    camera.close()
    camera.close()

    assert "0x80000003" in str(exc.value)
    assert adapter.actions.count("close") == 1


def test_hikvision_close_stops_stream_then_closes_device():
    adapter = FakeMvsAdapter(frame=_frame("BGR"))
    camera = HikvisionCamera(sdk_adapter=adapter)
    camera.open()

    camera.close()
    camera.close()

    assert adapter.actions[-2:] == ["stop", "close"]
    assert camera.is_open is False


def test_hikvision_rejects_invalid_frame_buffer_size():
    adapter = FakeMvsAdapter(
        frame=HikvisionSdkFrame(
            data=b"\x00\x01",
            width=2,
            height=1,
            pixel_format="BGR",
            frame_id=1,
        )
    )
    camera = HikvisionCamera(sdk_adapter=adapter)

    with pytest.raises(FrameFormatError) as exc:
        camera.read()

    assert "frame buffer" in str(exc.value).lower()
