from __future__ import annotations

import importlib.util
import os
import sys
import time
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Protocol

import numpy as np

from vision_platform.cameras.base import CameraBackend
from vision_platform.errors import (
    CameraUnavailableError,
    FrameFormatError,
    FrameTimeoutError,
)
from vision_platform.models import Frame


@dataclass(frozen=True)
class HikvisionSdkFrame:
    data: bytes
    width: int
    height: int
    pixel_format: str
    frame_id: int | None = None


class MvsSdkError(RuntimeError):
    def __init__(self, operation: str, code: int) -> None:
        self.operation = str(operation)
        self.code = int(code) & 0xFFFFFFFF
        super().__init__(
            f"海康 MVS {self.operation}失败，错误码 0x{self.code:08x}"
        )


class _SdkAdapter(Protocol):
    def enumerate_devices(self) -> list[dict[str, Any]]:
        ...

    def open_device(self, index: int) -> None:
        ...

    def start_grabbing(self) -> None:
        ...

    def read_frame(self, timeout_ms: int) -> HikvisionSdkFrame:
        ...

    def stop_grabbing(self) -> None:
        ...

    def close_device(self) -> None:
        ...


class _LegacyMvsAdapter:
    """Small wrapper around Hikvision's generated MvImport module."""

    def __init__(self, module: ModuleType) -> None:
        self.module = module
        self.device_list = None
        self.camera = None
        self.payload_size = 0
        self._started = False

    @staticmethod
    def _check(operation: str, code: int) -> None:
        if int(code) != 0:
            raise MvsSdkError(operation, int(code))

    def enumerate_devices(self) -> list[dict[str, Any]]:
        module = self.module
        device_list = module.MV_CC_DEVICE_INFO_LIST()
        layer_type = module.MV_GIGE_DEVICE | module.MV_USB_DEVICE
        self._check(
            "枚举设备",
            module.MvCamera.MV_CC_EnumDevices(layer_type, device_list),
        )
        self.device_list = device_list
        return [
            {"index": index, "transport": "MVS"}
            for index in range(int(device_list.nDeviceNum))
        ]

    def open_device(self, index: int) -> None:
        if self.device_list is None:
            raise RuntimeError("MVS devices must be enumerated before opening")
        if not 0 <= int(index) < int(self.device_list.nDeviceNum):
            raise CameraUnavailableError(
                f"海康相机索引 {index} 超出枚举范围",
                device_index=index,
                device_count=int(self.device_list.nDeviceNum),
            )
        module = self.module
        pointer = self.device_list.pDeviceInfo[int(index)]
        device_info = cast(
            pointer,
            POINTER(module.MV_CC_DEVICE_INFO),
        ).contents
        camera = module.MvCamera()
        self._check("创建句柄", camera.MV_CC_CreateHandle(device_info))
        self.camera = camera
        self._check(
            "打开设备",
            camera.MV_CC_OpenDevice(module.MV_ACCESS_Exclusive, 0),
        )
        trigger_off = getattr(module, "MV_TRIGGER_MODE_OFF", 0)
        self._check(
            "关闭触发模式",
            camera.MV_CC_SetEnumValue("TriggerMode", trigger_off),
        )
        payload = module.MVCC_INTVALUE()
        memset(byref(payload), 0, sizeof(payload))
        self._check(
            "读取 PayloadSize",
            camera.MV_CC_GetIntValue("PayloadSize", payload),
        )
        self.payload_size = int(payload.nCurValue)
        if self.payload_size <= 0:
            raise CameraUnavailableError(
                "海康 MVS 返回了无效的 PayloadSize",
                payload_size=self.payload_size,
            )

    def start_grabbing(self) -> None:
        if self.camera is None:
            raise RuntimeError("MVS device is not open")
        self._check("开始取流", self.camera.MV_CC_StartGrabbing())
        self._started = True

    def read_frame(self, timeout_ms: int) -> HikvisionSdkFrame:
        if self.camera is None or not self._started:
            raise RuntimeError("MVS stream is not running")
        module = self.module
        frame_info = module.MV_FRAME_OUT_INFO_EX()
        memset(byref(frame_info), 0, sizeof(frame_info))
        buffer = (c_ubyte * self.payload_size)()
        getter = getattr(self.camera, "MV_CC_GetImageForBGR", None)
        if not callable(getter):
            raise CameraUnavailableError(
                "当前 MVS Python SDK 不提供 MV_CC_GetImageForBGR"
            )
        code = int(
            getter(
                buffer,
                self.payload_size,
                frame_info,
                int(timeout_ms),
            )
        )
        no_data = int(getattr(module, "MV_E_NODATA", 0x80000007)) & 0xFFFFFFFF
        if (code & 0xFFFFFFFF) == no_data:
            raise TimeoutError("MVS returned no frame before timeout")
        self._check("读取图像", code)
        width = int(frame_info.nWidth)
        height = int(frame_info.nHeight)
        expected = width * height * 3
        frame_length = int(getattr(frame_info, "nFrameLen", expected) or expected)
        if frame_length < expected:
            raise FrameFormatError(
                "MVS BGR frame is shorter than width*height*3",
                width=width,
                height=height,
                frame_length=frame_length,
                expected=expected,
            )
        return HikvisionSdkFrame(
            data=bytes(buffer[:expected]),
            width=width,
            height=height,
            pixel_format="BGR",
            frame_id=int(getattr(frame_info, "nFrameNum", 0)),
        )

    def stop_grabbing(self) -> None:
        if self.camera is None or not self._started:
            return
        self._check("停止取流", self.camera.MV_CC_StopGrabbing())
        self._started = False

    def close_device(self) -> None:
        camera = self.camera
        if camera is None:
            return
        close_error = None
        try:
            code = int(camera.MV_CC_CloseDevice())
            if code != 0:
                close_error = MvsSdkError("关闭设备", code)
        finally:
            destroy_code = int(camera.MV_CC_DestroyHandle())
            self.camera = None
            self.payload_size = 0
        if close_error is not None:
            raise close_error
        self._check("销毁句柄", destroy_code)


def _default_search_roots() -> list[Path]:
    project_root = Path(__file__).resolve().parents[2]
    roots = [
        Path(r"C:\Program Files (x86)\MVS"),
        Path(r"C:\Program Files\MVS"),
        Path(r"C:\Program Files (x86)\Common Files\MVS"),
        Path(r"C:\Program Files\Common Files\MVS"),
        Path("/opt/MVS"),
    ]
    roots.extend(
        path.parent
        for path in project_root.glob("*/MvImport/MvCameraControl_class.py")
    )
    return roots


def _candidate_module_paths(root: Path) -> Iterable[Path]:
    yield root / "MvCameraControl_class.py"
    yield root / "MvImport" / "MvCameraControl_class.py"
    yield (
        root
        / "Development"
        / "Samples"
        / "Python"
        / "MvImport"
        / "MvCameraControl_class.py"
    )


def _candidate_runtime_paths(root: Path) -> Iterable[Path]:
    if os.name == "nt":
        name = "MvCameraControl.dll"
        yield root / name
        yield root / "Runtime" / "Win64_x64" / name
        yield root / "Development" / "MVS" / "Runtime" / "Win64_x64" / name
    else:
        names = ("libMvCameraControl.so", "libMvCameraControl.so.3.2.2.1")
        for name in names:
            yield root / name
            yield root / "lib" / "64" / name


def _load_mvs_adapter(
    *,
    mvs_root: str | Path | None,
    search_roots: Iterable[str | Path] | None,
) -> _LegacyMvsAdapter:
    roots: list[Path] = []
    selected_root = mvs_root or os.getenv("HIK_MVS_ROOT")
    if selected_root:
        roots.append(Path(selected_root).expanduser().resolve())
    if search_roots is None:
        roots.extend(_default_search_roots())
    else:
        roots.extend(Path(path).expanduser().resolve() for path in search_roots)

    module_path = next(
        (
            candidate
            for root in roots
            for candidate in _candidate_module_paths(root)
            if candidate.is_file()
        ),
        None,
    )
    runtime_path = next(
        (
            candidate
            for root in roots
            for candidate in _candidate_runtime_paths(root)
            if candidate.is_file()
        ),
        None,
    )
    if module_path is None or runtime_path is None:
        raise CameraUnavailableError(
            (
                "未找到可用的海康 MVS Python SDK 与运行库；"
                "请安装 MVS，并设置 HIK_MVS_ROOT"
            ),
            searched=[str(path) for path in roots],
            python_module=str(module_path) if module_path else None,
            runtime=str(runtime_path) if runtime_path else None,
        )

    dll_directory = None
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        dll_directory = os.add_dll_directory(str(runtime_path.parent))
    module_name = f"_vision_platform_mvs_{abs(hash(module_path))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise CameraUnavailableError(
            f"无法加载海康 MVS Python 模块：{module_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(module_path.parent))
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise CameraUnavailableError(
            f"海康 MVS SDK 加载失败：{error}",
            module=str(module_path),
            runtime=str(runtime_path),
        ) from error
    finally:
        if sys.path and sys.path[0] == str(module_path.parent):
            sys.path.pop(0)
        if dll_directory is not None:
            dll_directory.close()
    return _LegacyMvsAdapter(module)


class HikvisionCamera(CameraBackend):
    """Optional Hikvision MVS backend with lazy SDK loading."""

    def __init__(
        self,
        *,
        sdk_adapter: _SdkAdapter | None = None,
        mvs_root: str | Path | None = None,
        search_roots: Iterable[str | Path] | None = None,
        device_index: int = 0,
        timeout_ms: int = 1000,
    ) -> None:
        if device_index < 0:
            raise ValueError("device_index must be non-negative")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self._adapter = sdk_adapter
        self.mvs_root = mvs_root
        self.search_roots = search_roots
        self.device_index = int(device_index)
        self.timeout_ms = int(timeout_ms)
        self._is_open = False
        self._adapter_closed = False
        self._sequence_id = 0

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> None:
        if self._is_open:
            return
        if self._adapter is None:
            self._adapter = _load_mvs_adapter(
                mvs_root=self.mvs_root,
                search_roots=self.search_roots,
            )
        self._adapter_closed = False
        try:
            devices = self._adapter.enumerate_devices()
            if not devices:
                raise CameraUnavailableError(
                    "海康 MVS 未枚举到相机，请检查供电、网线/USB 和驱动"
                )
            if self.device_index >= len(devices):
                raise CameraUnavailableError(
                    f"海康相机索引 {self.device_index} 不存在",
                    device_index=self.device_index,
                    device_count=len(devices),
                )
            self._adapter.open_device(self.device_index)
            self._adapter.start_grabbing()
        except CameraUnavailableError:
            self._cleanup_after_failed_open()
            raise
        except Exception as error:
            self._cleanup_after_failed_open()
            raise CameraUnavailableError(
                f"海康 MVS 相机打开失败：{error}",
                device_index=self.device_index,
            ) from error
        self._is_open = True

    def read(self, timeout_s: float = 1.0) -> Frame:
        if not self._is_open:
            self.open()
        assert self._adapter is not None
        timeout_ms = (
            self.timeout_ms
            if timeout_s == 1.0
            else max(1, int(round(float(timeout_s) * 1000.0)))
        )
        try:
            packet = self._adapter.read_frame(timeout_ms)
        except TimeoutError as error:
            raise FrameTimeoutError(
                "海康相机在规定时间内未返回图像",
                timeout_ms=timeout_ms,
            ) from error
        except FrameFormatError:
            raise
        except Exception as error:
            raise CameraUnavailableError(
                f"海康 MVS 取帧失败：{error}",
                timeout_ms=timeout_ms,
            ) from error

        width = int(packet.width)
        height = int(packet.height)
        if width <= 0 or height <= 0:
            raise FrameFormatError(
                "Hikvision frame has invalid dimensions",
                width=width,
                height=height,
            )
        data = bytes(packet.data)
        expected = width * height * 3
        if len(data) != expected:
            raise FrameFormatError(
                (
                    f"Hikvision frame buffer has {len(data)} bytes; "
                    f"expected {expected}"
                ),
                actual_bytes=len(data),
                expected_bytes=expected,
            )
        image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        pixel_format = str(packet.pixel_format).upper()
        if pixel_format == "RGB":
            image_bgr = image[:, :, ::-1].copy()
        elif pixel_format == "BGR":
            image_bgr = image.copy()
        else:
            raise FrameFormatError(
                f"Unsupported Hikvision pixel format: {packet.pixel_format}",
                pixel_format=packet.pixel_format,
            )
        frame = Frame(
            image_bgr=image_bgr,
            width=width,
            height=height,
            timestamp_s=time.time(),
            source="hikvision",
            sequence_id=self._sequence_id,
            metadata={
                "device_index": self.device_index,
                "pixel_format": pixel_format,
                "sdk_frame_id": packet.frame_id,
            },
        )
        self._sequence_id += 1
        return frame

    def close(self) -> None:
        adapter = self._adapter
        if adapter is None or self._adapter_closed:
            self._is_open = False
            return
        try:
            if self._is_open:
                adapter.stop_grabbing()
        finally:
            adapter.close_device()
            self._adapter_closed = True
            self._is_open = False

    def _cleanup_after_failed_open(self) -> None:
        if self._adapter is None or self._adapter_closed:
            return
        try:
            self._adapter.close_device()
        except Exception:
            pass
        self._adapter_closed = True
        self._is_open = False
