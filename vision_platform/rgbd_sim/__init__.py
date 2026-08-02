"""Public boundary for the isolated D1-01 RGB-D simulator adapter."""

from .errors import RgbdSimContractError
from .intrinsics import derive_intrinsics, expected_intrinsics
from .capture import CoppeliaRgbdCapture
from .depth_model import (
    DepthAnchor,
    SourceDepthModelObservation,
    normalize_source_capture,
    observe_source_depth_model,
)
from .models import (
    RgbdSensorMetadata,
    RgbdSimCapture,
    RgbdSourceCapture,
    capture_to_dict,
    source_capture_to_dict,
)
from .preview import colorize_depth, copy_bgr, copy_bgr_preview, depth_summary, depth_to_preview
from .probe import ProbeReport, RoiSummary, build_probe_report, probe_report_to_dict
from .scene_binding import RoiSpec, SceneBinding, bind_scene_manifest, load_scene_binding

__all__ = [
    "RgbdSensorMetadata",
    "RgbdSimCapture",
    "RgbdSimContractError",
    "RgbdSourceCapture",
    "CoppeliaRgbdCapture",
    "DepthAnchor",
    "SourceDepthModelObservation",
    "capture_to_dict",
    "colorize_depth",
    "copy_bgr",
    "copy_bgr_preview",
    "derive_intrinsics",
    "depth_summary",
    "depth_to_preview",
    "ProbeReport",
    "RoiSpec",
    "RoiSummary",
    "SceneBinding",
    "bind_scene_manifest",
    "build_probe_report",
    "expected_intrinsics",
    "normalize_source_capture",
    "observe_source_depth_model",
    "source_capture_to_dict",
    "load_scene_binding",
    "probe_report_to_dict",
]
