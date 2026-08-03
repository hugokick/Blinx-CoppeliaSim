"""In-memory probe evidence for one bound D1-01 RGB-D capture."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .depth_model import (
    SourceDepthModelObservation,
    _anchor_digest,
    observe_source_depth_model,
)
from .errors import RgbdSimContractError
from .models import RgbdSimCapture, _assert_capture_trusted
from .preview import depth_summary
from .scene_binding import RoiSpec, SceneBinding


@dataclass(frozen=True)
class RoiSummary:
    median_depth_m: float | None
    valid_count: int
    invalid_count: int

    def __post_init__(self) -> None:
        if type(self.valid_count) is not int or type(self.invalid_count) is not int:
            raise RgbdSimContractError("RGBD_SIM_PROBE_INVALID", "ROI counts must be integers")
        if self.valid_count < 0 or self.invalid_count < 0:
            raise RgbdSimContractError("RGBD_SIM_PROBE_INVALID", "ROI counts must be non-negative")
        if self.median_depth_m is not None and (
            type(self.median_depth_m) not in {int, float}
            or not math.isfinite(float(self.median_depth_m))
            or float(self.median_depth_m) <= 0.0
        ):
            raise RgbdSimContractError("RGBD_SIM_PROBE_INVALID", "ROI median must be finite positive")
        object.__setattr__(self, "median_depth_m", None if self.median_depth_m is None else float(self.median_depth_m))


@dataclass(frozen=True)
class ProbeReport:
    status: str
    scene_path: str
    scene_sha256: str
    sensor_path: str
    resolution: tuple[int, int]
    expected_source_depth_model: str
    observed_source_depth_model: str
    output_depth_model: str
    intrinsics: Any
    sequence_id: int
    depth_summary: Mapping[str, Any]
    rois: Mapping[str, RoiSummary]
    source_model_evidence: tuple[Mapping[str, Any], ...]
    failure_code: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "FAIL"}:
            raise RgbdSimContractError("RGBD_SIM_PROBE_INVALID", "probe status is unsupported")
        if type(self.sequence_id) is not int or self.sequence_id < 0:
            raise RgbdSimContractError("RGBD_SIM_PROBE_INVALID", "sequence id is invalid")
        object.__setattr__(self, "resolution", tuple(self.resolution))
        object.__setattr__(self, "rois", MappingProxyType(dict(self.rois)))
        object.__setattr__(self, "depth_summary", MappingProxyType(dict(self.depth_summary)))
        object.__setattr__(self, "source_model_evidence", tuple(dict(item) for item in self.source_model_evidence))


def _roi_summary(depth: np.ndarray, roi: RoiSpec) -> RoiSummary:
    crop = depth[roi.y0 : roi.y1, roi.x0 : roi.x1]
    valid = crop[crop > 0.0]
    return RoiSummary(
        median_depth_m=None if valid.size == 0 else float(np.median(valid)),
        valid_count=int(valid.size),
        invalid_count=int(crop.size - valid.size),
    )


def _intrinsics_to_dict(intrinsics: Any) -> dict[str, float | int | str]:
    return {
        "width_px": int(intrinsics.width_px),
        "height_px": int(intrinsics.height_px),
        "fx_px": float(intrinsics.fx_px),
        "fy_px": float(intrinsics.fy_px),
        "cx_px": float(intrinsics.cx_px),
        "cy_px": float(intrinsics.cy_px),
        "pixel_center_convention": intrinsics.pixel_center_convention,
    }


def _evidence(observation: SourceDepthModelObservation, capture: RgbdSimCapture, binding: SceneBinding) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for anchor in binding.anchors:
        observed = float(capture.source.source_depth_m[anchor.v_px, anchor.u_px])
        rows.append(
            {
                "pixel": [anchor.u_px, anchor.v_px],
                "observed_source_depth_m": observed,
                "expected_optical_z_m": float(anchor.optical_z_m),
                "expected_ray_range_m": float(anchor.ray_range_m),
                "optical_error_m": abs(observed - float(anchor.optical_z_m)),
                "ray_range_error_m": abs(observed - float(anchor.ray_range_m)),
                "classified_model": observation.model,
            }
        )
    return tuple(rows)


def _failure_report(capture: RgbdSimCapture, binding: SceneBinding, code: str, message: str) -> ProbeReport:
    return ProbeReport(
        status="FAIL",
        scene_path=capture.source.metadata.scene_path,
        scene_sha256=capture.source.metadata.scene_sha256,
        sensor_path=capture.source.metadata.sensor_path,
        resolution=capture.source.metadata.resolution,
        expected_source_depth_model=binding.expected_source_depth_model,
        observed_source_depth_model="unknown",
        output_depth_model=capture.output_depth_model,
        intrinsics=capture.intrinsics,
        sequence_id=capture.source.metadata.sequence_id,
        depth_summary=depth_summary(capture.frame.depth_m),
        rois={name: _roi_summary(capture.frame.depth_m, roi) for name, roi in binding.rois.items()},
        source_model_evidence=(),
        failure_code=code,
        message=message,
    )


def build_probe_report(capture: RgbdSimCapture, binding: SceneBinding) -> ProbeReport:
    """Build deterministic evidence and return ``FAIL`` for contract mismatches."""

    _assert_capture_trusted(capture)
    if not isinstance(capture, RgbdSimCapture) or not isinstance(binding, SceneBinding):
        raise RgbdSimContractError("RGBD_SIM_PROBE_INVALID", "capture or scene binding type is invalid")
    if capture._proof_anchor_digest != _anchor_digest(binding.anchors):
        raise RgbdSimContractError(
            "RGBD_SIM_CAPTURE_BINDING_INVALID",
            "capture normalization proof does not match scene anchor contract",
        )
    metadata = capture.source.metadata
    if (
        metadata.scene_path != binding.scene_path
        or metadata.scene_sha256 != binding.scene_sha256
        or metadata.sensor_path != binding.sensor_path
        or metadata.resolution != binding.resolution
        or metadata.expected_source_depth_model != binding.expected_source_depth_model
    ):
        return _failure_report(capture, binding, "RGBD_SIM_PROBE_BINDING_MISMATCH", "capture metadata does not match scene binding")
    try:
        observation = observe_source_depth_model(capture.source, binding.anchors)
    except RgbdSimContractError as exc:
        return _failure_report(capture, binding, exc.code, str(exc))
    evidence = _evidence(observation, capture, binding)
    if observation.model != binding.expected_source_depth_model:
        return ProbeReport(
            status="FAIL",
            scene_path=metadata.scene_path,
            scene_sha256=metadata.scene_sha256,
            sensor_path=metadata.sensor_path,
            resolution=metadata.resolution,
            expected_source_depth_model=binding.expected_source_depth_model,
            observed_source_depth_model=observation.model,
            output_depth_model=capture.output_depth_model,
            intrinsics=capture.intrinsics,
            sequence_id=metadata.sequence_id,
            depth_summary=depth_summary(capture.frame.depth_m),
            rois={name: _roi_summary(capture.frame.depth_m, roi) for name, roi in binding.rois.items()},
            source_model_evidence=evidence,
            failure_code="RGBD_SIM_PROBE_DEPTH_MODEL_MISMATCH",
            message="observed source-depth model differs from manifest declaration",
        )

    summaries = {name: _roi_summary(capture.frame.depth_m, roi) for name, roi in binding.rois.items()}
    if capture.output_depth_model != "optical_z":
        return ProbeReport(
            status="FAIL",
            scene_path=metadata.scene_path,
            scene_sha256=metadata.scene_sha256,
            sensor_path=metadata.sensor_path,
            resolution=metadata.resolution,
            expected_source_depth_model=binding.expected_source_depth_model,
            observed_source_depth_model=observation.model,
            output_depth_model=capture.output_depth_model,
            intrinsics=capture.intrinsics,
            sequence_id=metadata.sequence_id,
            depth_summary=depth_summary(capture.frame.depth_m),
            rois=summaries,
            source_model_evidence=evidence,
            failure_code="RGBD_SIM_PROBE_OUTPUT_MODEL_INVALID",
            message="probe output must be optical_z",
        )

    near = summaries["near_block"].median_depth_m
    far = summaries["far_block"].median_depth_m
    low = summaries["step_low"].median_depth_m
    high = summaries["step_high"].median_depth_m
    if near is None or far is None or high is None or low is None:
        code = "RGBD_SIM_PROBE_DEPTH_ORDER_INVALID"
        message = "depth-order ROI has no valid samples"
    elif not (near + binding.depth_order_margin_m < far and high + binding.depth_order_margin_m < low):
        code = "RGBD_SIM_PROBE_DEPTH_ORDER_INVALID"
        message = "configured depth-order margins are not met"
    else:
        code = None
        message = None
    return ProbeReport(
        status="PASS" if code is None else "FAIL",
        scene_path=metadata.scene_path,
        scene_sha256=metadata.scene_sha256,
        sensor_path=metadata.sensor_path,
        resolution=metadata.resolution,
        expected_source_depth_model=binding.expected_source_depth_model,
        observed_source_depth_model=observation.model,
        output_depth_model=capture.output_depth_model,
        intrinsics=capture.intrinsics,
        sequence_id=metadata.sequence_id,
        depth_summary=depth_summary(capture.frame.depth_m),
        rois=summaries,
        source_model_evidence=evidence,
        failure_code=code,
        message=message,
    )


def probe_report_to_dict(report: ProbeReport) -> dict[str, Any]:
    if not isinstance(report, ProbeReport):
        raise RgbdSimContractError("RGBD_SIM_PROBE_INVALID", "probe report type is invalid")
    return {
        "status": report.status,
        "failure_code": report.failure_code,
        "message": report.message,
        "scene_path": report.scene_path,
        "scene_sha256": report.scene_sha256,
        "sensor_path": report.sensor_path,
        "resolution": [report.resolution[0], report.resolution[1]],
        "expected_source_depth_model": report.expected_source_depth_model,
        "observed_source_depth_model": report.observed_source_depth_model,
        "output_depth_model": report.output_depth_model,
        "intrinsics": _intrinsics_to_dict(report.intrinsics),
        "sequence_id": report.sequence_id,
        "depth_summary": dict(report.depth_summary),
        "rois": {
            name: {
                "median_depth_m": summary.median_depth_m,
                "valid_count": summary.valid_count,
                "invalid_count": summary.invalid_count,
            }
            for name, summary in report.rois.items()
        },
        "source_model_evidence": [dict(item) for item in report.source_model_evidence],
    }


__all__ = [
    "ProbeReport",
    "RoiSummary",
    "build_probe_report",
    "probe_report_to_dict",
]
