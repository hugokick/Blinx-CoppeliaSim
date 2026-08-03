"""Fixed-frame V1-09 surface defect analysis service."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np

from vision_platform.experiments.defect_assets import DefectAssets, load_defect_assets
from vision_platform.experiments.defect_sorting import (
    DefectObservation,
    DefectSortPlan,
    build_defect_sort_plan,
)
from vision_platform.vision2d.defect_detection import DefectConfig, DefectResult, detect_surface_defects


FRAME_SIZE = (1024, 1024)
FIXED_ROIS = {
    "reference": (416, 56, 192, 192),
    "entry_a": (80, 320, 192, 192),
    "entry_b": (416, 320, 192, 192),
    "entry_c": (752, 320, 192, 192),
    "entry_d": (80, 648, 192, 192),
    "entry_e": (416, 648, 192, 192),
    "entry_f": (752, 648, 192, 192),
}
_FIXED_ROUTES = (
    ("entry_a", "part_a", "route_qualified", "slot_qualified", (140.0, -16.0, 18.0), (132.0, -93.0, 22.0)),
    ("entry_b", "part_b", "route_missing", "slot_missing", (85.0, -16.0, 18.0), (85.0, -93.0, 22.0)),
    ("entry_c", "part_c", "route_hole", "slot_hole", (30.0, -16.0, 18.0), (38.0, -93.0, 22.0)),
    ("entry_d", "part_d", "route_foreign", "slot_foreign", (140.0, 38.0, 18.0), (132.0, 75.0, 22.0)),
    ("entry_e", "part_e", "route_broken", "slot_broken", (85.0, 38.0, 18.0), (85.0, 75.0, 22.0)),
    ("entry_f", "part_f", "route_dimension", "slot_dimension", (30.0, 38.0, 18.0), (38.0, 75.0, 22.0)),
)
FORMAL_DEFECT_CONFIG = DefectConfig(
    missing_ratio=0.01,
    hole_ratio=0.005,
    foreign_ratio=0.002,
    dimension_ratio=0.08,
    min_component_ratio=0.002,
    morphology_kernel_size=3,
    max_alignment_shift_px=8.0,
    min_contrast=4.0,
)


class DefectServiceError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _fail(code: str, message: str) -> DefectServiceError:
    return DefectServiceError(code, message)


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _crop(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = roi
    return np.ascontiguousarray(frame[y : y + height, x : x + width]).copy()


def _freeze_image(image: np.ndarray) -> np.ndarray:
    copy = np.ascontiguousarray(image).copy()
    copy.setflags(write=False)
    return copy


@dataclass(frozen=True)
class DefectServiceResult:
    results: tuple[DefectResult, ...]
    observations: tuple[DefectObservation, ...]
    plan: DefectSortPlan
    config: DefectConfig
    reference_crop: np.ndarray
    candidate_crops: Mapping[str, np.ndarray]
    annotated_frame: np.ndarray
    finding_masks: Mapping[str, np.ndarray]
    manifest_sha256: str
    scene_id: str
    run_id: str
    frame_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_crop", _freeze_image(self.reference_crop))
        object.__setattr__(self, "annotated_frame", _freeze_image(self.annotated_frame))
        object.__setattr__(self, "candidate_crops", MappingProxyType({key: _freeze_image(value) for key, value in self.candidate_crops.items()}))
        object.__setattr__(self, "finding_masks", MappingProxyType({key: _freeze_image(value) for key, value in self.finding_masks.items()}))

    @property
    def status(self) -> str:
        return self.plan.status


class DefectSortingService:
    def __init__(self, assets: DefectAssets, *, scene_id: str = "V1-09") -> None:
        if not isinstance(assets, DefectAssets):
            raise _fail("DEFECT_SORT_ASSET_INVALID", "assets must come from the hash-bound loader")
        self.assets = assets
        self.scene_id = scene_id
        self._runs: set[str] = set()

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, *, expected_scene_id: str = "V1-09") -> "DefectSortingService":
        return cls(load_defect_assets(manifest_path, expected_scene_id=expected_scene_id), scene_id=expected_scene_id)

    @property
    def manifest_sha256(self) -> str:
        return self.assets.manifest_sha256

    @staticmethod
    def _validate_roi(name: str, roi: Any, frame_size: tuple[int, int]) -> tuple[int, int, int, int]:
        if type(roi) not in {tuple, list} or len(roi) != 4 or any(type(value) is not int for value in roi):
            raise _fail("DEFECT_SORT_SCENE_INVALID", f"ROI {name} is invalid")
        x, y, width, height = roi
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > frame_size[0] or y + height > frame_size[1]:
            raise _fail("DEFECT_SORT_SCENE_INVALID", f"ROI {name} is outside the fixed frame")
        return int(x), int(y), int(width), int(height)

    def analyze(
        self,
        captured_frame: np.ndarray,
        *,
        reference_roi: tuple[int, int, int, int] = FIXED_ROIS["reference"],
        candidate_rois: Mapping[str, tuple[int, int, int, int]] = FIXED_ROIS,
        run_id: str,
        frame_id: str,
        scene_sha256: str,
    ) -> DefectServiceResult:
        if not isinstance(captured_frame, np.ndarray) or captured_frame.dtype != np.uint8 or captured_frame.ndim != 3 or captured_frame.shape[2] != 3 or captured_frame.shape[:2] != (FRAME_SIZE[1], FRAME_SIZE[0]):
            raise _fail("DEFECT_SORT_SCENE_INVALID", "captured frame must be 1024x1024 uint8 BGR")
        if type(run_id) is not str or not run_id or type(frame_id) is not str or not frame_id:
            raise _fail("DEFECT_SORT_RUN_MISMATCH", "run/frame identity is invalid")
        if type(scene_sha256) is not str or len(scene_sha256) != 64:
            raise _fail("DEFECT_SORT_SCENE_INVALID", "scene digest is invalid")
        if run_id in self._runs:
            raise _fail("DEFECT_SORT_PLAN_NOT_ACTIVE", "a run may be analyzed only once before reset")
        if not isinstance(candidate_rois, Mapping) or set(candidate_rois) != {f"entry_{letter}" for letter in "abcdef"}:
            raise _fail("DEFECT_SORT_SCENE_INVALID", "candidate ROIs must cover the six fixed entries")
        reference = self._validate_roi("reference", reference_roi, FRAME_SIZE)
        candidates = {key: self._validate_roi(key, value, FRAME_SIZE) for key, value in candidate_rois.items()}
        all_rois = [("reference", reference), *candidates.items()]
        for index, (_name, left) in enumerate(all_rois):
            lx, ly, lw, lh = left
            for _other, right in all_rois[index + 1 :]:
                rx, ry, rw, rh = right
                if not (lx + lw <= rx or rx + rw <= lx or ly + lh <= ry or ry + rh <= ly):
                    raise _fail("DEFECT_SORT_SCENE_INVALID", "fixed ROIs must not overlap")
        reference_crop = _crop(captured_frame, reference)
        observations: list[DefectObservation] = []
        results: list[DefectResult] = []
        crops: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}
        annotated = captured_frame.copy()
        config_payload = {
            "missing_ratio": FORMAL_DEFECT_CONFIG.missing_ratio,
            "hole_ratio": FORMAL_DEFECT_CONFIG.hole_ratio,
            "foreign_ratio": FORMAL_DEFECT_CONFIG.foreign_ratio,
            "dimension_ratio": FORMAL_DEFECT_CONFIG.dimension_ratio,
            "min_component_ratio": FORMAL_DEFECT_CONFIG.min_component_ratio,
            "morphology_kernel_size": FORMAL_DEFECT_CONFIG.morphology_kernel_size,
            "max_alignment_shift_px": FORMAL_DEFECT_CONFIG.max_alignment_shift_px,
            "min_contrast": FORMAL_DEFECT_CONFIG.min_contrast,
        }
        config_hash = _digest(config_payload)
        reference_hash = hashlib.sha256(reference_crop.tobytes()).hexdigest()
        for entry_id in (f"entry_{letter}" for letter in "abcdef"):
            roi = candidates[entry_id]
            candidate = _crop(captured_frame, roi)
            result = detect_surface_defects(reference_crop, candidate, config=FORMAL_DEFECT_CONFIG)
            if result.image_size != (roi[2], roi[3]):
                raise _fail("DEFECT_SORT_ANALYSIS_REJECTED", "detector result dimensions do not match the ROI crop")
            crop_hash = hashlib.sha256(candidate.tobytes()).hexdigest()
            observation = DefectObservation(
                entry_id=entry_id,
                part_id=f"part_{entry_id[-1]}",
                result=result,
                roi_px=roi,
                reference_crop_sha256=reference_hash,
                candidate_crop_sha256=crop_hash,
            )
            observations.append(observation)
            results.append(result)
            crops[entry_id] = candidate
            mask = np.zeros((roi[3], roi[2]), dtype=np.uint8)
            for finding in result.defects:
                bx, by, bw, bh = finding.bbox_px
                if bx >= 0 and by >= 0 and bw > 0 and bh > 0 and bx + bw <= roi[2] and by + bh <= roi[3]:
                    mask[by : by + bh, bx : bx + bw] = 255
            masks[entry_id] = mask
            x, y, width, height = roi
            cv2.rectangle(annotated, (x, y), (x + width - 1, y + height - 1), (0, 180, 0), 2)
            cv2.putText(annotated, entry_id, (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 0), 1, cv2.LINE_AA)
        try:
            plan = build_defect_sort_plan(
                {
                    "schema_version": 1,
                    "expected_count": 6,
                    "image_size": list(FRAME_SIZE),
                    "safe_z_mm": 110.0,
                    "speed_mm_s": 15.0,
                    "workspace": {"x_mm": [20.0, 155.0], "y_mm": [-95.0, 95.0], "z_mm": [10.0, 140.0], "safe_z_mm": 110.0},
                    "routes": [
                        {
                            "entry_id": entry_id,
                            "part_id": part_id,
                            "route_id": route_id,
                            "slot_id": slot_id,
                            "pick_xyz_mm": list(pick_xyz_mm),
                            "drop_xyz_mm": list(drop_xyz_mm),
                        }
                        for entry_id, part_id, route_id, slot_id, pick_xyz_mm, drop_xyz_mm in _FIXED_ROUTES
                    ],
                },
                tuple(observations),
                run_id=run_id,
                frame_id=frame_id,
                scene_sha256=scene_sha256,
                config_sha256=config_hash,
                asset_manifest_sha256=self.assets.manifest_sha256,
            )
        except Exception as exc:
            raise _fail(getattr(exc, "code", "DEFECT_SORT_ANALYSIS_REJECTED"), "defect plan validation failed") from exc
        self._runs.add(run_id)
        return DefectServiceResult(
            results=tuple(results),
            observations=tuple(observations),
            plan=plan,
            config=FORMAL_DEFECT_CONFIG,
            reference_crop=reference_crop,
            candidate_crops=crops,
            annotated_frame=annotated,
            finding_masks=masks,
            manifest_sha256=self.assets.manifest_sha256,
            scene_id=self.scene_id,
            run_id=run_id,
            frame_id=frame_id,
        )


__all__ = ["FIXED_ROIS", "FORMAL_DEFECT_CONFIG", "DefectServiceError", "DefectServiceResult", "DefectSortingService"]
