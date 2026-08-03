"""Host-only OCR training and fixed-ROI recognition service for V1-08."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np

from vision_platform.experiments.ocr_assets import (
    EXPECTED_SCENE_ID,
    IDENTIFIERS,
    OcrTrainingAssets,
    load_ocr_assets,
)
from vision_platform.experiments.ocr_sorting import (
    OcrAnalysis,
    OcrObservation,
    OcrSortPlan,
    build_ocr_sort_plan,
    validate_training_report,
)
from vision_platform.vision2d.ocr import OCRResult, TrainingReport, recognize_text, train_glyph_classifier


class OcrServiceError(ValueError):
    """Stable fail-closed error raised before a plan can be activated."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _fail(code: str, message: str) -> OcrServiceError:
    return OcrServiceError(code, message)


def _default_sort_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "expected_count": 4,
        "training_accuracy_min": 0.95,
        "confidence_min": 0.40,
        "safe_z_mm": 110.0,
        "speed_mm_s": 15.0,
        "workspace": {
            "x_mm": [20.0, 140.0],
            "y_mm": [-90.0, 90.0],
            "z_mm": [10.0, 140.0],
            "safe_z_mm": 110.0,
        },
        "routes": [
            {"entry_id": "entry_a", "part_id": "part_a", "identifier": "A1", "route_id": "route_alpha", "pick_xyz_mm": [40.0, -45.0, 18.0], "drop_xyz_mm": [116.0, -60.0, 22.0]},
            {"entry_id": "entry_b", "part_id": "part_b", "identifier": "A2", "route_id": "route_alpha", "pick_xyz_mm": [80.0, -45.0, 18.0], "drop_xyz_mm": [128.0, -60.0, 22.0]},
            {"entry_id": "entry_c", "part_id": "part_c", "identifier": "B1", "route_id": "route_beta", "pick_xyz_mm": [40.0, 5.0, 18.0], "drop_xyz_mm": [116.0, 60.0, 22.0]},
            {"entry_id": "entry_d", "part_id": "part_d", "identifier": "B2", "route_id": "route_beta", "pick_xyz_mm": [80.0, 5.0, 18.0], "drop_xyz_mm": [128.0, 60.0, 22.0]},
        ],
    }


@dataclass(frozen=True)
class OcrServiceResult:
    """Read-only recognition result; the classifier is intentionally absent."""

    results: tuple[OCRResult, ...]
    observations: tuple[OcrObservation, ...]
    plan: OcrSortPlan
    training_report: TrainingReport
    annotated_frame: np.ndarray
    manifest_sha256: str
    scene_id: str

    def __post_init__(self) -> None:
        if type(self.results) is not tuple or len(self.results) != 4:
            raise ValueError("results must contain exactly four OCR results")
        if type(self.observations) is not tuple or len(self.observations) != 4:
            raise ValueError("observations must contain exactly four items")
        image = np.ascontiguousarray(self.annotated_frame).copy()
        image.setflags(write=False)
        object.__setattr__(self, "annotated_frame", image)

    @property
    def status(self) -> str:
        return self.plan.status


class OcrSortingService:
    """Train one private model and recognize one fixed in-memory frame."""

    def __init__(
        self,
        assets: OcrTrainingAssets,
        *,
        minimum_accuracy: float,
        sort_config: Mapping[str, object] | None = None,
        scene_part_ids: Collection[str] | None = None,
    ) -> None:
        if not isinstance(assets, OcrTrainingAssets):
            raise _fail("OCR_SERVICE_ASSET_INVALID", "assets must be loaded by the hash-bound loader")
        if type(minimum_accuracy) not in {int, float}:
            raise _fail("OCR_SERVICE_CONFIG_INVALID", "minimum_accuracy must be a finite number")
        try:
            minimum = float(minimum_accuracy)
        except OverflowError:
            raise _fail("OCR_SERVICE_CONFIG_INVALID", "minimum_accuracy must be finite") from None
        if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
            raise _fail("OCR_SERVICE_CONFIG_INVALID", "minimum_accuracy is outside [0, 1]")
        if sort_config is not None and not isinstance(sort_config, Mapping):
            raise _fail("OCR_SERVICE_CONFIG_INVALID", "sort_config must be a mapping")
        config = dict(_default_sort_config() if sort_config is None else sort_config)
        config["training_accuracy_min"] = minimum
        expected_parts = {"part_a", "part_b", "part_c", "part_d"}
        try:
            raw_parts = tuple(expected_parts if scene_part_ids is None else scene_part_ids)
            if (
                len(raw_parts) != len(expected_parts)
                or any(type(part) is not str for part in raw_parts)
                or len(set(raw_parts)) != len(raw_parts)
                or set(raw_parts) != expected_parts
            ):
                raise _fail("OCR_SERVICE_SCENE_INVALID", "scene part IDs must be the four unique V1-08 parts")
            parts = frozenset(raw_parts)
        except OcrServiceError:
            raise
        except (TypeError, ValueError) as exc:
            raise _fail("OCR_SERVICE_SCENE_INVALID", "scene part IDs must be unique hashable strings") from exc
        try:
            model = train_glyph_classifier(
                assets.samples,
                method=str(assets.training_parameters["method"]),
                test_fraction=float(assets.training_parameters["test_fraction"]),
                seed=int(assets.training_parameters["seed"]),
            )
            report = validate_training_report(model.report, minimum)
        except Exception as exc:
            if isinstance(exc, OcrServiceError):
                raise
            raise _fail("OCR_TRAINING_INVALID", "OCR training failed closed") from exc
        self._assets = assets
        self._minimum_accuracy = minimum
        self._sort_config = config
        self._scene_part_ids = parts
        self._model = model  # private; never included in OcrServiceResult or JSON
        self._training_report = report

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        minimum_accuracy: float = 0.95,
        expected_scene_id: str = EXPECTED_SCENE_ID,
        sort_config: Mapping[str, object] | None = None,
        scene_part_ids: Collection[str] | None = None,
    ) -> "OcrSortingService":
        assets = load_ocr_assets(manifest_path, expected_scene_id=expected_scene_id)
        return cls(
            assets,
            minimum_accuracy=minimum_accuracy,
            sort_config=sort_config,
            scene_part_ids=scene_part_ids,
        )

    @property
    def training_report(self) -> TrainingReport:
        return self._training_report

    @property
    def manifest_sha256(self) -> str:
        return self._assets.manifest_sha256

    @property
    def scene_id(self) -> str:
        return self._assets.scene_id

    def analyze(
        self,
        captured_frame: np.ndarray,
        fixed_rois: Mapping[str, tuple[int, int, int, int]],
    ) -> OcrServiceResult:
        """Recognize all four bound ROIs and build a complete atomic plan."""

        if not isinstance(captured_frame, np.ndarray) or captured_frame.dtype != np.uint8 or captured_frame.ndim != 3 or captured_frame.shape[2] != 3 or captured_frame.size == 0:
            raise _fail("OCR_SERVICE_FRAME_INVALID", "captured_frame must be a non-empty uint8 BGR image")
        if not isinstance(fixed_rois, Mapping) or set(fixed_rois) != set(IDENTIFIERS):
            raise _fail("OCR_SERVICE_ROI_INVALID", "fixed_rois must contain exactly A1, A2, B1 and B2")
        frame_height, frame_width = (int(captured_frame.shape[0]), int(captured_frame.shape[1]))
        observations: list[OcrObservation] = []
        annotated = captured_frame.copy()
        for identifier in IDENTIFIERS:
            roi = fixed_rois[identifier]
            if type(roi) not in {tuple, list} or len(roi) != 4 or any(type(value) is not int for value in roi):
                raise _fail("OCR_SERVICE_ROI_INVALID", f"ROI for {identifier} is invalid")
            x, y, width, height = (int(value) for value in roi)
            if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > frame_width or y + height > frame_height:
                raise _fail("OCR_SERVICE_ROI_INVALID", f"ROI for {identifier} is outside the frame")
            # The caller supplies the fixed inner ROI around the glyph.  The
            # kernel may report a small deskew/interpolation fringe outside
            # that inner rectangle, so recognize against one deterministic
            # host-owned padded crop.  The padded ROI and its local coordinate
            # frame are retained in the observation; no clipping or
            # translation can turn malformed geometry into accepted evidence.
            pad = 8
            if x < pad or y < pad or x + width + pad > frame_width or y + height + pad > frame_height:
                raise _fail("OCR_SERVICE_ROI_INVALID", f"ROI for {identifier} cannot be padded inside the frame")
            expanded_roi = (x - pad, y - pad, width + 2 * pad, height + 2 * pad)
            expanded_x, expanded_y, expanded_width, expanded_height = expanded_roi
            crop = captured_frame[
                expanded_y : expanded_y + expanded_height,
                expanded_x : expanded_x + expanded_width,
            ].copy()
            try:
                result = recognize_text(crop, self._model, expected_text=identifier)
            except OcrServiceError:
                raise
            except Exception as exc:
                raise _fail("OCR_SORT_RESULT_INVALID", f"recognition failed for {identifier}") from exc
            if (
                not isinstance(result, OCRResult)
                or result.status != "PASS"
                or result.text != identifier
                or result.failure_code is not None
                or result.character_count != 2
                or len(result.characters) != 2
                or result.image_size != (expanded_width, expanded_height)
            ):
                raise _fail("OCR_SORT_RESULT_INVALID", f"recognition failed for {identifier}")
            for character in result.characters:
                bbox = getattr(character, "bbox_px", None)
                if (
                    type(bbox) is not tuple
                    or len(bbox) != 4
                    or any(type(value) is not int for value in bbox)
                ):
                    raise _fail("OCR_SORT_RESULT_INVALID", f"recognition geometry is invalid for {identifier}")
                bx, by, bw, bh = bbox
                if (
                    bx < 0
                    or by < 0
                    or bw <= 0
                    or bh <= 0
                    or bx + bw > expanded_width
                    or by + bh > expanded_height
                ):
                    raise _fail("OCR_SORT_RESULT_INVALID", f"recognition geometry is outside {identifier}")
            observation_roi = expanded_roi
            try:
                observation = OcrObservation(identifier=identifier, result=result, roi_px=observation_roi)
            except Exception as exc:
                raise _fail("OCR_SORT_RESULT_INVALID", f"recognition geometry is invalid for {identifier}") from exc
            observations.append(observation)
            cv2.rectangle(annotated, (expanded_x, expanded_y), (expanded_x + expanded_width - 1, expanded_y + expanded_height - 1), (0, 180, 0), 2)
            cv2.putText(annotated, identifier, (expanded_x + 2, max(16, expanded_y + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 0), 1, cv2.LINE_AA)
        analysis = OcrAnalysis(
            training_report=self._training_report,
            observations=tuple(observations),
            image_size=(frame_width, frame_height),
        )
        try:
            plan = build_ocr_sort_plan(
                self._sort_config,
                analysis,
                scene_part_ids=self._scene_part_ids,
            )
        except Exception as exc:
            code = getattr(exc, "code", "OCR_SORT_RESULT_INVALID")
            raise _fail(code, "OCR plan validation failed before activation") from exc
        annotated.setflags(write=False)
        return OcrServiceResult(
            results=tuple(item.result for item in observations),
            observations=tuple(observations),
            plan=plan,
            training_report=self._training_report,
            annotated_frame=annotated,
            manifest_sha256=self._assets.manifest_sha256,
            scene_id=self._assets.scene_id,
        )


__all__ = ["OcrServiceError", "OcrServiceResult", "OcrSortingService"]
