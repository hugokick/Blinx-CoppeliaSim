from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_platform.experiments.defect_service import (
    FORMAL_DEFECT_CONFIG,
    DefectServiceError,
    DefectSortingService,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "simulation" / "vision_defect_sorting_lab" / "defect_assets_manifest.json"
ROIS = {
    "reference": (416, 56, 192, 192),
    "entry_a": (80, 320, 192, 192),
    "entry_b": (416, 320, 192, 192),
    "entry_c": (752, 320, 192, 192),
    "entry_d": (80, 648, 192, 192),
    "entry_e": (416, 648, 192, 192),
    "entry_f": (752, 648, 192, 192),
}


def _frame() -> np.ndarray:
    assets = {}
    import json

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for record in manifest["assets"]:
        image = cv2.imread(str(MANIFEST.parent / record["path"]), cv2.IMREAD_COLOR)
        assert image is not None
        assets[record["asset_id"]] = cv2.resize(image, (192, 192), interpolation=cv2.INTER_AREA)
    frame = np.full((1024, 1024, 3), 245, dtype=np.uint8)
    for key, roi in ROIS.items():
        asset_id = "reference" if key == "reference" else key
        x, y, width, height = roi
        frame[y : y + height, x : x + width] = assets[asset_id]
    return frame


def test_service_analyzes_one_fixed_frame_and_freezes_six_decisions() -> None:
    service = DefectSortingService.from_manifest(MANIFEST)
    output = service.analyze(
        _frame(),
        reference_roi=ROIS["reference"],
        candidate_rois={key: value for key, value in ROIS.items() if key != "reference"},
        run_id="run-001",
        frame_id="frame-001",
        scene_sha256="1" * 64,
    )
    assert output.plan.status == "PASS"
    assert [entry.decision for entry in output.plan.entries] == [
        "qualified", "missing", "hole", "foreign", "broken", "dimension"
    ]
    assert output.config == FORMAL_DEFECT_CONFIG
    assert output.reference_crop.flags.writeable is False
    assert len(output.results) == 6
    assert output.annotated_frame.shape == (1024, 1024, 3)


def test_service_rejects_invalid_roi_before_analysis() -> None:
    service = DefectSortingService.from_manifest(MANIFEST)
    rois = {key: value for key, value in ROIS.items() if key != "reference"}
    rois["entry_a"] = (900, 900, 192, 192)
    with pytest.raises(DefectServiceError) as exc:
        service.analyze(_frame(), reference_roi=ROIS["reference"], candidate_rois=rois, run_id="run", frame_id="frame", scene_sha256="1" * 64)
    assert exc.value.code == "DEFECT_SORT_SCENE_INVALID"


def test_service_rejects_second_analysis_in_same_run() -> None:
    service = DefectSortingService.from_manifest(MANIFEST)
    kwargs = dict(reference_roi=ROIS["reference"], candidate_rois={key: value for key, value in ROIS.items() if key != "reference"}, run_id="run-001", frame_id="frame-001", scene_sha256="1" * 64)
    service.analyze(_frame(), **kwargs)
    with pytest.raises(DefectServiceError) as exc:
        service.analyze(_frame(), **kwargs)
    assert exc.value.code == "DEFECT_SORT_PLAN_NOT_ACTIVE"
