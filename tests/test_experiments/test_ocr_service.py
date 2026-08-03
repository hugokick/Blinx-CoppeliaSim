from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import vision_platform.experiments.ocr_service as ocr_service
from vision_platform.vision2d.ocr import OCRCharacter, OCRResult
from vision_platform.experiments.ocr_service import (
    OcrServiceError,
    OcrSortingService,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "simulation/vision_ocr_sorting_lab/ocr_assets_manifest.json"


def test_release_default_confidence_gate_is_strict():
    assert ocr_service._default_sort_config()["confidence_min"] == 0.90


def test_service_uses_scene_bound_roi_geometry_correction_without_score_remapping():
    frame, rois = _frame_and_rois()
    x, y, width, height = rois["A2"]
    crop = frame[y : y + height, x : x + width]

    corrected, matrix = ocr_service._prepare_ocr_roi(crop, "A2")

    assert corrected.shape == crop.shape
    assert corrected.dtype == np.uint8
    # The correction is a fixed camera/ROI geometry operation.  It changes
    # pixels before the existing KNN kernel, never the returned confidence or
    # confidence_method semantics.
    assert matrix.shape == (2, 3)
    assert matrix[0, 0] == pytest.approx(1.70, abs=0.02)
    assert matrix[1, 1] == pytest.approx(1.70, abs=0.02)


def _frame_and_rois() -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]:
    import json

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    labels = {
        item["identifier"]: cv2.imdecode(
            np.frombuffer((MANIFEST.parent / item["path"]).read_bytes(), dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        for item in manifest["labels"]
    }
    frame = np.full((240, 320, 3), 255, dtype=np.uint8)
    rois: dict[str, tuple[int, int, int, int]] = {}
    for index, identifier in enumerate(("A1", "A2", "B1", "B2")):
        image = labels[identifier]
        x = 30 + (index % 2) * 140
        y = 30 + (index // 2) * 90
        image_height, image_width = image.shape[:2]
        frame[y : y + image_height, x : x + image_width] = image
        rois[identifier] = (x, y, image_width, image_height)
    return frame, rois


def test_service_trains_once_and_returns_four_whitelisted_results() -> None:
    service = OcrSortingService.from_manifest(
        MANIFEST,
        minimum_accuracy=0.95,
        sort_config={**ocr_service._default_sort_config(), "confidence_min": 0.40},
    )
    frame, rois = _frame_and_rois()
    source = frame.copy()
    output = service.analyze(frame, rois)
    assert service.training_report.held_out_accuracy >= 0.95
    assert [result.text for result in output.results] == ["A1", "A2", "B1", "B2"]
    assert output.plan.status == "PASS"
    assert output.annotated_frame is not frame
    assert np.array_equal(frame, source)
    assert output.annotated_frame.shape == frame.shape
    assert not output.annotated_frame.flags.writeable
    assert not hasattr(output, "classifier")
    for identifier, observation in zip(("A1", "A2", "B1", "B2"), output.observations):
        x, y, width, height = rois[identifier]
        assert observation.roi_px == (x - 24, y - 24, width + 48, height + 48)
        assert observation.result.image_size == (width + 48, height + 48)
        assert all(
            0 <= bx and 0 <= by and bw > 0 and bh > 0
            and bx + bw <= width + 48 and by + bh <= height + 48
            for character in observation.result.characters
            for bx, by, bw, bh in (character.bbox_px,)
        )


def test_service_does_not_retrain_when_analyzing_again(monkeypatch: pytest.MonkeyPatch) -> None:
    import vision_platform.experiments.ocr_service as module

    calls = 0
    original = module.train_glyph_classifier

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "train_glyph_classifier", counted)
    service = OcrSortingService.from_manifest(
        MANIFEST,
        sort_config={**ocr_service._default_sort_config(), "confidence_min": 0.40},
    )
    frame, rois = _frame_and_rois()
    service.analyze(frame, rois)
    service.analyze(frame, rois)
    assert calls == 1


@pytest.mark.parametrize(
    "mutation,code",
    [
        (lambda frame, rois: rois.pop("A1"), "OCR_SERVICE_ROI_INVALID"),
        (lambda frame, rois: rois.__setitem__("C9", rois["A2"]), "OCR_SERVICE_ROI_INVALID"),
        (lambda frame, rois: rois.__setitem__("A1", (310, 0, 64, 96)), "OCR_SERVICE_ROI_INVALID"),
        (lambda frame, rois: frame.__setitem__((slice(30, 126), slice(30, 94)), 255), "OCR_SORT_RESULT_INVALID"),
    ],
)
def test_service_rejects_invalid_capture_before_plan_activation(mutation, code: str) -> None:
    service = OcrSortingService.from_manifest(MANIFEST)
    frame, rois = _frame_and_rois()
    mutation(frame, rois)
    with pytest.raises(OcrServiceError) as exc:
        service.analyze(frame, rois)
    assert exc.value.code == code


def test_service_requires_bgr_uint8_frame() -> None:
    service = OcrSortingService.from_manifest(MANIFEST)
    with pytest.raises(OcrServiceError) as exc:
        service.analyze(np.zeros((96, 64), dtype=np.uint8), {})
    assert exc.value.code == "OCR_SERVICE_FRAME_INVALID"


@pytest.mark.parametrize(
    "bbox",
    [(-1, 28, 30, 40), (2, 28, 120, 40)],
)
def test_service_rejects_out_of_bounds_kernel_geometry_without_plan(
    monkeypatch: pytest.MonkeyPatch, bbox: tuple[int, int, int, int]
) -> None:
    import vision_platform.experiments.ocr_service as module

    service = OcrSortingService.from_manifest(MANIFEST)
    frame, rois = _frame_and_rois()
    original = module.recognize_text

    def malformed(image, model, *, expected_text=None):
        result = original(image, model, expected_text=expected_text)
        character = result.characters[0]
        malformed_character = OCRCharacter(
            character=character.character,
            bbox_px=bbox,
            confidence=character.confidence,
            failure_code=character.failure_code,
            confidence_method=character.confidence_method,
        )
        return OCRResult(
            status=result.status,
            text=result.text,
            characters=(malformed_character, result.characters[1]),
            image_size=result.image_size,
            threshold_method=result.threshold_method,
            character_count=result.character_count,
            failure_code=result.failure_code,
            processing_ms=result.processing_ms,
            schema_version=result.schema_version,
            confidence_method=result.confidence_method,
        )

    monkeypatch.setattr(module, "recognize_text", malformed)
    with pytest.raises(OcrServiceError) as exc:
        service.analyze(frame, rois)
    assert exc.value.code == "OCR_SORT_RESULT_INVALID"


def test_service_rejects_unhashable_scene_part_ids() -> None:
    with pytest.raises(OcrServiceError) as exc:
        OcrSortingService.from_manifest(
            MANIFEST,
            scene_part_ids=[["part_a"], ["part_b"], ["part_c"], ["part_d"]],
        )
    assert exc.value.code == "OCR_SERVICE_SCENE_INVALID"


def test_service_rejects_duplicate_scene_part_ids() -> None:
    with pytest.raises(OcrServiceError) as exc:
        OcrSortingService.from_manifest(
            MANIFEST,
            scene_part_ids=["part_a", "part_b", "part_c", "part_d", "part_c"],
        )
    assert exc.value.code == "OCR_SERVICE_SCENE_INVALID"


def test_service_rejects_non_mapping_sort_config() -> None:
    with pytest.raises(OcrServiceError) as exc:
        OcrSortingService.from_manifest(MANIFEST, sort_config=object())
    assert exc.value.code == "OCR_SERVICE_CONFIG_INVALID"


def test_service_wraps_unexpected_recognizer_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vision_platform.experiments.ocr_service as module

    service = OcrSortingService.from_manifest(MANIFEST)
    frame, rois = _frame_and_rois()

    def broken(*_args, **_kwargs):
        raise RuntimeError("recognizer unavailable")

    monkeypatch.setattr(module, "recognize_text", broken)
    with pytest.raises(OcrServiceError) as exc:
        service.analyze(frame, rois)
    assert exc.value.code == "OCR_SORT_RESULT_INVALID"
