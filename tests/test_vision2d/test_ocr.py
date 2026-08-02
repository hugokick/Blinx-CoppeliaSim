from __future__ import annotations

import math

import numpy as np
import pytest

from vision_platform.vision2d.ocr import TrainingError, recognize_text, train_glyph_classifier

from .synthetic_v107_v109 import make_glyph, make_glyph_samples, make_text


def test_knn_training_has_reproducible_held_out_report() -> None:
    samples = make_glyph_samples("AB12", count=8, seed=3)

    first = train_glyph_classifier(samples, method="knn", test_fraction=0.25, seed=77)
    second = train_glyph_classifier(samples, method="knn", test_fraction=0.25, seed=77)

    assert first.method == "knn"
    assert first.labels == ("1", "2", "A", "B")
    assert first.report.train_count == 24
    assert first.report.test_count == 8
    assert first.report.train_count + first.report.test_count == 32
    assert first.report.train_count != 32
    assert first.report.seed == 77
    assert math.isfinite(first.report.held_out_accuracy)
    assert first.report == second.report


def test_svm_training_uses_same_isolated_contract() -> None:
    model = train_glyph_classifier(
        make_glyph_samples("AB", count=6, seed=12),
        method="svm",
        test_fraction=0.34,
        seed=9,
    )

    assert model.method == "svm"
    assert model.report.train_count > 0
    assert model.report.test_count > 0
    assert 0.0 <= model.report.held_out_accuracy <= 1.0


def test_svm_confidence_reports_explainable_pairwise_margin() -> None:
    model = train_glyph_classifier(
        make_glyph_samples("AB12", count=10, seed=13),
        method="svm",
        seed=13,
    )
    first = recognize_text(make_text("A", scale=4), model)
    second = recognize_text(make_text("2", scale=4), model)

    assert first.status == "PASS"
    assert second.status == "PASS"
    assert first.confidence_method == "svm_pairwise_margin"
    assert second.confidence_method == "svm_pairwise_margin"
    assert 0.0 <= first.characters[0].confidence <= 1.0
    assert 0.0 <= second.characters[0].confidence <= 1.0
    assert not (
        abs(first.characters[0].confidence - 0.80) < 1e-9
        and abs(second.characters[0].confidence - 0.80) < 1e-9
    )


def test_clean_text_is_recognized_in_order_with_bboxes_and_confidence() -> None:
    model = train_glyph_classifier(make_glyph_samples("AB12", count=10, seed=4), seed=4)
    result = recognize_text(make_text("AB12", scale=4), model, expected_text="AB12")

    assert result.status == "PASS"
    assert result.text == "AB12"
    assert result.character_count == 4
    assert [item.bbox_px[0] for item in result.characters] == sorted(
        item.bbox_px[0] for item in result.characters
    )
    assert all(0.0 <= item.confidence <= 1.0 for item in result.characters)
    assert all(math.isfinite(item.confidence) for item in result.characters)
    assert math.isfinite(result.processing_ms)


def test_text_survives_rotation_brightness_and_seeded_noise() -> None:
    model = train_glyph_classifier(make_glyph_samples("AB12", count=10, seed=5), seed=5)
    image = make_text("A2B1", scale=4, rotate_deg=11, noise_sigma=2.0, brightness=-13)

    result = recognize_text(image, model, expected_text="A2B1")

    assert result.status == "PASS"
    assert result.text == "A2B1"


def test_expected_text_mismatch_is_explicit() -> None:
    model = train_glyph_classifier(make_glyph_samples("AB", count=8, seed=6), seed=6)
    result = recognize_text(make_text("AB", scale=4), model, expected_text="BA")

    assert result.status in {"PARTIAL", "REJECTED"}
    assert result.failure_code == "EXPECTED_TEXT_MISMATCH"


def test_empty_noise_and_stuck_inputs_are_not_silent_success() -> None:
    model = train_glyph_classifier(make_glyph_samples("AB", count=8, seed=7), seed=7)
    empty = recognize_text(np.full((100, 260, 3), 255, dtype=np.uint8), model)
    rng = np.random.default_rng(8)
    noise = recognize_text(rng.integers(0, 256, (100, 260, 3), dtype=np.uint8), model)
    stuck_image = np.full((100, 260, 3), 255, dtype=np.uint8)
    stuck_image[25:75, 45:215] = 0
    stuck = recognize_text(stuck_image, model)

    assert empty.status == "NO_TARGETS"
    assert empty.failure_code == "NO_TEXT"
    assert noise.status in {"NO_TARGETS", "PARTIAL"}
    assert noise.failure_code in {"SEGMENTATION_NOISY", "SEGMENTATION_STUCK", "NO_TEXT"}
    assert stuck.status in {"PARTIAL", "REJECTED", "NO_TARGETS"}
    assert stuck.failure_code in {"SEGMENTATION_STUCK", "NO_TEXT", "LOW_CONFIDENCE"}


def test_broken_stroke_is_reported_as_partial_or_successful_text() -> None:
    model = train_glyph_classifier(make_glyph_samples("AB", count=10, seed=8), seed=8)
    result = recognize_text(make_text("AB", scale=4, broken_index=0), model)

    assert result.status in {"PASS", "PARTIAL"}
    if result.status != "PASS":
        assert result.failure_code in {"OCR_PARTIAL", "LOW_CONFIDENCE", "SEGMENTATION_STUCK"}


def test_invalid_image_model_and_training_samples_have_stable_failures() -> None:
    invalid = recognize_text(np.zeros((24, 24), dtype=np.uint8), None)
    assert invalid.status == "REJECTED"
    assert invalid.failure_code == "INPUT_INVALID"
    invalid_model = recognize_text(np.zeros((24, 24, 3), dtype=np.uint8), None)
    assert invalid_model.status == "REJECTED"
    assert invalid_model.failure_code == "MODEL_INVALID"

    with pytest.raises(TrainingError, match="at least two samples"):
        train_glyph_classifier({"A": [make_glyph("A")], "B": [make_glyph("B")]})
