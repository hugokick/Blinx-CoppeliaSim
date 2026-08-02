from __future__ import annotations

import cv2
import numpy as np
import pytest

from vision_platform.vision2d.template_matching import (
    TemplateMatchConfig,
    TemplateMatchError,
    annotate_template_match,
    match_template,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, TemplateMatchConfig]:
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    template = np.zeros((24, 32, 3), dtype=np.uint8)
    cv2.rectangle(template, (1, 1), (30, 22), (30, 210, 245), -1)
    cv2.line(template, (3, 20), (28, 4), (255, 255, 255), 2)
    x, y = 221, 205
    image[y : y + template.shape[0], x : x + template.shape[1]] = template
    config = TemplateMatchConfig(
        template_id="fixture",
        template_version="0.0.1",
        threshold=0.72,
        search_roi_px=(0, 0, 512, 512),
    )
    return image, template, config


def test_match_template_returns_original_coordinates_and_center() -> None:
    image, template, config = _fixture()

    result = match_template(image, template, config)

    assert result.status == "MATCHED"
    assert result.matched is True
    assert result.template_id == "fixture"
    assert result.bbox_px == (221, 205, 32, 24)
    assert result.center_px == (237.0, 217.0)
    assert result.image_size == (640, 640)
    assert result.search_roi_px == (0, 0, 512, 512)
    assert result.method == "TM_CCOEFF_NORMED"
    assert result.score == pytest.approx(1.0, abs=1e-6)


def test_match_template_reports_not_matched_when_score_is_below_threshold() -> None:
    image, template, config = _fixture()
    image[:] = (7, 11, 13)

    result = match_template(image, template, config)

    assert result.status == "NOT_MATCHED"
    assert result.matched is False
    assert result.bbox_px is not None
    assert result.score < config.threshold


def test_match_template_uses_smallest_y_then_x_for_equal_scores(monkeypatch) -> None:
    image, template, config = _fixture()
    response = np.zeros((config.search_roi_px[3] - template.shape[0] + 1,
                         config.search_roi_px[2] - template.shape[1] + 1),
                        dtype=np.float32)
    response[4, 10] = 0.9
    response[2, 15] = 0.9
    response[2, 5] = 0.9

    monkeypatch.setattr(cv2, "matchTemplate", lambda *_args, **_kwargs: response)
    result = match_template(image, template, config)

    assert result.bbox_px == (5, 2, 32, 24)


@pytest.mark.parametrize(
    ("image", "template", "config", "code"),
    [
        (np.zeros((10, 10, 3), dtype=np.float32), np.zeros((2, 2, 3), dtype=np.uint8), TemplateMatchConfig(), "VISION_TEMPLATE_INPUT_INVALID"),
        (np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((12, 2, 3), dtype=np.uint8), TemplateMatchConfig(search_roi_px=(0, 0, 10, 10)), "VISION_TEMPLATE_ASSET_INVALID"),
        (np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((2, 2, 3), dtype=np.uint8), TemplateMatchConfig(search_roi_px=(9, 9, 2, 2)), "VISION_TEMPLATE_INPUT_INVALID"),
    ],
)
def test_match_template_uses_stable_error_codes(image, template, config, code) -> None:
    with pytest.raises(TemplateMatchError) as exc_info:
        match_template(image, template, config)

    assert exc_info.value.code == code


def test_annotation_is_a_copy_and_draws_the_match() -> None:
    image, template, config = _fixture()
    result = match_template(image, template, config)

    annotated = annotate_template_match(image, result)

    assert annotated.shape == image.shape
    assert annotated.dtype == np.uint8
    assert np.array_equal(image[205:229, 221:253], template)
    assert not np.array_equal(annotated, image)
