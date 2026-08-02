"""Small deterministic OCR kernel using OpenCV KNN or linear SVM.

The module intentionally accepts caller-supplied glyph samples.  It does not
load fonts, OCR models, files or network resources.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Mapping, Sequence

import cv2
import numpy as np


class TrainingError(ValueError):
    """Raised when a reproducible train/test split cannot be formed."""


@dataclass(frozen=True)
class TrainingReport:
    train_count: int
    test_count: int
    held_out_accuracy: float
    seed: int

    def __post_init__(self) -> None:
        if self.train_count <= 0 or self.test_count <= 0:
            raise ValueError("train/test counts must be positive")
        if not 0.0 <= float(self.held_out_accuracy) <= 1.0:
            raise ValueError("held_out_accuracy must be in [0, 1]")


@dataclass(frozen=True)
class GlyphModel:
    method: str
    labels: tuple[str, ...]
    feature_size: tuple[int, int]
    report: TrainingReport
    classifier: object = field(repr=False, compare=False)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.method not in {"knn", "svm"}:
            raise ValueError("method must be knn or svm")
        if not self.labels or any(not isinstance(label, str) or len(label) != 1 for label in self.labels):
            raise ValueError("labels must be one-character strings")
        if self.feature_size != (20, 20):
            raise ValueError("feature_size must be 20x20")


@dataclass(frozen=True)
class OCRCharacter:
    character: str
    bbox_px: tuple[int, int, int, int]
    confidence: float
    failure_code: str | None = None
    confidence_method: str = "knn_neighbor_distance"

    def __post_init__(self) -> None:
        if len(self.character) != 1:
            raise ValueError("character must contain one symbol")
        if len(self.bbox_px) != 4 or self.bbox_px[2] <= 0 or self.bbox_px[3] <= 0:
            raise ValueError("bbox_px must be positive")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite in [0, 1]")
        if not self.confidence_method:
            raise ValueError("confidence_method must be non-empty")


@dataclass(frozen=True)
class OCRResult:
    status: str
    text: str
    characters: tuple[OCRCharacter, ...]
    image_size: tuple[int, int]
    threshold_method: str
    character_count: int
    failure_code: str | None
    processing_ms: float
    schema_version: int = 1
    confidence_method: str = "knn_neighbor_distance"

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "PARTIAL", "NO_TARGETS", "REJECTED"}:
            raise ValueError("unsupported OCR status")
        if self.character_count != len(self.characters):
            raise ValueError("character_count does not match characters")
        if len(self.image_size) != 2 or any(int(value) <= 0 for value in self.image_size):
            raise ValueError("image_size must be positive")
        if not math.isfinite(float(self.processing_ms)) or self.processing_ms < 0.0:
            raise ValueError("processing_ms must be finite")
        if not self.confidence_method:
            raise ValueError("confidence_method must be non-empty")


def _invalid_result(code: str, image_size: tuple[int, int] = (1, 1)) -> OCRResult:
    return OCRResult(
        status="REJECTED",
        text="",
        characters=(),
        image_size=image_size,
        threshold_method="none",
        character_count=0,
        failure_code=code,
        processing_ms=0.0,
        confidence_method="none",
    )


def _validate_image(image: object) -> tuple[np.ndarray | None, tuple[int, int]]:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        return None, (1, 1)
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        return None, (1, 1)
    return image, (int(image.shape[1]), int(image.shape[0]))


def _gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _normalize_glyph(image: np.ndarray) -> np.ndarray:
    gray = _gray(image)
    if gray.size == 0:
        raise TrainingError("glyph image is empty")
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _threshold, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if float(np.mean(mask > 0)) > 0.65:
        mask = cv2.bitwise_not(mask)
    points = cv2.findNonZero(mask)
    if points is None:
        raise TrainingError("glyph image contains no foreground")
    x, y, width, height = cv2.boundingRect(points)
    crop = mask[y : y + height, x : x + width]
    side = max(width, height) + 4
    square = np.zeros((side, side), dtype=np.uint8)
    left = (side - width) // 2
    top = (side - height) // 2
    square[top : top + height, left : left + width] = crop
    normalized = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
    return (normalized.astype(np.float32) / 255.0).reshape(1, -1)


def _split_samples(
    samples: Mapping[str, Sequence[np.ndarray]],
    *,
    test_fraction: float,
    seed: int,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(samples, Mapping) or len(samples) < 2:
        raise TrainingError("at least two labels are required")
    if isinstance(test_fraction, bool) or not 0.0 < float(test_fraction) < 0.5:
        raise TrainingError("test_fraction must be between 0 and 0.5")
    labels = tuple(sorted(str(label) for label in samples))
    if any(len(label) != 1 for label in labels):
        raise TrainingError("each label must be one character")
    rng = np.random.default_rng(int(seed))
    train_features: list[np.ndarray] = []
    train_targets: list[int] = []
    test_features: list[np.ndarray] = []
    test_targets: list[int] = []
    for class_index, label in enumerate(labels):
        raw_items = list(samples[label])
        if len(raw_items) < 2:
            raise TrainingError("at least two samples are required for every label")
        test_count = max(1, int(round(len(raw_items) * float(test_fraction))))
        if test_count >= len(raw_items):
            raise TrainingError("at least two samples are required for every label")
        order = rng.permutation(len(raw_items))
        test_indices = set(int(value) for value in order[:test_count])
        for index, item in enumerate(raw_items):
            feature = _normalize_glyph(item)
            if index in test_indices:
                test_features.append(feature)
                test_targets.append(class_index)
            else:
                train_features.append(feature)
                train_targets.append(class_index)
    return (
        labels,
        np.vstack(train_features).astype(np.float32),
        np.asarray(train_targets, dtype=np.float32).reshape(-1, 1),
        np.vstack(test_features).astype(np.float32),
        np.asarray(test_targets, dtype=np.float32).reshape(-1, 1),
    )


def train_glyph_classifier(
    samples: Mapping[str, Sequence[np.ndarray]],
    *,
    method: str = "knn",
    test_fraction: float = 0.25,
    seed: int = 20260802,
) -> GlyphModel:
    """Train a deterministic OpenCV KNN or linear SVM on a held-out split."""
    if method not in {"knn", "svm"}:
        raise TrainingError("method must be knn or svm")
    labels, train_x, train_y, test_x, test_y = _split_samples(
        samples,
        test_fraction=test_fraction,
        seed=seed,
    )
    if method == "knn":
        classifier = cv2.ml.KNearest_create()
        classifier.setDefaultK(3)
        classifier.setIsClassifier(True)
        classifier.train(train_x, cv2.ml.ROW_SAMPLE, train_y)
        _retval, predicted, _neighbors, _distances = classifier.findNearest(test_x, k=3)
    else:
        classifier = cv2.ml.SVM_create()
        classifier.setType(cv2.ml.SVM_C_SVC)
        classifier.setKernel(cv2.ml.SVM_LINEAR)
        classifier.setC(2.0)
        classifier.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 200, 1e-5))
        classifier.train(train_x, cv2.ml.ROW_SAMPLE, train_y.astype(np.int32))
        _retval, predicted = classifier.predict(test_x)
    accuracy = float(np.mean(predicted.reshape(-1) == test_y.reshape(-1)))
    report = TrainingReport(
        train_count=int(train_x.shape[0]),
        test_count=int(test_x.shape[0]),
        held_out_accuracy=accuracy,
        seed=int(seed),
    )
    return GlyphModel(
        method=method,
        labels=labels,
        feature_size=(20, 20),
        report=report,
        classifier=classifier,
    )


def _deskew(gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    points = cv2.findNonZero(mask)
    if points is None or len(points) < 8:
        return gray, mask, 0.0
    coordinates = points.reshape(-1, 2).astype(np.float32)
    _mean, _vectors, _values = cv2.PCACompute2(coordinates, mean=None)
    vector = _vectors[0]
    angle = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    if angle > 90.0:
        angle -= 180.0
    if angle < -90.0:
        angle += 180.0
    if abs(angle) < 1.5:
        return gray, mask, 0.0
    center = (gray.shape[1] / 2.0, gray.shape[0] / 2.0)
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    warped_gray = cv2.warpAffine(gray, matrix, gray.shape[::-1], borderValue=255)
    warped_mask = cv2.warpAffine(mask, matrix, gray.shape[::-1], flags=cv2.INTER_NEAREST)
    return warped_gray, warped_mask, -angle


def _projection_splits(component: np.ndarray, x: int, y: int, width: int, height: int) -> list[tuple[int, int, int, int]]:
    local = component[y : y + height, x : x + width]
    projection = np.sum(local > 0, axis=0)
    valleys = projection == 0
    groups: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_valley in enumerate(valleys):
        if is_valley and start is None:
            start = index
        elif not is_valley and start is not None:
            if index - start >= 1:
                groups.append((start, index))
            start = None
    if start is not None:
        groups.append((start, width))
    cuts = [0]
    for left, right in groups:
        if left > 0 and right < width and right - left <= max(3, width // 8):
            cuts.extend([left, right])
    cuts.append(width)
    cuts = sorted(set(cuts))
    boxes = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        if right - left < 3:
            continue
        region = local[:, left:right]
        points = cv2.findNonZero(region)
        if points is None:
            continue
        rx, ry, rw, rh = cv2.boundingRect(points)
        boxes.append((x + left + rx, y + ry, rw, rh))
    if len(boxes) <= 1 and width > int(height * 2.2):
        return []
    return boxes


def _segment_characters(mask: np.ndarray) -> tuple[list[tuple[int, int, int, int]], str | None]:
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(cleaned, 8)
    if count <= 1:
        return [], "NO_TEXT"
    max_height = int(np.max(stats[1:, cv2.CC_STAT_HEIGHT]))
    min_height = max(4, int(round(max_height * 0.35)))
    min_area = max(8, int(round(mask.size * 0.0003)))
    boxes: list[tuple[int, int, int, int]] = []
    stuck = False
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if height < min_height or area < min_area:
            continue
        if width > int(height * 1.7):
            split = _projection_splits(cleaned, x, y, width, height)
            if split:
                boxes.extend(split)
            else:
                stuck = True
        else:
            boxes.append((x, y, width, height))
    boxes.sort(key=lambda box: (box[0], box[1]))
    if not boxes:
        return [], "SEGMENTATION_STUCK" if stuck else "NO_TEXT"
    return boxes, "SEGMENTATION_STUCK" if stuck else None


def _map_bbox_back(
    bbox: tuple[int, int, int, int],
    *,
    angle_deg: float,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    if abs(angle_deg) < 1e-6:
        return bbox
    center = (image_shape[1] / 2.0, image_shape[0] / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    corners = np.asarray(
        [[[x, y], [x + width, y], [x + width, y + height], [x, y + height]]],
        dtype=np.float32,
    )
    mapped = cv2.transform(corners, matrix).reshape(-1, 2)
    min_x, min_y = np.floor(np.min(mapped, axis=0)).astype(int)
    max_x, max_y = np.ceil(np.max(mapped, axis=0)).astype(int)
    return int(min_x), int(min_y), max(1, int(max_x - min_x)), max(1, int(max_y - min_y))


def _svm_pairwise_confidence(
    classifier: object,
    feature: np.ndarray,
    label_count: int,
    predicted_index: int,
) -> float:
    """Convert linear SVM pairwise decision margins into a bounded score."""
    support_vectors = np.asarray(classifier.getSupportVectors(), dtype=np.float32)
    if support_vectors.ndim != 2 or support_vectors.shape[0] == 0:
        raise ValueError("SVM support vectors are unavailable")
    sample = feature.reshape(-1).astype(np.float32)
    votes = np.zeros(label_count, dtype=np.int32)
    target_margins: list[float] = []
    pair_index = 0
    for left, right in combinations(range(label_count), 2):
        rho, alpha, support_indices = classifier.getDecisionFunction(pair_index)
        coefficients = np.asarray(alpha, dtype=np.float64).reshape(-1)
        indices = np.asarray(support_indices, dtype=np.int32).reshape(-1)
        if coefficients.size != indices.size or np.any(indices < 0) or np.any(indices >= support_vectors.shape[0]):
            raise ValueError("SVM decision function support indices are invalid")
        score = float(
            np.sum(coefficients * (support_vectors[indices].astype(np.float64) @ sample.astype(np.float64)))
            - float(rho)
        )
        normalized_margin = abs(score) / (1.0 + abs(score))
        if score >= 0.0:
            votes[left] += 1
        else:
            votes[right] += 1
        if predicted_index in {left, right}:
            target_margins.append(normalized_margin)
        pair_index += 1
    if not target_margins:
        raise ValueError("SVM predicted class has no pairwise margins")
    vote_ratio = float(votes[predicted_index]) / max(1, label_count - 1)
    margin_strength = float(np.mean(target_margins))
    return float(np.clip(0.5 * vote_ratio + 0.5 * margin_strength, 0.0, 1.0))


def _predict(model: GlyphModel, feature: np.ndarray) -> tuple[str, float, str]:
    if model.method == "knn":
        _retval, prediction, _neighbors, distances = model.classifier.findNearest(feature, k=3)
        index = int(round(float(prediction.reshape(-1)[0])))
        mean_distance = float(np.mean(np.sqrt(np.maximum(distances.reshape(-1), 0.0))))
        # Feature distance is accumulated over 400 pixels; map it to a stable
        # bounded score rather than treating a single noisy pixel as failure.
        confidence = float(np.clip(math.exp(-mean_distance / 25.0), 0.0, 1.0))
        confidence_method = "knn_neighbor_distance"
    else:
        _retval, prediction = model.classifier.predict(feature)
        index = int(round(float(prediction.reshape(-1)[0])))
        confidence = _svm_pairwise_confidence(
            model.classifier,
            feature,
            len(model.labels),
            index,
        )
        confidence_method = "svm_pairwise_margin"
    if index < 0 or index >= len(model.labels):
        raise ValueError("classifier returned an unknown label")
    return model.labels[index], confidence, confidence_method


def recognize_text(
    image: object,
    model: GlyphModel | None,
    *,
    expected_text: str | None = None,
) -> OCRResult:
    """Segment and recognize a text line using a previously trained model."""
    started = time.perf_counter()
    validated, image_size = _validate_image(image)
    if validated is None:
        return _invalid_result("INPUT_INVALID", image_size)
    if not isinstance(model, GlyphModel):
        return _invalid_result("MODEL_INVALID", image_size)
    gray = _gray(validated)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    threshold = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )
    if float(np.mean(threshold > 0)) > 0.22:
        return OCRResult(
            status="NO_TARGETS",
            text="",
            characters=(),
            image_size=image_size,
            threshold_method="adaptive_gaussian",
            character_count=0,
            failure_code="SEGMENTATION_NOISY",
            processing_ms=(time.perf_counter() - started) * 1000.0,
        )
    warped_gray, warped_mask, deskew_angle = _deskew(gray, threshold)
    boxes, segmentation_failure = _segment_characters(warped_mask)
    if not boxes:
        failure = segmentation_failure or "NO_TEXT"
        return OCRResult(
            status="NO_TARGETS",
            text="",
            characters=(),
            image_size=image_size,
            threshold_method="adaptive_gaussian",
            character_count=0,
            failure_code=failure,
            processing_ms=(time.perf_counter() - started) * 1000.0,
            confidence_method="none",
        )
    characters: list[OCRCharacter] = []
    confidence_method = "svm_pairwise_margin" if model.method == "svm" else "knn_neighbor_distance"
    for bbox in boxes:
        x, y, width, height = bbox
        crop = warped_gray[max(0, y) : y + height, max(0, x) : x + width]
        try:
            feature = _normalize_glyph(crop)
            character, confidence, character_confidence_method = _predict(model, feature)
        except (TrainingError, cv2.error, ValueError):
            character, confidence, character_confidence_method = "?", 0.0, "none"
        mapped_bbox = _map_bbox_back(
            bbox,
            angle_deg=-deskew_angle,
            image_shape=(gray.shape[0], gray.shape[1]),
        )
        characters.append(
            OCRCharacter(
                character=character,
                bbox_px=mapped_bbox,
                confidence=confidence,
                failure_code="CLASSIFICATION_FAILED" if character == "?" else None,
                confidence_method=character_confidence_method,
            )
        )
    text = "".join(item.character for item in characters)
    low_confidence = any(item.confidence < 0.40 or item.character == "?" for item in characters)
    if expected_text is not None and text != expected_text:
        status = "PARTIAL"
        failure_code = "EXPECTED_TEXT_MISMATCH"
    elif low_confidence:
        status = "PARTIAL"
        failure_code = "LOW_CONFIDENCE"
    elif segmentation_failure is not None:
        status = "PARTIAL"
        failure_code = segmentation_failure
    else:
        status = "PASS"
        failure_code = None
    return OCRResult(
        status=status,
        text=text,
        characters=tuple(characters),
        image_size=image_size,
        threshold_method="adaptive_gaussian",
        character_count=len(characters),
        failure_code=failure_code,
        processing_ms=(time.perf_counter() - started) * 1000.0,
        confidence_method=confidence_method,
    )


__all__ = [
    "GlyphModel",
    "OCRCharacter",
    "OCRResult",
    "TrainingError",
    "TrainingReport",
    "recognize_text",
    "train_glyph_classifier",
]
