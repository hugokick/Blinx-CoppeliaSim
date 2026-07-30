import cv2
import numpy as np

from vision_platform.recognition.color_shape import ColorShapeRecognizer


def test_recognizer_finds_red_square_and_blue_circle():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (80, 100), (180, 200), (0, 0, 255), -1)
    cv2.circle(image, (420, 260), 55, (255, 0, 0), -1)

    detections = ColorShapeRecognizer().detect(image)

    assert [(d.color, d.shape) for d in detections] == [
        ("red", "square"),
        ("blue", "circle"),
    ]
    assert detections[0].center_px == (130.0, 150.0)
    assert detections[1].center_px == (420.0, 260.0)


def test_recognizer_finds_green_triangle_and_yellow_rectangle():
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    triangle = np.array([[100, 220], [160, 100], [220, 220]], dtype=np.int32)
    cv2.fillPoly(image, [triangle], (0, 255, 0))
    cv2.rectangle(image, (340, 250), (520, 330), (0, 255, 255), -1)

    detections = ColorShapeRecognizer().detect(image)

    assert [(d.color, d.shape) for d in detections] == [
        ("green", "triangle"),
        ("yellow", "rectangle"),
    ]


def test_small_noise_is_filtered_using_image_relative_area():
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    cv2.circle(image, (30, 30), 5, (0, 0, 255), -1)
    cv2.rectangle(image, (300, 300), (450, 450), (0, 0, 255), -1)

    detections = ColorShapeRecognizer(min_area_ratio=0.002).detect(image)

    assert len(detections) == 1
    assert detections[0].center_px == (375.0, 375.0)


def test_object_touching_image_border_does_not_raise():
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(image, (0, 30), (90, 130), (255, 0, 0), -1)

    detections = ColorShapeRecognizer().detect(image)

    assert len(detections) == 1
    assert detections[0].color == "blue"


def test_empty_image_returns_empty_list():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    assert ColorShapeRecognizer().detect(image) == []


def test_detections_have_stable_ids_after_spatial_sorting():
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(image, (300, 300), 35, (255, 0, 0), -1)
    cv2.circle(image, (100, 100), 35, (0, 0, 255), -1)

    detections = ColorShapeRecognizer().detect(image)

    assert [item.detection_id for item in detections] == ["det-001", "det-002"]
    assert [item.center_px for item in detections] == [
        (100.0, 100.0),
        (300.0, 300.0),
    ]


def test_annotate_returns_copy_without_modifying_input():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(image, (100, 100), 40, (0, 0, 255), -1)
    recognizer = ColorShapeRecognizer()
    detections = recognizer.detect(image)
    before = image.copy()

    annotated = recognizer.annotate(image, detections)

    assert np.array_equal(image, before)
    assert not np.array_equal(annotated, image)
