from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from simulation.training_scenes.build_scene import (
    _SEGMENT_POSES,
    _SEGMENTS,
)
from student_programs.templates import (
    r1_01_robot_basics,
    r1_02_teach_points,
    r1_05_visual_stacking,
    r1_06_digit_sort,
    r1_07_component_sort,
    r1_common,
)
from student_programs.templates.r1_common import (
    MIN_VISUAL_CONFIDENCE,
    VisualObject,
    detect_colored_objects,
    detect_digit_objects,
    pick_and_place,
    pixel_to_world,
)
from vision_platform.student.validator import validate_program


ROOT = Path(__file__).resolve().parents[2]
FORMAL_LABEL_ROOT = (
    ROOT / "simulation" / "logistics_lab" / "assets" / "labels"
)
TEMPLATES = (
    "r1_01_robot_basics.py",
    "r1_02_teach_points.py",
    "r1_05_visual_stacking.py",
    "r1_06_digit_sort.py",
    "r1_07_component_sort.py",
)


class FakeRobot:
    def __init__(self, *, pose=(100.0, 0.0, 120.0), fail_move=None):
        self.moves = []
        self.pose_value = pose
        self.pose_calls = 0
        self.fail_move = fail_move
        self.home_calls = 0

    def home(self):
        self.home_calls += 1

    def pose(self):
        self.pose_calls += 1
        return self.pose_value

    def move_world(self, x, y, z, *, speed):
        if len(self.moves) + 1 == self.fail_move:
            raise RuntimeError("primary motion failure")
        self.moves.append((x, y, z, speed))


class FakeTool:
    def __init__(self, *, fail_off=False):
        self.events = []
        self.fail_off = fail_off

    def on(self):
        self.events.append("on")

    def off(self):
        self.events.append("off")
        if self.fail_off:
            raise RuntimeError("cleanup release failure")


class FakeContext:
    def __init__(self, *, robot=None, tool=None, parameters=None):
        self.robot = robot or FakeRobot()
        self.tool = tool or FakeTool()
        self.experiment = FakeExperiment(parameters or {})
        self.camera = FakeCamera()
        self.logs = []
        self.checkpoints = []

    def log(self, message):
        self.logs.append(message)

    def checkpoint(self, label):
        self.checkpoints.append(label)


class FakeExperiment:
    def __init__(self, parameters):
        self.parameters = parameters

    def info(self):
        return {
            "experiment_id": "R1-test",
            "public_parameters": self.parameters,
            "hardware_status": "PENDING_HARDWARE",
        }


class FakeCamera:
    def capture(self):
        return type(
            "Frame",
            (),
            {"image_bgr": np.zeros((40, 40, 3), dtype=np.uint8)},
        )()


def _formal_parameters(experiment_id):
    payload = json.loads(
        (ROOT / "config" / "experiments" / f"{experiment_id}.json")
        .read_text(encoding="utf-8")
    )
    return deepcopy(payload["public_parameters"])


def _visual_object(
    index,
    *,
    color="red",
    shape="block",
    confidence=0.9,
):
    return VisualObject(
        center_px=(float(index), 20.0),
        world_xy_mm=(45.0 + index * 5.0, -55.0),
        color=color,
        shape=shape,
        confidence=confidence,
    )


def _assert_no_robot_or_tool_action(ctx):
    assert ctx.robot.home_calls == 0
    assert ctx.robot.pose_calls == 0
    assert ctx.robot.moves == []
    assert ctx.tool.events == []


def test_all_first_batch_templates_pass_student_validator():
    for name in TEMPLATES:
        result = validate_program(
            ROOT / "student_programs" / "templates" / name
        )
        assert result.ok, (name, result.issues)


def test_pixel_to_world_applies_two_by_three_affine_matrix():
    matrix = [[0.2, 0.0, 35.0], [0.0, -0.3, 70.0]]

    assert pixel_to_world(matrix, (100.0, 50.0)) == (55.0, 55.0)


@pytest.mark.parametrize(
    ("matrix", "center"),
    (
        ([[1.0, 0.0, 0.0]], (1.0, 2.0)),
        ([[1.0, 0.0, np.nan], [0.0, 1.0, 0.0]], (1.0, 2.0)),
        ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], (np.inf, 2.0)),
        ([[True, 0.0, 0.0], [0.0, 1.0, 0.0]], (1.0, 2.0)),
    ),
)
def test_pixel_to_world_rejects_malformed_or_non_finite_values(
    matrix,
    center,
):
    with pytest.raises(ValueError):
        pixel_to_world(matrix, center)


@pytest.mark.parametrize(
    "image",
    (
        np.empty((0, 0, 3), dtype=np.uint8),
        np.zeros((20, 20), dtype=np.uint8),
        np.zeros((20, 20, 4), dtype=np.uint8),
    ),
)
@pytest.mark.parametrize(
    "detector",
    (detect_colored_objects, detect_digit_objects),
)
def test_detectors_reject_empty_or_non_bgr_images(
    detector,
    image,
    tmp_path,
):
    arguments = {
        "calibration_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "pick_x_max_mm": 100.0,
    }
    if detector is detect_digit_objects:
        arguments["reference_dir"] = tmp_path

    with pytest.raises(ValueError):
        detector(image, **arguments)


@pytest.mark.parametrize(
    ("detector", "extra"),
    (
        (detect_colored_objects, {"minimum_area_px": np.nan}),
        (detect_colored_objects, {"pick_x_max_mm": np.inf}),
        (detect_digit_objects, {"pick_x_max_mm": np.inf}),
    ),
)
def test_detectors_reject_non_finite_parameters(detector, extra, tmp_path):
    arguments = {
        "calibration_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "pick_x_max_mm": 100.0,
    }
    arguments.update(extra)
    if detector is detect_digit_objects:
        arguments["reference_dir"] = tmp_path

    with pytest.raises(ValueError):
        detector(np.zeros((40, 40, 3), dtype=np.uint8), **arguments)


def test_colored_detector_distinguishes_four_colors_and_two_shapes():
    image = np.zeros((120, 300, 3), dtype=np.uint8)
    cv2.rectangle(image, (10, 20), (60, 70), (0, 0, 255), -1)
    cv2.rectangle(image, (80, 20), (130, 70), (255, 0, 0), -1)
    cv2.circle(image, (190, 45), 25, (0, 255, 0), -1)
    cv2.circle(image, (260, 45), 25, (0, 255, 255), -1)

    objects = detect_colored_objects(
        image,
        calibration_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        pick_x_max_mm=299.0,
        minimum_area_px=100.0,
    )

    assert [(item.color, item.shape) for item in objects] == [
        ("red", "block"),
        ("blue", "block"),
        ("green", "cylinder"),
        ("yellow", "cylinder"),
    ]


def _write_digit_references(directory, *, shape=(96, 64)):
    directory.mkdir(parents=True, exist_ok=True)
    for digit in (1, 2, 3):
        image = np.full(shape, 255, dtype=np.uint8)
        cv2.putText(
            image,
            str(digit),
            (8, shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            0,
            3,
        )
        assert cv2.imwrite(str(directory / f"{digit}.png"), image)


def _render_scene_digit_boards(scale_px_per_mm):
    plate_width_mm = 20
    plate_height_mm = 30
    margin = 10 * scale_px_per_mm
    plate_width = plate_width_mm * scale_px_per_mm
    plate_height = plate_height_mm * scale_px_per_mm
    canvas = np.zeros(
        (
            plate_height + 2 * margin,
            3 * plate_width + 4 * margin,
            3,
        ),
        dtype=np.uint8,
    )
    center_y = margin + plate_height // 2
    for index, digit in enumerate((1, 2, 3)):
        center_x = margin + plate_width // 2 + index * (
            plate_width + margin
        )
        plate_top_left = (
            center_x - plate_width // 2,
            center_y - plate_height // 2,
        )
        plate_bottom_right = (
            plate_top_left[0] + plate_width - 1,
            plate_top_left[1] + plate_height - 1,
        )
        cv2.rectangle(
            canvas,
            plate_top_left,
            plate_bottom_right,
            (235, 235, 235),
            thickness=-1,
        )
        for segment in _SEGMENTS[digit]:
            offset_mm, size_mm = _SEGMENT_POSES[segment]
            segment_width = size_mm[0] * scale_px_per_mm
            segment_height = size_mm[1] * scale_px_per_mm
            segment_center_x = center_x + offset_mm[0] * scale_px_per_mm
            segment_center_y = center_y - offset_mm[1] * scale_px_per_mm
            top_left = (
                segment_center_x - segment_width // 2,
                segment_center_y - segment_height // 2,
            )
            bottom_right = (
                top_left[0] + segment_width - 1,
                top_left[1] + segment_height - 1,
            )
            cv2.rectangle(
                canvas,
                top_left,
                bottom_right,
                (10, 10, 10),
                thickness=-1,
            )
    return canvas


def test_formal_digit_reference_bytes_and_manifest_hash_stay_frozen():
    expected = {
        "digits/1.png": (
            "75705d3239ff0dc900760ccc155270a7e"
            "d5d7d81827e6b81dbc6f54bfa2b522e"
        ),
        "digits/2.png": (
            "6317e231b0b4c0e8f36080f25fd9594c"
            "49d26eb6ed2ac51a55a9ae6083a99c85"
        ),
        "digits/3.png": (
            "5561eb48523ff7c8f142666e7a8a6ec2"
            "b3a3c783c3b667cda522f76c81b5eb40"
        ),
        "manifest.json": (
            "0f0b42e7f23e21a90bd520b505c4b0da"
            "942fc4b6e9fa2c92b5c4db0fd174d02d"
        ),
    }

    actual = {
        relative: hashlib.sha256(
            (FORMAL_LABEL_ROOT / relative).read_bytes()
        ).hexdigest()
        for relative in expected
    }
    manifest = json.loads(
        (FORMAL_LABEL_ROOT / "manifest.json").read_text(encoding="utf-8")
    )

    assert actual == expected
    assert {
        item["path"]: item["sha256"] for item in manifest["files"]
    } == {key: value for key, value in expected.items() if key != "manifest.json"}


@pytest.mark.parametrize("scale_px_per_mm", (3, 4, 5))
def test_digit_detector_matches_real_scene_geometry_to_formal_references(
    scale_px_per_mm,
):
    image = _render_scene_digit_boards(scale_px_per_mm)

    detected = detect_digit_objects(
        image,
        calibration_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        pick_x_max_mm=float(image.shape[1]),
        reference_dir=FORMAL_LABEL_ROOT / "digits",
    )

    assert [(digit,) for digit, _xy, _confidence in detected] == [
        (1,),
        (2,),
        (3,),
    ]
    assert all(
        confidence >= MIN_VISUAL_CONFIDENCE
        for _digit, _xy, confidence in detected
    )


def test_digit_detector_rejects_reference_with_wrong_dimensions(tmp_path):
    _write_digit_references(tmp_path, shape=(48, 32))
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (50, 70), (255, 255, 255), -1)

    with pytest.raises(RuntimeError, match="64.*96|96.*64"):
        detect_digit_objects(
            image,
            calibration_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            pick_x_max_mm=100.0,
            reference_dir=tmp_path,
        )


def test_digit_detector_rejects_non_finite_template_score(
    tmp_path,
    monkeypatch,
):
    _write_digit_references(tmp_path)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (50, 70), (255, 255, 255), -1)
    monkeypatch.setattr(
        r1_common.cv2,
        "matchTemplate",
        lambda *_args, **_kwargs: np.array([[np.nan]], dtype=np.float32),
    )

    with pytest.raises(RuntimeError, match="置信度"):
        detect_digit_objects(
            image,
            calibration_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            pick_x_max_mm=100.0,
            reference_dir=tmp_path,
        )


@pytest.mark.parametrize(
    ("best_score", "accepted"),
    ((0.499999, False), (0.5, True)),
)
def test_digit_detector_requires_inclusive_minimum_confidence(
    tmp_path,
    monkeypatch,
    best_score,
    accepted,
):
    _write_digit_references(tmp_path)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (50, 70), (255, 255, 255), -1)
    scores = iter((0.1, best_score, 0.2))
    monkeypatch.setattr(
        r1_common.cv2,
        "matchTemplate",
        lambda *_args, **_kwargs: np.array(
            [[next(scores)]],
            dtype=np.float32,
        ),
    )

    if not accepted:
        with pytest.raises(RuntimeError, match="置信度"):
            detect_digit_objects(
                image,
                calibration_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                pick_x_max_mm=100.0,
                reference_dir=tmp_path,
            )
        return
    detected = detect_digit_objects(
        image,
        calibration_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        pick_x_max_mm=100.0,
        reference_dir=tmp_path,
    )
    assert detected[0][2] == pytest.approx(0.5)


def test_pick_and_place_always_lifts_before_horizontal_motion():
    ctx = FakeContext()

    pick_and_place(
        ctx,
        pick_xy=(50.0, -40.0),
        drop_xyz=(118.0, 45.0, 38.0),
        pick_z_mm=20.0,
        safe_z_mm=100.0,
        speed=12.0,
    )

    assert ctx.robot.moves == [
        (50.0, -40.0, 100.0, 12.0),
        (50.0, -40.0, 20.0, 8.0),
        (50.0, -40.0, 100.0, 12.0),
        (118.0, 45.0, 100.0, 12.0),
        (118.0, 45.0, 38.0, 8.0),
        (118.0, 45.0, 100.0, 12.0),
    ]
    assert ctx.tool.events == ["on", "off"]


def test_pick_and_place_lifts_at_current_xy_before_horizontal_motion():
    ctx = FakeContext(robot=FakeRobot(pose=(20.0, 30.0, 40.0)))

    pick_and_place(
        ctx,
        pick_xy=(50.0, -40.0),
        drop_xyz=(118.0, 45.0, 38.0),
        pick_z_mm=20.0,
        safe_z_mm=100.0,
        speed=12.0,
    )

    assert ctx.robot.moves[:2] == [
        (20.0, 30.0, 100.0, 12.0),
        (50.0, -40.0, 100.0, 12.0),
    ]


def test_pick_and_place_releases_tool_without_masking_motion_failure():
    ctx = FakeContext(
        robot=FakeRobot(fail_move=3),
        tool=FakeTool(fail_off=True),
    )

    with pytest.raises(RuntimeError, match="primary motion failure"):
        pick_and_place(
            ctx,
            pick_xy=(50.0, -40.0),
            drop_xyz=(118.0, 45.0, 38.0),
            pick_z_mm=20.0,
            safe_z_mm=100.0,
            speed=12.0,
        )

    assert ctx.tool.events == ["on", "off"]


@pytest.mark.parametrize(
    "overrides",
    (
        {"pick_xy": (np.nan, -40.0)},
        {"drop_xyz": (118.0, 45.0, np.inf)},
        {"safe_z_mm": 20.0},
        {"speed": np.nan},
    ),
)
def test_pick_and_place_rejects_unsafe_parameters_before_robot_access(
    overrides,
):
    ctx = FakeContext()
    arguments = {
        "pick_xy": (50.0, -40.0),
        "drop_xyz": (118.0, 45.0, 38.0),
        "pick_z_mm": 20.0,
        "safe_z_mm": 100.0,
        "speed": 12.0,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError):
        pick_and_place(ctx, **arguments)

    assert ctx.robot.pose_calls == 0
    assert ctx.robot.moves == []
    assert ctx.tool.events == []


@pytest.mark.parametrize(
    "observation",
    (
        (100.0, 0.0),
        (100.0, np.nan, 120.0),
        (100.0, 0.0, np.inf),
    ),
)
def test_r1_01_rejects_invalid_observation_before_home(observation):
    parameters = _formal_parameters("R1-01")
    parameters["observation_pose_mm"] = observation
    ctx = FakeContext(parameters=parameters)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        r1_01_robot_basics.main(ctx)

    _assert_no_robot_or_tool_action(ctx)


@pytest.mark.parametrize(
    "teach_points",
    (
        [[60.0, -50.0, 120.0]] * 3,
        [
            [60.0, -50.0, 120.0],
            [110.0, -50.0, 120.0],
            [110.0, np.nan, 120.0],
            [60.0, 50.0, 120.0],
        ],
    ),
)
def test_r1_02_requires_exactly_four_finite_points_before_home(
    teach_points,
):
    parameters = _formal_parameters("R1-02")
    parameters["teach_points_mm"] = teach_points
    ctx = FakeContext(parameters=parameters)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        r1_02_teach_points.main(ctx)

    _assert_no_robot_or_tool_action(ctx)


@pytest.mark.parametrize(
    ("slot_count", "object_count"),
    ((5, 6), (6, 5), (7, 6), (6, 7)),
)
def test_r1_05_requires_exactly_six_objects_and_slots_before_home(
    slot_count,
    object_count,
    monkeypatch,
):
    parameters = _formal_parameters("R1-05")
    parameters["stack_slots_mm"] = parameters["stack_slots_mm"][:slot_count]
    if slot_count == 7:
        parameters["stack_slots_mm"].append([118.0, 0.0, 74.0])
    objects = [_visual_object(index) for index in range(object_count)]
    monkeypatch.setattr(
        r1_05_visual_stacking,
        "detect_colored_objects",
        lambda *_args, **_kwargs: objects,
    )
    ctx = FakeContext(parameters=parameters)

    with pytest.raises(RuntimeError):
        r1_05_visual_stacking.main(ctx)

    _assert_no_robot_or_tool_action(ctx)


@pytest.mark.parametrize("bad_value", (np.nan, np.inf))
def test_r1_05_rejects_non_finite_plan_before_home(bad_value, monkeypatch):
    parameters = _formal_parameters("R1-05")
    parameters["stack_slots_mm"][5][2] = bad_value
    objects = [_visual_object(index) for index in range(6)]
    monkeypatch.setattr(
        r1_05_visual_stacking,
        "detect_colored_objects",
        lambda *_args, **_kwargs: objects,
    )
    ctx = FakeContext(parameters=parameters)

    with pytest.raises((ValueError, RuntimeError)):
        r1_05_visual_stacking.main(ctx)

    _assert_no_robot_or_tool_action(ctx)


@pytest.mark.parametrize("confidence", (0.0, 0.499999))
@pytest.mark.parametrize(
    "module",
    (
        r1_05_visual_stacking,
        r1_06_digit_sort,
        r1_07_component_sort,
    ),
)
def test_visual_templates_reject_low_confidence_before_any_action(
    module,
    confidence,
    monkeypatch,
):
    if module is r1_05_visual_stacking:
        parameters = _formal_parameters("R1-05")
        detected = [
            _visual_object(
                index,
                confidence=confidence if index == 0 else 0.9,
            )
            for index in range(6)
        ]
        monkeypatch.setattr(
            module,
            "detect_colored_objects",
            lambda *_args, **_kwargs: detected,
        )
    elif module is r1_06_digit_sort:
        parameters = _formal_parameters("R1-06")
        detected = [
            (1, (50.0, -55.0), confidence),
            (2, (75.0, -55.0), 0.9),
            (3, (100.0, -55.0), 0.9),
        ]
        monkeypatch.setattr(
            module,
            "detect_digit_objects",
            lambda *_args, **_kwargs: detected,
        )
    else:
        parameters = _formal_parameters("R1-07")
        detected = _class_objects(confidence=confidence)
        monkeypatch.setattr(
            module,
            "detect_colored_objects",
            lambda *_args, **_kwargs: detected,
        )
    ctx = FakeContext(parameters=parameters)

    with pytest.raises(RuntimeError, match="置信度"):
        module.main(ctx)

    _assert_no_robot_or_tool_action(ctx)


@pytest.mark.parametrize(
    ("order", "slot_count", "detected"),
    (
        ([1, 1, 2], 3, [(1, (50.0, -55.0), 0.9), (1, (60.0, -55.0), 0.8), (2, (70.0, -55.0), 0.9)]),
        ([1, 2, 3], 2, [(1, (50.0, -55.0), 0.9), (2, (75.0, -55.0), 0.9), (3, (100.0, -55.0), 0.9)]),
        ([1, 2, 3], 3, [(1, (50.0, -55.0), 0.9), (1, (60.0, -55.0), 0.8), (2, (75.0, -55.0), 0.9), (3, (100.0, -55.0), 0.9)]),
        ([1, 2, 3], 3, [(1, (50.0, -55.0), 0.9), (2, (75.0, -55.0), 0.9)]),
        ([1, 2, 3], 3, [(1, (50.0, -55.0), np.nan), (2, (75.0, -55.0), 0.9), (3, (100.0, -55.0), 0.9)]),
        ([1, 2, 3], 3, [(1, (50.0, -55.0), 1.1), (2, (75.0, -55.0), 0.9), (3, (100.0, -55.0), 0.9)]),
    ),
)
def test_r1_06_rejects_invalid_order_slots_or_detection_before_home(
    order,
    slot_count,
    detected,
    monkeypatch,
):
    parameters = _formal_parameters("R1-06")
    parameters["order"] = order
    parameters["drop_slots_mm"] = parameters["drop_slots_mm"][:slot_count]
    monkeypatch.setattr(
        r1_06_digit_sort,
        "detect_digit_objects",
        lambda *_args, **_kwargs: detected,
    )
    ctx = FakeContext(parameters=parameters)

    with pytest.raises(RuntimeError):
        r1_06_digit_sort.main(ctx)

    _assert_no_robot_or_tool_action(ctx)


def _class_objects(*, duplicate=False, confidence=0.9):
    objects = [
        _visual_object(0, color="red", shape="block", confidence=confidence),
        _visual_object(1, color="blue", shape="block"),
        _visual_object(2, color="green", shape="cylinder"),
        _visual_object(3, color="yellow", shape="cylinder"),
    ]
    if duplicate:
        objects.append(_visual_object(4, color="red", shape="block"))
    return objects


@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate_expected_class",
        "missing_route",
        "duplicate_detected_class",
        "missing_detected_class",
        "extra_detected_class",
        "non_finite_confidence",
        "out_of_range_confidence",
    ),
)
def test_r1_07_requires_one_to_one_finite_class_routes_before_home(
    mutation,
    monkeypatch,
):
    parameters = _formal_parameters("R1-07")
    objects = _class_objects()
    if mutation == "duplicate_expected_class":
        parameters["classes"].append("red_block")
    elif mutation == "missing_route":
        parameters["drop_poses_mm"].pop("yellow_cylinder")
    elif mutation == "duplicate_detected_class":
        objects = _class_objects(duplicate=True)
    elif mutation == "missing_detected_class":
        objects.pop()
    elif mutation == "extra_detected_class":
        objects.append(_visual_object(4, color="red", shape="cylinder"))
    elif mutation == "non_finite_confidence":
        objects = _class_objects(confidence=np.nan)
    elif mutation == "out_of_range_confidence":
        objects = _class_objects(confidence=1.1)
    monkeypatch.setattr(
        r1_07_component_sort,
        "detect_colored_objects",
        lambda *_args, **_kwargs: objects,
    )
    ctx = FakeContext(parameters=parameters)

    with pytest.raises(RuntimeError):
        r1_07_component_sort.main(ctx)

    _assert_no_robot_or_tool_action(ctx)


def test_all_five_templates_keep_a_runnable_happy_path(monkeypatch):
    contexts = []
    basics = FakeContext(parameters=_formal_parameters("R1-01"))
    r1_01_robot_basics.main(basics)
    contexts.append(basics)
    teach = FakeContext(parameters=_formal_parameters("R1-02"))
    r1_02_teach_points.main(teach)
    contexts.append(teach)

    stack = FakeContext(parameters=_formal_parameters("R1-05"))
    monkeypatch.setattr(
        r1_05_visual_stacking,
        "detect_colored_objects",
        lambda *_args, **_kwargs: [
            _visual_object(index) for index in range(6)
        ],
    )
    r1_05_visual_stacking.main(stack)
    contexts.append(stack)

    digits = FakeContext(parameters=_formal_parameters("R1-06"))
    monkeypatch.setattr(
        r1_06_digit_sort,
        "detect_digit_objects",
        lambda *_args, **_kwargs: [
            (1, (50.0, -55.0), 0.9),
            (2, (75.0, -55.0), 0.9),
            (3, (100.0, -55.0), 0.9),
        ],
    )
    r1_06_digit_sort.main(digits)
    contexts.append(digits)

    classes = FakeContext(parameters=_formal_parameters("R1-07"))
    monkeypatch.setattr(
        r1_07_component_sort,
        "detect_colored_objects",
        lambda *_args, **_kwargs: _class_objects(),
    )
    r1_07_component_sort.main(classes)
    contexts.append(classes)

    assert all(ctx.robot.home_calls == 2 for ctx in contexts)


def test_visual_confidence_threshold_is_inclusive_for_all_templates(
    monkeypatch,
):
    threshold = 0.5
    stack = FakeContext(parameters=_formal_parameters("R1-05"))
    monkeypatch.setattr(
        r1_05_visual_stacking,
        "detect_colored_objects",
        lambda *_args, **_kwargs: [
            _visual_object(index, confidence=threshold) for index in range(6)
        ],
    )
    r1_05_visual_stacking.main(stack)

    digits = FakeContext(parameters=_formal_parameters("R1-06"))
    monkeypatch.setattr(
        r1_06_digit_sort,
        "detect_digit_objects",
        lambda *_args, **_kwargs: [
            (1, (50.0, -55.0), threshold),
            (2, (75.0, -55.0), threshold),
            (3, (100.0, -55.0), threshold),
        ],
    )
    r1_06_digit_sort.main(digits)

    classes = FakeContext(parameters=_formal_parameters("R1-07"))
    monkeypatch.setattr(
        r1_07_component_sort,
        "detect_colored_objects",
        lambda *_args, **_kwargs: _class_objects(confidence=threshold),
    )
    r1_07_component_sort.main(classes)

    assert all(
        ctx.robot.home_calls == 2 for ctx in (stack, digits, classes)
    )


def test_visual_object_is_immutable():
    item = VisualObject(
        center_px=(10.0, 20.0),
        world_xy_mm=(40.0, 50.0),
        color="red",
        shape="block",
        confidence=0.9,
    )

    with pytest.raises(Exception):
        item.color = "blue"
