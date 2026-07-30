from __future__ import annotations

from pathlib import Path

import pytest

from student_programs.templates.r1_common import (
    VisualObject,
    pick_and_place,
    pixel_to_world,
)
from vision_platform.student.validator import validate_program


ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = (
    "r1_01_robot_basics.py",
    "r1_02_teach_points.py",
    "r1_05_visual_stacking.py",
    "r1_06_digit_sort.py",
    "r1_07_component_sort.py",
)


class FakeRobot:
    def __init__(self):
        self.moves = []

    def move_world(self, x, y, z, *, speed):
        self.moves.append((x, y, z, speed))


class FakeTool:
    def __init__(self):
        self.events = []

    def on(self):
        self.events.append("on")

    def off(self):
        self.events.append("off")


class FakeContext:
    def __init__(self):
        self.robot = FakeRobot()
        self.tool = FakeTool()


def test_all_first_batch_templates_pass_student_validator():
    for name in TEMPLATES:
        result = validate_program(
            ROOT / "student_programs" / "templates" / name
        )
        assert result.ok, (name, result.issues)


def test_pixel_to_world_applies_two_by_three_affine_matrix():
    matrix = [[0.2, 0.0, 35.0], [0.0, -0.3, 70.0]]

    assert pixel_to_world(matrix, (100.0, 50.0)) == (55.0, 55.0)


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
