from __future__ import annotations

import ast
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from vision_platform.experiments.catalog import ExperimentCatalog
from vision_platform.student.validator import validate_program


ROOT = Path(__file__).resolve().parents[2]
DEFINITION = ROOT / "config" / "experiments" / "V1-03.json"
GUIDE = ROOT / "docs" / "experiments" / "V1-03.md"
TEMPLATE = ROOT / "student_programs/templates/v1_03_pose_measurement.py"
CATALOG = ROOT / "config" / "experiments" / "catalog.json"


def _load(path):
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    assert not text.startswith("\ufeff")
    assert text.encode("utf-8") == raw
    return json.loads(text)


def _expected_parameters():
    return {
        "baseline_profile_id": "standard",
        "allowed_profile_ids": ["standard", "wide_dim", "detail_bright"],
        "camera_path": "/VisionQualityLab/CameraRig/Camera",
        "analysis_focus": "pose",
        "expected_target_count": 3,
        "expected_shapes": ["circle", "rectangle", "triangle"],
        "vision2d": {
            "profile_id": "standard",
            "roi_px": [180, 180, 335, 340],
            "min_area_ratio": 0.002,
            "max_area_ratio": 0.05,
            "saturation_min": 60,
            "value_min": 40,
            "pixel_scale_mm": None,
        },
    }


def _context():
    events = []
    targets = (
        {
            "detection_id": "det-001",
            "shape": "circle",
            "center_px": (214.5, 214.5),
            "rotated_box_px": ((190.0, 214.0),) * 4,
            "angle_deg": None,
            "quality_flags": ("ANGLE_UNDEFINED_FOR_CIRCLE",),
        },
        {
            "detection_id": "det-002",
            "shape": "rectangle",
            "center_px": (296.5, 215.0),
            "rotated_box_px": ((269.0, 201.0),) * 4,
            "angle_deg": 0.0,
            "quality_flags": (),
        },
        {
            "detection_id": "det-003",
            "shape": "triangle",
            "center_px": (296.7, 297.5),
            "rotated_box_px": ((273.0, 282.0),) * 4,
            "angle_deg": 0.0,
            "quality_flags": (),
        },
    )

    class Experiment:
        def info(self):
            events.append("experiment.info")
            return {"public_parameters": _expected_parameters()}

    class Camera:
        def apply_profile(self, profile_id):
            events.append(("camera.apply", profile_id))

        def reset_profile(self):
            events.append("camera.reset")

    class Vision2D:
        def analyze(self):
            events.append("vision2d.analyze")
            return SimpleNamespace(status="PASS", targets=targets)

    class Context:
        experiment = Experiment()
        camera = Camera()
        vision2d = Vision2D()

        def log(self, message):
            events.append(("log", message))

        def checkpoint(self, label):
            events.append(("checkpoint", label))

    return Context(), events


def test_v1_03_definition_publishes_pose_without_physical_scale():
    payload = _load(DEFINITION)

    assert payload["experiment_id"] == "V1-03"
    assert payload["title"] == "物体定位与旋转角测量"
    assert payload["scene"].endswith("BL23_vision_quality_lab.ttt")
    assert payload["public_parameters"] == _expected_parameters()
    assert "vision2d.analysis" in payload["capabilities"]
    assert payload["acceptance"]["automated_checks"]
    assert payload["acceptance"]["human_checks"]
    assert payload["hardware_status"] == "PENDING_HARDWARE"


def test_v1_03_is_registered_after_v1_02_and_loads_strictly():
    catalog = ExperimentCatalog.load(CATALOG, project_root=ROOT)

    start = catalog.ids.index("V1-01")
    assert catalog.ids[start : start + 3] == ("V1-01", "V1-02", "V1-03")
    assert catalog.require("V1-03").student_template == TEMPLATE.resolve()


def test_v1_03_guide_preserves_algorithm_and_acceptance_boundaries():
    text = GUIDE.read_text(encoding="utf-8")
    for phrase in (
        "实验目标",
        "安全边界",
        "统一入口",
        "学生步骤",
        "中心",
        "旋转框",
        "旋转角",
        "ANGLE_UNDEFINED_FOR_CIRCLE",
        "ANGLE_AMBIGUOUS_FOR_SQUARE",
        "原创合成图",
        "五层",
        "automated checks",
        "human checks",
        "不是课程成绩",
        "PENDING_HUMAN_ACCEPTANCE",
        "PENDING_HARDWARE",
        "真实机械臂",
    ):
        assert phrase in text


def test_v1_03_template_uses_public_sdk_and_handles_circle_angle():
    source = TEMPLATE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        for node in ast.walk(tree)
    )
    assert "ctx.vision2d.analyze" in source
    assert "ANGLE_UNDEFINED_FOR_CIRCLE" in source
    assert validate_program(TEMPLATE).ok is True

    context, events = _context()
    runpy.run_path(str(TEMPLATE))["main"](context)

    assert events[:3] == [
        "experiment.info",
        ("camera.apply", "standard"),
        "vision2d.analyze",
    ]
    assert events[-1] == "camera.reset"
    logs = [event[1] for event in events if event[0] == "log"]
    assert len(logs) == 3
    assert "angle=undefined" in logs[0]
    assert all("center=" in message for message in logs)


def test_v1_03_template_rejects_defined_circle_angle_and_resets():
    context, events = _context()
    original_analyze = context.vision2d.analyze

    def invalid_analysis():
        result = original_analyze()
        changed = tuple(dict(target) for target in result.targets)
        changed[0]["angle_deg"] = 0.0
        return SimpleNamespace(status="PASS", targets=changed)

    context.vision2d.analyze = invalid_analysis

    with pytest.raises(RuntimeError, match="圆形角度"):
        runpy.run_path(str(TEMPLATE))["main"](context)

    assert events[-1] == "camera.reset"


@pytest.mark.parametrize("path", [DEFINITION, GUIDE, TEMPLATE, Path(__file__)])
def test_v1_03_materials_have_no_overlong_lines(path):
    assert [
        (number, line)
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        )
        if len(line) > 88
    ] == []
