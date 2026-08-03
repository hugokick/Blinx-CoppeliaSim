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
DEFINITION = ROOT / "config/experiments/V1-05.json"
GUIDE = ROOT / "docs/experiments/V1-05.md"
TEMPLATE = ROOT / "student_programs/templates/v1_05_color_shape.py"
CATALOG = ROOT / "config/experiments/catalog.json"


def _parameters():
    return {
        "baseline_profile_id": "standard",
        "allowed_profile_ids": ["standard", "wide_dim", "detail_bright"],
        "camera_path": "/VisionQualityLab/CameraRig/Camera",
        "analysis_focus": "appearance",
        "expected_target_count": 3,
        "expected_shapes": ["circle", "rectangle", "triangle"],
        "expected_colors": ["blue", "red", "green"],
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


def _context(*, unknown=False):
    events = []
    colors = ("unknown", "red", "green") if unknown else (
        "blue",
        "red",
        "green",
    )
    targets = tuple(
        {
            "detection_id": f"det-{index:03d}",
            "color": color,
            "shape": shape,
            "vertex_count": 3 if shape == "triangle" else 4,
            "circularity": 0.6 + index * 0.1,
            "aspect_ratio": 0.5 + index * 0.1,
            "contour_px": ((1, 1), (2, 2), (3, 1)),
            "quality_flags": (),
        }
        for index, (color, shape) in enumerate(
            zip(colors, ("circle", "rectangle", "triangle"), strict=True),
            start=1,
        )
    )

    class Context:
        experiment = SimpleNamespace(
            info=lambda: {"public_parameters": _parameters()}
        )
        camera = SimpleNamespace(
            apply_profile=lambda profile_id: events.append(
                ("apply", profile_id)
            ),
            reset_profile=lambda: events.append("reset"),
        )
        vision2d = SimpleNamespace(
            analyze=lambda: SimpleNamespace(status="PASS", targets=targets)
        )

        def log(self, message):
            events.append(("log", message))

        def checkpoint(self, label):
            events.append(("checkpoint", label))

    return Context(), events


def test_v1_05_definition_publishes_appearance_without_physical_scale():
    payload = json.loads(DEFINITION.read_text(encoding="utf-8"))

    assert payload["experiment_id"] == "V1-05"
    assert payload["title"] == "颜色、形状与轮廓识别"
    assert payload["public_parameters"] == _parameters()
    assert payload["public_parameters"]["vision2d"][
        "pixel_scale_mm"
    ] is None
    assert "vision2d.analysis" in payload["capabilities"]
    assert payload["hardware_status"] == "PENDING_HARDWARE"


def test_v1_05_is_registered_after_v1_04_and_loads_strictly():
    catalog = ExperimentCatalog.load(CATALOG, project_root=ROOT)

    assert catalog.ids[-8:-3] == (
        "V1-01",
        "V1-02",
        "V1-03",
        "V1-04",
        "V1-05",
    )
    assert catalog.ids[-3:] == ("V1-06", "V1-07", "V1-08")
    assert catalog.require("V1-05").student_template == TEMPLATE.resolve()


def test_v1_05_guide_has_appearance_evidence_and_boundaries():
    text = GUIDE.read_text(encoding="utf-8")
    for phrase in (
        "实验目标",
        "安全边界",
        "统一入口",
        "学生步骤",
        "HSV",
        "颜色",
        "形状",
        "轮廓",
        "unknown",
        "原创合成图",
        "五层",
        "automated checks",
        "human checks",
        "不是课程成绩",
        "PENDING_HUMAN_ACCEPTANCE",
        "PENDING_HARDWARE",
        "真实识别准确率",
    ):
        assert phrase in text


def test_v1_05_template_uses_public_sdk_and_preserves_labels():
    source = TEMPLATE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        for node in ast.walk(tree)
    )
    assert "ctx.vision2d.analyze" in source
    assert "vision_platform.recognition" not in source
    assert "unknown" in source
    assert validate_program(TEMPLATE).ok is True

    context, events = _context()
    runpy.run_path(str(TEMPLATE))["main"](context)

    assert events[-1] == "reset"
    logs = [event[1] for event in events if event[0] == "log"]
    assert len(logs) == 3
    assert any("blue/circle" in message for message in logs)
    assert any("red/rectangle" in message for message in logs)
    assert any("green/triangle" in message for message in logs)


def test_v1_05_template_does_not_coerce_unknown_label_and_resets():
    context, events = _context(unknown=True)

    with pytest.raises(RuntimeError, match="unknown"):
        runpy.run_path(str(TEMPLATE))["main"](context)

    assert events[-1] == "reset"


@pytest.mark.parametrize("path", [DEFINITION, GUIDE, TEMPLATE, Path(__file__)])
def test_v1_05_materials_have_no_overlong_lines(path):
    assert [
        (number, line)
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        )
        if len(line) > 88
    ] == []
