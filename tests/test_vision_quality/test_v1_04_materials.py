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
DEFINITION = ROOT / "config/experiments/V1-04.json"
GUIDE = ROOT / "docs/experiments/V1-04.md"
TEMPLATE = ROOT / "student_programs/templates/v1_04_geometry_measurement.py"
CATALOG = ROOT / "config/experiments/catalog.json"


def _parameters():
    return {
        "baseline_profile_id": "standard",
        "allowed_profile_ids": ["standard", "wide_dim", "detail_bright"],
        "camera_path": "/VisionQualityLab/CameraRig/Camera",
        "analysis_focus": "geometry",
        "expected_target_count": 3,
        "expected_shapes": ["circle", "rectangle", "triangle"],
        "vision2d": {
            "profile_id": "standard",
            "roi_px": [180, 180, 335, 340],
            "min_area_ratio": 0.002,
            "max_area_ratio": 0.05,
            "saturation_min": 60,
            "value_min": 40,
            "pixel_scale_mm": [
                1.4433756729740643,
                1.4433756729740643,
            ],
        },
    }


def _context(*, missing_physical=False):
    events = []
    targets = tuple(
        {
            "detection_id": f"det-{index:03d}",
            "shape": shape,
            "perimeter_px": 100.0 + index,
            "area_px2": 900.0 + index,
            "perimeter_mm": None if missing_physical else 145.0 + index,
            "area_mm2": None if missing_physical else 1875.0 + index,
            "circularity": 0.6 + index * 0.1,
            "aspect_ratio": 0.5 + index * 0.1,
        }
        for index, shape in enumerate(
            ("circle", "rectangle", "triangle"),
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


def test_v1_04_definition_publishes_geometry_with_explicit_scale():
    payload = json.loads(DEFINITION.read_text(encoding="utf-8"))

    assert payload["experiment_id"] == "V1-04"
    assert payload["title"] == "边缘长度、周长和面积测量"
    assert payload["public_parameters"] == _parameters()
    assert payload["public_parameters"]["vision2d"][
        "pixel_scale_mm"
    ] is not None
    assert "vision2d.analysis" in payload["capabilities"]
    assert payload["hardware_status"] == "PENDING_HARDWARE"


def test_v1_04_is_registered_after_v1_03_and_loads_strictly():
    catalog = ExperimentCatalog.load(CATALOG, project_root=ROOT)

    start = catalog.ids.index("V1-01")
    assert catalog.ids[start : start + 4] == (
        "V1-01",
        "V1-02",
        "V1-03",
        "V1-04",
    )
    assert catalog.require("V1-04").guide == GUIDE.resolve()


def test_v1_04_guide_has_geometry_evidence_and_boundaries():
    text = GUIDE.read_text(encoding="utf-8")
    for phrase in (
        "实验目标",
        "安全边界",
        "统一入口",
        "学生步骤",
        "边缘长度",
        "周长",
        "面积",
        "圆度",
        "长宽比",
        "原创合成图",
        "五层",
        "automated checks",
        "human checks",
        "不是课程成绩",
        "PENDING_HUMAN_ACCEPTANCE",
        "PENDING_HARDWARE",
        "真实测量精度",
    ):
        assert phrase in text


def test_v1_04_template_uses_public_sdk_and_logs_both_unit_systems():
    source = TEMPLATE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        for node in ast.walk(tree)
    )
    assert "ctx.vision2d.analyze" in source
    assert validate_program(TEMPLATE).ok is True

    context, events = _context()
    runpy.run_path(str(TEMPLATE))["main"](context)

    assert events[-1] == "reset"
    logs = [event[1] for event in events if event[0] == "log"]
    assert len(logs) == 3
    assert all("px" in message and "mm" in message for message in logs)
    assert all("circularity=" in message for message in logs)
    assert all("aspect=" in message for message in logs)


def test_v1_04_template_rejects_missing_physical_geometry_and_resets():
    context, events = _context(missing_physical=True)

    with pytest.raises(RuntimeError, match="毫米周长或面积"):
        runpy.run_path(str(TEMPLATE))["main"](context)

    assert events[-1] == "reset"


@pytest.mark.parametrize("path", [DEFINITION, GUIDE, TEMPLATE, Path(__file__)])
def test_v1_04_materials_have_no_overlong_lines(path):
    assert [
        (number, line)
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        )
        if len(line) > 88
    ] == []
