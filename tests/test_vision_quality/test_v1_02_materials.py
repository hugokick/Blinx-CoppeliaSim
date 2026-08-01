from __future__ import annotations

import ast
import json
from pathlib import Path
import runpy

import pytest

from vision_platform.experiments.catalog import ExperimentCatalog
from vision_platform.student.protocol import ResponseMessage
from vision_platform.student.sdk import StudentContext
from vision_platform.student.validator import validate_program


ROOT = Path(__file__).resolve().parents[2]
DEFINITION = ROOT / "config" / "experiments" / "V1-02.json"
GUIDE = ROOT / "docs" / "experiments" / "V1-02.md"
TEMPLATE = (
    ROOT / "student_programs" / "templates" / "v1_02_size_measurement.py"
)
CATALOG = ROOT / "config" / "experiments" / "catalog.json"

VISION2D_CONFIG = {
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
}
PUBLIC_PARAMETERS = {
    "baseline_profile_id": "standard",
    "allowed_profile_ids": ["standard", "wide_dim", "detail_bright"],
    "camera_path": "/VisionQualityLab/CameraRig/Camera",
    "analysis_focus": "size",
    "expected_target_count": 3,
    "expected_shapes": ["circle", "rectangle", "triangle"],
    "vision2d": VISION2D_CONFIG,
}
EXPECTED_DEFINITION = {
    "schema_version": 1,
    "experiment_id": "V1-02",
    "pack_id": "V1",
    "title": "像素尺寸与构件尺寸测量",
    "version": "2.2.0",
    "scene": "simulation/vision_quality_lab/BL23_vision_quality_lab.ttt",
    "scene_manifest": "simulation/vision_quality_lab/scene_manifest.json",
    "student_template": (
        "student_programs/templates/v1_02_size_measurement.py"
    ),
    "guide": "docs/experiments/V1-02.md",
    "capabilities": [
        "camera.rgb",
        "camera.profile",
        "lighting.profile",
        "vision2d.analysis",
        "experiment.info",
        "scene.probe",
    ],
    "workspace": {
        "x_mm": [20, 140],
        "y_mm": [-90, 90],
        "z_mm": [10, 140],
        "safe_z_mm": 100,
    },
    "public_parameters": PUBLIC_PARAMETERS,
    "acceptance": {
        "probe_kind": "vision_profile_observation",
        "automated_checks": [
            "standard_profile",
            "three_targets",
            "size_fields",
            "five_layer_evidence",
        ],
        "human_checks": [
            "学生解释像素尺寸与比例尺的关系",
            "学生说明测量平面与误差来源",
        ],
    },
    "hardware_status": "PENDING_HARDWARE",
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_strict_json(path: Path):
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    assert not text.startswith("\ufeff")
    assert text.encode("utf-8") == raw
    return json.loads(text, object_pairs_hook=_unique_object)


def _analysis_value():
    targets = []
    for index, (shape, center, size_px, size_mm) in enumerate(
        (
            ("circle", [210.0, 215.0], [44.0, 44.0], [63.5, 63.5]),
            ("rectangle", [296.0, 215.0], [65.0, 30.0], [93.8, 43.3]),
            ("triangle", [298.0, 304.0], [51.0, 44.0], [73.6, 63.5]),
        ),
        start=1,
    ):
        targets.append(
            {
                "detection_id": f"det-{index:03d}",
                "center_px": center,
                "long_side_px": size_px[0],
                "short_side_px": size_px[1],
                "area_px2": 1200.0,
                "size_mm": size_mm,
                "area_mm2": 2500.0,
                "shape": shape,
            }
        )
    return {
        "snapshot_id": "frame-000001",
        "vision_bundle_path": "vision-bundle-V1-02-frame-000001.json",
        "profile_id": "standard",
        "status": "PASS",
        "image_size": [512, 512],
        "targets": targets,
        "rejected_targets": [],
    }


def _public_context():
    sent = []

    class ProtocolSpy:
        __slots__ = ()

        def send(self, payload):
            sent.append(payload)

        def recv(self):
            command = sent[-1]
            name = command["name"]
            if name == "experiment.info":
                value = {
                    "experiment_id": "V1-02",
                    "public_parameters": PUBLIC_PARAMETERS,
                    "hardware_status": "PENDING_HARDWARE",
                }
            elif name in {"camera.profile.apply", "camera.profile.reset"}:
                value = {
                    "profile_id": "standard",
                    "resolution": [512, 512],
                    "perspective_angle_deg": 60.0,
                    "camera_rig_z_m": 0.7,
                    "key_diffuse_rgb": [0.8, 0.8, 0.8],
                    "fill_diffuse_rgb": [0.35, 0.35, 0.35],
                }
            elif name == "vision2d.analyze":
                value = _analysis_value()
            else:
                value = None
            return ResponseMessage(
                command_id=command["command_id"],
                status="PASS",
                value=value,
                error=None,
            ).to_dict()

    return StudentContext(ProtocolSpy()), lambda: tuple(sent)


def test_v1_02_definition_is_exact_strict_utf8_json():
    assert _load_strict_json(DEFINITION) == EXPECTED_DEFINITION


def test_v1_02_loads_through_real_catalog_and_reuses_formal_scene():
    catalog = ExperimentCatalog.load(CATALOG, project_root=ROOT)
    definition = catalog.require("V1-02")

    start = catalog.ids.index("V1-01")
    assert catalog.ids[start : start + 2] == ("V1-01", "V1-02")
    assert definition.scene == (
        ROOT / "simulation/vision_quality_lab/BL23_vision_quality_lab.ttt"
    ).resolve()
    assert definition.public_parameters["vision2d"]["roi_px"] == tuple(
        VISION2D_CONFIG["roi_px"]
    )
    assert definition.hardware_status == "PENDING_HARDWARE"


def test_v1_02_guide_has_complete_course_and_acceptance_boundaries():
    text = GUIDE.read_text(encoding="utf-8")
    for phrase in (
        "实验目标",
        "安全边界",
        "统一入口",
        "学生步骤",
        "五层",
        "像素尺寸",
        "比例尺",
        "测量平面",
        "automated checks",
        "human checks",
        "不是课程成绩",
        "PENDING_HUMAN_ACCEPTANCE",
        "PENDING_HARDWARE",
        "海康相机",
        "真实机械臂",
        "急停",
        "气路",
        "物理抓取",
        "真实测量精度",
    ):
        assert phrase in text


def test_v1_02_template_uses_only_public_sdk_and_validates():
    source = TEMPLATE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        for node in ast.walk(tree)
    )
    assert "ctx.vision2d.analyze" in source
    assert "ctx.camera.apply_profile" in source
    assert "ctx.camera.reset_profile" in source
    for forbidden in (
        "RemoteAPIClient",
        "subprocess",
        "socket",
        "cv2",
        "vision_platform.vision2d",
    ):
        assert forbidden not in source
    assert validate_program(TEMPLATE).ok is True


def test_v1_02_template_runs_exact_public_command_sequence():
    context, read_commands = _public_context()
    main = runpy.run_path(str(TEMPLATE))["main"]

    main(context)

    commands = read_commands()
    assert [command["name"] for command in commands] == [
        "experiment.info",
        "camera.profile.apply",
        "vision2d.analyze",
        "context.log",
        "context.log",
        "context.log",
        "context.checkpoint",
        "camera.profile.reset",
    ]
    assert commands[1]["args"] == {"profile_id": "standard"}
    assert all(
        "px" in command["args"]["message"]
        and "mm" in command["args"]["message"]
        for command in commands
        if command["name"] == "context.log"
    )
    assert commands[-2]["args"]["label"] == "已记录 3 个构件尺寸"


def test_v1_02_template_rejects_missing_physical_measurement_but_resets():
    class Experiment:
        def info(self):
            return {"public_parameters": PUBLIC_PARAMETERS}

    class Camera:
        def __init__(self):
            self.reset_calls = 0

        def apply_profile(self, profile_id):
            assert profile_id == "standard"

        def reset_profile(self):
            self.reset_calls += 1

    class Vision2D:
        def analyze(self):
            value = _analysis_value()
            value["targets"][0]["size_mm"] = None
            return StudentContext(
                type(
                    "Connection",
                    (),
                    {
                        "send": lambda self, payload: setattr(
                            self, "payload", payload
                        ),
                        "recv": lambda self: ResponseMessage(
                            command_id=self.payload["command_id"],
                            status="PASS",
                            value=value,
                            error=None,
                        ).to_dict(),
                    },
                )()
            ).vision2d.analyze()

    context = type(
        "Context",
        (),
        {
            "experiment": Experiment(),
            "camera": Camera(),
            "vision2d": Vision2D(),
            "log": lambda self, message: None,
            "checkpoint": lambda self, label: None,
        },
    )()
    main = runpy.run_path(str(TEMPLATE))["main"]

    with pytest.raises(RuntimeError, match="毫米尺寸"):
        main(context)

    assert context.camera.reset_calls == 1


@pytest.mark.parametrize("path", [DEFINITION, GUIDE, TEMPLATE, Path(__file__)])
def test_v1_02_materials_have_no_overlong_lines(path):
    assert [
        (number, line)
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        )
        if len(line) > 88
    ] == []
