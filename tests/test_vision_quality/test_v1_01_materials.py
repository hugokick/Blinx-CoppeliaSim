from __future__ import annotations

import ast
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vision_platform.experiments.catalog import ExperimentCatalog
from vision_platform.student.protocol import ResponseMessage
from vision_platform.student.sdk import StudentContext
from vision_platform.student.validator import validate_program


ROOT = Path(__file__).resolve().parents[2]
DEFINITION = ROOT / "config" / "experiments" / "V1-01.json"
GUIDE = ROOT / "docs" / "experiments" / "V1-01.md"
TEMPLATE = (
    ROOT / "student_programs" / "templates" / "v1_01_virtual_vision.py"
)
CATALOG = ROOT / "config" / "experiments" / "catalog.json"
PROFILES = ROOT / "simulation" / "vision_quality_lab" / "profiles.json"

EXPECTED_DEFINITION = {
    "schema_version": 1,
    "experiment_id": "V1-01",
    "pack_id": "V1",
    "title": "虚拟视觉系统认知",
    "version": "2.2.0",
    "scene": "simulation/vision_quality_lab/BL23_vision_quality_lab.ttt",
    "scene_manifest": (
        "simulation/vision_quality_lab/scene_manifest.json"
    ),
    "student_template": (
        "student_programs/templates/v1_01_virtual_vision.py"
    ),
    "guide": "docs/experiments/V1-01.md",
    "capabilities": [
        "camera.rgb",
        "camera.profile",
        "lighting.profile",
        "experiment.info",
        "scene.probe",
    ],
    "workspace": {
        "x_mm": [20, 140],
        "y_mm": [-90, 90],
        "z_mm": [10, 140],
        "safe_z_mm": 100,
    },
    "public_parameters": {
        "baseline_profile_id": "standard",
        "allowed_profile_ids": [
            "standard",
            "wide_dim",
            "detail_bright",
        ],
        "comparison_order": [
            "standard",
            "wide_dim",
            "detail_bright",
        ],
        "camera_path": "/VisionQualityLab/CameraRig/Camera",
    },
    "acceptance": {
        "probe_kind": "vision_profile_observation",
        "automated_checks": [
            "profile_readback",
            "three_captures",
            "baseline_reset",
            "evidence_hashes",
        ],
        "human_checks": [
            "学生解释分辨率与视场差异",
            "学生解释光照变化对图像的影响",
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


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _public_profile(profile):
    return {
        key: value
        for key, value in profile.items()
        if key
        in {
            "profile_id",
            "resolution",
            "perspective_angle_deg",
            "camera_rig_z_m",
            "key_diffuse_rgb",
            "fill_diffuse_rgb",
        }
    }


def _png(width: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".png",
        np.zeros((width, width, 3), dtype=np.uint8),
    )
    assert ok
    return encoded.tobytes()


def _real_public_context():
    sent = []
    current_profile = [None]
    sequence = [0]
    profile_payload = _load_strict_json(PROFILES)
    profiles = {
        item["profile_id"]: item for item in profile_payload["profiles"]
    }
    png_by_id = {
        profile_id: _png(profile["resolution"][0])
        for profile_id, profile in profiles.items()
    }

    class ProtocolSpy:
        __slots__ = ()

        def send(self, payload):
            sent.append(payload)

        def recv(self):
            command = sent[-1]
            name = command["name"]
            if name == "experiment.info":
                value = {
                    "experiment_id": "V1-01",
                    "public_parameters": EXPECTED_DEFINITION[
                        "public_parameters"
                    ],
                    "hardware_status": "PENDING_HARDWARE",
                }
            elif name == "camera.profile.apply":
                current_profile[0] = command["args"]["profile_id"]
                value = _public_profile(profiles[current_profile[0]])
            elif name == "camera.capture":
                profile_id = current_profile[0]
                width = profiles[profile_id]["resolution"][0]
                sequence[0] += 1
                value = {
                    "snapshot_id": f"v1-01-{sequence[0]}",
                    "png_bytes": png_by_id[profile_id],
                    "width": width,
                    "height": width,
                    "source": "coppeliasim",
                    "sequence_id": sequence[0],
                }
            elif name == "camera.profile.reset":
                current_profile[0] = profile_payload["baseline_profile_id"]
                value = _public_profile(profiles[current_profile[0]])
            else:
                value = None
            return ResponseMessage(
                command_id=command["command_id"],
                status="PASS",
                value=value,
                error=None,
            ).to_dict()

    return StudentContext(ProtocolSpy()), lambda: tuple(sent)


def _strict_failure_context(*, failure_stage, primary_error, reset_error):
    events = []
    current_profile = [None]
    widths = {"standard": 512, "wide_dim": 256, "detail_bright": 768}
    angles = {"standard": 60.0, "wide_dim": 75.0, "detail_bright": 40.0}
    heights = {"standard": 0.7, "wide_dim": 0.8, "detail_bright": 0.6}

    class StrictExperiment:
        __slots__ = ()

        def info(self):
            return {
                "public_parameters": EXPECTED_DEFINITION[
                    "public_parameters"
                ]
            }

    class StrictCamera:
        __slots__ = ()

        def apply_profile(self, profile_id):
            events.append(("apply", profile_id))
            current_profile[0] = profile_id
            if failure_stage == "apply" and profile_id == "wide_dim":
                raise primary_error
            return SimpleNamespace(
                perspective_angle_deg=angles[profile_id],
                camera_rig_z_m=heights[profile_id],
            )

        def capture(self):
            profile_id = current_profile[0]
            events.append(("capture", profile_id))
            if failure_stage == "capture" and profile_id == "wide_dim":
                raise primary_error
            width = widths[profile_id]
            return SimpleNamespace(width=width, height=width)

        def reset_profile(self):
            events.append(("reset", "standard"))
            if reset_error is not None:
                raise reset_error

    class StrictContext:
        __slots__ = ("camera", "experiment")

        def __init__(self):
            self.camera = StrictCamera()
            self.experiment = StrictExperiment()

        def log(self, message):
            events.append(("log", message))

        def checkpoint(self, label):
            events.append(("checkpoint", label))

    context = StrictContext()
    assert not hasattr(context.camera, "__dict__")
    for leaked_name in ("_WIDTHS", "events", "failure"):
        assert not hasattr(context.camera, leaked_name)
    return context, lambda: tuple(events)


def _template_main():
    namespace = runpy.run_path(str(TEMPLATE))
    return namespace["main"]


def _guide_profile_rows():
    rows = []
    for line in GUIDE.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        resolution = tuple(
            int(component.strip()) for component in cells[1].split("×")
        )
        lights = tuple(
            float(component.strip()) for component in cells[4].split("/")
        )
        rows.append(
            {
                "profile_id": cells[0].strip("`"),
                "resolution": resolution,
                "perspective_angle_deg": float(cells[2].removesuffix("°")),
                "camera_rig_z_m": float(cells[3].removesuffix(" m")),
                "lights": lights,
            }
        )
    return rows


def test_v1_01_definition_is_exact_utf8_json_without_duplicate_keys():
    assert _load_strict_json(DEFINITION) == EXPECTED_DEFINITION


def test_v1_01_definition_loads_through_real_experiment_catalog(tmp_path):
    payload = _load_strict_json(DEFINITION)
    definition = tmp_path / "config" / "experiments" / "V1-01.json"
    definition.parent.mkdir(parents=True)
    definition.write_bytes(DEFINITION.read_bytes())

    dependencies = {
        payload["scene"]: b"placeholder scene",
        payload["scene_manifest"]: b"{}",
        payload["student_template"]: TEMPLATE.read_bytes(),
        payload["guide"]: GUIDE.read_bytes(),
    }
    for relative_path, content in dependencies.items():
        selected = tmp_path / relative_path
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_bytes(content)

    catalog_path = definition.parent / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {"schema_version": 1, "experiments": ["V1-01.json"]}
        ),
        encoding="utf-8",
    )

    catalog = ExperimentCatalog.load(catalog_path, project_root=tmp_path)
    loaded = catalog.require("V1-01")
    assert catalog.ids == ("V1-01",)
    assert loaded.experiment_id == "V1-01"
    assert loaded.capabilities == tuple(payload["capabilities"])
    assert loaded.public_parameters["comparison_order"] == tuple(
        payload["public_parameters"]["comparison_order"]
    )
    assert loaded.hardware_status == "PENDING_HARDWARE"


def test_released_catalog_preserves_r1_then_appends_v1_in_order():
    assert _load_strict_json(CATALOG) == {
        "schema_version": 1,
        "experiments": [
            "R1-01.json",
            "R1-02.json",
            "R1-05.json",
            "R1-06.json",
            "R1-07.json",
            "V1-01.json",
            "V1-02.json",
            "V1-03.json",
            "V1-04.json",
            "V1-05.json",
        ],
    }


def test_guide_profile_table_matches_profiles_and_definition_order():
    definition = _load_strict_json(DEFINITION)
    profile_payload = _load_strict_json(PROFILES)
    profiles = profile_payload["profiles"]
    rows = _guide_profile_rows()

    profile_ids = [profile["profile_id"] for profile in profiles]
    row_ids = [row["profile_id"] for row in rows]
    parameters = definition["public_parameters"]
    assert row_ids == profile_ids
    assert row_ids == parameters["allowed_profile_ids"]
    assert row_ids == parameters["comparison_order"]
    assert parameters["baseline_profile_id"] == profile_payload[
        "baseline_profile_id"
    ]

    for row, profile in zip(rows, profiles, strict=True):
        assert row["resolution"] == tuple(profile["resolution"])
        assert row["perspective_angle_deg"] == profile[
            "perspective_angle_deg"
        ]
        assert row["camera_rig_z_m"] == profile["camera_rig_z_m"]
        assert row["lights"] == (
            profile["key_diffuse_rgb"][0],
            profile["fill_diffuse_rgb"][0],
        )
        assert len(set(profile["key_diffuse_rgb"])) == 1
        assert len(set(profile["fill_diffuse_rgb"])) == 1


def test_v1_01_guide_contains_complete_learning_and_acceptance_boundaries():
    text = GUIDE.read_text(encoding="utf-8")
    for phrase in (
        "目标",
        "安全边界",
        "三档配置",
        "学生步骤",
        "预期证据",
        "问题",
        "automated checks",
        "human checks",
        "分辨率",
        "视场角",
        "相机高度",
        "光照",
        "不是课程成绩",
        "PENDING_HUMAN_ACCEPTANCE",
        "PENDING_HARDWARE",
        "不连接海康相机",
        "不驱动真实机械臂",
        "急停",
        "气路",
        "物理抓取",
        "真实光学精度验收",
    ):
        assert phrase in text


def test_v1_01_guide_documents_gated_unified_entrypoints_without_pass_claim():
    text = GUIDE.read_text(encoding="utf-8")
    normalized = " ".join(text.replace("`", "").split())
    for phrase in (
        "Task 10 正式注册后",
        "powershell -ExecutionPolicy Bypass -File "
        ".\\tools\\vision_lab\\run_pyqt.ps1",
        "“实验目录”选择 V1-01",
        "powershell -ExecutionPolicy Bypass -File "
        ".\\tools\\vision_lab\\python.ps1 -m vision_platform.cli "
        "experiment-run --experiment V1-01 --port 23005",
        "run_experiment.ps1 -Experiment V1-01 -Port 23005",
        "ValidateSet",
        "入口说明不等于在线 PASS",
    ):
        assert phrase in normalized


def test_v1_01_guide_assigns_automated_evidence_to_the_correct_tasks():
    text = GUIDE.read_text(encoding="utf-8")
    for phrase in (
        "Task 8",
        "只验证起始和结束状态均为 `standard`",
        "运行命令和证据包",
        "三档 apply/capture/hash",
        "Task 11 在线测试汇总",
    ):
        assert phrase in text
    assert "正式探针应执行的 automated checks 清单" not in text


def test_v1_01_template_uses_only_the_public_sdk_surface():
    source = TEMPLATE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "main"
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        for node in ast.walk(tree)
    )

    forbidden = {
        "RemoteAPIClient",
        "getObject",
        "setObject",
        "subprocess",
        "socket",
    }
    identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    identifiers.update(
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    )
    assert forbidden.isdisjoint(identifiers)

    allowed_attributes = {
        "ctx.experiment",
        "ctx.experiment.info",
        "ctx.camera",
        "ctx.camera.apply_profile",
        "ctx.camera.capture",
        "ctx.camera.reset_profile",
        "ctx.log",
        "ctx.checkpoint",
        "frame.width",
        "frame.height",
        "state.perspective_angle_deg",
        "state.camera_rig_z_m",
        "primary_error.add_note",
    }
    attribute_paths = {
        _dotted_name(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    assert attribute_paths <= allowed_attributes
    assert not any(
        node.attr.startswith("_")
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    )

    allowed_calls = {
        "RuntimeError",
        "set",
        "tuple",
        "ctx.experiment.info",
        "ctx.camera.apply_profile",
        "ctx.camera.capture",
        "ctx.log",
        "ctx.checkpoint",
        "ctx.camera.reset_profile",
        "primary_error.add_note",
    }
    assert {
        _dotted_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    } <= allowed_calls
    assert validate_program(TEMPLATE).ok is True


def test_v1_01_template_runs_only_public_student_context_commands():
    context, read_commands = _real_public_context()

    _template_main()(context)

    commands = read_commands()
    assert [command["name"] for command in commands] == [
        "experiment.info",
        "camera.profile.apply",
        "camera.capture",
        "context.log",
        "context.checkpoint",
        "camera.profile.apply",
        "camera.capture",
        "context.log",
        "context.checkpoint",
        "camera.profile.apply",
        "camera.capture",
        "context.log",
        "context.checkpoint",
        "camera.profile.reset",
    ]
    assert [
        command["args"]["profile_id"]
        for command in commands
        if command["name"] == "camera.profile.apply"
    ] == ["standard", "wide_dim", "detail_bright"]
    assert [
        command["args"]["message"]
        for command in commands
        if command["name"] == "context.log"
    ] == [
        "standard: 512x512, FOV=60.0 deg, camera_z=0.70 m",
        "wide_dim: 256x256, FOV=75.0 deg, camera_z=0.80 m",
        "detail_bright: 768x768, FOV=40.0 deg, camera_z=0.60 m",
    ]


@pytest.mark.parametrize("failure_stage", ["apply", "capture"])
def test_v1_01_template_preserves_primary_when_reset_also_fails(
    failure_stage,
):
    primary = RuntimeError(f"{failure_stage} failed")
    reset = RuntimeError("reset failed")
    context, read_events = _strict_failure_context(
        failure_stage=failure_stage,
        primary_error=primary,
        reset_error=reset,
    )

    with pytest.raises(RuntimeError) as raised:
        _template_main()(context)

    assert raised.value is primary
    assert read_events()[-1] == ("reset", "standard")
    assert any(
        "reset failed" in note for note in getattr(primary, "__notes__", ())
    )


def test_v1_01_template_propagates_reset_failure_when_no_primary_exists():
    reset = RuntimeError("reset failed without primary")
    context, read_events = _strict_failure_context(
        failure_stage=None,
        primary_error=None,
        reset_error=reset,
    )

    with pytest.raises(RuntimeError) as raised:
        _template_main()(context)

    assert raised.value is reset
    assert read_events()[-1] == ("reset", "standard")


@pytest.mark.parametrize(
    "path",
    [DEFINITION, GUIDE, TEMPLATE, Path(__file__)],
)
def test_task7_materials_have_no_overlong_lines(path):
    overlong = [
        (line_number, line)
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        )
        if len(line) > 88
    ]
    assert overlong == []
