from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from vision_platform.experiments.catalog import ExperimentCatalog


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CAPABILITIES = {
    "camera.rgb",
    "camera.profile",
    "lighting.profile",
    "vision2d.ocr_sorting",
}
EXPECTED_IDENTIFIERS = ("A1", "A2", "B1", "B2")
EXPECTED_ENTRIES = ("entry_a", "entry_b", "entry_c", "entry_d")


def _definition() -> dict[str, object]:
    return json.loads(
        (ROOT / "config" / "experiments" / "V1-08.json").read_text(
            encoding="utf-8"
        )
    )


def test_v1_08_definition_is_catalogued_with_fixed_scene_profile_and_assets() -> None:
    catalog = ExperimentCatalog.load(
        ROOT / "config" / "experiments" / "catalog.json",
        project_root=ROOT,
    )
    assert catalog.ids[-1] == "V1-08"
    definition = catalog.require("V1-08")

    payload = _definition()
    assert payload["experiment_id"] == "V1-08"
    assert payload["version"] == "2.2.0"
    assert payload["scene"] == (
        "simulation/vision_ocr_sorting_lab/BL23_vision_ocr_sorting_lab.ttt"
    )
    assert payload["scene_manifest"] == (
        "simulation/vision_ocr_sorting_lab/scene_manifest.json"
    )
    assert payload["student_template"] == (
        "student_programs/templates/v1_08_ocr_sorting.py"
    )
    assert payload["guide"] == "docs/experiments/V1-08.md"
    assert set(definition.capabilities) == EXPECTED_CAPABILITIES
    assert not any(
        capability.startswith(("robot.", "tool."))
        for capability in definition.capabilities
    )
    assert payload["hardware_status"] == "PENDING_HARDWARE"
    assert definition.workspace["safe_z_mm"] == 110

    parameters = payload["public_parameters"]
    assert parameters["baseline_profile_id"] == "standard"
    assert parameters["allowed_profile_ids"] == ["standard"]
    ocr = parameters["ocr_sorting"]
    assert ocr["profile_id"] == "standard"
    assert ocr["image_size_px"] == [1024, 1024]
    assert ocr["training_manifest"] == (
        "simulation/vision_ocr_sorting_lab/ocr_assets_manifest.json"
    )
    manifest_path = ROOT / ocr["training_manifest"]
    assert ocr["training_manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert ocr["expected_identifiers"] == list(EXPECTED_IDENTIFIERS)
    fixed_rois = ocr["fixed_rois_px"]
    assert tuple(fixed_rois) == EXPECTED_IDENTIFIERS
    for roi in fixed_rois.values():
        x, y, width, height = roi
        assert x >= 0 and y >= 0 and width > 0 and height > 0
        assert x + width <= 1024 and y + height <= 1024

    sort_config = ocr["sort_config"]
    assert sort_config["schema_version"] == 1
    assert sort_config["expected_count"] == 4
    routes = sort_config["routes"]
    assert tuple(route["entry_id"] for route in routes) == EXPECTED_ENTRIES
    assert tuple(route["identifier"] for route in routes) == EXPECTED_IDENTIFIERS
    assert {route["route_id"] for route in routes} == {"route_alpha", "route_beta"}
    assert len({route["drop_xyz_mm"][0:2].__str__() for route in routes}) == 4

    acceptance = payload["acceptance"]
    assert acceptance["probe_kind"] == "ocr_sort_occupancy"
    assert {
        "standard_profile",
        "training_manifest_hash",
        "four_identifiers",
        "complete_sort_plan",
        "no_motion_before_plan",
        "four_guarded_transfers",
        "final_bin_occupancy",
        "robot_home",
        "tool_off",
        "same_run_evidence",
    } <= set(acceptance["automated_checks"])
    assert acceptance["human_checks"]


def test_v1_08_template_calls_host_commands_only_in_safe_order() -> None:
    payload = _definition()
    source = (ROOT / payload["student_template"]).read_text(encoding="utf-8")
    tree = ast.parse(source)

    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.append(node.func.attr)
    assert calls[0] == "ocr_sorting"
    assert calls.count("ocr_sorting") == 1
    # One call site inside the checked four-entry loop is the intended template
    # shape; runtime execution invokes it once per approved entry.
    assert calls.count("sort_ocr_entry") == 1
    assert "entry_id" in source
    assert all(token not in source for token in ("move_world", "home(", "tool.", "robot."))
    assert all(token not in source for token in ("pick_xyz", "drop_xyz", "world", "subprocess", "os.system"))
    for entry_id in EXPECTED_ENTRIES:
        assert entry_id in source
    assert "docs/experiments/V1-08.md" in source or "V1-08.md" in source


def test_v1_08_template_executes_against_typed_sdk_results() -> None:
    payload = _definition()
    source = (ROOT / payload["student_template"]).read_text(encoding="utf-8")
    namespace: dict[str, object] = {}
    exec(compile(source, str(ROOT / payload["student_template"]), "exec"), namespace)

    class Vision:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def ocr_sorting(self):
            self.calls.append("ocr_sorting")
            return type(
                "TypedRecognition",
                (),
                {
                    "status": "PASS",
                    "entries": tuple(
                        type(
                            "TypedEntry",
                            (),
                            {
                                "entry_id": entry_id,
                                "status": "APPROVED",
                            },
                        )()
                        for entry_id in EXPECTED_ENTRIES
                    ),
                },
            )()

        def sort_ocr_entry(self, entry_id: str):
            self.calls.append(entry_id)
            return type("TypedReceipt", (), {"status": "COMPLETED"})()

    class Context:
        def __init__(self) -> None:
            self.vision2d = Vision()
            self.messages: list[str] = []

        def log(self, message: str) -> None:
            self.messages.append(message)

        def checkpoint(self, message: str) -> None:
            self.messages.append(message)

    context = Context()
    namespace["SELECTED_ENTRY_ORDER"] = EXPECTED_ENTRIES
    namespace["main"](context)
    assert context.vision2d.calls == ["ocr_sorting", *EXPECTED_ENTRIES]
    assert ".get(" not in source


def test_v1_08_template_allows_only_a_permutation_of_approved_entry_ids() -> None:
    payload = _definition()
    source = (ROOT / payload["student_template"]).read_text(encoding="utf-8")
    namespace: dict[str, object] = {}
    exec(compile(source, str(ROOT / payload["student_template"]), "exec"), namespace)
    chosen_order = ("entry_c", "entry_a", "entry_d", "entry_b")
    calls: list[str] = []

    class Vision:
        def ocr_sorting(self):
            calls.append("ocr_sorting")
            return SimpleNamespace(
                status="PASS",
                entries=tuple(
                    SimpleNamespace(entry_id=entry_id, status="APPROVED")
                    for entry_id in EXPECTED_ENTRIES
                ),
            )

        def sort_ocr_entry(self, entry_id: str):
            calls.append(entry_id)
            return SimpleNamespace(status="COMPLETED")

    context = SimpleNamespace(
        vision2d=Vision(),
        log=lambda _message: None,
        checkpoint=lambda _message: None,
    )
    namespace["SELECTED_ENTRY_ORDER"] = chosen_order
    namespace["main"](context)
    assert calls == ["ocr_sorting", *chosen_order]
    assert "SELECTED_ENTRY_ORDER" in source


def test_v1_08_guide_is_chinese_and_preserves_pending_acceptance_boundaries() -> None:
    guide = (ROOT / "docs" / "experiments" / "V1-08.md").read_text(
        encoding="utf-8"
    )
    for heading in (
        "## 实验目标",
        "## 训练集与测试集",
        "## 字符分割",
        "## 置信度与白名单",
        "## 失败分析",
        "## 安全复位",
        "## 证据说明",
        "## 思考题",
        "## 人工验收边界",
        "## 真机迁移边界",
    ):
        assert heading in guide
    for text in ("A1", "A2", "B1", "B2", "ocr_sorting", "sort_ocr_entry", "V1-08.json"):
        assert text in guide
    assert "PENDING_HUMAN_ACCEPTANCE" in guide
    assert "PENDING_HARDWARE" in guide
    assert "仿真通过不代表" in guide
    assert "评分" not in guide


def test_v1_08_formal_files_are_retained() -> None:
    retained = {
        line.strip()
        for line in (ROOT / "RETAINED_FILES.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert {
        "config/experiments/V1-08.json",
        "config/experiments/catalog.json",
        "student_programs/templates/v1_08_ocr_sorting.py",
        "docs/experiments/V1-08.md",
        "tests/test_vision_quality/test_v1_08_materials.py",
    } <= retained
