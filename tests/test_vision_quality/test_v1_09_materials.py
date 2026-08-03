from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

from vision_platform.experiments.catalog import ExperimentCatalog


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config" / "experiments"


def test_v1_09_resolves_its_independent_materials_and_public_capabilities() -> None:
    catalog = ExperimentCatalog.load(CONFIG_DIR / "catalog.json", project_root=ROOT)
    definition = catalog.require("V1-09")
    assert definition.capabilities == (
        "camera.rgb", "camera.profile", "lighting.profile", "vision2d.surface_defects"
    )
    assert definition.scene == ROOT / "simulation/vision_defect_sorting_lab/BL23_vision_defect_sorting_lab.ttt"
    assert definition.scene_manifest == ROOT / "simulation/vision_defect_sorting_lab/scene_manifest.json"
    assert definition.student_template == ROOT / "student_programs/templates/v1_09_surface_defects.py"
    assert definition.guide == ROOT / "docs/experiments/V1-09.md"
    assert definition.hardware_status == "PENDING_HARDWARE"
    parameters = definition.public_parameters["defect_sorting"]
    assert parameters["profile_id"] == "standard"
    assert parameters["image_size_px"] == (1024, 1024)
    assert tuple(parameters["entry_ids"]) == tuple(f"entry_{letter}" for letter in "abcdef")
    assert set(parameters["fixed_rois_px"]) == {"reference", *(f"entry_{letter}" for letter in "abcdef")}
    assert "confidence_min" not in parameters


def test_v1_09_definition_keeps_pending_human_and_hardware_boundaries() -> None:
    payload = json.loads((CONFIG_DIR / "V1-09.json").read_text(encoding="utf-8"))
    assert payload["hardware_status"] == "PENDING_HARDWARE"
    assert payload["acceptance"]["human_checks"]
    assert "PENDING_HUMAN_ACCEPTANCE" in payload["acceptance"]["human_checks"]
    assert "PENDING_HARDWARE" in payload["acceptance"]["human_checks"]


def test_v1_09_student_template_executes_with_strict_log_contract() -> None:
    decisions = {
        "entry_a": "qualified",
        "entry_b": "missing",
        "entry_c": "hole",
        "entry_d": "foreign",
        "entry_e": "broken",
        "entry_f": "dimension",
    }
    entry_ids = tuple(decisions)
    entry_calls: list[str] = []

    class Vision2D:
        def surface_defects(self):
            return SimpleNamespace(
                plan_id="a" * 64,
                entries=tuple(SimpleNamespace(entry_id=entry_id) for entry_id in entry_ids),
            )

        def defect_sort_entry(self, entry_id: str):
            entry_calls.append(entry_id)
            return SimpleNamespace(
                entry_id=entry_id,
                decision=decisions[entry_id],
                status="COMPLETED",
            )

    class Context:
        def __init__(self) -> None:
            self.vision2d = Vision2D()
            self.logs: list[str] = []

        def log(self, message: str) -> None:
            assert isinstance(message, str)
            self.logs.append(message)

    namespace = runpy.run_path(str(ROOT / "student_programs/templates/v1_09_surface_defects.py"))
    context = Context()

    namespace["main"](context)

    assert entry_calls == list(entry_ids)
    assert len(context.logs) == 7
    assert context.logs[0] == "缺陷计划已冻结 plan_id=" + ("a" * 64) + " entries=6"
    assert context.logs[1:] == [
        f"条目完成 entry_id={entry_id} decision={decisions[entry_id]} status=COMPLETED"
        for entry_id in entry_ids
    ]
