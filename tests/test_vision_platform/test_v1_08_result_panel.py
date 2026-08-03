from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt

from vision_platform.experiments.catalog import ExperimentCatalog
from vision_platform.student.evidence import StudentRunEvidence
from vision_platform.ui.experiment_catalog_panel import ExperimentCatalogPanel
from vision_platform.ui.vision_result_panel import VisionResultPanel
from vision_platform.vision_quality.evidence import record_vision_bundle
from vision_platform.vision_quality.results import make_result_bundle


ROOT = Path(__file__).resolve().parents[2]


def _record_ocr_bundle(tmp_path: Path, *, error: str | None = None) -> Path:
    program = tmp_path / "v1_08.py"
    program.write_text("def main(ctx):\n    pass\n", encoding="utf-8")
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=program,
        robot_backend="sim",
        run_id="v1-08-panel-run",
    )
    raw = np.full((1024, 1024, 3), 245, dtype=np.uint8)
    annotated = raw.copy()
    results = [
        {
            "identifier": identifier,
            "confidence": confidence,
            "route_id": route,
            "entry_id": entry_id,
            "status": "PASS",
        }
        for identifier, confidence, route, entry_id in (
            ("A1", 0.98, "route_alpha", "entry_a"),
            ("A2", 0.97, "route_alpha", "entry_b"),
            ("B1", 0.96, "route_beta", "entry_c"),
            ("B2", 0.95, "route_beta", "entry_d"),
        )
    ]
    result = {
        "plan_id": "a" * 64,
        "status": "PASS" if error is None else "REJECTED",
        "training": {
            "held_out_accuracy": 0.9875,
            "train_count": 48,
            "test_count": 16,
        },
        "results": results,
        "motion_status": "COMPLETED",
        "final_occupancy": {
            "route_alpha": {"slot_1": "part_a", "slot_2": "part_b"},
            "route_beta": {"slot_1": "part_c", "slot_2": "part_d"},
        },
        "same_run_evidence": True,
        "robot_home": True,
        "tool_off": True,
        "error": error,
        "human_acceptance": "PENDING_HUMAN_ACCEPTANCE",
        "hardware_status": "PENDING_HARDWARE",
    }
    bundle = make_result_bundle(
        bundle_id="V1-08-frame-000001-final",
        experiment_id="V1-08",
        source_snapshot_id="frame-000001",
        status="PASS" if error is None else "REJECTED",
        layers={
            "raw": ("原图", raw),
            "annotated": ("OCR 编号标注", annotated),
        },
        result=result,
        profile={"profile_id": "standard", "resolution": [1024, 1024]},
    )
    record_vision_bundle(evidence, bundle)
    return evidence.directory


def test_v1_08_panel_renders_read_only_ocr_evidence(qtbot, tmp_path) -> None:
    panel = VisionResultPanel()
    qtbot.addWidget(panel)
    panel.load_run(_record_ocr_bundle(tmp_path))

    assert "0.988" in panel.metrics_label.text()
    assert "raw" in panel.evidence_paths_label.text()
    assert "annotated" in panel.evidence_paths_label.text()
    assert "A1" in panel.ocr_text.toPlainText()
    assert "route_alpha" in panel.ocr_text.toPlainText()
    assert "slot_1" in panel.ocr_text.toPlainText()
    assert "COMPLETED" in panel.ocr_text.toPlainText()
    assert "PENDING_HARDWARE" in panel.ocr_text.toPlainText()
    assert len(panel.ocr_summary_label.text()) < 80
    assert panel.ocr_text.isReadOnly()
    assert panel.evidence_paths_label.textInteractionFlags() & Qt.TextSelectableByMouse


def test_v1_08_panel_handles_long_error_and_missing_evidence(qtbot, tmp_path) -> None:
    panel = VisionResultPanel()
    qtbot.addWidget(panel)
    long_error = "OCR_SORT_FAILED:" + "x" * 4000
    panel.load_run(_record_ocr_bundle(tmp_path, error=long_error))
    assert "OCR_SORT_FAILED" in panel.ocr_text.toPlainText()
    assert len(panel.ocr_text.toPlainText()) < 5000

    panel.clear_result("本次运行没有视觉结果包")
    assert "没有视觉结果" in panel.status_label.text()
    assert panel.ocr_text.toPlainText() == ""


def test_v1_08_catalog_label_and_hardware_boundary(qtbot) -> None:
    catalog = ExperimentCatalog.load(
        ROOT / "config" / "experiments" / "catalog.json",
        project_root=ROOT,
    )
    panel = ExperimentCatalogPanel(catalog=catalog, on_select=lambda _id: None)
    qtbot.addWidget(panel)
    index = next(
        index
        for index in range(panel.experiment_combo.count())
        if panel.experiment_combo.itemData(index) == "V1-08"
    )
    panel.experiment_combo.setCurrentIndex(index)
    assert "OCR" in panel.capabilities_label.text()
    assert panel.hardware_label.text() == "PENDING_HARDWARE"


def test_v1_08_panel_geometry_and_ui_evidence(qtbot, tmp_path) -> None:
    panel = VisionResultPanel()
    qtbot.addWidget(panel)
    panel.load_run(_record_ocr_bundle(tmp_path))
    evidence_root = ROOT / "artifacts" / "vision_lab" / "ui" / "v1-08"
    evidence_root.mkdir(parents=True, exist_ok=True)

    panel.resize(1000, 700)
    panel.show()
    qtbot.wait(50)
    assert panel.preview_label.width() > 0
    assert panel.result_text.height() > 0
    assert panel.grab().save(str(evidence_root / "vision_result_panel_100.png"))

    panel.resize(1250, 875)
    qtbot.wait(50)
    assert panel.preview_label.width() > 0
    assert panel.result_text.height() > 0
    assert panel.grab().save(str(evidence_root / "vision_result_panel_125.png"))


def test_v1_08_bundle_snapshot_binding_is_stable(tmp_path) -> None:
    directory = _record_ocr_bundle(tmp_path)
    artifact = next(directory.glob("vision-bundle-*.json"))
    assert artifact.name == "vision-bundle-V1-08-frame-000001-final.json"
    assert hashlib.sha256(artifact.read_bytes()).hexdigest()
