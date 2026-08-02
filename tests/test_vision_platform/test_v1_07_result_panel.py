from __future__ import annotations

import numpy as np

from vision_platform.student.evidence import StudentRunEvidence
from vision_platform.ui.vision_result_panel import VisionResultPanel
from vision_platform.vision_quality.evidence import record_vision_bundle
from vision_platform.vision_quality.results import make_result_bundle


def _record_route_bundle(tmp_path):
    program = tmp_path / "v1_07.py"
    program.write_text("def main(ctx):\n    pass\n", encoding="utf-8")
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs", program_path=program,
        robot_backend="sim", run_id="v1-07-panel-run",
    )
    raw = np.full((1024, 1024, 3), 255, dtype=np.uint8)
    annotated = raw.copy()
    result = {
        "plan_id": "a" * 64,
        "status": "PASS",
        "entries": [
            {
                "entry_id": "entry_a", "part_id": "part_a", "code_type": "qr",
                "payload": "[V1-07-A](file:///C:/unsafe)", "route_id": "route_red",
                "pick_xyz_mm": [40.0, -45.0, 18.0],
                "drop_xyz_mm": [116.0, -60.0, 22.0], "confidence": 0.98,
            }
        ],
        "motion_events": [{"name": "tool.off", "status": "PASS"}],
        "completed_entry_ids": ["entry_a"],
        "final_occupancy": {"route_red": ["part_a"], "route_blue": []},
        "error": None,
        "human_acceptance": "PENDING_HUMAN_ACCEPTANCE",
        "hardware_status": "PENDING_HARDWARE",
    }
    bundle = make_result_bundle(
        bundle_id="V1-07-frame-000001-final", experiment_id="V1-07",
        source_snapshot_id="frame-000001", status="PASS",
        layers={"raw": ("原图", raw), "annotated": ("代码与仓位标注", annotated)},
        result=result,
        profile={"profile_id": "standard", "resolution": [1024, 1024]},
    )
    record_vision_bundle(evidence, bundle)
    return evidence.directory


def test_panel_renders_v1_07_routes_as_inert_read_only_text(qtbot, tmp_path) -> None:
    panel = VisionResultPanel()
    qtbot.addWidget(panel)
    panel.load_run(_record_route_bundle(tmp_path))
    assert tuple(
        panel.layer_combo.itemData(index) for index in range(panel.layer_combo.count())
    ) == ("raw", "annotated")
    assert panel.route_text.isReadOnly()
    rendered = panel.route_text.toPlainText()
    assert "[V1-07-A](file:///C:/unsafe)" in rendered
    assert "part_a" in rendered and "route_red" in rendered
    assert "40.0, -45.0, 18.0" in rendered
    assert "tool.off" in rendered
    assert "PENDING_HUMAN_ACCEPTANCE" in rendered
    assert "PENDING_HARDWARE" in rendered
