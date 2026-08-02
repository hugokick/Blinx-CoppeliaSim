from __future__ import annotations

import numpy as np

from vision_platform.student.evidence import StudentRunEvidence
from vision_platform.ui.vision_result_panel import VisionResultPanel
from vision_platform.vision_quality.evidence import record_vision_bundle
from vision_platform.vision_quality.results import make_result_bundle


def _record_template_bundle(tmp_path):
    program = tmp_path / "v1_06_template.py"
    program.write_text("def main(ctx):\n    pass\n", encoding="utf-8")
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=program,
        robot_backend="sim",
        run_id="v1-06-panel-run",
    )
    raw = np.full((512, 512, 3), 255, dtype=np.uint8)
    template = np.full((40, 67, 3), 255, dtype=np.uint8)
    template[5:34, 5:61] = (71, 71, 255)
    raw[201:241, 269:336] = template
    annotated = raw.copy()
    annotated[201:241, 269:336] = (0, 255, 0)
    bundle = make_result_bundle(
        bundle_id="V1-06-frame-000001",
        experiment_id="V1-06",
        source_snapshot_id="frame-000001",
        status="PASS",
        layers={
            "raw": ("原图", raw),
            "template": ("固定模板", template),
            "annotated": ("模板匹配标注", annotated),
        },
        result={
            "snapshot_id": "frame-000001",
            "template_id": "v1_06_red_rectangle",
            "template_version": "1.0.0",
            "status": "MATCHED",
            "matched": True,
            "score": 0.9767,
            "threshold": 0.72,
            "bbox_px": [269, 201, 67, 40],
            "center_px": [302.5, 221.0],
            "image_size": [512, 512],
            "search_roi_px": [0, 0, 512, 512],
            "method": "TM_CCOEFF_NORMED",
        },
        profile={
            "profile_id": "standard",
            "resolution": [512, 512],
            "perspective_angle_deg": 60.0,
            "camera_rig_z_m": 0.70,
            "key_diffuse_rgb": [0.8, 0.8, 0.8],
            "fill_diffuse_rgb": [0.35, 0.35, 0.35],
        },
    )
    record_vision_bundle(evidence, bundle)
    return evidence.directory


def test_panel_renders_v1_06_template_metrics_and_three_layers(qtbot, tmp_path):
    panel = VisionResultPanel()
    qtbot.addWidget(panel)

    panel.load_run(_record_template_bundle(tmp_path))

    assert tuple(
        panel.layer_combo.itemData(index)
        for index in range(panel.layer_combo.count())
    ) == ("raw", "template", "annotated")
    assert "v1_06_red_rectangle" in panel.metrics_label.text()
    assert "0.977" in panel.metrics_label.text()
    assert "269, 201, 67, 40" in panel.metrics_label.text()
    assert "302.5, 221.0" in panel.metrics_label.text()
    assert '"template_id": "v1_06_red_rectangle"' in panel.result_text.toPlainText()
    for index in range(panel.layer_combo.count()):
        panel.layer_combo.setCurrentIndex(index)
        assert panel.preview_label.pixmap() is not None
        assert not panel.preview_label.pixmap().isNull()
