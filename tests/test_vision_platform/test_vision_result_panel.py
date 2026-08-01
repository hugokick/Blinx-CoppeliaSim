from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vision_platform.student.evidence import StudentRunEvidence
from vision_platform.student.protocol import RunState
from vision_platform.ui.vision_result_panel import VisionResultPanel
from vision_platform.vision_quality.evidence import record_vision_bundle
from vision_platform.vision_quality.results import make_result_bundle


def _record_two_layer_bundle(tmp_path):
    program = tmp_path / "student.py"
    program.write_text("def main(ctx):\n    pass\n", encoding="utf-8")
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=program,
        robot_backend="sim",
        run_id="vision-panel-run",
    )
    raw = np.zeros((24, 32, 3), dtype=np.uint8)
    raw[:, :] = (20, 80, 180)
    annotated = raw.copy()
    annotated[4:20, 8:24] = (30, 210, 80)
    bundle = make_result_bundle(
        bundle_id="frame-000001",
        experiment_id="V1-01",
        source_snapshot_id="frame-000001",
        status="PASS",
        layers={
            "raw": ("原始图像", raw),
            "annotated": ("标注结果", annotated),
        },
        result={
            "experiment_id": "V1-01",
            "observation": "三档配置比较完成",
        },
        profile={
            "profile_id": "standard",
            "resolution": [32, 24],
            "perspective_angle_deg": 60.0,
            "camera_rig_z_m": 0.70,
            "key_diffuse_rgb": [0.8, 0.8, 0.8],
            "fill_diffuse_rgb": [0.35, 0.35, 0.35],
        },
    )
    record_vision_bundle(evidence, bundle)
    return evidence.directory


class FakeController:
    def __init__(self):
        self.state = RunState.EMPTY
        self.handlers = []
        self.unsubscribe_calls = 0

    def subscribe(self, handler):
        self.handlers.append(handler)

        def unsubscribe():
            if handler in self.handlers:
                self.handlers.remove(handler)
                self.unsubscribe_calls += 1

        return unsubscribe

    def emit(self, *, state, evidence_dir):
        self.state = state
        snapshot = SimpleNamespace(
            state=state,
            evidence_dir=evidence_dir,
        )
        for handler in tuple(self.handlers):
            handler(snapshot)


def test_panel_has_stable_empty_state_and_pending_boundaries(qtbot):
    panel = VisionResultPanel()
    qtbot.addWidget(panel)

    assert "尚无视觉结果" in panel.status_label.text()
    assert panel.layer_combo.count() == 0
    assert "PENDING_HARDWARE" in panel.boundary_label.text()
    assert "不是课程成绩" in panel.boundary_label.text()
    assert panel.result_text.isReadOnly() is True


def test_panel_loads_latest_bundle_and_switches_layers(qtbot, tmp_path):
    run = _record_two_layer_bundle(tmp_path)
    panel = VisionResultPanel()
    qtbot.addWidget(panel)

    panel.load_run(run)

    assert panel.layer_combo.count() == 2
    assert panel.layer_combo.itemData(0) == "raw"
    assert "standard" in panel.profile_label.text()
    assert "60" in panel.profile_label.text()
    assert "0.70" in panel.profile_label.text()
    assert "主光 RGB：0.80, 0.80, 0.80" in panel.profile_label.text()
    assert "补光 RGB：0.35, 0.35, 0.35" in panel.profile_label.text()
    assert '"experiment_id": "V1-01"' in panel.result_text.toPlainText()
    assert panel.preview_label.pixmap() is not None
    assert not panel.preview_label.pixmap().isNull()
    panel.layer_combo.setCurrentIndex(1)
    assert panel.layer_combo.currentData() == "annotated"
    assert not panel.preview_label.pixmap().isNull()


def test_panel_contains_invalid_bundle_error(qtbot, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "vision-bundle-frame-000001.json").write_text(
        "{}",
        encoding="utf-8",
    )
    panel = VisionResultPanel()
    qtbot.addWidget(panel)

    panel.load_run(run)

    assert "VISION_RESULT_EVIDENCE_INVALID" in panel.status_label.text()
    assert panel.layer_combo.count() == 0
    assert panel.preview_label.pixmap() is None


def test_panel_does_not_expose_local_path_in_error_status(qtbot, tmp_path):
    missing = tmp_path / "private-user" / "secret-runs" / "missing"
    panel = VisionResultPanel()
    qtbot.addWidget(panel)

    panel.load_run(missing)

    assert (
        panel.status_label.text()
        == "VISION_RESULT_EVIDENCE_INVALID：无法载入视觉结果证据"
    )
    assert "private-user" not in panel.status_label.text()
    assert str(tmp_path) not in panel.status_label.text()


def test_panel_terminal_snapshot_without_bundle_has_stable_state(qtbot, tmp_path):
    controller = FakeController()
    run = tmp_path / "empty-run"
    run.mkdir()
    panel = VisionResultPanel(controller=controller)
    qtbot.addWidget(panel)

    controller.emit(state=RunState.PASSED, evidence_dir=run)

    qtbot.waitUntil(
        lambda: "本次运行没有视觉结果包" in panel.status_label.text()
    )
    assert panel.layer_combo.count() == 0


def test_panel_callback_contains_hostile_bundle_error_and_unsubscribes(
    qtbot,
    tmp_path,
):
    controller = FakeController()
    run = tmp_path / "hostile-run"
    run.mkdir()
    (run / "vision-bundle-frame-000001.json").write_text(
        '{"bundle_id":".."}',
        encoding="utf-8",
    )
    panel = VisionResultPanel(controller=controller)
    qtbot.addWidget(panel)

    controller.emit(state=RunState.FAILED, evidence_dir=run)

    qtbot.waitUntil(
        lambda: "VISION_RESULT_EVIDENCE_INVALID"
        in panel.status_label.text()
    )
    assert panel.layer_combo.count() == 0
    panel.release_subscription()
    panel.release_subscription()
    assert controller.unsubscribe_calls == 1
    assert controller.handlers == []
