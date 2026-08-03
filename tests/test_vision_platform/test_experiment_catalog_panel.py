from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from PyQt5.QtCore import Qt

from vision_platform.experiments.catalog import ExperimentCatalog
from vision_platform.ui.experiment_catalog_panel import ExperimentCatalogPanel


ROOT = Path(__file__).resolve().parents[2]


def _definition(
    experiment_id: str,
    title: str,
    capabilities: tuple[str, ...],
    human_check: str,
):
    return SimpleNamespace(
        experiment_id=experiment_id,
        title=title,
        capabilities=capabilities,
        hardware_status="PENDING_HARDWARE",
        acceptance=SimpleNamespace(human_checks=(human_check,)),
    )


class FakeCatalog:
    definitions = (
        _definition(
            "R1-01",
            "机械臂认知和基础操作",
            ("robot.home", "robot.pose"),
            "学生指出六个关节",
        ),
        _definition(
            "R1-05",
            "基于视觉的物体码垛",
            ("camera.rgb", "tool.suction"),
            "学生解释码垛顺序",
        ),
    )

    def require(self, experiment_id):
        return next(
            item
            for item in self.definitions
            if item.experiment_id == experiment_id
        )


def test_catalog_panel_shows_hardware_boundary_and_selects(qtbot):
    selected = []
    panel = ExperimentCatalogPanel(
        catalog=FakeCatalog(),
        on_select=selected.append,
    )
    qtbot.addWidget(panel)

    assert panel.experiment_combo.count() == 2
    assert "PENDING_HARDWARE" in panel.hardware_label.text()
    assert "不代表真机" in panel.boundary_label.text()
    panel.experiment_combo.setCurrentIndex(1)
    qtbot.mouseClick(panel.select_button, Qt.LeftButton)

    assert selected == ["R1-05"]
    assert "相机" in panel.capabilities_label.text()
    assert "学生解释码垛顺序" in panel.human_checks.toPlainText()


def test_catalog_panel_preserves_formal_order_through_v1_07(qtbot):
    catalog = ExperimentCatalog.load(
        ROOT / "config" / "experiments" / "catalog.json",
        project_root=ROOT,
    )
    panel = ExperimentCatalogPanel(catalog=catalog, on_select=lambda _id: None)
    qtbot.addWidget(panel)

    visible_ids = tuple(
        panel.experiment_combo.itemData(index)
        for index in range(panel.experiment_combo.count())
    )
    assert visible_ids[-9:-3] == (
        "V1-01",
        "V1-02",
        "V1-03",
        "V1-04",
        "V1-05",
        "V1-06",
    )
    assert visible_ids[-3:] == ("V1-07", "V1-08", "V1-09")
    panel.experiment_combo.setCurrentIndex(panel.experiment_combo.count() - 4)
    assert "模板匹配" in panel.capabilities_label.text()


def test_catalog_panel_empty_catalog_is_stably_disabled(qtbot):
    panel = ExperimentCatalogPanel(
        catalog=SimpleNamespace(definitions=()),
        on_select=lambda _experiment_id: None,
    )
    qtbot.addWidget(panel)

    assert panel.experiment_combo.count() == 0
    assert panel.select_button.isEnabled() is False
    assert "不可用" in panel.status_label.text()


def test_catalog_panel_contains_callback_failure(qtbot):
    def fail(_experiment_id):
        raise RuntimeError("scene-switch-failed")

    panel = ExperimentCatalogPanel(catalog=FakeCatalog(), on_select=fail)
    qtbot.addWidget(panel)

    qtbot.mouseClick(panel.select_button, Qt.LeftButton)

    assert "载入失败" in panel.status_label.text()
    assert "scene-switch-failed" in panel.status_label.toolTip()


def test_catalog_panel_keeps_scheduled_selection_pending_until_result(qtbot):
    panel = ExperimentCatalogPanel(
        catalog=FakeCatalog(),
        on_select=lambda _experiment_id: True,
    )
    qtbot.addWidget(panel)

    qtbot.mouseClick(panel.select_button, Qt.LeftButton)

    assert panel.select_button.isEnabled() is False
    assert panel.experiment_combo.isEnabled() is False
    assert "正在载入" in panel.status_label.text()

    panel.show_selection_success("R1-01")

    assert panel.select_button.isEnabled() is True
    assert panel.experiment_combo.isEnabled() is True
    assert "已载入" in panel.status_label.text()
