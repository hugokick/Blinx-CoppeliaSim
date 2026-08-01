from __future__ import annotations

from typing import Any

from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


_CAPABILITY_LABELS = {
    "robot.home": "机械臂回零",
    "robot.pose": "TCP 查询",
    "robot.move_world": "世界坐标运动",
    "tool.suction": "吸盘",
    "camera.rgb": "RGB 相机",
    "experiment.info": "实验参数",
    "scene.probe": "场景探针",
}


def _safe_message(error: BaseException) -> str:
    try:
        return str(error)[:2000]
    except BaseException:
        return "<unprintable>"


class ExperimentCatalogPanel(QWidget):
    """Read-only course catalog; scene probes are not student grading."""

    def __init__(self, *, catalog: Any, on_select: Any, parent=None) -> None:
        super().__init__(parent)
        self.catalog = catalog
        self.on_select = on_select
        self._selection_busy = False
        self.experiment_combo = QComboBox()
        self.title_label = QLabel("—")
        self.hardware_label = QLabel("PENDING_HARDWARE")
        self.hardware_label.setObjectName("hardwareBoundary")
        self.hardware_label.setStyleSheet(
            "color: #991B1B; background: #FEE2E2; "
            "border: 1px solid #FCA5A5; border-radius: 6px; "
            "padding: 6px 10px; font-weight: 700;"
        )
        self.capabilities_label = QLabel("—")
        self.capabilities_label.setWordWrap(True)
        self.human_checks = QTextBrowser()
        self.human_checks.setMaximumHeight(110)
        self.select_button = QPushButton("载入实验")
        self.select_button.setMinimumHeight(44)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.boundary_label = QLabel(
            "场景探针仅记录仿真状态，不自动评分；"
            "仿真结果不代表真机或教学效果验收。"
        )
        self.boundary_label.setWordWrap(True)
        self.boundary_label.setStyleSheet("color: #92400E; font-weight: 600;")

        form = QFormLayout()
        form.addRow("实验", self.experiment_combo)
        form.addRow("名称", self.title_label)
        form.addRow("所需能力", self.capabilities_label)
        form.addRow("硬件状态", self.hardware_label)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("教师人工复核点"))
        layout.addWidget(self.human_checks)
        layout.addWidget(self.boundary_label)
        layout.addWidget(self.select_button)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        self.experiment_combo.currentIndexChanged.connect(self._render)
        self.select_button.clicked.connect(self._select)
        self._load_catalog()

    def _load_catalog(self) -> None:
        try:
            definitions = tuple(self.catalog.definitions)
            for item in definitions:
                experiment_id = str(item.experiment_id)
                title = str(item.title)
                if not experiment_id or not title:
                    raise ValueError("实验编号和名称不能为空")
                self.experiment_combo.addItem(
                    f"{experiment_id}  {title}",
                    experiment_id,
                )
            if not definitions:
                raise ValueError("实验目录为空")
            if not callable(self.on_select):
                raise TypeError("实验载入回调不可用")
            self._render()
        except BaseException as error:
            self._set_unavailable(error)

    def _definition(self):
        experiment_id = self.experiment_combo.currentData()
        if not isinstance(experiment_id, str) or not experiment_id:
            raise ValueError("没有可选实验")
        return self.catalog.require(experiment_id)

    def _render(self, *_args: Any) -> None:
        try:
            item = self._definition()
            capabilities = tuple(item.capabilities)
            human_checks = tuple(item.acceptance.human_checks)
            hardware_status = str(item.hardware_status)
            if hardware_status != "PENDING_HARDWARE":
                raise ValueError("硬件状态必须保持 PENDING_HARDWARE")
            self.title_label.setText(str(item.title))
            self.hardware_label.setText(hardware_status)
            self.capabilities_label.setText(
                "、".join(
                    _CAPABILITY_LABELS.get(str(name), str(name))
                    for name in capabilities
                )
                or "—"
            )
            self.human_checks.setPlainText(
                "\n".join(f"• {text}" for text in human_checks)
            )
            self.select_button.setEnabled(not self._selection_busy)
            self.experiment_combo.setEnabled(not self._selection_busy)
            self.status_label.clear()
            self.status_label.setToolTip("")
        except BaseException as error:
            self._set_unavailable(error)

    def _set_unavailable(self, error: BaseException) -> None:
        message = _safe_message(error)
        self.title_label.setText("—")
        self.capabilities_label.setText("—")
        self.human_checks.clear()
        self.select_button.setEnabled(False)
        self.status_label.setText("实验目录不可用")
        self.status_label.setToolTip(message)

    def _select(self, *_args: Any) -> None:
        try:
            result = self.on_select(self._definition().experiment_id)
            if result is False:
                self.show_selection_failure("实验载入失败，请查看窗口状态")
                return
        except BaseException as error:
            self.show_selection_failure("实验载入失败", error=error)
            return
        self.set_selection_busy(True)
        self.status_label.setText("实验正在载入，请稍候…")
        self.status_label.setToolTip("")

    def set_selection_busy(self, busy: bool) -> None:
        if type(busy) is not bool:
            raise TypeError("busy must be a boolean")
        self._selection_busy = busy
        self.experiment_combo.setEnabled(not busy)
        self.select_button.setEnabled(not busy)

    def show_selection_success(self, experiment_id: str) -> None:
        self.set_selection_busy(False)
        self.status_label.setText(
            f"已载入 {experiment_id}，可进入学生编程页"
        )
        self.status_label.setToolTip("")

    def show_selection_failure(
        self,
        message: str,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.set_selection_busy(False)
        self.status_label.setText(str(message))
        self.status_label.setToolTip(
            "" if error is None else _safe_message(error)
        )
