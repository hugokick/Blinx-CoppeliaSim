from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from PyQt5.QtCore import (
    QCoreApplication,
    QEvent,
    QMetaObject,
    QObject,
    QThread,
    Qt,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vision_platform.student.experiment_gateway import (
    FileBytesSeal,
    validate_experiment_binding,
)
from vision_platform.student.protocol import RunState
from vision_platform.student.runner import StudentProgramController
from vision_platform.student.safety import StudentExecutionPolicy
from vision_platform.ui.experiment_catalog_panel import ExperimentCatalogPanel
from vision_platform.ui.student_program_panel import StudentProgramPanel
from vision_platform.ui.vision_result_panel import VisionResultPanel
from vision_platform.ui.view_model import (
    VisionLabSnapshot,
    VisionLabViewModel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UI_OUTPUT = PROJECT_ROOT / "artifacts" / "vision_lab" / "ui"


@dataclass(frozen=True)
class _ActionOutcome:
    action: str
    payload: Any = None
    frame: Any = None


class _ViewModelBridge(QObject):
    updated = pyqtSignal(object)


class _SessionBridge(QObject):
    replaced = pyqtSignal(object)


@dataclass(frozen=True)
class _PreparedExperimentSelection:
    context: Any
    definition: Any
    canonical_manifest: dict[str, Any]
    manifest_seal: FileBytesSeal
    template_path: Path
    template_source: str
    template_seal: FileBytesSeal


@dataclass(frozen=True)
class _ExperimentSelectionFailure:
    error: BaseException
    rollback_error: BaseException | None
    session_restored: bool
    restored_context: Any = None


@dataclass(frozen=True)
class _ExperimentSelectionTransaction:
    experiment_id: str
    session_snapshot: Any
    controller_snapshot: Any
    panel_snapshot: Any


class _ExperimentSelectionControl:
    """Thread-safe decision handle that outlives the worker QObject."""

    def __init__(self) -> None:
        self.cancel_requested = threading.Event()
        self.decision_ready = threading.Event()
        self.lock = threading.Lock()
        self.commit_accepted: bool | None = None
        self.commit_error: BaseException | None = None

    def cancel(self) -> None:
        self.cancel_requested.set()
        self.reject_commit(RuntimeError("实验载入已因窗口关闭而取消"))

    def accept_commit(self) -> bool:
        with self.lock:
            if self.commit_accepted is not None:
                return False
            self.commit_accepted = True
            self.decision_ready.set()
            return True

    def reject_commit(self, error: BaseException) -> bool:
        with self.lock:
            if self.commit_accepted is not None:
                return False
            self.commit_accepted = False
            self.commit_error = error
            self.decision_ready.set()
            return True

    def decision(self) -> tuple[bool | None, BaseException | None]:
        with self.lock:
            return self.commit_accepted, self.commit_error


def _prepare_experiment_selection(
    *,
    catalog: Any,
    experiment_session: Any,
    experiment_id: str,
    before_select: Any | None = None,
) -> _PreparedExperimentSelection:
    definition = catalog.require(experiment_id)
    if definition.experiment_id != experiment_id:
        raise ValueError("实验目录返回了不匹配的实验定义")
    initial_manifest = json.loads(
        definition.scene_manifest.read_bytes().decode("utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant: {value}")
        ),
    )
    if not isinstance(initial_manifest, dict):
        raise ValueError("场景清单必须是 JSON 对象")
    template_bytes = definition.student_template.read_bytes()
    if before_select is not None:
        before_select()
    context = experiment_session.select(experiment_id)
    if getattr(context, "experiment_id", None) != experiment_id:
        raise ValueError("实验会话返回了不匹配的运行上下文")
    fresh_manifest_bytes = definition.scene_manifest.read_bytes()
    fresh_manifest = json.loads(
        fresh_manifest_bytes.decode("utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant: {value}")
        ),
    )
    if not isinstance(fresh_manifest, dict):
        raise ValueError("场景清单必须是 JSON 对象")
    canonical_manifest = validate_experiment_binding(
        context,
        definition,
        fresh_manifest,
        scene_manifest_file_bytes=fresh_manifest_bytes,
    )
    if canonical_manifest is None:  # pragma: no cover - contract
        raise RuntimeError("实验绑定验证未返回规范场景清单")
    fresh_template_bytes = definition.student_template.read_bytes()
    if fresh_template_bytes != template_bytes:
        raise RuntimeError("学生模板在实验切换期间发生变化")
    template_source = fresh_template_bytes.decode("utf-8")
    return _PreparedExperimentSelection(
        context=context,
        definition=definition,
        canonical_manifest=canonical_manifest,
        manifest_seal=FileBytesSeal.capture(
            definition.scene_manifest,
            content=fresh_manifest_bytes,
        ),
        template_path=definition.student_template,
        template_source=template_source,
        template_seal=FileBytesSeal.capture(
            definition.student_template,
            content=fresh_template_bytes,
        ),
    )


class _ExperimentSelectWorker(QObject):
    prepared = pyqtSignal(object)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(object)

    def __init__(
        self,
        *,
        catalog: Any,
        experiment_session: Any,
        experiment_id: str,
        session_snapshot: Any,
        control: _ExperimentSelectionControl,
    ) -> None:
        super().__init__()
        self.catalog = catalog
        self.experiment_session = experiment_session
        self.experiment_id = experiment_id
        self.session_snapshot = session_snapshot
        self.control = control

    def _restore_session(self) -> tuple[bool, Any, BaseException | None]:
        try:
            context = self.experiment_session.restore_snapshot(
                self.session_snapshot
            )
        except BaseException as rollback_error:
            return False, None, rollback_error
        return True, context, None

    @pyqtSlot()
    def run(self) -> None:
        selection_started = False

        def mark_selection_started() -> None:
            nonlocal selection_started
            selection_started = True

        try:
            prepared = _prepare_experiment_selection(
                catalog=self.catalog,
                experiment_session=self.experiment_session,
                experiment_id=self.experiment_id,
                before_select=mark_selection_started,
            )
            if self.control.cancel_requested.is_set():
                raise RuntimeError("实验载入已因窗口关闭而取消")
        except BaseException as error:
            restored, restored_context, rollback_error = (
                self._restore_session()
                if selection_started
                else (False, None, None)
            )
            self.failed.emit(
                _ExperimentSelectionFailure(
                    error,
                    rollback_error,
                    restored,
                    restored_context,
                )
            )
            return

        self.prepared.emit(prepared)
        while not self.control.decision_ready.wait(timeout=0.05):
            if self.control.cancel_requested.is_set():
                self.control.reject_commit(
                    RuntimeError("实验载入已因窗口关闭而取消")
                )
        accepted, commit_error = self.control.decision()
        if accepted:
            self.succeeded.emit(prepared)
            return
        error = commit_error or RuntimeError("实验载入提交被拒绝")
        restored, restored_context, rollback_error = self._restore_session()
        self.failed.emit(
            _ExperimentSelectionFailure(
                error,
                rollback_error,
                restored,
                restored_context,
            )
        )


class _ActionWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str, str)

    def __init__(
        self,
        *,
        application: Any,
        action: str,
        step_mode: bool,
        output_dir: Path,
    ) -> None:
        super().__init__()
        self.application = application
        self.action = action
        self.step_mode = bool(step_mode)
        self.output_dir = output_dir
        self.cancel_event = threading.Event()
        self.pause_event = threading.Event()
        self.step_event = threading.Event()
        if self.step_mode:
            self.step_event.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            if self.action == "connect":
                self.application.open()
                frame = self.application.camera.read(timeout_s=5.0)
                outcome = _ActionOutcome("connect", frame=frame)
            elif self.action == "calibrate":
                from vision_platform.acceptance import (
                    run_calibration_acceptance,
                )

                report = run_calibration_acceptance(
                    self.application,
                    output_dir=self.output_dir / "calibration",
                )
                frame = self.application.camera.read(timeout_s=5.0)
                outcome = _ActionOutcome(
                    "calibrate",
                    payload=report,
                    frame=frame,
                )
            elif self.action == "classify":
                from vision_platform.acceptance import (
                    run_calibration_acceptance,
                )

                self.application.open()
                if self.application.calibration is None:
                    run_calibration_acceptance(
                        self.application,
                        output_dir=self.output_dir / "calibration",
                    )
                task = self.application.create_classification_task(
                    cancel_event=self.cancel_event,
                    step_waiter=self._wait_for_permission,
                )
                count = len(self.application.scene_spec.get("objects", [])) or 1
                result = task.run(max_objects=count)
                frame = self.application.camera.read(timeout_s=5.0)
                outcome = _ActionOutcome(
                    "classify",
                    payload=result,
                    frame=frame,
                )
            elif self.action == "reset":
                self.application.robot.move_home()
                outcome = _ActionOutcome("reset")
            else:
                raise ValueError(f"Unsupported UI action: {self.action}")
            self.finished.emit(outcome)
        except Exception as error:
            code = str(getattr(error, "code", "UI_ACTION_FAILED"))
            self.failed.emit(code, str(error))

    def pause(self) -> None:
        self.pause_event.set()

    def resume(self) -> None:
        self.pause_event.clear()
        self.step_event.set()

    def cancel(self) -> None:
        self.cancel_event.set()
        self.pause_event.clear()
        self.step_event.set()

    def _wait_for_permission(self, _event) -> None:
        while self.pause_event.is_set() and not self.cancel_event.is_set():
            time.sleep(0.03)
        if self.step_mode and not self.cancel_event.is_set():
            while (
                not self.step_event.wait(timeout=0.05)
                and not self.cancel_event.is_set()
            ):
                pass
            self.step_event.clear()


class VisionLabWindow(QMainWindow):
    """High-readability desktop teaching console for the vision lab."""

    def __init__(
        self,
        *,
        application: Any,
        session: Any | None = None,
        view_model: VisionLabViewModel | None = None,
        output_dir: str | Path = DEFAULT_UI_OUTPUT,
        experiment_catalog: Any | None = None,
        experiment_session: Any | None = None,
    ) -> None:
        super().__init__()
        self.session = session
        self.application = (
            session.application if session is not None else application
        )
        self.view_model = view_model or VisionLabViewModel()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self._thread: QThread | None = None
        self._worker: _ActionWorker | None = None
        self._experiment_thread: QThread | None = None
        self._experiment_worker: _ExperimentSelectWorker | None = None
        self._experiment_control: _ExperimentSelectionControl | None = None
        self._experiment_transaction: _ExperimentSelectionTransaction | None = None
        self._session_bridge_error: BaseException | None = None
        self._lifecycle_lock = threading.RLock()
        self._closing = False
        self._close_complete = False
        self._owner_close_started = False
        self._owner_close_finished = False
        self._owner_close_error: Exception | None = None
        self._cleanup_in_progress: set[str] = set()
        self._session_unsubscribe = None
        self._event_unsubscribe = None
        self._event_subscription_application = None
        self._pending_event_unsubscribes = []
        self._view_unsubscribe = None
        self._student_panel_unsubscribe = None
        self._vision_result_unsubscribe = None
        self._student_stop_pending_reported = False
        self._experiment_switch_quarantined = False
        self.student_controller: StudentProgramController | None = None
        self.student_program_panel: QWidget | None = None
        self.vision_result_panel: QWidget | None = None

        self.setWindowTitle("BL23 机器人视觉虚拟仿真实验台")
        self.setMinimumSize(1180, 760)
        self.resize(1440, 900)
        self._build_ui()
        if experiment_catalog is not None and experiment_session is not None:
            self.experiment_catalog = experiment_catalog
            self.experiment_session = experiment_session
            self.experiment_catalog_panel = ExperimentCatalogPanel(
                catalog=experiment_catalog,
                on_select=self._request_experiment_select,
            )
            self.tabs.insertTab(
                0,
                self.experiment_catalog_panel,
                "实验目录",
            )
        self._apply_design_system()

        self._bridge = _ViewModelBridge(self)
        self._bridge.updated.connect(self._render_snapshot)
        self._view_unsubscribe = self.view_model.subscribe(
            self._bridge.updated.emit
        )
        if self.session is not None:
            self._session_bridge = _SessionBridge(self)
            self._session_bridge.replaced.connect(
                self._session_application_replaced
            )
            self._session_unsubscribe = self.session.subscribe(
                self._session_bridge.replaced.emit
            )
            self._replace_application(self.session.application)
        else:
            self._replace_application(self.application)
        self._render_snapshot(self.view_model.snapshot())

    @pyqtSlot(object)
    def _session_application_replaced(self, application: Any) -> None:
        if self._lifecycle_is_closing():
            return
        try:
            self._replace_application(application)
        except BaseException as error:
            self._session_bridge_error = error
            control = self._experiment_control
            if control is not None:
                control.reject_commit(error)
            self.view_model.set_error(
                "EXPERIMENT_APPLICATION_REPLACE_FAILED",
                str(error)[:2000],
            )
            return
        self._session_bridge_error = None

    def _replace_application(self, application: Any) -> None:
        first_error: Exception | None = None
        with self._lifecycle_lock:
            if self._closing or self._close_complete:
                return
            previous_unsubscribe = self._event_unsubscribe
            self._event_unsubscribe = None
            self._event_subscription_application = None
            if previous_unsubscribe is not None:
                try:
                    previous_unsubscribe()
                except Exception as error:
                    self._pending_event_unsubscribes.append(
                        previous_unsubscribe
                    )
                    first_error = error
            self.application = application
            event_bus = getattr(application, "event_bus", None)
            if event_bus is not None:
                try:
                    self._event_unsubscribe = event_bus.subscribe(
                        self.view_model.apply
                    )
                    self._event_subscription_application = application
                except Exception as error:
                    if first_error is None:
                        first_error = error
            else:
                self._event_subscription_application = application
        if first_error is not None:
            raise first_error

    def _current_application(self) -> Any:
        if self.session is not None:
            return self.session.application
        return self.application

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(20, 16, 20, 18)
        root_layout.setSpacing(14)

        header = QHBoxLayout()
        title_column = QVBoxLayout()
        title = QLabel("BL23 机器人视觉虚拟仿真实验台")
        title.setObjectName("title")
        self.subtitle_label = QLabel(
            "仿真标定 · 视觉识别 · 学生编程 · 六轴机械臂分类闭环"
        )
        self.subtitle_label.setObjectName("subtitle")
        title_column.addWidget(title)
        title_column.addWidget(self.subtitle_label)
        header.addLayout(title_column)
        header.addStretch(1)
        self.connection_badge = QLabel("未连接")
        self.connection_badge.setObjectName("statusBadge")
        self.task_badge = QLabel("等待任务")
        self.task_badge.setObjectName("neutralBadge")
        header.addWidget(self.connection_badge)
        header.addWidget(self.task_badge)
        root_layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter, 1)

        preview_group = QGroupBox("相机视图与坐标反馈")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel()
        self.preview_label.setObjectName("preview")
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setPixmap(self._placeholder_pixmap())
        preview_layout.addWidget(self.preview_label, 1)
        coordinates = QHBoxLayout()
        self.pixel_coordinate_label = QLabel("像素坐标：—")
        self.world_coordinate_label = QLabel("世界坐标：—")
        coordinates.addWidget(self.pixel_coordinate_label)
        coordinates.addWidget(self.world_coordinate_label)
        coordinates.addStretch(1)
        preview_layout.addLayout(coordinates)
        splitter.addWidget(preview_group)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        backend_group = QGroupBox("实验后端")
        backend_layout = QGridLayout(backend_group)
        backend_layout.addWidget(QLabel("相机"), 0, 0)
        self.camera_backend_combo = QComboBox()
        self.camera_backend_combo.addItems(
            ["CoppeliaSim", "回放数据", "海康相机"]
        )
        backend_layout.addWidget(self.camera_backend_combo, 0, 1)
        backend_layout.addWidget(QLabel("机械臂"), 1, 0)
        self.robot_backend_combo = QComboBox()
        self.robot_backend_combo.addItems(["CoppeliaSim", "真实机械臂"])
        backend_layout.addWidget(self.robot_backend_combo, 1, 1)
        self.connect_button = QPushButton("连接实验平台")
        self.connect_button.setObjectName("primaryButton")
        self.connect_button.clicked.connect(
            lambda: self._start_action("connect")
        )
        backend_layout.addWidget(self.connect_button, 2, 0, 1, 2)
        capability = QLabel(
            "仿真：可用　回放：可用　海康：连接硬件后检测"
        )
        capability.setObjectName("muted")
        capability.setWordWrap(True)
        backend_layout.addWidget(capability, 3, 0, 1, 2)
        right_layout.addWidget(backend_group)

        self.tabs = QTabWidget()
        calibration_tab = QWidget()
        calibration_layout = QVBoxLayout(calibration_tab)
        self.calibration_table = QTableWidget(3, 4)
        self.calibration_table.setHorizontalHeaderLabels(
            ["标定点", "像素 X", "像素 Y", "世界 XY / mm"]
        )
        self.calibration_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        for row in range(3):
            self.calibration_table.setItem(
                row,
                0,
                QTableWidgetItem(f"P{row + 1}"),
            )
        calibration_layout.addWidget(self.calibration_table)
        self.calibration_summary = QLabel("标定状态：未标定")
        calibration_layout.addWidget(self.calibration_summary)
        self.calibrate_button = QPushButton("执行三点标定与验证")
        self.calibrate_button.setObjectName("secondaryButton")
        self.calibrate_button.clicked.connect(
            lambda: self._start_action("calibrate")
        )
        calibration_layout.addWidget(self.calibrate_button)
        self.tabs.addTab(calibration_tab, "标定")

        detection_tab = QWidget()
        detection_layout = QVBoxLayout(detection_tab)
        self.detection_table = QTableWidget(0, 6)
        self.detection_table.setHorizontalHeaderLabels(
            ["编号", "颜色", "形状", "置信度", "像素中心", "世界坐标"]
        )
        self.detection_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        detection_layout.addWidget(self.detection_table)
        self.tabs.addTab(detection_tab, "识别结果")

        report_tab = QWidget()
        report_layout = QVBoxLayout(report_tab)
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setPlaceholderText("自动验收结果将在此显示")
        report_layout.addWidget(self.report_text)
        self.tabs.addTab(report_tab, "验收")
        self._build_student_program_tab()
        right_layout.addWidget(self.tabs, 1)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        control_group = QGroupBox("任务控制")
        controls = QHBoxLayout(control_group)
        controls.addWidget(QLabel("运行模式"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["全自动", "单步教学"])
        controls.addWidget(self.mode_combo)
        self.start_button = QPushButton("启动六件分类")
        self.start_button.setObjectName("ctaButton")
        self.start_button.clicked.connect(
            lambda: self._start_action("classify")
        )
        controls.addWidget(self.start_button)
        self.pause_button = QPushButton("暂停")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self._pause)
        controls.addWidget(self.pause_button)
        self.resume_button = QPushButton("继续 / 下一步")
        self.resume_button.setEnabled(False)
        self.resume_button.clicked.connect(self._resume)
        controls.addWidget(self.resume_button)
        self.reset_button = QPushButton("复位")
        self.reset_button.clicked.connect(lambda: self._start_action("reset"))
        controls.addWidget(self.reset_button)
        self.emergency_button = QPushButton("急停")
        self.emergency_button.setObjectName("dangerButton")
        self.emergency_button.clicked.connect(self._emergency_stop)
        controls.addWidget(self.emergency_button)
        root_layout.addWidget(control_group)

        status_row = QHBoxLayout()
        self.status_label = QLabel("未连接：请选择后端并连接实验平台")
        self.status_label.setObjectName("statusText")
        status_row.addWidget(self.status_label, 2)
        self.error_label = QLabel("错误码：—")
        self.error_label.setObjectName("errorText")
        status_row.addWidget(self.error_label)
        root_layout.addLayout(status_row)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setPlaceholderText("状态机日志")
        root_layout.addWidget(self.log_text)

    def _build_student_program_tab(self) -> None:
        setup_error: Exception | None = None
        if self.session is not None:
            try:
                config = self.application.config
                student = config.student
                speed_range = student["speed_range"]
                policy = StudentExecutionPolicy(
                    min_speed=float(speed_range[0]),
                    max_speed=float(speed_range[1]),
                    max_runtime_s=float(student["max_runtime_s"]),
                    max_commands=int(student["max_commands"]),
                    command_timeout_s=float(
                        student["command_timeout_s"]
                    ),
                    max_sleep_s=float(student["max_sleep_s"]),
                    tool_on_max_z_mm=float(
                        student["tool_on_max_z_mm"]
                    ),
                )
                resolver = self.application._resolve_project_path
                student_output = resolver(config, student["output"])
                controller = StudentProgramController(
                    session=self.session,
                    execution_policy=policy,
                    output_root=student_output,
                )
                self.student_controller = controller
                panel = StudentProgramPanel(controller=controller)
                self.student_program_panel = panel
                self._student_panel_unsubscribe = (
                    panel.release_subscription
                )
                result_panel = VisionResultPanel(controller=controller)
                self.vision_result_panel = result_panel
                self._vision_result_unsubscribe = (
                    result_panel.release_subscription
                )
            except Exception as error:
                setup_error = error
                cleanup_errors: list[Exception] = []
                self._attempt_unsubscribe(
                    "_vision_result_unsubscribe",
                    cleanup_errors,
                )
                self._attempt_unsubscribe(
                    "_student_panel_unsubscribe",
                    cleanup_errors,
                )
                if (
                    self._vision_result_unsubscribe is None
                    and self._student_panel_unsubscribe is None
                ):
                    self.student_controller = None
                    self.student_program_panel = None
                    self.vision_result_panel = None
            else:
                self.tabs.addTab(panel, "学生编程")
                self.tabs.addTab(result_panel, "视觉结果")
                return

        unavailable = QLabel("当前窗口未配置学生程序会话")
        unavailable.setAlignment(Qt.AlignCenter)
        if setup_error is not None:
            unavailable.setToolTip(str(setup_error))
        self.student_program_panel = unavailable
        index = self.tabs.addTab(unavailable, "学生编程")
        self.tabs.setTabEnabled(index, False)

    def _select_experiment(self, experiment_id: str) -> bool:
        transaction = None
        selection_started = False
        try:
            transaction = self._capture_experiment_transaction(experiment_id)

            def mark_selection_started() -> None:
                nonlocal selection_started
                selection_started = True

            prepared = _prepare_experiment_selection(
                catalog=self.experiment_catalog,
                experiment_session=self.experiment_session,
                experiment_id=experiment_id,
                before_select=mark_selection_started,
            )
            self._commit_prepared_experiment(prepared)
        except BaseException as error:
            rollback_error = None
            restored_context = None
            if selection_started and transaction is not None:
                try:
                    restored_context = self.experiment_session.restore_snapshot(
                        transaction.session_snapshot
                    )
                except BaseException as failure:
                    rollback_error = failure
                self._restore_local_experiment_transaction(
                    transaction,
                    restored_context=restored_context,
                    rollback_error=rollback_error,
                )
            try:
                message = str(error)[:2000]
            except BaseException:
                message = "<unprintable>"
            if rollback_error is not None:
                try:
                    rollback_message = str(rollback_error)[:1000]
                except BaseException:
                    rollback_message = "<unprintable>"
                message = f"{message}；回滚失败：{rollback_message}"
            self.view_model.set_error(
                "EXPERIMENT_SELECT_FAILED",
                f"实验载入失败：{message}",
            )
            return False

        self.status_label.setText(
            f"已载入 {prepared.definition.experiment_id}："
            f"{prepared.definition.title}"
        )
        self.error_label.setText("错误码：—")
        return True

    def _capture_experiment_transaction(
        self,
        experiment_id: str,
    ) -> _ExperimentSelectionTransaction:
        controller = self.student_controller
        panel = self.student_program_panel
        if controller is None or not isinstance(panel, StudentProgramPanel):
            raise RuntimeError("当前窗口没有可用的学生程序控制器")
        self._ensure_experiment_switch_allowed()
        return _ExperimentSelectionTransaction(
            experiment_id=experiment_id,
            session_snapshot=self.experiment_session.capture_snapshot(),
            controller_snapshot=controller.capture_experiment_snapshot(),
            panel_snapshot=panel.capture_template_snapshot(),
        )

    def _commit_prepared_experiment(
        self,
        prepared: _PreparedExperimentSelection,
    ) -> None:
        self._ensure_experiment_switch_allowed()
        if self._session_bridge_error is not None:
            raise RuntimeError("实验应用切换通知处理失败") from (
                self._session_bridge_error
            )
        if self.session is not None and self.application is not self.session.application:
            raise RuntimeError("实验应用尚未在界面线程完成接管")
        controller = self.student_controller
        panel = self.student_program_panel
        if controller is None or not isinstance(panel, StudentProgramPanel):
            raise RuntimeError("当前窗口没有可用的学生程序控制器")
        controller.bind_experiment(
            context=prepared.context,
            definition=prepared.definition,
            scene_manifest=prepared.canonical_manifest,
            program_path=prepared.template_path,
            scene_manifest_seal=prepared.manifest_seal,
        )
        panel.load_template(
            prepared.template_path,
            source=prepared.template_source,
            controller_already_bound=True,
            template_seal=prepared.template_seal,
        )

    def _restore_local_experiment_transaction(
        self,
        transaction: _ExperimentSelectionTransaction,
        *,
        restored_context: Any,
        rollback_error: BaseException | None,
    ) -> None:
        controller = self.student_controller
        panel = self.student_program_panel
        local_errors = []
        if controller is not None:
            if rollback_error is None:
                try:
                    controller.restore_experiment_snapshot(
                        transaction.controller_snapshot,
                        restored_context=restored_context,
                    )
                except BaseException as error:
                    local_errors.append(error)
            else:
                local_errors.append(rollback_error)
            if local_errors:
                try:
                    controller.quarantine_experiment_binding(local_errors[0])
                except BaseException as error:
                    local_errors.append(error)
        if isinstance(panel, StudentProgramPanel):
            try:
                panel.restore_template_snapshot(transaction.panel_snapshot)
            except BaseException as error:
                local_errors.append(error)
                if controller is not None:
                    try:
                        controller.quarantine_experiment_binding(error)
                    except BaseException as quarantine_error:
                        local_errors.append(quarantine_error)
        if local_errors:
            self._enter_experiment_switch_quarantine(local_errors[0])

    def _enter_experiment_switch_quarantine(
        self,
        error: BaseException,
    ) -> None:
        self._experiment_switch_quarantined = True
        controller = self.student_controller
        if (
            controller is not None
            and not controller.experiment_binding_quarantined
        ):
            try:
                controller.quarantine_experiment_binding(error)
            except BaseException:
                pass
        self._disable_operation_controls()

    def _request_experiment_select(self, experiment_id: str) -> bool:
        thread: QThread | None = None
        worker: _ExperimentSelectWorker | None = None
        control: _ExperimentSelectionControl | None = None
        try:
            with self._lifecycle_lock:
                if (
                    self._closing
                    or self._close_complete
                    or self._experiment_switch_quarantined
                    or self._experiment_thread is not None
                ):
                    return False
            transaction = self._capture_experiment_transaction(experiment_id)
            thread = QThread(self)
            control = _ExperimentSelectionControl()
            thread.finished.connect(self._experiment_thread_finished)
            thread.finished.connect(thread.deleteLater)
            self._experiment_transaction = transaction
            self._experiment_thread = thread
            self._experiment_control = control
            self._set_experiment_switch_busy(True)
            thread.start()
            worker = _ExperimentSelectWorker(
                catalog=self.experiment_catalog,
                experiment_session=self.experiment_session,
                experiment_id=experiment_id,
                session_snapshot=transaction.session_snapshot,
                control=control,
            )
            worker.moveToThread(thread)
            worker.prepared.connect(self._experiment_prepared)
            worker.succeeded.connect(self._experiment_succeeded)
            worker.failed.connect(self._experiment_failed)
            worker.succeeded.connect(worker.deleteLater)
            worker.failed.connect(worker.deleteLater)
            worker.succeeded.connect(thread.quit, Qt.DirectConnection)
            worker.failed.connect(thread.quit, Qt.DirectConnection)
            self._experiment_worker = worker
            QMetaObject.invokeMethod(worker, "run", Qt.QueuedConnection)
            return True
        except BaseException as error:
            if control is not None:
                control.cancel()
            if worker is not None:
                try:
                    worker.deleteLater()
                except RuntimeError:
                    pass
            if thread is not None:
                try:
                    if thread.isRunning():
                        thread.quit()
                        thread.wait(5000)
                finally:
                    thread.deleteLater()
            self._experiment_transaction = None
            self._experiment_thread = None
            self._experiment_worker = None
            self._experiment_control = None
            self._set_experiment_switch_busy(False)
            self.view_model.set_error(
                "EXPERIMENT_SELECT_FAILED",
                f"实验载入失败：{str(error)[:2000]}",
            )
            return False

    @pyqtSlot(object)
    def _experiment_prepared(
        self,
        prepared: _PreparedExperimentSelection,
    ) -> None:
        control = self._experiment_control
        if control is None:
            return
        if self._lifecycle_is_closing():
            control.reject_commit(RuntimeError("窗口正在关闭"))
            return
        try:
            self._commit_prepared_experiment(prepared)
        except BaseException as error:
            control.reject_commit(error)
            return
        control.accept_commit()

    @pyqtSlot(object)
    def _experiment_succeeded(
        self,
        prepared: _PreparedExperimentSelection,
    ) -> None:
        if self._lifecycle_is_closing():
            return
        self.status_label.setText(
            f"已载入 {prepared.definition.experiment_id}："
            f"{prepared.definition.title}"
        )
        self.error_label.setText("错误码：—")
        self.experiment_catalog_panel.show_selection_success(
            prepared.definition.experiment_id
        )

    @pyqtSlot(object)
    def _experiment_failed(
        self,
        failure: _ExperimentSelectionFailure,
    ) -> None:
        if self._lifecycle_is_closing():
            return
        transaction = self._experiment_transaction
        rollback_error = failure.rollback_error
        if failure.session_restored and rollback_error is None:
            rollback_error = self._rollback_bridge_error()
        if transaction is not None and (
            failure.session_restored or rollback_error is not None
        ):
            self._restore_local_experiment_transaction(
                transaction,
                restored_context=failure.restored_context,
                rollback_error=rollback_error,
            )
        try:
            message = str(failure.error)[:2000]
        except BaseException:
            message = "<unprintable>"
        if rollback_error is not None:
            try:
                rollback_message = str(rollback_error)[:1000]
            except BaseException:
                rollback_message = "<unprintable>"
            message = f"{message}；回滚失败：{rollback_message}"
        self.view_model.set_error(
            "EXPERIMENT_SELECT_FAILED",
            f"实验载入失败：{message}",
        )
        self.experiment_catalog_panel.show_selection_failure(
            "实验载入失败，请查看窗口状态",
            error=failure.error,
        )

    def _rollback_bridge_error(self) -> BaseException | None:
        if self._session_bridge_error is not None:
            return self._session_bridge_error
        if self._pending_event_unsubscribes:
            return RuntimeError(
                "回滚应用仍有未完成的事件退订"
            )
        if self.session is None:
            return None
        current = self.session.application
        if self.application is not current:
            return RuntimeError(
                "回滚应用尚未在界面线程完成接管"
            )
        if self._event_subscription_application is not current:
            return RuntimeError(
                "回滚应用事件订阅尚未完成接管"
            )
        return None

    @pyqtSlot()
    def _experiment_thread_finished(self) -> None:
        thread = self.sender()
        if thread is not self._experiment_thread:
            return
        self._experiment_thread = None
        self._experiment_worker = None
        self._experiment_control = None
        self._experiment_transaction = None
        if not self._lifecycle_is_closing() and not self._experiment_switch_quarantined:
            self._set_experiment_switch_busy(False)

    def _set_experiment_switch_busy(self, busy: bool) -> None:
        panel = getattr(self, "experiment_catalog_panel", None)
        if busy:
            if panel is not None:
                panel.set_selection_busy(True)
            self._disable_operation_controls()
            return
        if self._lifecycle_is_closing() or self._experiment_switch_quarantined:
            self._disable_operation_controls()
            return
        for control in (
            self.camera_backend_combo,
            self.robot_backend_combo,
            self.mode_combo,
            self.emergency_button,
        ):
            control.setEnabled(True)
        self._set_controls_running(self._thread is not None)
        if self.student_program_panel is not None:
            self.student_program_panel.setEnabled(True)
        if panel is not None:
            panel.setEnabled(True)
            panel.set_selection_busy(False)

    def _ensure_experiment_switch_allowed(self) -> None:
        if self._experiment_switch_quarantined:
            raise RuntimeError("实验切换已进入安全隔离状态，不能继续操作")
        if self._thread is not None:
            raise RuntimeError("平台任务运行期间不能切换实验")
        panel = self.student_program_panel
        if (
            isinstance(panel, StudentProgramPanel)
            and panel.operation_in_progress
        ):
            raise RuntimeError("学生程序控制操作期间不能切换实验")
        controller = self.student_controller
        if controller is None:
            raise RuntimeError("当前窗口没有可用的学生程序控制器")
        controller.ensure_experiment_switch_allowed()

    def _apply_design_system(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #F8FAFC;
                color: #1E293B;
                font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
                font-size: 14px;
            }
            QLabel#title {
                font-size: 24px;
                font-weight: 700;
                color: #0F172A;
            }
            QLabel#subtitle, QLabel#muted {
                color: #64748B;
            }
            QLabel#statusBadge, QLabel#neutralBadge {
                padding: 7px 12px;
                border-radius: 12px;
                font-weight: 600;
            }
            QLabel#statusBadge {
                color: #9A3412;
                background: #FFEDD5;
                border: 1px solid #FDBA74;
            }
            QLabel#neutralBadge {
                color: #334155;
                background: #E2E8F0;
                border: 1px solid #CBD5E1;
            }
            QGroupBox {
                border: 1px solid #CBD5E1;
                border-radius: 10px;
                margin-top: 12px;
                padding: 10px;
                font-weight: 650;
                background: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QLabel#preview {
                background: #111827;
                border: 2px solid #334155;
                border-radius: 8px;
            }
            QPushButton, QComboBox {
                min-height: 44px;
                border: 1px solid #CBD5E1;
                border-radius: 7px;
                padding: 0 14px;
                background: #FFFFFF;
            }
            QPushButton:hover, QComboBox:hover {
                border-color: #3B82F6;
                background: #EFF6FF;
            }
            QPushButton:focus, QComboBox:focus {
                border: 2px solid #2563EB;
            }
            QPushButton:disabled {
                color: #94A3B8;
                background: #F1F5F9;
            }
            QPushButton#primaryButton {
                color: #FFFFFF;
                background: #2563EB;
                border-color: #2563EB;
                font-weight: 650;
            }
            QPushButton#secondaryButton {
                color: #1D4ED8;
                background: #DBEAFE;
                border-color: #93C5FD;
                font-weight: 650;
            }
            QPushButton#ctaButton {
                color: #FFFFFF;
                background: #F97316;
                border-color: #EA580C;
                font-weight: 700;
            }
            QPushButton#dangerButton {
                color: #FFFFFF;
                background: #DC2626;
                border-color: #B91C1C;
                font-weight: 700;
            }
            QTableWidget, QTextEdit, QTabWidget::pane {
                background: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                selection-background-color: #DBEAFE;
                selection-color: #1E3A8A;
            }
            QHeaderView::section {
                background: #E2E8F0;
                color: #334155;
                border: none;
                border-right: 1px solid #CBD5E1;
                padding: 7px;
                font-weight: 650;
            }
            QLabel#errorText {
                color: #B91C1C;
                font-weight: 600;
            }
            """
        )
        for button in (
            self.connect_button,
            self.calibrate_button,
            self.start_button,
            self.pause_button,
            self.resume_button,
            self.reset_button,
            self.emergency_button,
        ):
            button.setMinimumHeight(44)

    def _placeholder_pixmap(self) -> QPixmap:
        pixmap = QPixmap(640, 480)
        pixmap.fill(QColor("#111827"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#334155"), 1))
        for x in range(0, 641, 64):
            painter.drawLine(x, 0, x, 480)
        for y in range(0, 481, 48):
            painter.drawLine(0, y, 640, y)
        painter.setPen(QColor("#CBD5E1"))
        painter.drawText(
            pixmap.rect(),
            Qt.AlignCenter,
            "等待相机画面\n640 × 480",
        )
        painter.end()
        return pixmap

    def _start_action(self, action: str) -> None:
        with self._lifecycle_lock:
            if (
                self._closing
                or self._close_complete
                or self._thread is not None
            ):
                return
            step_mode = self.mode_combo.currentText() == "单步教学"
            thread = QThread(self)
            worker = _ActionWorker(
                application=self._current_application(),
                action=action,
                step_mode=step_mode,
                output_dir=self.output_dir,
            )
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(self._action_finished)
            worker.failed.connect(self._action_failed)
            worker.finished.connect(thread.quit)
            worker.failed.connect(thread.quit)
            thread.finished.connect(worker.deleteLater)
            thread.finished.connect(self._thread_finished)
            self._thread = thread
            self._worker = worker
            self.view_model.set_running(True)
            self._set_controls_running(True)
            thread.start()

    @pyqtSlot(object)
    def _action_finished(self, outcome: _ActionOutcome) -> None:
        if outcome.frame is not None:
            self.view_model.set_frame(outcome.frame)
        if outcome.action == "connect":
            self.view_model.set_connection(
                connected=True,
                camera_backend=self._selected_camera_backend(),
                robot_backend=self._selected_robot_backend(),
                message="实验平台已连接，可以开始标定",
            )
        elif outcome.action == "calibrate":
            report = outcome.payload
            self._populate_calibration_table(report.calibration_path)
            self.view_model.set_calibration(
                status=report.status,
                rms_error_mm=report.rms_error_mm,
                max_error_mm=report.max_error_mm,
            )
            self.report_text.setPlainText(
                (
                    f"标定：{report.status}\n"
                    f"RMS 误差：{report.rms_error_mm:.3f} mm\n"
                    f"最大误差：{report.max_error_mm:.3f} mm\n"
                    f"报告：{report.report_path}"
                )
            )
        elif outcome.action == "classify":
            result = outcome.payload
            self.view_model.set_acceptance(result.status)
            self.report_text.setPlainText(
                (
                    f"分类闭环：{result.status}\n"
                    f"完成对象：{result.metrics.get('objects_completed', 0)}\n"
                    f"吸附成功：{result.metrics.get('attach_success', 0)}\n"
                    f"释放成功：{result.metrics.get('release_success', 0)}\n"
                    f"分类区验证：{result.metrics.get('correct_zone', 0)}"
                )
            )
        elif outcome.action == "reset":
            self.view_model.reset()
        self.view_model.set_running(False)

    def _populate_calibration_table(self, calibration_path: Path) -> None:
        payload = json.loads(
            Path(calibration_path).read_text(encoding="utf-8")
        )
        pixels = payload.get("pixel_points", [])
        worlds = payload.get("world_points_mm", [])
        row_count = min(3, len(pixels), len(worlds))
        for row in range(row_count):
            values = (
                f"{float(pixels[row][0]):.1f}",
                f"{float(pixels[row][1]):.1f}",
                (
                    f"({float(worlds[row][0]):.1f}, "
                    f"{float(worlds[row][1]):.1f})"
                ),
            )
            for column, value in enumerate(values, start=1):
                self.calibration_table.setItem(
                    row,
                    column,
                    QTableWidgetItem(value),
                )

    @pyqtSlot(str, str)
    def _action_failed(self, code: str, message: str) -> None:
        self.view_model.set_error(code, message)
        self.report_text.setPlainText(f"执行失败\n{code}\n{message}")

    @pyqtSlot()
    def _thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._set_controls_running(False)

    def _pause(self) -> None:
        if self._lifecycle_is_closing():
            return
        if self._worker is None:
            return
        self._worker.pause()
        self.view_model.set_running(True, paused=True)

    def _resume(self) -> None:
        if self._lifecycle_is_closing():
            return
        if self._worker is None:
            return
        self._worker.resume()
        self.view_model.set_running(True, paused=False)

    def _emergency_stop(self) -> None:
        if self._lifecycle_is_closing():
            return
        if self._worker is not None:
            self._worker.cancel()
        try:
            tool = getattr(self._current_application(), "tool", None)
            if tool is not None:
                tool.off()
        except Exception:
            pass
        self.view_model.set_error("EMERGENCY_STOP", "已请求急停并关闭吸盘")

    def _set_controls_running(self, running: bool) -> None:
        if self._lifecycle_is_closing():
            self._disable_operation_controls()
            return
        self.connect_button.setEnabled(not running)
        self.calibrate_button.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.reset_button.setEnabled(not running)
        self.pause_button.setEnabled(running)
        self.resume_button.setEnabled(running)

    def _lifecycle_is_closing(self) -> bool:
        with self._lifecycle_lock:
            return self._closing or self._close_complete

    def _disable_operation_controls(self) -> None:
        for control in (
            self.camera_backend_combo,
            self.robot_backend_combo,
            self.mode_combo,
            self.connect_button,
            self.calibrate_button,
            self.start_button,
            self.pause_button,
            self.resume_button,
            self.reset_button,
            self.emergency_button,
        ):
            control.setEnabled(False)
        if self.student_program_panel is not None:
            self.student_program_panel.setEnabled(False)
        catalog_panel = getattr(self, "experiment_catalog_panel", None)
        if catalog_panel is not None:
            catalog_panel.setEnabled(False)

    def _selected_camera_backend(self) -> str:
        return ("sim", "replay", "hik")[self.camera_backend_combo.currentIndex()]

    def _selected_robot_backend(self) -> str:
        return ("sim", "real")[self.robot_backend_combo.currentIndex()]

    @pyqtSlot(object)
    def _render_snapshot(self, snapshot: VisionLabSnapshot) -> None:
        self.status_label.setText(snapshot.status_text)
        self.error_label.setText(
            f"错误码：{snapshot.error_code or '—'}"
        )
        self.connection_badge.setText(
            "已连接" if snapshot.connected else "未连接"
        )
        self.task_badge.setText(snapshot.state)
        if snapshot.pixel_coordinate is None:
            self.pixel_coordinate_label.setText("像素坐标：—")
        else:
            self.pixel_coordinate_label.setText(
                "像素坐标："
                f"({snapshot.pixel_coordinate[0]:.1f}, "
                f"{snapshot.pixel_coordinate[1]:.1f})"
            )
        if snapshot.world_coordinate_mm is None:
            self.world_coordinate_label.setText("世界坐标：—")
        else:
            point = snapshot.world_coordinate_mm
            self.world_coordinate_label.setText(
                f"世界坐标：({point[0]:.1f}, {point[1]:.1f}, "
                f"{point[2]:.1f}) mm"
            )
        self.calibration_summary.setText(
            (
                f"标定状态：{snapshot.calibration_status}　"
                f"RMS：{self._format_metric(snapshot.calibration_rms_error_mm)}　"
                f"MAX：{self._format_metric(snapshot.calibration_max_error_mm)}"
            )
        )
        self.log_text.setPlainText("\n".join(snapshot.logs[-100:]))
        self.log_text.moveCursor(self.log_text.textCursor().End)
        if snapshot.frame_bgr is not None:
            self._render_frame(snapshot.frame_bgr)
        self._render_detection_table(snapshot.detections)

    @staticmethod
    def _format_metric(value: float | None) -> str:
        return "—" if value is None else f"{value:.3f} mm"

    def _render_frame(self, image_bgr) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = QImage(
            rgb.data,
            width,
            height,
            int(rgb.strides[0]),
            QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(pixmap)

    def _render_detection_table(self, detections) -> None:
        self.detection_table.setRowCount(len(detections))
        for row, detection in enumerate(detections):
            values = (
                detection.get("detection_id", ""),
                detection.get("color", ""),
                detection.get("shape", ""),
                detection.get("confidence", ""),
                detection.get("center_px", ""),
                detection.get("world_mm", ""),
            )
            for column, value in enumerate(values):
                self.detection_table.setItem(
                    row,
                    column,
                    QTableWidgetItem(str(value)),
                )

    def _attempt_unsubscribe(
        self,
        attribute: str,
        errors: list[Exception],
    ) -> None:
        with self._lifecycle_lock:
            callback = getattr(self, attribute)
            if callback is None or attribute in self._cleanup_in_progress:
                return
            setattr(self, attribute, None)
            self._cleanup_in_progress.add(attribute)

        failure: Exception | None = None
        try:
            callback()
        except Exception as error:
            failure = error
            errors.append(error)
        finally:
            with self._lifecycle_lock:
                self._cleanup_in_progress.discard(attribute)
                if failure is not None and getattr(self, attribute) is None:
                    setattr(self, attribute, callback)

    def _attempt_event_unsubscribes(
        self,
        errors: list[Exception],
    ) -> None:
        cleanup_key = "event_subscriptions"
        with self._lifecycle_lock:
            if cleanup_key in self._cleanup_in_progress:
                return
            callbacks = list(self._pending_event_unsubscribes)
            if self._event_unsubscribe is not None:
                callbacks.append(self._event_unsubscribe)
            if not callbacks:
                return
            self._pending_event_unsubscribes.clear()
            self._event_unsubscribe = None
            self._event_subscription_application = None
            self._cleanup_in_progress.add(cleanup_key)

        failed_callbacks = []
        try:
            for callback in callbacks:
                try:
                    callback()
                except Exception as error:
                    failed_callbacks.append(callback)
                    errors.append(error)
        finally:
            with self._lifecycle_lock:
                self._cleanup_in_progress.discard(cleanup_key)
                self._pending_event_unsubscribes.extend(failed_callbacks)

    def _attempt_owner_close(self, errors: list[Exception]) -> None:
        with self._lifecycle_lock:
            if self._owner_close_started:
                return
            self._owner_close_started = True
            owner = (
                self.session
                if self.session is not None
                else self.application
            )

        failure: Exception | None = None
        try:
            owner.close()
        except Exception as error:
            failure = error
            errors.append(error)
        finally:
            with self._lifecycle_lock:
                self._owner_close_finished = True
                if failure is not None:
                    self._owner_close_error = failure

    def _finish_close_if_ready(self) -> tuple[bool, Exception | None]:
        with self._lifecycle_lock:
            cleanup_pending = (
                self._student_panel_unsubscribe is not None
                or self._vision_result_unsubscribe is not None
                or self._session_unsubscribe is not None
                or self._event_unsubscribe is not None
                or bool(self._pending_event_unsubscribes)
                or self._view_unsubscribe is not None
                or bool(self._cleanup_in_progress)
            )
            ready = (
                self._owner_close_finished
                and not cleanup_pending
            )
            if ready:
                self._close_complete = True
            return ready, self._owner_close_error

    def _report_close_failure(self, code: str, error: Exception) -> None:
        try:
            self.view_model.set_error(code, str(error))
            self._render_snapshot(self.view_model.snapshot())
        except Exception:
            pass

    def _student_backend_ready_for_close(
        self,
    ) -> tuple[bool, Exception | None]:
        controller = self.student_controller
        if controller is None:
            return True, None
        try:
            panel = self.student_program_panel
            if isinstance(panel, StudentProgramPanel):
                if controller.state in {
                    RunState.RUNNING,
                    RunState.PAUSED,
                }:
                    if panel.request_stop():
                        self._student_stop_pending_reported = False
                if panel.operation_in_progress:
                    return (
                        False,
                        RuntimeError(
                            "student program control operation "
                            "is still running"
                        ),
                    )
            if controller.wait_for_quiescence(0):
                return True, None
        except Exception as error:
            return False, error
        return (
            False,
            RuntimeError(
                "student program backend is still stopping; "
                "close the window again after it becomes quiescent"
            ),
        )

    @property
    def owner_shutdown_complete(self) -> bool:
        with self._lifecycle_lock:
            return bool(
                self._close_complete
                and self._owner_close_error is None
            )

    def shutdown(self, *, timeout_ms: int = 5000) -> None:
        """Idempotently close all workers before releasing the owner."""
        if type(timeout_ms) is not int or timeout_ms < 0:
            raise TypeError("timeout_ms must be a non-negative integer")
        deadline = time.monotonic() + (timeout_ms / 1000.0)
        while True:
            with self._lifecycle_lock:
                owner_error = self._owner_close_error
                close_complete = self._close_complete
            if owner_error is not None:
                raise owner_error
            if close_complete:
                return

            self.close()
            QApplication.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            QApplication.processEvents()

            with self._lifecycle_lock:
                owner_error = self._owner_close_error
                close_complete = self._close_complete
            if owner_error is not None:
                raise owner_error
            if close_complete:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "window owner shutdown did not complete before timeout"
                )
            time.sleep(0.01)

    def closeEvent(self, event) -> None:
        with self._lifecycle_lock:
            if self._close_complete:
                event.accept()
                return

        try:
            experiment_control = self._experiment_control
            experiment_thread = self._experiment_thread
            if experiment_control is not None:
                experiment_control.cancel()
            if experiment_thread is not None:
                experiment_thread.quit()
                if not bool(experiment_thread.wait(5000)):
                    self._report_close_failure(
                        "UI_EXPERIMENT_THREAD_STOP_TIMEOUT",
                        RuntimeError(
                            "experiment selection worker did not stop "
                            "within 5 seconds"
                        ),
                    )
                    event.ignore()
                    return
            if self._worker is not None:
                self._worker.cancel()
            thread = self._thread
            if thread is not None:
                thread.quit()
                if not bool(thread.wait(5000)):
                    self._report_close_failure(
                        "UI_THREAD_STOP_TIMEOUT",
                        RuntimeError(
                            "UI worker did not stop within 5 seconds"
                        ),
                    )
                    event.ignore()
                    return
        except Exception as error:
            self._report_close_failure("UI_CLOSE_FAILED", error)
            event.ignore()
            return

        student_ready, student_error = (
            self._student_backend_ready_for_close()
        )
        if not student_ready:
            if not self._student_stop_pending_reported:
                self._student_stop_pending_reported = True
                self._report_close_failure(
                    "UI_STUDENT_STOP_PENDING",
                    student_error
                    or RuntimeError(
                        "student program backend is still stopping"
                    ),
                )
            event.ignore()
            return

        with self._lifecycle_lock:
            if self._close_complete:
                event.accept()
                return
            self._closing = True
        self._disable_operation_controls()

        errors: list[Exception] = []
        error_reported = False
        self._attempt_unsubscribe("_vision_result_unsubscribe", errors)
        self._attempt_unsubscribe("_student_panel_unsubscribe", errors)
        self._attempt_unsubscribe("_session_unsubscribe", errors)
        self._attempt_owner_close(errors)
        with self._lifecycle_lock:
            owner_finished = self._owner_close_finished
            owner_error = self._owner_close_error
        if owner_error is not None:
            self._report_close_failure("UI_CLOSE_FAILED", owner_error)
            error_reported = True
        elif errors:
            self._report_close_failure("UI_CLOSE_FAILED", errors[0])
            error_reported = True
        if owner_finished:
            self._attempt_event_unsubscribes(errors)
            if errors and not error_reported:
                self._report_close_failure("UI_CLOSE_FAILED", errors[0])
                error_reported = True
            self._attempt_unsubscribe("_view_unsubscribe", errors)
            if errors and not error_reported:
                self._report_close_failure("UI_CLOSE_FAILED", errors[0])
                error_reported = True

        ready, persistent_error = self._finish_close_if_ready()
        if ready:
            event.accept()
            return

        error = (
            errors[0]
            if errors
            else persistent_error
            or RuntimeError("UI close cleanup is still in progress")
        )
        if not error_reported:
            self._report_close_failure("UI_CLOSE_FAILED", error)
        event.ignore()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vision-lab-pyqt")
    parser.add_argument("--config")
    parser.add_argument(
        "--camera",
        choices=("sim", "replay", "hik"),
        default="sim",
    )
    parser.add_argument(
        "--robot",
        choices=("sim", "real"),
        default="sim",
    )
    parser.add_argument("--output", default=str(DEFAULT_UI_OUTPUT))
    return parser


def _window_has_background_work(window: Any) -> bool:
    for attribute in ("_experiment_thread", "_thread"):
        thread = getattr(window, attribute, None)
        if thread is None:
            continue
        is_running = getattr(thread, "isRunning", None)
        if not callable(is_running) or bool(is_running()):
            return True
    panel = getattr(window, "student_program_panel", None)
    if bool(getattr(panel, "operation_in_progress", False)):
        return True
    controller = getattr(window, "student_controller", None)
    wait_for_quiescence = getattr(controller, "wait_for_quiescence", None)
    if callable(wait_for_quiescence):
        try:
            if not bool(wait_for_quiescence(0)):
                return True
        except BaseException:
            return True
    return False


def _close_direct_owner(session: Any, application: Any) -> None:
    owner = session if session is not None else application
    if owner is None:
        return
    close = getattr(owner, "close", None)
    if not callable(close):
        raise RuntimeError("owned application resource has no close method")
    close()


def _shutdown_main_owner(
    *,
    window: Any,
    session: Any,
    application: Any,
) -> None:
    if window is None:
        _close_direct_owner(session, application)
        return

    if getattr(window, "owner_shutdown_complete", False) is True:
        return
    shutdown = getattr(window, "shutdown", None)
    if callable(shutdown):
        shutdown()
        if getattr(window, "owner_shutdown_complete", False) is not True:
            raise RuntimeError(
                "window shutdown returned before owner cleanup completed"
            )
        return

    close = getattr(window, "close", None)
    if not callable(close):
        if _window_has_background_work(window):
            raise RuntimeError(
                "window has background work but no safe shutdown method"
            )
        _close_direct_owner(session, application)
        return

    try:
        accepted = close()
    except BaseException as close_error:
        if _window_has_background_work(window):
            raise
        try:
            _close_direct_owner(session, application)
        except BaseException as owner_error:
            raise close_error from owner_error
        raise
    if accepted is False:
        if _window_has_background_work(window):
            raise RuntimeError(
                "window close was rejected while background work is active"
            )
        _close_direct_owner(session, application)
        return
    completion = getattr(window, "owner_shutdown_complete", None)
    if completion is False:
        if _window_has_background_work(window):
            raise RuntimeError(
                "window accepted close before background work completed"
            )
        _close_direct_owner(session, application)


def main(argv=None) -> int:
    from vision_platform.application import VisionLabApplication
    from vision_platform.config import load_config
    from vision_platform.session import VisionLabSession

    args = build_parser().parse_args(argv)
    environ = dict(os.environ)
    environ["VISION_BACKEND"] = args.camera
    environ["ROBOT_BACKEND"] = args.robot
    config = load_config(
        args.config,
        project_root=PROJECT_ROOT,
        environ=environ,
    )
    def factory():
        return VisionLabApplication.from_config(config)

    application = None
    session = None
    window = None
    result: int | None = None
    primary_error: BaseException | None = None
    try:
        application = factory()
        session = VisionLabSession(application=application, factory=factory)
        experiment_catalog = None
        experiment_session = None
        window_holder: dict[str, VisionLabWindow] = {}
        if args.camera == "sim" and args.robot == "sim":
            from vision_platform.experiments.catalog import ExperimentCatalog
            from vision_platform.experiments.session import ExperimentSession

            experiment_catalog = ExperimentCatalog.load(
                PROJECT_ROOT / "config" / "experiments" / "catalog.json",
                project_root=PROJECT_ROOT,
            )

            def student_is_idle() -> bool:
                selected_window = window_holder.get("window")
                if selected_window is None:
                    return False
                try:
                    if getattr(selected_window, "_thread", None) is not None:
                        return False
                    panel = getattr(
                        selected_window,
                        "student_program_panel",
                        None,
                    )
                    if bool(getattr(panel, "operation_in_progress", False)):
                        return False
                    controller = getattr(
                        selected_window,
                        "student_controller",
                        None,
                    )
                    if controller is None:
                        return False
                    controller.ensure_experiment_switch_allowed()
                except (AttributeError, RuntimeError):
                    return False
                return True

            experiment_session = ExperimentSession(
                catalog=experiment_catalog,
                vision_session=session,
                base_config=config,
                application_factory=VisionLabApplication.from_config,
                student_is_idle=student_is_idle,
            )
        qt_application = QApplication.instance() or QApplication(sys.argv)
        window = VisionLabWindow(
            application=application,
            session=session,
            output_dir=args.output,
            experiment_catalog=experiment_catalog,
            experiment_session=experiment_session,
        )
        window_holder["window"] = window
        window.show()
        result = int(qt_application.exec_())
    except BaseException as error:
        primary_error = error

    cleanup_error: BaseException | None = None
    try:
        _shutdown_main_owner(
            window=window,
            session=session,
            application=application,
        )
    except BaseException as error:
        cleanup_error = error

    if primary_error is not None:
        if cleanup_error is not None:
            raise primary_error from cleanup_error
        raise primary_error
    if cleanup_error is not None:
        raise cleanup_error
    assert result is not None
    return result


if __name__ == "__main__":
    raise SystemExit(main())
