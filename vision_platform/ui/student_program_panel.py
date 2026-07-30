from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vision_platform.student.protocol import RunState
from vision_platform.student.validator import validate_program


_ACTIVE_STATES = frozenset({RunState.RUNNING, RunState.PAUSED})
_TERMINAL_STATES = frozenset(
    {RunState.PASSED, RunState.FAILED, RunState.CANCELLED}
)


class _SnapshotBridge(QObject):
    updated = pyqtSignal(object)


class _ControllerOperationWorker(QObject):
    succeeded = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)

    def __init__(
        self,
        *,
        controller: Any,
        operation: str,
        staged_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.operation = operation
        self.staged_path = staged_path

    @pyqtSlot()
    def run(self) -> None:
        try:
            if self.operation == "cancel":
                self.controller.cancel()
                result = None
            elif self.operation == "reset":
                application = self.controller.reset()
                validation = None
                loaded_path = None
                if self.staged_path is not None:
                    loaded_path = self.controller.load(
                        self.staged_path
                    )
                    validation = self.controller.validate(
                        loaded_path
                    )
                result = {
                    "application": application,
                    "program_path": loaded_path,
                    "validation": validation,
                }
            else:
                raise ValueError(
                    f"unsupported controller operation: "
                    f"{self.operation}"
                )
        except BaseException as error:
            try:
                message = str(error)
            except BaseException:
                message = "<unprintable>"
            self.failed.emit(self.operation, message[:2000])
            return
        self.succeeded.emit(self.operation, result)


class StudentProgramPanel(QWidget):
    def __init__(self, *, controller: Any, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.program_path: Path | None = None
        self._pending_program_path: Path | None = None
        self._unsubscribe = None
        self._operation_thread: QThread | None = None
        self._operation_worker: _ControllerOperationWorker | None = None
        self._operation_name: str | None = None
        self._cancel_requested_for_run = False
        self._last_state: RunState | None = None
        self._close_pending_reported = False
        self._build_ui()

        self._bridge = _SnapshotBridge(self)
        self._bridge.updated.connect(self._render_snapshot)
        self._unsubscribe = controller.subscribe(
            self._bridge.updated.emit
        )
        self._render_snapshot(
            SimpleNamespace(
                state=controller.state,
                current_command=None,
                command_count=0,
                elapsed_seconds=0.0,
                error=None,
                evidence_dir=None,
                tcp_mm=None,
            )
        )

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        file_toolbar = QHBoxLayout()
        run_toolbar = QHBoxLayout()
        self.open_button = QPushButton("打开程序")
        self.save_button = QPushButton("保存")
        self.save_as_button = QPushButton("另存为")
        self.validate_button = QPushButton("检查代码")
        self.run_button = QPushButton("运行")
        self.pause_button = QPushButton("暂停")
        self.resume_button = QPushButton("继续")
        self.step_button = QPushButton("下一步")
        self.stop_button = QPushButton("停止")
        self.reset_button = QPushButton("复位场景")
        file_buttons = (
            self.open_button,
            self.save_button,
            self.save_as_button,
            self.validate_button,
        )
        run_buttons = (
            self.run_button,
            self.pause_button,
            self.resume_button,
            self.step_button,
            self.stop_button,
            self.reset_button,
        )
        for button in (*file_buttons, *run_buttons):
            button.setMinimumHeight(38)
        for button in file_buttons:
            file_toolbar.addWidget(button)
        for button in run_buttons:
            run_toolbar.addWidget(button)
        layout.addLayout(file_toolbar)
        layout.addLayout(run_toolbar)

        self.path_label = QLabel("尚未打开学生程序")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.path_label)

        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText(
            "打开或粘贴包含 def main(ctx): 的 UTF-8 Python 程序"
        )
        self.editor.setTabStopDistance(32)
        layout.addWidget(self.editor, 3)

        status = QGridLayout()
        self.state_label = QLabel("EMPTY")
        self.state_label.setObjectName("studentStateBadge")
        self.command_label = QLabel("当前命令：—")
        self.metrics_label = QLabel("命令数：0　运行时间：0.0 s")
        self.tcp_label = QLabel("TCP：—")
        self.evidence_label = QLabel("证据目录：—")
        status.addWidget(QLabel("运行状态："), 0, 0)
        status.addWidget(self.state_label, 0, 1)
        status.addWidget(self.command_label, 1, 0, 1, 2)
        status.addWidget(self.metrics_label, 2, 0, 1, 2)
        status.addWidget(self.tcp_label, 3, 0, 1, 2)
        status.addWidget(self.evidence_label, 4, 0, 1, 2)
        layout.addLayout(status)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(130)
        layout.addWidget(self.console, 1)

        self.open_button.clicked.connect(self._open_program)
        self.save_button.clicked.connect(self._save)
        self.save_as_button.clicked.connect(self._save_as)
        self.validate_button.clicked.connect(self._validate)
        self.run_button.clicked.connect(self._run)
        self.pause_button.clicked.connect(
            lambda: self._call_controller("pause")
        )
        self.resume_button.clicked.connect(
            lambda: self._call_controller("resume")
        )
        self.step_button.clicked.connect(
            lambda: self._call_controller("step")
        )
        self.stop_button.clicked.connect(self.request_stop)
        self.reset_button.clicked.connect(self.request_reset)

    @pyqtSlot()
    def _open_program(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开学生程序",
            "",
            "Python (*.py)",
        )
        if not path:
            return
        selected = Path(path).expanduser().resolve()
        try:
            source = selected.read_text(encoding="utf-8")
        except Exception as error:
            self._show_error("STUDENT_FILE_OPEN_FAILED", error)
            return
        self.editor.setPlainText(source)
        self.program_path = selected
        self.path_label.setText(str(self.program_path))
        if self.controller.state in _TERMINAL_STATES:
            self._pending_program_path = selected
            self.console.append(f"已暂存：{self.program_path}")
            self._apply_state(self.controller.state)
            return
        try:
            loaded = self.controller.load(selected)
        except Exception as error:
            self._pending_program_path = selected
            self._show_error("STUDENT_PROGRAM_SYNC_FAILED", error)
        else:
            self.program_path = Path(loaded).resolve()
            self._pending_program_path = None
        self.console.append(f"已载入：{self.program_path}")
        self._apply_state(self.controller.state)

    @pyqtSlot()
    def _save(self) -> bool:
        if self.program_path is None:
            return self._save_as()
        return self._save_to_path(self.program_path)

    def _save_to_path(self, path: Path) -> bool:
        selected = Path(path).expanduser().resolve()
        try:
            selected.write_text(
                self.editor.toPlainText(),
                encoding="utf-8",
                newline="\n",
            )
        except Exception as error:
            self._show_error("STUDENT_FILE_SAVE_FAILED", error)
            return False
        self.program_path = selected
        self.path_label.setText(str(self.program_path))
        self.console.append(f"已保存：{self.program_path}")
        if self.controller.state in _TERMINAL_STATES:
            self._pending_program_path = self.program_path
            self._apply_state(self.controller.state)
            return True
        try:
            loaded = self.controller.load(self.program_path)
        except Exception as error:
            self._pending_program_path = self.program_path
            self._show_error("STUDENT_PROGRAM_SYNC_FAILED", error)
        else:
            self.program_path = Path(loaded).resolve()
            self._pending_program_path = None
        self._apply_state(self.controller.state)
        return True

    @pyqtSlot()
    def _save_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "另存学生程序",
            "student_program.py",
            "Python (*.py)",
        )
        if not path:
            return False
        selected = Path(path).expanduser()
        if selected.suffix.lower() != ".py":
            selected = selected.with_suffix(".py")
        return self._save_to_path(selected)

    @pyqtSlot()
    def _validate(self) -> None:
        if not self._save():
            return
        try:
            if (
                self.controller.state in _TERMINAL_STATES
                or self._pending_program_path is not None
            ):
                result = validate_program(self.program_path)
            else:
                result = self.controller.validate(self.program_path)
        except Exception as error:
            self._show_error("STUDENT_VALIDATION_FAILED", error)
            return
        self._render_validation(result)

    @pyqtSlot()
    def _run(self) -> None:
        if not self._save():
            return
        if self._pending_program_path is not None:
            self._show_error(
                "STUDENT_PROGRAM_SYNC_PENDING",
                RuntimeError(
                    "程序尚未同步到控制器，不能运行"
                ),
            )
            return
        try:
            result = self.controller.validate(self.program_path)
            if not result.ok:
                self._render_validation(result)
                return
            self.controller.start()
        except Exception as error:
            self._show_error("STUDENT_RUN_START_FAILED", error)

    def _render_validation(self, result: Any) -> None:
        if result.ok:
            self.console.append("代码检查：PASS")
            return
        self.console.append("代码检查：FAIL")
        for issue in result.issues:
            location = (
                f"第 {issue.line} 行"
                if issue.line is not None
                else "文件"
            )
            self.console.append(
                f"[{issue.code}] {location}：{issue.message}"
            )

    def _call_controller(self, operation: str) -> None:
        try:
            getattr(self.controller, operation)()
        except Exception as error:
            self._show_error(
                f"STUDENT_{operation.upper()}_FAILED",
                error,
            )

    @property
    def pending_program_path(self) -> Path | None:
        return self._pending_program_path

    @property
    def operation_in_progress(self) -> bool:
        return self._operation_thread is not None

    def request_stop(self) -> bool:
        if self.controller.state not in _ACTIVE_STATES:
            return False
        if (
            self._cancel_requested_for_run
            or self.operation_in_progress
        ):
            return False
        self._cancel_requested_for_run = True
        try:
            if self._start_operation("cancel"):
                return True
        except Exception as error:
            self._show_error("STUDENT_CANCEL_FAILED", error)
        self._cancel_requested_for_run = False
        return False

    def request_reset(self) -> bool:
        if (
            self.controller.state not in _TERMINAL_STATES
            or self.operation_in_progress
        ):
            return False
        try:
            return self._start_operation(
                "reset",
                staged_path=self._pending_program_path,
            )
        except Exception as error:
            self._show_error("STUDENT_RESET_FAILED", error)
            return False

    def _start_operation(
        self,
        operation: str,
        *,
        staged_path: Path | None = None,
    ) -> bool:
        if self.operation_in_progress:
            return False
        thread = QThread(self)
        worker = _ControllerOperationWorker(
            controller=self.controller,
            operation=operation,
            staged_path=staged_path,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(self._operation_succeeded)
        worker.failed.connect(self._operation_failed)
        worker.succeeded.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.succeeded.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._operation_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._operation_thread = thread
        self._operation_worker = worker
        self._operation_name = operation
        self._apply_state(self.controller.state)
        try:
            thread.start()
        except Exception:
            self._operation_thread = None
            self._operation_worker = None
            self._operation_name = None
            thread.deleteLater()
            try:
                self._apply_state(self.controller.state)
            except Exception:
                pass
            raise
        return True

    @pyqtSlot(str, object)
    def _operation_succeeded(
        self,
        operation: str,
        payload: Any,
    ) -> None:
        if operation == "cancel":
            self.console.append("停止请求已完成")
            return
        if operation != "reset":
            return
        if isinstance(payload, Mapping):
            program_path = payload.get("program_path")
            validation = payload.get("validation")
            if program_path is not None:
                self.program_path = Path(program_path).resolve()
                self.path_label.setText(str(self.program_path))
                self._pending_program_path = None
            if validation is not None:
                self._render_validation(validation)
        self.console.append("场景复位完成")

    @pyqtSlot(str, str)
    def _operation_failed(
        self,
        operation: str,
        message: str,
    ) -> None:
        if operation == "cancel":
            self._cancel_requested_for_run = False
            code = "STUDENT_CANCEL_FAILED"
        else:
            code = "STUDENT_RESET_FAILED"
        self._show_error(code, RuntimeError(message))

    @pyqtSlot()
    def _operation_thread_finished(self) -> None:
        thread = self.sender()
        if thread is not self._operation_thread:
            return
        self._operation_thread = None
        self._operation_worker = None
        self._operation_name = None
        self._apply_state(self.controller.state)

    @pyqtSlot(object)
    def _render_snapshot(self, snapshot: Any) -> None:
        try:
            state = snapshot.state
            if not isinstance(state, RunState):
                state = RunState(str(state))
            command = snapshot.current_command or "—"
            command_count = int(snapshot.command_count)
            elapsed = float(snapshot.elapsed_seconds)
            evidence = snapshot.evidence_dir or "—"
            tcp = getattr(snapshot, "tcp_mm", None)
            if tcp is not None:
                tcp_values = tuple(float(value) for value in tcp)
                if len(tcp_values) != 3:
                    raise ValueError("tcp_mm must contain X, Y and Z")
            else:
                tcp_values = None
        except Exception as error:
            self._show_error("STUDENT_SNAPSHOT_INVALID", error)
            return

        previous_state = self._last_state
        if (
            state in _ACTIVE_STATES
            and previous_state not in _ACTIVE_STATES
        ):
            self._cancel_requested_for_run = False
        self._last_state = state
        self.state_label.setText(state.value)
        self._style_state_badge(state)
        self.command_label.setText(f"当前命令：{command}")
        self.metrics_label.setText(
            f"命令数：{command_count}　运行时间：{elapsed:.1f} s"
        )
        if tcp_values is None:
            self.tcp_label.setText("TCP：—")
        else:
            self.tcp_label.setText(
                f"TCP：X={tcp_values[0]:.1f} mm　"
                f"Y={tcp_values[1]:.1f} mm　"
                f"Z={tcp_values[2]:.1f} mm"
            )
        self.evidence_label.setText(f"证据目录：{evidence}")
        error = snapshot.error
        if isinstance(error, Mapping):
            self.console.append(
                f"[{error.get('code', 'FAILED')}] "
                f"{error.get('message', '运行失败')}"
            )
        self._apply_state(state)

    def _style_state_badge(self, state: RunState) -> None:
        colors = {
            RunState.EMPTY: ("#475569", "#E2E8F0", "#CBD5E1"),
            RunState.LOADED: ("#1E40AF", "#DBEAFE", "#93C5FD"),
            RunState.VALIDATED: ("#1D4ED8", "#DBEAFE", "#60A5FA"),
            RunState.RUNNING: ("#166534", "#DCFCE7", "#86EFAC"),
            RunState.PAUSED: ("#92400E", "#FEF3C7", "#FCD34D"),
            RunState.PASSED: ("#166534", "#DCFCE7", "#4ADE80"),
            RunState.FAILED: ("#991B1B", "#FEE2E2", "#FCA5A5"),
            RunState.CANCELLED: ("#334155", "#E2E8F0", "#94A3B8"),
            RunState.RESETTING: ("#6B21A8", "#F3E8FF", "#D8B4FE"),
        }
        foreground, background, border = colors[state]
        self.state_label.setProperty("runState", state.value)
        self.state_label.setStyleSheet(
            "padding: 4px 10px; border-radius: 10px; "
            "font-weight: 700; "
            f"color: {foreground}; background: {background}; "
            f"border: 1px solid {border};"
        )

    def _apply_state(self, state: RunState) -> None:
        if self.operation_in_progress:
            self.editor.setReadOnly(True)
            for button in (
                self.open_button,
                self.save_button,
                self.save_as_button,
                self.validate_button,
                self.run_button,
                self.pause_button,
                self.resume_button,
                self.step_button,
                self.stop_button,
                self.reset_button,
            ):
                button.setEnabled(False)
            return
        editable = state in {
            RunState.EMPTY,
            RunState.LOADED,
            RunState.VALIDATED,
            RunState.PASSED,
            RunState.FAILED,
            RunState.CANCELLED,
        }
        self.editor.setReadOnly(not editable)
        self.open_button.setEnabled(editable)
        self.save_button.setEnabled(editable)
        self.save_as_button.setEnabled(editable)
        self.validate_button.setEnabled(
            state
            in {
                RunState.LOADED,
                RunState.VALIDATED,
                RunState.PASSED,
                RunState.FAILED,
                RunState.CANCELLED,
            }
        )
        self.run_button.setEnabled(
            state is RunState.VALIDATED
            and self._pending_program_path is None
        )
        self.pause_button.setEnabled(state is RunState.RUNNING)
        self.resume_button.setEnabled(state is RunState.PAUSED)
        self.step_button.setEnabled(state is RunState.PAUSED)
        self.stop_button.setEnabled(
            state in {RunState.RUNNING, RunState.PAUSED}
        )
        self.reset_button.setEnabled(
            state in _TERMINAL_STATES
        )

    def release_subscription(self) -> None:
        callback = self._unsubscribe
        if callback is None:
            return
        self._unsubscribe = None
        try:
            callback()
        except Exception:
            self._unsubscribe = callback
            raise

    def _show_error(self, code: str, error: BaseException) -> None:
        try:
            message = str(error)
        except BaseException:
            message = "<unprintable>"
        self.console.append(f"[{code}] {message[:2000]}")

    def closeEvent(self, event) -> None:
        try:
            if self.controller.state in _ACTIVE_STATES:
                self.request_stop()
            if self.operation_in_progress:
                raise RuntimeError(
                    "学生程序控制操作仍在执行"
                )
            if not self.controller.wait_for_quiescence(0):
                raise RuntimeError(
                    "学生程序后端尚未完成安全收尾"
                )
        except Exception as error:
            if not self._close_pending_reported:
                self._close_pending_reported = True
                self._show_error("STUDENT_CLOSE_PENDING", error)
            event.ignore()
            return
        try:
            self.release_subscription()
        except Exception as error:
            self._show_error("STUDENT_UNSUBSCRIBE_FAILED", error)
            event.ignore()
            return
        event.accept()
