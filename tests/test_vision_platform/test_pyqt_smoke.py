from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import vision_platform.ui.pyqt_app as pyqt_app
import pytest
from PyQt5.QtCore import QCoreApplication, QEvent, QPoint, QThread, QTimer, Qt
from PyQt5.QtWidgets import QApplication
from vision_platform.experiments.models import (
    ExperimentAcceptance,
    ExperimentDefinition,
    ExperimentRunContext,
)
from vision_platform.robot.safety import WorkspacePolicy
from vision_platform.student.protocol import RunState
from vision_platform.student.runner import StudentProgramController
from vision_platform.ui.pyqt_app import VisionLabWindow
from vision_platform.ui.student_program_panel import StudentProgramPanel


class FakeEventBus:
    def __init__(
        self,
        *,
        unsubscribe_failures=0,
        subscribe_fail_on_attempts=(),
        subscribe_entered=None,
        allow_subscribe=None,
    ):
        self.handlers = []
        self.unsubscribe_calls = 0
        self.unsubscribe_attempts = 0
        self.unsubscribe_failures = unsubscribe_failures
        self.subscribe_attempts = 0
        self.subscribe_fail_on_attempts = frozenset(
            subscribe_fail_on_attempts
        )
        self.subscribe_entered = subscribe_entered
        self.allow_subscribe = allow_subscribe

    def subscribe(self, handler):
        self.subscribe_attempts += 1
        if self.subscribe_attempts in self.subscribe_fail_on_attempts:
            raise RuntimeError("event subscribe failed")
        if self.subscribe_entered is not None:
            self.subscribe_entered.set()
        if self.allow_subscribe is not None:
            assert self.allow_subscribe.wait(timeout=2.0)
        self.handlers.append(handler)
        unsubscribed = False

        def unsubscribe():
            nonlocal unsubscribed
            if unsubscribed:
                return
            self.unsubscribe_attempts += 1
            if self.unsubscribe_failures:
                self.unsubscribe_failures -= 1
                raise RuntimeError("event unsubscribe failed")
            unsubscribed = True
            self.unsubscribe_calls += 1
            if handler in self.handlers:
                self.handlers.remove(handler)

        return unsubscribe


class FakeTool:
    def __init__(self):
        self.off_calls = 0

    def off(self):
        self.off_calls += 1


class DisconnectedApplication:
    def __init__(self, *, with_event_bus=False, event_bus=None):
        self.close_calls = 0
        self.event_bus = (
            event_bus
            if event_bus is not None
            else FakeEventBus() if with_event_bus else None
        )
        self.tool = FakeTool()

    def close(self):
        self.close_calls += 1


class FullStudentApplication(DisconnectedApplication):
    def __init__(self, project_root, *, event_bus=None):
        super().__init__(event_bus=event_bus)
        self.workspace = WorkspacePolicy(
            x_mm=(-250.0, 250.0),
            y_mm=(-250.0, 250.0),
            z_mm=(0.0, 300.0),
            safe_z_mm=80.0,
        )
        self.config = SimpleNamespace(
            student={
                "speed_range": [1, 30],
                "max_runtime_s": 60,
                "max_commands": 200,
                "command_timeout_s": 10,
                "max_sleep_s": 5,
                "tool_on_max_z_mm": 35,
                "output": "student-runs",
            },
            project_root=Path(project_root).resolve(),
            robot_backend="sim",
        )

    @staticmethod
    def _resolve_project_path(config, value):
        selected = Path(value).expanduser()
        if not selected.is_absolute():
            selected = config.project_root / selected
        return selected.resolve()


class FakeSession:
    def __init__(
        self,
        application,
        *,
        unsubscribe_failures=0,
        close_failures=0,
    ):
        self.application = application
        self.handlers = []
        self.close_calls = 0
        self.unsubscribe_attempts = 0
        self.unsubscribe_failures = unsubscribe_failures
        self.close_failures = close_failures

    def subscribe(self, handler):
        self.handlers.append(handler)
        unsubscribed = False

        def unsubscribe():
            nonlocal unsubscribed
            if unsubscribed:
                return
            self.unsubscribe_attempts += 1
            if self.unsubscribe_failures:
                self.unsubscribe_failures -= 1
                raise RuntimeError("session unsubscribe failed")
            unsubscribed = True
            if handler in self.handlers:
                self.handlers.remove(handler)

        return unsubscribe

    def replace(self, application):
        self.application = application
        for handler in tuple(self.handlers):
            handler(application)

    def close(self):
        self.close_calls += 1
        if self.close_failures:
            self.close_failures -= 1
            raise RuntimeError("session close failed")
        self.application.close()


class FakeCloseEvent:
    def __init__(self):
        self.accepted = False
        self.ignored = False

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


def operation_controls(window):
    return (
        window.camera_backend_combo,
        window.robot_backend_combo,
        window.mode_combo,
        window.connect_button,
        window.calibrate_button,
        window.start_button,
        window.pause_button,
        window.resume_button,
        window.reset_button,
        window.emergency_button,
    )


def test_pyqt_window_constructs_without_camera_or_robot_connection(qtbot):
    application = DisconnectedApplication()
    window = VisionLabWindow(application=application)
    qtbot.addWidget(window)

    assert window.camera_backend_combo.count() == 3
    assert window.robot_backend_combo.count() == 2
    assert "未连接" in window.status_label.text()
    assert window.preview_label.minimumWidth() >= 640
    assert window.start_button.minimumHeight() >= 44
    assert window.emergency_button.minimumHeight() >= 44


def test_pyqt_window_exposes_teaching_workflow_controls(qtbot):
    window = VisionLabWindow(application=DisconnectedApplication())
    qtbot.addWidget(window)

    assert window.mode_combo.currentText() == "全自动"
    assert window.calibration_table.columnCount() == 4
    assert window.detection_table.columnCount() == 6
    assert window.pause_button.isEnabled() is False
    assert window.resume_button.isEnabled() is False
    assert "像素坐标" in window.pixel_coordinate_label.text()
    assert "世界坐标" in window.world_coordinate_label.text()


def test_pyqt_window_adds_disabled_student_tab_without_full_session(qtbot):
    window = VisionLabWindow(application=DisconnectedApplication())
    qtbot.addWidget(window)

    labels = [
        window.tabs.tabText(index)
        for index in range(window.tabs.count())
    ]
    student_index = labels.index("学生编程")

    assert window.tabs.isTabEnabled(student_index) is False
    assert (
        window.student_program_panel.text()
        == "当前窗口未配置学生程序会话"
    )
    assert window.student_controller is None
    assert (
        window.subtitle_label.text()
        == "仿真标定 · 视觉识别 · 学生编程 · 六轴机械臂分类闭环"
    )


def test_pyqt_window_disables_student_tab_for_incomplete_session(qtbot):
    application = DisconnectedApplication()
    session = FakeSession(application)
    window = VisionLabWindow(application=application, session=session)
    qtbot.addWidget(window)

    student_index = window.tabs.indexOf(window.student_program_panel)

    assert student_index >= 0
    assert window.tabs.isTabEnabled(student_index) is False
    assert window.student_controller is None


def test_pyqt_window_builds_student_workspace_from_session_config(
    qtbot,
    tmp_path,
):
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    window = VisionLabWindow(application=application, session=session)
    qtbot.addWidget(window)

    student_index = window.tabs.indexOf(window.student_program_panel)

    assert student_index >= 0
    assert window.tabs.isTabEnabled(student_index) is True
    assert isinstance(window.student_program_panel, StudentProgramPanel)
    assert isinstance(window.student_controller, StudentProgramController)
    assert window.student_controller._session is session
    assert (
        window.student_controller._output_root
        == (tmp_path / "student-runs").resolve()
    )

    window.close()


def test_student_toolbar_buttons_fit_supported_minimum_window(
    qtbot,
    tmp_path,
):
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    window = VisionLabWindow(application=application, session=session)
    qtbot.addWidget(window)
    window.resize(1180, 760)
    student_index = window.tabs.indexOf(window.student_program_panel)
    window.tabs.setCurrentIndex(student_index)
    window.show()
    QApplication.processEvents()
    qtbot.waitUntil(
        lambda: (
            window.size().width() == 1180
            and window.size().height() == 760
            and window.student_program_panel.isVisible()
        ),
        timeout=2000,
    )

    panel = window.student_program_panel
    assert isinstance(panel, StudentProgramPanel)
    buttons = (
        panel.open_button,
        panel.save_button,
        panel.save_as_button,
        panel.validate_button,
        panel.run_button,
        panel.pause_button,
        panel.resume_button,
        panel.step_button,
        panel.stop_button,
        panel.reset_button,
    )
    sensible_horizontal_padding = 24

    for button in buttons:
        required_width = (
            button.fontMetrics().horizontalAdvance(button.text())
            + sensible_horizontal_padding
        )
        assert button.width() >= required_width
        button_left = button.mapTo(panel, QPoint(0, 0)).x()
        button_right = button_left + button.width()
        assert button_left >= panel.contentsRect().left()
        assert button_right <= panel.contentsRect().right() + 1


def test_pyqt_close_keeps_owner_alive_until_student_backend_is_quiescent(
    qtbot,
    tmp_path,
    monkeypatch,
):
    controllers = []
    cancel_entered = Event()
    cancel_release = Event()

    class FakeLifecycleStudentController:
        def __init__(
            self,
            *,
            session,
            execution_policy,
            output_root,
        ):
            self.session = session
            self.execution_policy = execution_policy
            self.output_root = Path(output_root)
            self.state = RunState.RUNNING
            self.handlers = []
            self.cancel_calls = 0
            self.unsubscribe_calls = 0
            self.backend_active = True
            self.quiescence_timeouts = []
            controllers.append(self)

        def subscribe(self, handler):
            self.handlers.append(handler)
            unsubscribed = False

            def unsubscribe():
                nonlocal unsubscribed
                if unsubscribed:
                    return
                unsubscribed = True
                self.unsubscribe_calls += 1
                if handler in self.handlers:
                    self.handlers.remove(handler)

            return unsubscribe

        def cancel(self):
            self.cancel_calls += 1
            cancel_entered.set()
            assert cancel_release.wait(timeout=2)
            self.state = RunState.CANCELLED

        def wait_for_quiescence(self, timeout_s):
            self.quiescence_timeouts.append(timeout_s)
            return not self.backend_active

        def load(self, path):
            return Path(path)

        def validate(self, _path=None):
            raise AssertionError("close attempted validation")

        def start(self, *, paused=False):
            raise AssertionError(f"close attempted start: {paused}")

        def pause(self):
            raise AssertionError("close attempted pause")

        def step(self):
            raise AssertionError("close attempted step")

        def reset(self):
            raise AssertionError("close attempted reset")

    monkeypatch.setattr(
        pyqt_app,
        "StudentProgramController",
        FakeLifecycleStudentController,
    )
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    window = VisionLabWindow(application=application, session=session)
    qtbot.addWidget(window)
    controller = controllers[0]
    first_event = FakeCloseEvent()
    second_event = FakeCloseEvent()
    final_event = FakeCloseEvent()
    first_close_returned = Event()
    second_close_returned = Event()
    close_observations = []

    def release_after_repeated_close() -> None:
        assert cancel_entered.wait(timeout=2)
        close_observations.append(
            first_close_returned.wait(timeout=0.25)
        )
        close_observations.append(
            second_close_returned.wait(timeout=0.25)
        )
        cancel_release.set()

    gate = Thread(target=release_after_repeated_close, daemon=True)
    gate.start()
    window.closeEvent(first_event)
    first_close_returned.set()
    assert cancel_entered.wait(timeout=2)

    assert first_event.ignored is True
    assert first_event.accepted is False
    assert controller.cancel_calls == 1
    assert session.close_calls == 0
    assert application.close_calls == 0
    assert len(session.handlers) == 1
    assert len(controller.handlers) == 1
    assert window._closing is False
    assert (
        window.view_model.snapshot().error_code
        == "UI_STUDENT_STOP_PENDING"
    )
    first_logs = window.view_model.snapshot().logs

    window.closeEvent(second_event)
    second_close_returned.set()
    gate.join(timeout=2)

    assert second_event.ignored is True
    assert second_event.accepted is False
    assert close_observations == [True, True]
    assert controller.cancel_calls == 1
    assert window.view_model.snapshot().logs == first_logs
    assert session.close_calls == 0
    assert application.close_calls == 0
    qtbot.waitUntil(
        lambda: window.student_program_panel.operation_in_progress is False,
        timeout=2000,
    )

    controller.backend_active = False
    window.closeEvent(final_event)

    assert final_event.accepted is True
    assert final_event.ignored is False
    assert controller.cancel_calls == 1
    assert controller.quiescence_timeouts[-1:] == [0]
    assert controller.handlers == []
    assert controller.unsubscribe_calls == 1
    assert session.handlers == []
    assert session.close_calls == 1
    assert application.close_calls == 1


def test_pyqt_close_reuses_prior_async_stop_request(
    qtbot,
    tmp_path,
    monkeypatch,
):
    controllers = []
    cancel_entered = Event()
    cancel_release = Event()

    class StopBeforeCloseController:
        def __init__(self, **_kwargs):
            self.state = RunState.RUNNING
            self.handlers = []
            self.cancel_calls = 0
            self.backend_active = True
            controllers.append(self)

        def subscribe(self, handler):
            self.handlers.append(handler)

            def unsubscribe():
                if handler in self.handlers:
                    self.handlers.remove(handler)

            return unsubscribe

        def cancel(self):
            self.cancel_calls += 1
            cancel_entered.set()
            assert cancel_release.wait(timeout=2)
            self.state = RunState.CANCELLED
            self.backend_active = False

        def wait_for_quiescence(self, timeout_s):
            assert timeout_s == 0
            return not self.backend_active

        def load(self, path):
            return Path(path)

        def validate(self, _path=None):
            raise AssertionError("unexpected validate")

        def start(self, *, paused=False):
            raise AssertionError(f"unexpected start: {paused}")

        def pause(self):
            raise AssertionError("unexpected pause")

        def resume(self):
            raise AssertionError("unexpected resume")

        def step(self):
            raise AssertionError("unexpected step")

        def reset(self):
            raise AssertionError("unexpected reset")

    monkeypatch.setattr(
        pyqt_app,
        "StudentProgramController",
        StopBeforeCloseController,
    )
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    window = VisionLabWindow(application=application, session=session)
    qtbot.addWidget(window)
    controller = controllers[0]
    click_returned = Event()
    close_returned = Event()
    click_observations = []

    def release_after_stop_click() -> None:
        assert cancel_entered.wait(timeout=2)
        click_observations.append(
            click_returned.wait(timeout=0.25)
        )
        click_observations.append(
            close_returned.wait(timeout=0.25)
        )
        cancel_release.set()

    gate = Thread(target=release_after_stop_click, daemon=True)
    gate.start()
    qtbot.mouseClick(
        window.student_program_panel.stop_button,
        Qt.LeftButton,
    )
    click_returned.set()
    close_event = FakeCloseEvent()
    window.closeEvent(close_event)
    close_returned.set()
    gate.join(timeout=2)

    assert click_observations == [True, True]
    assert close_event.ignored is True
    assert controller.cancel_calls == 1
    assert session.close_calls == 0
    assert application.close_calls == 0

    qtbot.waitUntil(
        lambda: window.student_program_panel.operation_in_progress is False,
        timeout=2000,
    )
    final_event = FakeCloseEvent()
    window.closeEvent(final_event)

    assert final_event.accepted is True
    assert controller.cancel_calls == 1
    assert session.close_calls == 1
    assert application.close_calls == 1


def test_pyqt_window_close_calls_application_close(qtbot):
    application = DisconnectedApplication()
    window = VisionLabWindow(application=application)
    qtbot.addWidget(window)

    window.close()
    window.close()

    assert application.close_calls == 1


def test_pyqt_window_populates_calibration_table_from_report(qtbot, tmp_path):
    application = DisconnectedApplication()
    window = VisionLabWindow(application=application)
    qtbot.addWidget(window)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "pixel_points": [[100, 120], [200, 120], [100, 220]],
                "world_points_mm": [[35, -70], [100, -70], [35, 70]],
            }
        ),
        encoding="utf-8",
    )
    report = SimpleNamespace(
        status="PASS",
        rms_error_mm=0.2,
        max_error_mm=0.4,
        calibration_path=calibration_path,
        report_path=tmp_path / "report.json",
    )

    window._action_finished(
        SimpleNamespace(action="calibrate", payload=report, frame=None)
    )

    assert window.calibration_table.item(0, 1).text() == "100.0"
    assert window.calibration_table.item(2, 3).text() == "(35.0, 70.0)"


def test_pyqt_window_rebinds_replaced_application_event_bus(qtbot):
    first = DisconnectedApplication(with_event_bus=True)
    second = DisconnectedApplication(with_event_bus=True)
    session = FakeSession(first)
    window = VisionLabWindow(application=first, session=session)
    qtbot.addWidget(window)

    assert len(first.event_bus.handlers) == 1

    session.replace(second)

    assert window.application is second
    assert first.event_bus.handlers == []
    assert first.event_bus.unsubscribe_calls == 1
    assert len(second.event_bus.handlers) == 1


def test_pyqt_window_session_close_is_used_exactly_once(qtbot):
    first = DisconnectedApplication()
    second = DisconnectedApplication()
    session = FakeSession(first)
    window = VisionLabWindow(application=first, session=session)
    qtbot.addWidget(window)
    session.replace(second)

    window.close()
    window.close()

    assert session.handlers == []
    assert session.close_calls == 1
    assert first.close_calls == 0
    assert second.close_calls == 1


def test_pyqt_new_worker_reads_current_session_application(
    qtbot,
    monkeypatch,
):
    first = DisconnectedApplication()
    second = DisconnectedApplication()
    session = FakeSession(first)
    window = VisionLabWindow(application=first, session=session)
    qtbot.addWidget(window)
    session.application = second
    created = []

    class FakeSignal:
        def connect(self, _handler):
            pass

    class FakeThread:
        def __init__(self, _parent):
            self.started = FakeSignal()
            self.finished = FakeSignal()

        def start(self):
            pass

        def quit(self):
            pass

        def wait(self, _timeout):
            return True

    class FakeWorker:
        def __init__(self, **kwargs):
            self.application = kwargs["application"]
            self.finished = FakeSignal()
            self.failed = FakeSignal()
            created.append(self)

        def moveToThread(self, _thread):
            pass

        def run(self):
            pass

        def deleteLater(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setattr(pyqt_app, "QThread", FakeThread)
    monkeypatch.setattr(pyqt_app, "_ActionWorker", FakeWorker)

    window._start_action("reset")

    assert created[0].application is second
    window.close()


def test_pyqt_emergency_stop_reads_current_session_application(qtbot):
    first = DisconnectedApplication()
    second = DisconnectedApplication()
    session = FakeSession(first)
    window = VisionLabWindow(application=first, session=session)
    qtbot.addWidget(window)
    session.application = second

    window._emergency_stop()

    assert first.tool.off_calls == 0
    assert second.tool.off_calls == 1


def test_pyqt_close_serializes_with_in_flight_application_replacement(qtbot):
    subscribe_entered = Event()
    allow_subscribe = Event()
    first = DisconnectedApplication(with_event_bus=True)
    second_bus = FakeEventBus(
        subscribe_entered=subscribe_entered,
        allow_subscribe=allow_subscribe,
    )
    second = DisconnectedApplication(event_bus=second_bus)
    session = FakeSession(first)
    window = VisionLabWindow(application=first, session=session)
    qtbot.addWidget(window)
    errors = []

    def replace() -> None:
        try:
            session.replace(second)
        except BaseException as error:
            errors.append(error)

    close_event = FakeCloseEvent()
    close_started = Event()

    def close() -> None:
        try:
            assert subscribe_entered.wait(timeout=2.0)
            close_started.set()
            window.closeEvent(close_event)
        except BaseException as error:
            errors.append(error)

    replace_thread = Thread(target=replace)
    replace_thread.start()
    replace_thread.join(timeout=2.0)
    close_thread = Thread(target=close)
    release_thread = Thread(
        target=lambda: (
            close_started.wait(timeout=2.0),
            time.sleep(0.05),
            allow_subscribe.set(),
        )
    )
    close_thread.start()
    release_thread.start()
    QApplication.processEvents()
    close_thread.join(timeout=2.0)
    release_thread.join(timeout=2.0)

    assert not replace_thread.is_alive()
    assert not close_thread.is_alive()
    assert not release_thread.is_alive()
    assert errors == []
    assert close_event.accepted is True
    assert close_event.ignored is False
    assert second_bus.handlers == []
    assert window._event_unsubscribe is None
    assert session.close_calls == 1
    assert second.close_calls == 1


def test_pyqt_close_retries_failed_unsubscribes_without_reclosing_owner(
    qtbot,
    monkeypatch,
):
    event_bus = FakeEventBus(unsubscribe_failures=1)
    application = DisconnectedApplication(event_bus=event_bus)
    session = FakeSession(application, unsubscribe_failures=1)
    window = VisionLabWindow(application=application, session=session)
    qtbot.addWidget(window)
    original_view_unsubscribe = window._view_unsubscribe
    view_unsubscribe_attempts = 0

    def flaky_view_unsubscribe() -> None:
        nonlocal view_unsubscribe_attempts
        view_unsubscribe_attempts += 1
        if view_unsubscribe_attempts == 1:
            raise RuntimeError("view unsubscribe failed")
        original_view_unsubscribe()

    window._view_unsubscribe = flaky_view_unsubscribe
    first_event = FakeCloseEvent()
    second_event = FakeCloseEvent()

    window.closeEvent(first_event)

    assert first_event.ignored is True
    assert first_event.accepted is False
    assert session.close_calls == 1
    assert application.close_calls == 1
    assert session.unsubscribe_attempts == 1
    assert event_bus.unsubscribe_attempts == 1
    assert view_unsubscribe_attempts == 1
    assert window.view_model.snapshot().error_code == "UI_CLOSE_FAILED"
    assert "UI_CLOSE_FAILED" in window.error_label.text()
    assert all(
        control.isEnabled() is False
        for control in operation_controls(window)
    )

    def reject_worker_creation(**_kwargs):
        raise AssertionError("closing window created a worker")

    monkeypatch.setattr(pyqt_app, "_ActionWorker", reject_worker_creation)
    window._start_action("reset")
    assert window._worker is None
    assert window._thread is None
    window._thread_finished()
    assert all(
        control.isEnabled() is False
        for control in operation_controls(window)
    )

    window.closeEvent(second_event)

    assert second_event.accepted is True
    assert second_event.ignored is False
    assert session.close_calls == 1
    assert application.close_calls == 1
    assert session.unsubscribe_attempts == 2
    assert event_bus.unsubscribe_attempts == 2
    assert view_unsubscribe_attempts == 2
    assert session.handlers == []
    assert event_bus.handlers == []
    assert window._session_unsubscribe is None
    assert window._event_unsubscribe is None
    assert window._view_unsubscribe is None
    assert window._close_complete is True


def test_pyqt_shutdown_retries_ignored_close_without_reclosing_owner(qtbot):
    event_bus = FakeEventBus(unsubscribe_failures=1)
    application = DisconnectedApplication(event_bus=event_bus)
    session = FakeSession(application, unsubscribe_failures=1)
    window = VisionLabWindow(application=application, session=session)
    qtbot.addWidget(window)

    window.shutdown(timeout_ms=1000)
    window.shutdown(timeout_ms=1000)

    assert window.owner_shutdown_complete is True
    assert session.close_calls == 1
    assert application.close_calls == 1
    assert session.unsubscribe_attempts == 2
    assert event_bus.unsubscribe_attempts == 2


def test_pyqt_shutdown_surfaces_owner_close_error_without_retry(qtbot):
    application = DisconnectedApplication()
    session = FakeSession(application, close_failures=1)
    window = VisionLabWindow(application=application, session=session)
    qtbot.addWidget(window)

    with pytest.raises(RuntimeError, match="session close failed"):
        window.shutdown(timeout_ms=1000)
    with pytest.raises(RuntimeError, match="session close failed"):
        window.shutdown(timeout_ms=1000)

    assert session.close_calls == 1
    assert application.close_calls == 0


def test_pyqt_owner_close_error_is_visible_and_does_not_block_process_exit(
    qtbot,
):
    event_bus = FakeEventBus()
    application = DisconnectedApplication(event_bus=event_bus)
    session = FakeSession(application, close_failures=1)
    window = VisionLabWindow(application=application, session=session)
    qtbot.addWidget(window)
    first_event = FakeCloseEvent()
    second_event = FakeCloseEvent()

    window.closeEvent(first_event)

    assert first_event.accepted is True
    assert first_event.ignored is False
    assert session.close_calls == 1
    assert application.close_calls == 0
    assert session.handlers == []
    assert event_bus.handlers == []
    assert window._session_unsubscribe is None
    assert window._event_unsubscribe is None
    assert window._view_unsubscribe is None
    assert window.view_model.snapshot().error_code == "UI_CLOSE_FAILED"
    assert "UI_CLOSE_FAILED" in window.error_label.text()
    assert window._close_complete is True
    for control in operation_controls(window):
        assert control.isEnabled() is False

    window.closeEvent(second_event)

    assert second_event.accepted is True
    assert second_event.ignored is False
    assert session.close_calls == 1
    assert application.close_calls == 0
    assert window._close_complete is True


def test_pyqt_owner_close_error_has_priority_over_unsubscribe_error(qtbot):
    application = DisconnectedApplication()
    session = FakeSession(
        application,
        unsubscribe_failures=1,
        close_failures=1,
    )
    window = VisionLabWindow(application=application, session=session)
    qtbot.addWidget(window)
    first_event = FakeCloseEvent()
    second_event = FakeCloseEvent()

    window.closeEvent(first_event)

    assert first_event.ignored is True
    assert first_event.accepted is False
    snapshot = window.view_model.snapshot()
    assert snapshot.error_code == "UI_CLOSE_FAILED"
    assert "session close failed" in snapshot.status_text
    assert "session unsubscribe failed" not in snapshot.status_text
    assert session.close_calls == 1

    window.closeEvent(second_event)

    assert second_event.accepted is True
    assert second_event.ignored is False
    assert session.close_calls == 1
    assert window._close_complete is True


def test_pyqt_event_unsubscribe_error_is_visible_before_view_unsubscribe(
    qtbot,
):
    event_bus = FakeEventBus(unsubscribe_failures=1)
    application = DisconnectedApplication(event_bus=event_bus)
    session = FakeSession(application)
    window = VisionLabWindow(application=application, session=session)
    qtbot.addWidget(window)
    first_event = FakeCloseEvent()
    second_event = FakeCloseEvent()

    window.closeEvent(first_event)

    assert first_event.ignored is True
    assert first_event.accepted is False
    assert "UI_CLOSE_FAILED" in window.error_label.text()
    assert window.view_model.snapshot().error_code == "UI_CLOSE_FAILED"
    assert window._view_unsubscribe is None
    assert session.close_calls == 1

    window.closeEvent(second_event)

    assert second_event.accepted is True
    assert second_event.ignored is False
    assert session.close_calls == 1
    assert event_bus.handlers == []


def test_pyqt_view_unsubscribe_error_remains_visible_and_retryable(qtbot):
    application = DisconnectedApplication()
    session = FakeSession(application)
    window = VisionLabWindow(application=application, session=session)
    qtbot.addWidget(window)
    original_view_unsubscribe = window._view_unsubscribe
    attempts = 0

    def unsubscribe_then_fail_once() -> None:
        nonlocal attempts
        attempts += 1
        original_view_unsubscribe()
        if attempts == 1:
            raise RuntimeError("view unsubscribe failed after removal")

    window._view_unsubscribe = unsubscribe_then_fail_once
    first_event = FakeCloseEvent()
    second_event = FakeCloseEvent()

    window.closeEvent(first_event)

    assert first_event.ignored is True
    assert first_event.accepted is False
    assert "UI_CLOSE_FAILED" in window.error_label.text()
    assert window.view_model.snapshot().error_code == "UI_CLOSE_FAILED"
    assert window._view_unsubscribe is unsubscribe_then_fail_once
    assert session.close_calls == 1

    window.closeEvent(second_event)

    assert second_event.accepted is True
    assert second_event.ignored is False
    assert attempts == 2
    assert session.close_calls == 1
    assert window._view_unsubscribe is None


def test_pyqt_close_wait_timeout_preserves_owner_and_subscriptions_for_retry(
    qtbot,
):
    event_bus = FakeEventBus()
    application = DisconnectedApplication(event_bus=event_bus)
    session = FakeSession(application)
    window = VisionLabWindow(application=application, session=session)
    qtbot.addWidget(window)

    class FakeWorker:
        def __init__(self):
            self.cancel_calls = 0

        def cancel(self):
            self.cancel_calls += 1

    class FakeThread:
        def __init__(self):
            self.quit_calls = 0
            self.wait_results = [False, True]

        def quit(self):
            self.quit_calls += 1

        def wait(self, _timeout):
            return self.wait_results.pop(0)

    worker = FakeWorker()
    thread = FakeThread()
    window._worker = worker
    window._thread = thread
    first_event = FakeCloseEvent()
    second_event = FakeCloseEvent()
    enabled_before = [
        control.isEnabled()
        for control in operation_controls(window)
    ]

    window.closeEvent(first_event)

    assert first_event.ignored is True
    assert first_event.accepted is False
    assert window.view_model.snapshot().error_code == "UI_THREAD_STOP_TIMEOUT"
    assert session.close_calls == 0
    assert application.close_calls == 0
    assert len(session.handlers) == 1
    assert len(event_bus.handlers) == 1
    assert window._session_unsubscribe is not None
    assert window._event_unsubscribe is not None
    assert window._view_unsubscribe is not None
    assert window._closing is False
    assert [
        control.isEnabled()
        for control in operation_controls(window)
    ] == enabled_before

    window.closeEvent(second_event)

    assert second_event.accepted is True
    assert second_event.ignored is False
    assert session.close_calls == 1
    assert application.close_calls == 1
    assert worker.cancel_calls == 2
    assert thread.quit_calls == 2


def _task14_catalog(tmp_path):
    definitions = []
    for experiment_id in ("R1-01", "R1-02", "R1-05", "R1-06", "R1-07"):
        scene = (tmp_path / f"{experiment_id}.ttt").resolve()
        scene.write_bytes(f"scene:{experiment_id}".encode("utf-8"))
        scene_sha256 = hashlib.sha256(scene.read_bytes()).hexdigest()
        manifest = (tmp_path / f"{experiment_id}.scene.json").resolve()
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "scene": {
                        "path": str(scene),
                        "sha256": scene_sha256,
                    },
                    "task_contracts": {},
                    "generation": "before-select",
                }
            ),
            encoding="utf-8",
        )
        template = tmp_path / f"{experiment_id}.py"
        template.write_text(
            f"def main(ctx):\n    ctx.log('{experiment_id}')\n",
            encoding="utf-8",
        )
        guide = tmp_path / f"{experiment_id}.md"
        guide.write_text(f"# {experiment_id}\n", encoding="utf-8")
        definitions.append(
            ExperimentDefinition(
                experiment_id=experiment_id,
                pack_id="robot-basics",
                title=f"实验 {experiment_id}",
                version="1.0.0",
                scene=scene,
                scene_manifest=manifest,
                student_template=template.resolve(),
                guide=guide.resolve(),
                capabilities=("robot.home", "scene.probe"),
                workspace={
                    "x_mm": [-250, 250],
                    "y_mm": [-250, 250],
                    "z_mm": [0, 300],
                    "safe_z_mm": 80,
                },
                public_parameters={"camera_path": "/Vision_sensor"},
                hardware_status="PENDING_HARDWARE",
                acceptance=ExperimentAcceptance(
                    probe_kind="scene",
                    automated_checks=(),
                    human_checks=(f"人工检查 {experiment_id}",)
                ),
            )
        )

    class Catalog:
        def __init__(self):
            self.definitions = tuple(definitions)

        def require(self, experiment_id):
            return next(
                item
                for item in self.definitions
                if item.experiment_id == experiment_id
            )

    return Catalog()


class FakeExperimentSession:
    def __init__(
        self,
        vision_session,
        replacement=None,
        failure=None,
        *,
        catalog=None,
        on_select=None,
        restore_failure=None,
        select_delay_s=0.0,
        select_entered=None,
        select_release=None,
        fresh_restore_context=False,
        restore_application=None,
    ):
        self.vision_session = vision_session
        self.replacement = replacement
        self.failure = failure
        self.catalog = catalog
        self.on_select = on_select
        self.restore_failure = restore_failure
        self.select_delay_s = float(select_delay_s)
        self.select_entered = select_entered
        self.select_release = select_release
        self.fresh_restore_context = fresh_restore_context
        self.restore_application = restore_application
        self.select_calls = []
        self.restore_calls = []
        self.current = None

    def capture_snapshot(self):
        return self.current

    def restore_snapshot(self, snapshot):
        self.restore_calls.append(snapshot)
        if self.restore_failure is not None:
            raise self.restore_failure
        restored = snapshot
        if snapshot is not None and self.fresh_restore_context:
            restored = ExperimentRunContext(
                experiment_id=snapshot.experiment_id,
                experiment_version=snapshot.experiment_version,
                scene_path=snapshot.scene_path,
                scene_sha256=snapshot.scene_sha256,
                scene_manifest_path=snapshot.scene_manifest_path,
                public_parameters=snapshot.public_parameters,
                hardware_status=snapshot.hardware_status,
            )
        self.current = restored
        if self.restore_application is not None:
            self.vision_session.replace(self.restore_application)
        if restored is not None:
            self.select_calls.append(restored.experiment_id)
        return restored

    def select(self, experiment_id):
        self.select_calls.append(experiment_id)
        if self.select_entered is not None:
            self.select_entered.set()
        if self.select_release is not None:
            if not self.select_release.wait(timeout=5):
                raise TimeoutError("experiment selection release timed out")
        if self.select_delay_s:
            time.sleep(self.select_delay_s)
        if self.failure is not None:
            raise self.failure
        if self.replacement is not None:
            self.vision_session.replace(self.replacement)
        definition = self.catalog.require(experiment_id)
        context = ExperimentRunContext(
            experiment_id=definition.experiment_id,
            experiment_version=definition.version,
            scene_path=definition.scene,
            scene_sha256=hashlib.sha256(
                definition.scene.read_bytes()
            ).hexdigest(),
            scene_manifest_path=definition.scene_manifest,
            public_parameters=definition.public_parameters,
            hardware_status=definition.hardware_status,
        )
        self.current = context
        if self.on_select is not None:
            self.on_select(experiment_id, definition)
        return context


def test_pyqt_window_inserts_catalog_first_and_binds_selected_template(
    qtbot,
    tmp_path,
):
    catalog = _task14_catalog(tmp_path)
    application = FullStudentApplication(tmp_path)
    replacement = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    experiment_session = FakeExperimentSession(
        session,
        replacement,
        catalog=catalog,
    )
    window = VisionLabWindow(
        application=application,
        session=session,
        experiment_catalog=catalog,
        experiment_session=experiment_session,
    )
    qtbot.addWidget(window)

    assert window.tabs.tabText(0) == "实验目录"
    assert window.experiment_catalog_panel.experiment_combo.count() == 5
    window.experiment_catalog_panel.experiment_combo.setCurrentIndex(2)
    qtbot.mouseClick(
        window.experiment_catalog_panel.select_button,
        Qt.LeftButton,
    )
    qtbot.waitUntil(
        lambda: window._experiment_thread is None,
        timeout=3000,
    )

    definition = catalog.require("R1-05")
    assert experiment_session.select_calls == ["R1-05"]
    assert window.application is replacement
    assert window.student_controller.program_path == (
        definition.student_template.resolve()
    )
    assert "R1-05" in window.student_program_panel.editor.toPlainText()
    assert "已载入 R1-05" in window.status_label.text()


def test_experiment_select_binds_fresh_manifest_rewritten_during_select(
    qtbot,
    tmp_path,
):
    catalog = _task14_catalog(tmp_path)
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)

    def rewrite_manifest(experiment_id, definition):
        assert experiment_id == "R1-05"
        payload = json.loads(
            definition.scene_manifest.read_text(encoding="utf-8")
        )
        payload["generation"] = "after-select"
        replacement = definition.scene_manifest.with_suffix(".new")
        replacement.write_text(json.dumps(payload), encoding="utf-8")
        replacement.replace(definition.scene_manifest)

    experiment_session = FakeExperimentSession(
        session,
        catalog=catalog,
        on_select=rewrite_manifest,
    )
    window = VisionLabWindow(
        application=application,
        session=session,
        experiment_catalog=catalog,
        experiment_session=experiment_session,
    )
    qtbot.addWidget(window)

    assert window._select_experiment("R1-05") is True

    assert window.student_controller._scene_manifest["generation"] == (
        "after-select"
    )
    assert "已载入 R1-05" in window.status_label.text()


def test_experiment_select_rejects_inconsistent_fresh_manifest_and_rolls_back(
    qtbot,
    tmp_path,
):
    catalog = _task14_catalog(tmp_path)
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    experiment_session = FakeExperimentSession(session, catalog=catalog)
    window = VisionLabWindow(
        application=application,
        session=session,
        experiment_catalog=catalog,
        experiment_session=experiment_session,
    )
    qtbot.addWidget(window)
    assert window._select_experiment("R1-01") is True
    old_context = window.student_controller._experiment_context
    old_manifest = dict(window.student_controller._scene_manifest)
    old_program = window.student_controller.program_path
    old_panel_path = window.student_program_panel.program_path
    old_source = window.student_program_panel.editor.toPlainText()

    def corrupt_manifest(experiment_id, definition):
        if experiment_id != "R1-05":
            return
        payload = json.loads(
            definition.scene_manifest.read_text(encoding="utf-8")
        )
        payload["scene"]["sha256"] = "f" * 64
        replacement = definition.scene_manifest.with_suffix(".new")
        replacement.write_text(json.dumps(payload), encoding="utf-8")
        replacement.replace(definition.scene_manifest)

    experiment_session.on_select = corrupt_manifest

    assert window._select_experiment("R1-05") is False

    assert experiment_session.select_calls == ["R1-01", "R1-05", "R1-01"]
    assert experiment_session.current.experiment_id == "R1-01"
    assert window.student_controller._experiment_context is old_context
    assert window.student_controller._scene_manifest == old_manifest
    assert window.student_controller.program_path == old_program
    assert window.student_program_panel.program_path == old_panel_path
    assert window.student_program_panel.editor.toPlainText() == old_source
    assert "已载入 R1-05" not in window.status_label.text()
    assert "载入失败" in window.status_label.text()


@pytest.mark.parametrize(
    "replacement_bytes",
    (
        b"def main(ctx):\n    ctx.log('changed')\n",
        b"\xff\xfe",
    ),
)
def test_experiment_select_rejects_template_rewrite_and_keeps_previous_binding(
    qtbot,
    tmp_path,
    replacement_bytes,
):
    catalog = _task14_catalog(tmp_path)
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    experiment_session = FakeExperimentSession(session, catalog=catalog)
    window = VisionLabWindow(
        application=application,
        session=session,
        experiment_catalog=catalog,
        experiment_session=experiment_session,
    )
    qtbot.addWidget(window)
    assert window._select_experiment("R1-01") is True
    old_context = window.student_controller._experiment_context
    old_manifest = dict(window.student_controller._scene_manifest)
    old_program = window.student_controller.program_path
    old_panel_path = window.student_program_panel.program_path
    old_source = window.student_program_panel.editor.toPlainText()

    def rewrite_template(experiment_id, definition):
        if experiment_id != "R1-05":
            return
        replacement = definition.student_template.with_suffix(".new")
        replacement.write_bytes(replacement_bytes)
        replacement.replace(definition.student_template)

    experiment_session.on_select = rewrite_template

    assert window._select_experiment("R1-05") is False

    assert experiment_session.select_calls == ["R1-01", "R1-05", "R1-01"]
    assert experiment_session.current.experiment_id == "R1-01"
    assert window.student_controller._experiment_context is old_context
    assert window.student_controller._scene_manifest == old_manifest
    assert window.student_controller.program_path == old_program
    assert window.student_program_panel.program_path == old_panel_path
    assert window.student_program_panel.editor.toPlainText() == old_source
    assert "已载入 R1-05" not in window.status_label.text()
    assert "载入失败" in window.status_label.text()


def test_experiment_select_panel_commit_failure_restores_previous_transaction(
    qtbot,
    tmp_path,
    monkeypatch,
):
    catalog = _task14_catalog(tmp_path)
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    experiment_session = FakeExperimentSession(
        session,
        catalog=catalog,
        fresh_restore_context=True,
    )
    window = VisionLabWindow(
        application=application,
        session=session,
        experiment_catalog=catalog,
        experiment_session=experiment_session,
    )
    qtbot.addWidget(window)
    assert window._select_experiment("R1-01") is True
    previous_session = experiment_session.current
    previous_context = window.student_controller._experiment_context
    previous_program = window.student_controller.program_path
    previous_source = window.student_program_panel.editor.toPlainText()
    previous_panel_path = window.student_program_panel.program_path
    previous_console = window.student_program_panel.console.toPlainText()
    real_load_template = window.student_program_panel.load_template

    def commit_then_fail(*args, **kwargs):
        real_load_template(*args, **kwargs)
        raise RuntimeError("panel-commit-failed")

    monkeypatch.setattr(
        window.student_program_panel,
        "load_template",
        commit_then_fail,
    )

    assert window._select_experiment("R1-05") is False

    assert experiment_session.restore_calls == [previous_session]
    assert experiment_session.current is not previous_session
    assert (
        window.student_controller._experiment_context
        is experiment_session.current
    )
    assert (
        window.student_controller._experiment_context.experiment_id
        == previous_context.experiment_id
    )
    assert window.student_controller.program_path == previous_program
    assert window.student_program_panel.program_path == previous_panel_path
    assert window.student_program_panel.editor.toPlainText() == previous_source
    assert window.student_program_panel.console.toPlainText() == previous_console
    assert "已载入 R1-05" not in window.status_label.text()
    assert "载入失败" in window.status_label.text()


def test_first_experiment_panel_commit_failure_restores_base_and_empty_ui(
    qtbot,
    tmp_path,
    monkeypatch,
):
    catalog = _task14_catalog(tmp_path)
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    experiment_session = FakeExperimentSession(session, catalog=catalog)
    window = VisionLabWindow(
        application=application,
        session=session,
        experiment_catalog=catalog,
        experiment_session=experiment_session,
    )
    qtbot.addWidget(window)
    monkeypatch.setattr(
        window.student_program_panel,
        "load_template",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("panel-commit-failed")
        ),
    )

    assert window._select_experiment("R1-05") is False

    assert experiment_session.restore_calls == [None]
    assert experiment_session.current is None
    assert window.student_controller._experiment_context is None
    assert window.student_controller.program_path is None
    assert window.student_controller.state is RunState.EMPTY
    assert window.student_program_panel.program_path is None
    assert window.student_program_panel.editor.toPlainText() == ""
    assert "已载入 R1-05" not in window.status_label.text()


def test_experiment_rollback_failure_quarantines_all_conflicting_controls(
    qtbot,
    tmp_path,
    monkeypatch,
):
    catalog = _task14_catalog(tmp_path)
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    experiment_session = FakeExperimentSession(session, catalog=catalog)
    window = VisionLabWindow(
        application=application,
        session=session,
        experiment_catalog=catalog,
        experiment_session=experiment_session,
    )
    qtbot.addWidget(window)
    assert window._select_experiment("R1-01") is True
    experiment_session.restore_failure = RuntimeError("base-restore-failed")
    monkeypatch.setattr(
        window.student_program_panel,
        "load_template",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("panel-commit-failed")
        ),
    )

    assert window._select_experiment("R1-05") is False

    assert window._experiment_switch_quarantined is True
    assert window.student_controller.experiment_binding_quarantined is True
    assert window.student_controller._experiment_context is None
    assert window.student_controller.program_path is None
    assert window.experiment_catalog_panel.isEnabled() is False
    assert window.student_program_panel.isEnabled() is False
    assert all(not control.isEnabled() for control in operation_controls(window))
    assert "base-restore-failed" in window.status_label.text()
    assert "已载入 R1-05" not in window.status_label.text()


def test_catalog_click_schedules_slow_selection_without_blocking_qt(
    qtbot,
    tmp_path,
):
    catalog = _task14_catalog(tmp_path)
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    experiment_session = FakeExperimentSession(
        session,
        catalog=catalog,
        select_delay_s=0.35,
    )
    window = VisionLabWindow(
        application=application,
        session=session,
        experiment_catalog=catalog,
        experiment_session=experiment_session,
    )
    qtbot.addWidget(window)
    timer_fired = []
    QTimer.singleShot(50, lambda: timer_fired.append(time.monotonic()))

    started = time.monotonic()
    qtbot.mouseClick(window.experiment_catalog_panel.select_button, Qt.LeftButton)
    click_elapsed = time.monotonic() - started

    assert click_elapsed < 0.15
    qtbot.waitUntil(lambda: bool(timer_fired), timeout=1000)
    qtbot.waitUntil(
        lambda: window._experiment_thread is None,
        timeout=3000,
    )
    assert "已载入 R1-01" in window.status_label.text()


def test_experiment_selection_rejects_duplicates_and_disables_conflicts(
    qtbot,
    tmp_path,
):
    catalog = _task14_catalog(tmp_path)
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    entered = Event()
    release = Event()
    experiment_session = FakeExperimentSession(
        session,
        catalog=catalog,
        select_entered=entered,
        select_release=release,
    )
    window = VisionLabWindow(
        application=application,
        session=session,
        experiment_catalog=catalog,
        experiment_session=experiment_session,
    )
    qtbot.addWidget(window)

    assert window._request_experiment_select("R1-01") is True
    assert entered.wait(timeout=1)
    assert window._request_experiment_select("R1-05") is False
    assert window.experiment_catalog_panel.select_button.isEnabled() is False
    assert window.student_program_panel.isEnabled() is False
    assert all(not control.isEnabled() for control in operation_controls(window))
    release.set()
    qtbot.waitUntil(
        lambda: window._experiment_thread is None,
        timeout=3000,
    )

    assert experiment_session.select_calls == ["R1-01"]
    assert window.student_program_panel.isEnabled() is True
    assert window.experiment_catalog_panel.select_button.isEnabled() is True


def test_async_application_notification_reaches_gui_before_binding_commit(
    qtbot,
    tmp_path,
):
    catalog = _task14_catalog(tmp_path)
    application = FullStudentApplication(tmp_path)
    replacement = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    experiment_session = FakeExperimentSession(
        session,
        replacement,
        catalog=catalog,
        select_delay_s=0.05,
    )

    class TrackingWindow(VisionLabWindow):
        def __init__(self, **kwargs):
            self.replacement_on_gui = []
            super().__init__(**kwargs)

        def _replace_application(self, selected_application):
            self.replacement_on_gui.append(
                QThread.currentThread() is self.thread()
            )
            return super()._replace_application(selected_application)

    window = TrackingWindow(
        application=application,
        session=session,
        experiment_catalog=catalog,
        experiment_session=experiment_session,
    )
    qtbot.addWidget(window)
    binding_observations = []
    real_bind = window.student_controller.bind_experiment

    def tracked_bind(**kwargs):
        binding_observations.append(
            (
                QThread.currentThread() is window.thread(),
                window.application is replacement,
                session.application is replacement,
            )
        )
        return real_bind(**kwargs)

    window.student_controller.bind_experiment = tracked_bind

    assert window._request_experiment_select("R1-05") is True
    qtbot.waitUntil(
        lambda: window._experiment_thread is None,
        timeout=3000,
    )

    assert window.replacement_on_gui == [True, True]
    assert binding_observations == [(True, True, True)]


def test_async_rollback_bridge_subscribe_failure_quarantines_binding(
    qtbot,
    tmp_path,
    monkeypatch,
):
    catalog = _task14_catalog(tmp_path)
    original_bus = FakeEventBus(subscribe_fail_on_attempts=(2,))
    original = FullStudentApplication(tmp_path, event_bus=original_bus)
    replacement = FullStudentApplication(
        tmp_path,
        event_bus=FakeEventBus(),
    )
    session = FakeSession(original)
    experiment_session = FakeExperimentSession(
        session,
        catalog=catalog,
        restore_application=original,
    )
    window = VisionLabWindow(
        application=original,
        session=session,
        experiment_catalog=catalog,
        experiment_session=experiment_session,
    )
    qtbot.addWidget(window)
    assert window._select_experiment("R1-01") is True
    experiment_session.replacement = replacement
    monkeypatch.setattr(
        window.student_program_panel,
        "load_template",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("panel-commit-failed")
        ),
    )

    assert window._request_experiment_select("R1-05") is True
    qtbot.waitUntil(
        lambda: window._experiment_thread is None,
        timeout=3000,
    )

    assert original_bus.subscribe_attempts == 2
    assert window.application is session.application is original
    assert window._event_unsubscribe is None
    assert window._experiment_switch_quarantined is True
    assert window.student_controller.experiment_binding_quarantined is True
    assert window.student_controller._experiment_context is None
    assert window.student_controller.program_path is None
    assert window.experiment_catalog_panel.isEnabled() is False
    assert window.student_program_panel.isEnabled() is False
    assert all(not control.isEnabled() for control in operation_controls(window))
    assert "已载入 R1-05" not in window.status_label.text()


def test_async_rollback_with_pending_event_unsubscribe_is_quarantined(
    qtbot,
    tmp_path,
):
    catalog = _task14_catalog(tmp_path)
    original_bus = FakeEventBus(unsubscribe_failures=1)
    original = FullStudentApplication(tmp_path, event_bus=original_bus)
    replacement = FullStudentApplication(
        tmp_path,
        event_bus=FakeEventBus(),
    )
    session = FakeSession(original)
    experiment_session = FakeExperimentSession(
        session,
        catalog=catalog,
        restore_application=original,
    )
    window = VisionLabWindow(
        application=original,
        session=session,
        experiment_catalog=catalog,
        experiment_session=experiment_session,
    )
    qtbot.addWidget(window)
    assert window._select_experiment("R1-01") is True
    experiment_session.replacement = replacement

    assert window._request_experiment_select("R1-05") is True
    qtbot.waitUntil(
        lambda: window._experiment_thread is None,
        timeout=3000,
    )

    assert window.application is session.application is original
    assert window._event_subscription_application is original
    assert window._event_unsubscribe is not None
    assert len(window._pending_event_unsubscribes) == 1
    assert window._experiment_switch_quarantined is True
    assert window.student_controller.experiment_binding_quarantined is True
    assert window.student_controller._experiment_context is None
    assert window.student_controller.program_path is None
    assert window.experiment_catalog_panel.isEnabled() is False
    assert window.student_program_panel.isEnabled() is False
    assert all(not control.isEnabled() for control in operation_controls(window))
    assert "已载入 R1-05" not in window.status_label.text()


def test_async_manifest_change_at_controller_handoff_rolls_back(
    qtbot,
    tmp_path,
    monkeypatch,
):
    catalog = _task14_catalog(tmp_path)
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    experiment_session = FakeExperimentSession(session, catalog=catalog)
    window = VisionLabWindow(
        application=application,
        session=session,
        experiment_catalog=catalog,
        experiment_session=experiment_session,
    )
    qtbot.addWidget(window)
    assert window._select_experiment("R1-01") is True
    previous_context = window.student_controller._experiment_context
    previous_manifest = dict(window.student_controller._scene_manifest)
    previous_program = window.student_controller.program_path
    previous_source = window.student_program_panel.editor.toPlainText()
    real_bind = window.student_controller.bind_experiment

    def rewrite_manifest_then_bind(**kwargs):
        manifest_path = kwargs["definition"].scene_manifest
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["handoff_generation"] = "changed"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return real_bind(**kwargs)

    monkeypatch.setattr(
        window.student_controller,
        "bind_experiment",
        rewrite_manifest_then_bind,
    )

    assert window._request_experiment_select("R1-05") is True
    qtbot.waitUntil(
        lambda: window._experiment_thread is None,
        timeout=3000,
    )

    assert experiment_session.current.experiment_id == "R1-01"
    assert window.student_controller._experiment_context is previous_context
    assert window.student_controller._scene_manifest == previous_manifest
    assert window.student_controller.program_path == previous_program
    assert window.student_program_panel.editor.toPlainText() == previous_source
    assert "已载入 R1-05" not in window.status_label.text()
    assert "载入失败" in window.status_label.text()


def test_async_template_change_at_panel_handoff_rolls_back(
    qtbot,
    tmp_path,
    monkeypatch,
):
    catalog = _task14_catalog(tmp_path)
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    experiment_session = FakeExperimentSession(session, catalog=catalog)
    window = VisionLabWindow(
        application=application,
        session=session,
        experiment_catalog=catalog,
        experiment_session=experiment_session,
    )
    qtbot.addWidget(window)
    assert window._select_experiment("R1-01") is True
    previous_context = window.student_controller._experiment_context
    previous_program = window.student_controller.program_path
    previous_source = window.student_program_panel.editor.toPlainText()
    previous_console = window.student_program_panel.console.toPlainText()
    real_load = window.student_program_panel.load_template

    def rewrite_template_then_load(path, **kwargs):
        Path(path).write_bytes(
            b"def main(ctx):\n    ctx.log('handoff-changed')\n"
        )
        return real_load(path, **kwargs)

    monkeypatch.setattr(
        window.student_program_panel,
        "load_template",
        rewrite_template_then_load,
    )

    assert window._request_experiment_select("R1-05") is True
    qtbot.waitUntil(
        lambda: window._experiment_thread is None,
        timeout=3000,
    )

    assert experiment_session.current.experiment_id == "R1-01"
    assert window.student_controller._experiment_context is previous_context
    assert window.student_controller.program_path == previous_program
    assert window.student_program_panel.editor.toPlainText() == previous_source
    assert window.student_program_panel.console.toPlainText() == previous_console
    assert "已载入 R1-05" not in window.status_label.text()
    assert "载入失败" in window.status_label.text()


def test_async_panel_commit_failure_restores_all_previous_snapshots(
    qtbot,
    tmp_path,
    monkeypatch,
):
    catalog = _task14_catalog(tmp_path)
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    experiment_session = FakeExperimentSession(session, catalog=catalog)
    window = VisionLabWindow(
        application=application,
        session=session,
        experiment_catalog=catalog,
        experiment_session=experiment_session,
    )
    qtbot.addWidget(window)
    assert window._select_experiment("R1-01") is True
    previous_session = experiment_session.current
    previous_context = window.student_controller._experiment_context
    previous_program = window.student_controller.program_path
    previous_source = window.student_program_panel.editor.toPlainText()
    previous_console = window.student_program_panel.console.toPlainText()
    real_load = window.student_program_panel.load_template

    def commit_then_fail(*args, **kwargs):
        real_load(*args, **kwargs)
        raise RuntimeError("async-panel-commit-failed")

    monkeypatch.setattr(
        window.student_program_panel,
        "load_template",
        commit_then_fail,
    )

    assert window._request_experiment_select("R1-05") is True
    qtbot.waitUntil(
        lambda: window._experiment_thread is None,
        timeout=3000,
    )

    assert experiment_session.current is previous_session
    assert window.student_controller._experiment_context is previous_context
    assert window.student_controller.program_path == previous_program
    assert window.student_program_panel.editor.toPlainText() == previous_source
    assert window.student_program_panel.console.toPlainText() == previous_console
    assert "已载入 R1-05" not in window.status_label.text()
    assert "载入失败" in window.status_label.text()


def test_async_rollback_failure_invalidates_binding_and_stays_quarantined(
    qtbot,
    tmp_path,
    monkeypatch,
):
    catalog = _task14_catalog(tmp_path)
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    experiment_session = FakeExperimentSession(session, catalog=catalog)
    window = VisionLabWindow(
        application=application,
        session=session,
        experiment_catalog=catalog,
        experiment_session=experiment_session,
    )
    qtbot.addWidget(window)
    assert window._select_experiment("R1-01") is True
    experiment_session.restore_failure = RuntimeError("async-restore-failed")
    real_load = window.student_program_panel.load_template

    def commit_then_fail(*args, **kwargs):
        real_load(*args, **kwargs)
        raise RuntimeError("async-panel-commit-failed")

    monkeypatch.setattr(
        window.student_program_panel,
        "load_template",
        commit_then_fail,
    )

    assert window._request_experiment_select("R1-05") is True
    qtbot.waitUntil(
        lambda: window._experiment_thread is None,
        timeout=3000,
    )

    assert window._experiment_switch_quarantined is True
    assert window.student_controller.experiment_binding_quarantined is True
    assert window.student_controller._experiment_context is None
    assert window.student_controller.program_path is None
    assert window._request_experiment_select("R1-01") is False
    assert window.student_program_panel.isEnabled() is False
    assert window.experiment_catalog_panel.isEnabled() is False
    assert "async-restore-failed" in window.status_label.text()


def test_repeated_async_selections_release_qthread_children(
    qtbot,
    tmp_path,
):
    catalog = _task14_catalog(tmp_path)
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    experiment_session = FakeExperimentSession(session, catalog=catalog)
    window = VisionLabWindow(
        application=application,
        session=session,
        experiment_catalog=catalog,
        experiment_session=experiment_session,
    )
    qtbot.addWidget(window)
    initial_threads = len(window.findChildren(QThread))

    for experiment_id in ("R1-01", "R1-02", "R1-05", "R1-06", "R1-07"):
        assert window._request_experiment_select(experiment_id) is True
        qtbot.waitUntil(
            lambda: window._experiment_thread is None,
            timeout=3000,
        )
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QApplication.processEvents()

    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()
    assert len(window.findChildren(QThread)) == initial_threads


def test_repeated_experiment_thread_start_failures_release_local_ownership(
    qtbot,
    tmp_path,
    monkeypatch,
):
    catalog = _task14_catalog(tmp_path)
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    experiment_session = FakeExperimentSession(session, catalog=catalog)
    window = VisionLabWindow(
        application=application,
        session=session,
        experiment_catalog=catalog,
        experiment_session=experiment_session,
    )
    qtbot.addWidget(window)
    initial_threads = len(window.findChildren(QThread))
    start_calls = []
    destroyed_threads = []
    worker_construction_calls = []
    real_worker = pyqt_app._ExperimentSelectWorker

    class StartFailThread(QThread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.destroyed.connect(
                lambda: destroyed_threads.append("destroyed")
            )

        def start(self, *args, **kwargs):
            del args, kwargs
            start_calls.append("start")
            raise RuntimeError("experiment-thread-start-failed")

    def track_worker_construction(**kwargs):
        worker_construction_calls.append("worker")
        return real_worker(**kwargs)

    monkeypatch.setattr(pyqt_app, "QThread", StartFailThread)
    monkeypatch.setattr(
        pyqt_app,
        "_ExperimentSelectWorker",
        track_worker_construction,
    )

    for _ in range(5):
        assert window._request_experiment_select("R1-01") is False
        assert window._experiment_thread is None
        assert window._experiment_worker is None
        assert window._experiment_control is None
        assert window._experiment_transaction is None
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QApplication.processEvents()

    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()
    assert start_calls == ["start"] * 5
    assert worker_construction_calls == []
    assert destroyed_threads == ["destroyed"] * 5
    assert len(window.findChildren(QThread)) == initial_threads
    assert window.experiment_catalog_panel.select_button.isEnabled() is True
    assert window.student_program_panel.isEnabled() is True


def test_close_waits_for_experiment_worker_without_use_after_close(
    qtbot,
    tmp_path,
):
    catalog = _task14_catalog(tmp_path)
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    entered = Event()
    release = Event()
    experiment_session = FakeExperimentSession(
        session,
        catalog=catalog,
        select_entered=entered,
        select_release=release,
    )
    window = VisionLabWindow(
        application=application,
        session=session,
        experiment_catalog=catalog,
        experiment_session=experiment_session,
    )
    qtbot.addWidget(window)
    assert window._request_experiment_select("R1-05") is True
    assert entered.wait(timeout=1)
    experiment_thread = window._experiment_thread
    experiment_finished = Event()
    experiment_thread.finished.connect(
        experiment_finished.set,
        Qt.DirectConnection,
    )
    releaser = Thread(
        target=lambda: (time.sleep(0.05), release.set()),
        daemon=True,
    )
    releaser.start()
    event = FakeCloseEvent()

    window.closeEvent(event)
    releaser.join(timeout=1)

    assert event.accepted is True
    assert event.ignored is False
    assert experiment_finished.is_set()
    assert experiment_session.current is None
    assert session.close_calls == 1
    QApplication.processEvents()
    assert "已载入 R1-05" not in window.status_label.text()


def test_catalog_panel_fits_supported_minimum_window(qtbot, tmp_path):
    catalog = _task14_catalog(tmp_path)
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    window = VisionLabWindow(
        application=application,
        session=session,
        experiment_catalog=catalog,
        experiment_session=FakeExperimentSession(session, catalog=catalog),
    )
    qtbot.addWidget(window)
    window.resize(1180, 760)
    window.tabs.setCurrentIndex(0)
    window.show()
    QApplication.processEvents()
    qtbot.waitUntil(
        lambda: window.experiment_catalog_panel.isVisible(),
        timeout=2000,
    )

    panel = window.experiment_catalog_panel
    bounds = panel.contentsRect()
    for widget in (
        panel.experiment_combo,
        panel.hardware_label,
        panel.capabilities_label,
        panel.human_checks,
        panel.boundary_label,
        panel.select_button,
    ):
        top_left = widget.mapTo(panel, QPoint(0, 0))
        bottom_right = top_left + QPoint(widget.width(), widget.height())
        assert top_left.x() >= bounds.left()
        assert top_left.y() >= bounds.top()
        assert bottom_right.x() <= bounds.right() + 1
        assert bottom_right.y() <= bounds.bottom() + 1
    assert panel.select_button.height() >= 44


def test_pyqt_window_rejects_active_switch_before_scene_select(
    qtbot,
    tmp_path,
):
    catalog = _task14_catalog(tmp_path)
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    experiment_session = FakeExperimentSession(session, catalog=catalog)
    window = VisionLabWindow(
        application=application,
        session=session,
        experiment_catalog=catalog,
        experiment_session=experiment_session,
    )
    qtbot.addWidget(window)
    old_template = catalog.require("R1-01").student_template
    window.student_program_panel.load_template(old_template)
    old_source = window.student_program_panel.editor.toPlainText()
    with window.student_controller._condition:
        window.student_controller._state = RunState.RUNNING

    window.experiment_catalog_panel.experiment_combo.setCurrentIndex(2)
    qtbot.mouseClick(
        window.experiment_catalog_panel.select_button,
        Qt.LeftButton,
    )

    assert experiment_session.select_calls == []
    assert window.student_program_panel.editor.toPlainText() == old_source
    assert "载入失败" in window.status_label.text()


@pytest.mark.parametrize("active_operation", ("legacy", "student-control"))
def test_pyqt_window_rejects_other_ui_operations_before_scene_select(
    qtbot,
    tmp_path,
    active_operation,
):
    catalog = _task14_catalog(tmp_path)
    application = FullStudentApplication(tmp_path)
    session = FakeSession(application)
    experiment_session = FakeExperimentSession(session, catalog=catalog)
    window = VisionLabWindow(
        application=application,
        session=session,
        experiment_catalog=catalog,
        experiment_session=experiment_session,
    )
    qtbot.addWidget(window)
    if active_operation == "legacy":
        window._thread = object()
    else:
        window.student_program_panel._operation_thread = object()

    try:
        qtbot.mouseClick(
            window.experiment_catalog_panel.select_button,
            Qt.LeftButton,
        )
    finally:
        window._thread = None
        window.student_program_panel._operation_thread = None

    assert experiment_session.select_calls == []
    assert "载入失败" in window.status_label.text()


def test_pyqt_window_without_experiment_services_remains_compatible(qtbot):
    window = VisionLabWindow(application=DisconnectedApplication())
    qtbot.addWidget(window)

    assert not hasattr(window, "experiment_catalog_panel")
    assert window.camera_backend_combo.count() == 3
    assert window.robot_backend_combo.count() == 2


def test_pyqt_main_builds_formal_experiment_services_for_default_sim(
    monkeypatch,
    tmp_path,
):
    import vision_platform.application as application_module
    import vision_platform.config as config_module
    import vision_platform.experiments.catalog as catalog_module
    import vision_platform.experiments.session as experiment_session_module
    import vision_platform.session as session_module

    constructed = {}
    base_config = SimpleNamespace(project_root=tmp_path)
    base_application = DisconnectedApplication()
    formal_catalog = object()

    class FakeVisionLabApplication:
        @classmethod
        def from_config(cls, config):
            constructed.setdefault("application_configs", []).append(config)
            return base_application

    class MainVisionSession:
        def __init__(self, *, application, factory):
            self.application = application
            self.factory = factory
            self.close_calls = 0
            constructed["vision_session"] = self

        def close(self):
            self.close_calls += 1
            self.application.close()

    class MainExperimentCatalog:
        @classmethod
        def load(cls, path, *, project_root):
            constructed["catalog_load"] = (Path(path), Path(project_root))
            return formal_catalog

    class MainExperimentSession:
        def __init__(self, **kwargs):
            self.student_is_idle = kwargs["student_is_idle"]
            constructed["experiment_session_kwargs"] = kwargs

    class MainWindow:
        def __init__(self, **kwargs):
            constructed["window_kwargs"] = kwargs
            self.session = kwargs["session"]
            self.closed = False
            self._thread = object()
            self.student_program_panel = SimpleNamespace(
                operation_in_progress=False
            )
            self.student_controller = SimpleNamespace(
                ensure_experiment_switch_allowed=lambda: None
            )

        def show(self):
            experiment_session = constructed["window_kwargs"][
                "experiment_session"
            ]
            constructed["idle_during_ui_action"] = (
                experiment_session.student_is_idle()
            )
            self._thread = None
            constructed["idle_after_window"] = (
                experiment_session.student_is_idle()
            )

        @property
        def owner_shutdown_complete(self):
            return self.closed

        def close(self):
            if not self.closed:
                self.closed = True
                self.session.close()
            return True

    class MainQtApplication:
        @classmethod
        def instance(cls):
            return None

        def __init__(self, _argv):
            pass

        def exec_(self):
            return 0

    monkeypatch.setattr(
        application_module,
        "VisionLabApplication",
        FakeVisionLabApplication,
    )
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda *args, **kwargs: base_config,
    )
    monkeypatch.setattr(session_module, "VisionLabSession", MainVisionSession)
    monkeypatch.setattr(catalog_module, "ExperimentCatalog", MainExperimentCatalog)
    monkeypatch.setattr(
        experiment_session_module,
        "ExperimentSession",
        MainExperimentSession,
    )
    monkeypatch.setattr(pyqt_app, "VisionLabWindow", MainWindow)
    monkeypatch.setattr(pyqt_app, "QApplication", MainQtApplication)

    assert pyqt_app.main([]) == 0

    catalog_path, project_root = constructed["catalog_load"]
    assert catalog_path == (
        pyqt_app.PROJECT_ROOT / "config/experiments/catalog.json"
    )
    assert project_root == pyqt_app.PROJECT_ROOT
    assert constructed["window_kwargs"]["experiment_catalog"] is formal_catalog
    assert constructed["window_kwargs"]["experiment_session"] is not None
    assert constructed["idle_during_ui_action"] is False
    assert constructed["idle_after_window"] is True


@pytest.mark.parametrize(
    "argv",
    (["--camera", "replay"], ["--robot", "real"]),
)
def test_pyqt_main_does_not_mount_sim_catalog_for_non_sim_combinations(
    monkeypatch,
    argv,
):
    import vision_platform.application as application_module
    import vision_platform.config as config_module
    import vision_platform.experiments.catalog as catalog_module
    import vision_platform.session as session_module

    windows = []

    class FakeVisionLabApplication:
        close_calls = 0

        @classmethod
        def from_config(cls, _config):
            return cls()

        def close(self):
            self.close_calls += 1

    class MainVisionSession:
        def __init__(self, *, application, factory):
            self.application = application
            self.factory = factory
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            self.application.close()

    class MainWindow:
        def __init__(self, **kwargs):
            windows.append(kwargs)
            self.session = kwargs["session"]
            self.closed = False

        def show(self):
            pass

        @property
        def owner_shutdown_complete(self):
            return self.closed

        def close(self):
            if not self.closed:
                self.closed = True
                self.session.close()
            return True

    class MainQtApplication:
        @classmethod
        def instance(cls):
            return None

        def __init__(self, _argv):
            pass

        def exec_(self):
            return 0

    monkeypatch.setattr(
        application_module,
        "VisionLabApplication",
        FakeVisionLabApplication,
    )
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(session_module, "VisionLabSession", MainVisionSession)
    monkeypatch.setattr(
        catalog_module.ExperimentCatalog,
        "load",
        lambda *args, **kwargs: pytest.fail("sim catalog must not load"),
    )
    monkeypatch.setattr(pyqt_app, "VisionLabWindow", MainWindow)
    monkeypatch.setattr(pyqt_app, "QApplication", MainQtApplication)

    assert pyqt_app.main(argv) == 0
    assert windows[0]["experiment_catalog"] is None
    assert windows[0]["experiment_session"] is None


def _install_main_lifecycle_fakes(
    monkeypatch,
    tmp_path,
    *,
    fail_stage=None,
    close_during_exec=False,
    hidden_after_show=False,
    window_close_behavior="success",
    active_background=False,
):
    import vision_platform.application as application_module
    import vision_platform.config as config_module
    import vision_platform.experiments.catalog as catalog_module
    import vision_platform.experiments.session as experiment_session_module
    import vision_platform.session as session_module

    state = SimpleNamespace(
        applications=[],
        sessions=[],
        windows=[],
    )

    class OwnedApplication:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    class OwnedApplicationFactory:
        @classmethod
        def from_config(cls, _config):
            application = OwnedApplication()
            state.applications.append(application)
            return application

    class OwnedSession:
        def __init__(self, *, application, factory):
            del factory
            self.application = application
            self.close_calls = 0
            self.closed = False
            state.sessions.append(self)

        def close(self):
            self.close_calls += 1
            if self.closed:
                return
            self.closed = True
            self.application.close()

    class MainCatalog:
        @classmethod
        def load(cls, *_args, **_kwargs):
            if fail_stage == "catalog":
                raise RuntimeError("catalog-load-failed")
            return object()

    class MainExperimentSession:
        def __init__(self, **_kwargs):
            if fail_stage == "experiment_session":
                raise RuntimeError("experiment-session-construct-failed")

    class OwnedWindow:
        def __init__(self, **kwargs):
            if fail_stage == "window":
                raise RuntimeError("window-construct-failed")
            self.session = kwargs["session"]
            self.close_calls = 0
            self.visible = False
            self.closed = False
            self._thread = object() if active_background else None
            self.student_program_panel = SimpleNamespace(
                operation_in_progress=False
            )
            self.student_controller = SimpleNamespace(
                ensure_experiment_switch_allowed=lambda: None
            )
            state.windows.append(self)

        def show(self):
            if fail_stage == "show":
                raise RuntimeError("window-show-failed")
            self.visible = not hidden_after_show

        @property
        def owner_shutdown_complete(self):
            return self.closed

        def isVisible(self):
            return self.visible

        def close(self):
            self.close_calls += 1
            if self.closed:
                return True
            if window_close_behavior == "raise":
                raise RuntimeError("window-close-failed")
            if window_close_behavior == "false":
                return False
            if window_close_behavior == "true_without_owner":
                return True
            self.closed = True
            self.visible = False
            self.session.close()
            return True

    class MainQtApplication:
        @classmethod
        def instance(cls):
            return None

        def __init__(self, _argv):
            if fail_stage == "qapplication":
                raise RuntimeError("qapplication-construct-failed")

        def exec_(self):
            if close_during_exec:
                state.windows[-1].close()
            if fail_stage == "exec":
                raise RuntimeError("qt-exec-failed")
            return 0

    monkeypatch.setattr(
        application_module,
        "VisionLabApplication",
        OwnedApplicationFactory,
    )
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda *args, **kwargs: SimpleNamespace(project_root=tmp_path),
    )
    monkeypatch.setattr(session_module, "VisionLabSession", OwnedSession)
    monkeypatch.setattr(catalog_module, "ExperimentCatalog", MainCatalog)
    monkeypatch.setattr(
        experiment_session_module,
        "ExperimentSession",
        MainExperimentSession,
    )
    monkeypatch.setattr(pyqt_app, "VisionLabWindow", OwnedWindow)
    monkeypatch.setattr(pyqt_app, "QApplication", MainQtApplication)
    return state


@pytest.mark.parametrize(
    ("fail_stage", "message"),
    (
        ("catalog", "catalog-load-failed"),
        ("experiment_session", "experiment-session-construct-failed"),
        ("qapplication", "qapplication-construct-failed"),
        ("window", "window-construct-failed"),
        ("show", "window-show-failed"),
        ("exec", "qt-exec-failed"),
    ),
)
def test_pyqt_main_closes_owned_application_once_on_failure(
    monkeypatch,
    tmp_path,
    fail_stage,
    message,
):
    state = _install_main_lifecycle_fakes(
        monkeypatch,
        tmp_path,
        fail_stage=fail_stage,
    )

    with pytest.raises(RuntimeError, match=message):
        pyqt_app.main([])

    assert state.sessions[0].close_calls == 1
    assert state.applications[0].close_calls == 1
    if fail_stage in {"show", "exec"}:
        assert state.windows[0].close_calls == 1


@pytest.mark.parametrize("close_during_exec", (False, True))
def test_pyqt_main_normal_window_close_does_not_double_close_owner(
    monkeypatch,
    tmp_path,
    close_during_exec,
):
    state = _install_main_lifecycle_fakes(
        monkeypatch,
        tmp_path,
        close_during_exec=close_during_exec,
    )

    assert pyqt_app.main([]) == 0

    assert state.windows[0].close_calls == 1
    assert state.sessions[0].close_calls == 1
    assert state.applications[0].close_calls == 1


def test_pyqt_main_hidden_window_still_shuts_down_owner_once(
    monkeypatch,
    tmp_path,
):
    state = _install_main_lifecycle_fakes(
        monkeypatch,
        tmp_path,
        hidden_after_show=True,
    )

    assert pyqt_app.main([]) == 0

    assert state.windows[0].close_calls == 1
    assert state.sessions[0].close_calls == 1
    assert state.applications[0].close_calls == 1


def test_pyqt_main_close_false_without_background_falls_back_to_session(
    monkeypatch,
    tmp_path,
):
    state = _install_main_lifecycle_fakes(
        monkeypatch,
        tmp_path,
        window_close_behavior="false",
    )

    assert pyqt_app.main([]) == 0

    assert state.windows[0].close_calls == 1
    assert state.sessions[0].close_calls == 1
    assert state.applications[0].close_calls == 1


def test_pyqt_main_close_true_requires_explicit_owner_completion(
    monkeypatch,
    tmp_path,
):
    state = _install_main_lifecycle_fakes(
        monkeypatch,
        tmp_path,
        window_close_behavior="true_without_owner",
    )

    assert pyqt_app.main([]) == 0

    assert state.windows[0].close_calls == 1
    assert state.sessions[0].close_calls == 1
    assert state.applications[0].close_calls == 1


def test_pyqt_main_close_false_with_background_never_races_owner_close(
    monkeypatch,
    tmp_path,
):
    state = _install_main_lifecycle_fakes(
        monkeypatch,
        tmp_path,
        window_close_behavior="false",
        active_background=True,
    )

    with pytest.raises(RuntimeError, match="background work"):
        pyqt_app.main([])

    assert state.windows[0].close_calls == 1
    assert state.sessions[0].close_calls == 0
    assert state.applications[0].close_calls == 0


def test_pyqt_main_normal_close_error_is_not_retried_or_ignored(
    monkeypatch,
    tmp_path,
):
    state = _install_main_lifecycle_fakes(
        monkeypatch,
        tmp_path,
        window_close_behavior="raise",
    )

    with pytest.raises(RuntimeError, match="window-close-failed"):
        pyqt_app.main([])

    assert state.windows[0].close_calls == 1
    assert state.sessions[0].close_calls == 1
    assert state.applications[0].close_calls == 1


def test_pyqt_main_preserves_exec_error_when_cleanup_also_fails(
    monkeypatch,
    tmp_path,
):
    state = _install_main_lifecycle_fakes(
        monkeypatch,
        tmp_path,
        fail_stage="exec",
        window_close_behavior="raise",
    )

    with pytest.raises(RuntimeError, match="qt-exec-failed") as captured:
        pyqt_app.main([])

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert str(captured.value.__cause__) == "window-close-failed"
    assert state.windows[0].close_calls == 1
    assert state.sessions[0].close_calls == 1
    assert state.applications[0].close_calls == 1


def test_pyqt_main_exec_error_survives_close_false_fallback(
    monkeypatch,
    tmp_path,
):
    state = _install_main_lifecycle_fakes(
        monkeypatch,
        tmp_path,
        fail_stage="exec",
        window_close_behavior="false",
    )

    with pytest.raises(RuntimeError, match="qt-exec-failed") as captured:
        pyqt_app.main([])

    assert captured.value.__cause__ is None
    assert state.windows[0].close_calls == 1
    assert state.sessions[0].close_calls == 1
    assert state.applications[0].close_calls == 1
