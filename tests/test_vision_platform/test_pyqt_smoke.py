from __future__ import annotations

import json
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import vision_platform.ui.pyqt_app as pyqt_app
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtWidgets import QApplication
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
        subscribe_entered=None,
        allow_subscribe=None,
    ):
        self.handlers = []
        self.unsubscribe_calls = 0
        self.unsubscribe_attempts = 0
        self.unsubscribe_failures = unsubscribe_failures
        self.subscribe_entered = subscribe_entered
        self.allow_subscribe = allow_subscribe

    def subscribe(self, handler):
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

    def close() -> None:
        try:
            window.closeEvent(close_event)
        except BaseException as error:
            errors.append(error)

    replace_thread = Thread(target=replace)
    replace_thread.start()
    assert subscribe_entered.wait(timeout=2.0)
    close_thread = Thread(target=close)
    close_thread.start()
    allow_subscribe.set()
    replace_thread.join(timeout=2.0)
    close_thread.join(timeout=2.0)

    assert not replace_thread.is_alive()
    assert not close_thread.is_alive()
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
