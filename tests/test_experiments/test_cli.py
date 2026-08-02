from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import vision_platform.application as application_module
import vision_platform.cli as cli_module
import vision_platform.config as config_module
import vision_platform.experiments.session as experiment_session_module
import vision_platform.session as session_module
import vision_platform.student.runner as runner_module
from vision_platform.cli import build_parser, main
from vision_platform.student.validator import (
    ValidationIssue,
    ValidationResult,
)


def _one_json_line(capsys):
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1
    payload = json.loads(
        lines[0],
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant: {value}")
        ),
    )
    return payload, captured.err


def _student_config():
    return {
        "speed_range": [2, 24],
        "max_runtime_s": 12,
        "max_commands": 80,
        "command_timeout_s": 3,
        "max_sleep_s": 2,
        "tool_on_max_z_mm": 30,
    }


def _install_fake_experiment_runtime(
    monkeypatch,
    tmp_path,
    *,
    validation=None,
    result=None,
    fail_stage=None,
    runtime_stdout=None,
):
    state = SimpleNamespace(
        operations=[],
        applications=[],
        sessions=[],
        experiment_sessions=[],
        controllers=[],
        load_config_calls=[],
    )
    template = tmp_path / "student template.py"
    template.write_text("def main(ctx):\n    pass\n", encoding="utf-8")
    scene = tmp_path / "formal scene.ttt"
    scene.write_bytes(b"scene")
    manifest = tmp_path / "scene manifest.json"
    manifest_payload = {"schema_version": 1, "task_contracts": {}}
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    guide = tmp_path / "guide.md"
    guide.write_text("# Guide\n", encoding="utf-8")
    definition = SimpleNamespace(
        experiment_id="R1-05",
        title="基于视觉的物体码垛",
        version="2.2.0",
        scene=scene,
        scene_manifest=manifest,
        student_template=template,
        guide=guide,
        capabilities=("robot.move_world", "scene.probe"),
        public_parameters={},
        hardware_status="PENDING_HARDWARE",
        acceptance=SimpleNamespace(
            automated_checks=("six_objects",),
            human_checks=("人工观察",),
        ),
    )

    class FakeCatalog:
        definitions = (definition,)

        def require(self, experiment_id):
            state.operations.append(("catalog.require", experiment_id))
            if experiment_id != definition.experiment_id:
                raise KeyError(f"Unknown experiment: {experiment_id}")
            return definition

    config = SimpleNamespace(student=_student_config())

    def fake_load_config(config_path, *, project_root, environ):
        state.load_config_calls.append(
            {
                "config_path": config_path,
                "project_root": project_root,
                "environ": dict(environ),
            }
        )
        if fail_stage == "config":
            raise RuntimeError("config exploded")
        return config

    class FakeApplication:
        def __init__(self):
            self.config = config
            self.close_calls = 0
            self.close_quarantined_calls = 0
            self.closed = False

        @classmethod
        def from_config(cls, received):
            assert received is config
            if fail_stage == "application":
                raise RuntimeError("application exploded")
            application = cls()
            state.applications.append(application)
            state.operations.append("application.create")
            return application

        def close(self):
            self.close_calls += 1
            self.closed = True
            state.operations.append("application.close")

        def close_quarantined(self):
            self.close_quarantined_calls += 1
            self.closed = True
            state.operations.append("application.close_quarantined")

    class FakeVisionSession:
        def __init__(self, *, application, factory):
            self.application = application
            self.factory = factory
            self.close_calls = 0
            self.close_quarantined_calls = 0
            self.closed = False
            state.sessions.append(self)

        def close(self):
            self.close_calls += 1
            self.closed = True
            state.operations.append("session.close")
            self.application.close()
            if fail_stage == "close":
                raise RuntimeError("close exploded")

        def close_quarantined(self):
            self.close_quarantined_calls += 1
            self.closed = True
            state.operations.append("session.close_quarantined")
            self.application.close_quarantined()
            if fail_stage == "close_quarantined":
                raise RuntimeError("quarantined close exploded")

    context = SimpleNamespace(
        experiment_id=definition.experiment_id,
        hardware_status="PENDING_HARDWARE",
    )

    class FakeExperimentSession:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            state.experiment_sessions.append(self)

        def select(self, experiment_id):
            state.operations.append(("experiment.select", experiment_id))
            if runtime_stdout:
                print(runtime_stdout)
            if fail_stage in {"load_scene", "select", "capabilities"}:
                raise RuntimeError(f"{fail_stage} exploded")
            return context

    if validation is None:
        validation = ValidationResult(template.resolve(), True, ())
    if result is None:
        summary = tmp_path / "summary.json"
        summary.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "scene_probe_status": "PASS",
                }
            ),
            encoding="utf-8",
        )
        result = SimpleNamespace(
            status="PASS",
            summary_path=summary,
            evidence_dir=tmp_path / "evidence",
            error=None,
        )

    class FakeController:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.process_is_alive = False
            self.cancel_calls = 0
            self.quiescence_calls = []
            state.controllers.append(self)

        def load(self, program):
            state.operations.append(("controller.load", Path(program)))
            if fail_stage == "load":
                raise RuntimeError("load exploded")

        def validate(self):
            state.operations.append("controller.validate")
            if fail_stage == "validate":
                raise RuntimeError("validate exploded")
            return validation

        def start(self):
            state.operations.append("controller.start")
            self.process_is_alive = True
            if fail_stage == "start":
                raise RuntimeError("start exploded")

        def wait(self, *, timeout_s):
            state.operations.append(("controller.wait", timeout_s))
            if fail_stage in {"wait", "wait_cancel"}:
                raise RuntimeError("wait exploded")
            if fail_stage == "timeout":
                raise TimeoutError("wait timed out")
            self.process_is_alive = False
            return result

        def cancel(self):
            self.cancel_calls += 1
            state.operations.append("controller.cancel")
            self.process_is_alive = False
            if fail_stage in {"cancel", "wait_cancel"}:
                raise RuntimeError("cancel exploded")

        def wait_for_quiescence(self, timeout_s):
            self.quiescence_calls.append(timeout_s)
            state.operations.append(
                ("controller.wait_for_quiescence", timeout_s)
            )
            return True

    monkeypatch.setattr(
        cli_module,
        "_experiment_catalog",
        FakeCatalog,
        raising=False,
    )
    monkeypatch.setattr(config_module, "load_config", fake_load_config)
    monkeypatch.setattr(
        application_module,
        "VisionLabApplication",
        FakeApplication,
    )
    monkeypatch.setattr(
        session_module,
        "VisionLabSession",
        FakeVisionSession,
    )
    monkeypatch.setattr(
        experiment_session_module,
        "ExperimentSession",
        FakeExperimentSession,
    )
    monkeypatch.setattr(
        runner_module,
        "StudentProgramController",
        FakeController,
    )
    return state, definition, context, manifest_payload, result


def _install_sequential_cleanup_failures(
    monkeypatch,
    state,
    *,
    cleanup_base_error,
):
    controller_class = runner_module.StudentProgramController
    session_class = session_module.VisionLabSession
    first_cleanup_error = RuntimeError("process liveness cleanup failed")
    base_getattribute = controller_class.__getattribute__
    liveness_read_failed = False

    def failing_getattribute(self, name):
        nonlocal liveness_read_failed
        if name == "process_is_alive" and not liveness_read_failed:
            liveness_read_failed = True
            state.operations.append("controller.process_is_alive.cleanup")
            raise first_cleanup_error
        return base_getattribute(self, name)

    def finishing_quiescence(self, timeout_s):
        self.quiescence_calls.append(timeout_s)
        state.operations.append(
            ("controller.wait_for_quiescence", timeout_s)
        )
        self.process_is_alive = False
        return True

    base_close_quarantined = session_class.close_quarantined

    def interrupting_close_quarantined(self):
        base_close_quarantined(self)
        raise cleanup_base_error

    monkeypatch.setattr(
        controller_class,
        "__getattribute__",
        failing_getattribute,
    )
    monkeypatch.setattr(
        controller_class,
        "wait_for_quiescence",
        finishing_quiescence,
    )
    monkeypatch.setattr(
        session_class,
        "close_quarantined",
        interrupting_close_quarantined,
    )
    return first_cleanup_error


def test_experiment_run_parser_uses_catalog_id_not_arbitrary_scene():
    args = build_parser().parse_args(
        [
            "experiment-run",
            "--experiment",
            "R1-05",
            "--program",
            "student_programs/my_stack.py",
        ]
    )

    assert args.experiment == "R1-05"
    assert args.program == "student_programs/my_stack.py"
    assert not hasattr(args, "robot")
    assert not hasattr(args, "scene")


def test_experiment_list_prints_formal_items(capsys):
    code = main(["experiment-list"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["status"] == "PASS"
    assert [item["experiment_id"] for item in payload["experiments"]] == [
        "R1-01",
        "R1-02",
        "R1-05",
        "R1-06",
        "R1-07",
        "V1-01",
        "V1-02",
        "V1-03",
        "V1-04",
        "V1-05",
        "V1-06",
        "V1-07",
    ]
    assert all(
        item["hardware_status"] == "PENDING_HARDWARE"
        for item in payload["experiments"]
    )


def test_experiment_list_and_show_include_v1_labs(capsys):
    assert main(["experiment-list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [
        item["experiment_id"] for item in listed["experiments"][-7:]
    ] == ["V1-01", "V1-02", "V1-03", "V1-04", "V1-05", "V1-06", "V1-07"]
    expected_templates = {
        "V1-01": "v1_01_virtual_vision.py",
        "V1-02": "v1_02_size_measurement.py",
        "V1-03": "v1_03_pose_measurement.py",
        "V1-04": "v1_04_geometry_measurement.py",
        "V1-05": "v1_05_color_shape.py",
        "V1-06": "v1_06_template_matching.py",
        "V1-07": "v1_07_code_routing.py",
    }
    for experiment_id, template in expected_templates.items():
        assert main(
            ["experiment-show", "--experiment", experiment_id]
        ) == 0
        shown = json.loads(capsys.readouterr().out)
        assert shown["student_template"].endswith(template)
        assert shown["hardware_status"] == "PENDING_HARDWARE"


def test_experiment_show_does_not_claim_hardware_pass(capsys):
    code = main(["experiment-show", "--experiment", "R1-07"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["experiment_id"] == "R1-07"
    assert payload["hardware_status"] == "PENDING_HARDWARE"
    assert payload["scene"] == (
        "simulation/logistics_lab/BL23_logistics_lab.ttt"
    )
    assert payload["student_template"] == (
        "student_programs/templates/r1_07_component_sort.py"
    )
    assert payload["guide"] == "docs/experiments/R1-07.md"
    assert payload["capabilities"]
    assert payload["automated_checks"]
    assert payload["human_checks"]


@pytest.mark.parametrize("command", ["experiment-show", "experiment-run"])
def test_unknown_experiment_is_one_json_validation_failure(command, capsys):
    code = main([command, "--experiment", "R9-99"])
    payload, _ = _one_json_line(capsys)

    assert code == 2
    assert payload["status"] == "FAIL"
    assert payload["error"]["code"] == "EXPERIMENT_NOT_FOUND"
    assert payload["hardware_status"] == "PENDING_HARDWARE"


@pytest.mark.parametrize("port", ["0", "not-a-port"])
def test_experiment_run_rejects_bad_port_before_connecting(
    tmp_path,
    monkeypatch,
    capsys,
    port,
):
    state, _, _, _, _ = _install_fake_experiment_runtime(
        monkeypatch,
        tmp_path,
    )

    code = main(
        ["experiment-run", "--experiment", "R1-05", "--port", port]
    )
    payload, _ = _one_json_line(capsys)

    assert code == 2
    assert payload["error"]["code"] == "EXPERIMENT_ARGUMENT_INVALID"
    assert not state.applications
    assert not state.sessions


def test_experiment_run_rejects_missing_program_before_connecting(
    tmp_path,
    monkeypatch,
    capsys,
):
    state, _, _, _, _ = _install_fake_experiment_runtime(
        monkeypatch,
        tmp_path,
    )

    code = main(
        [
            "experiment-run",
            "--experiment",
            "R1-05",
            "--program",
            str(tmp_path / "missing.py"),
        ]
    )
    payload, _ = _one_json_line(capsys)

    assert code == 2
    assert payload["error"]["code"] == "EXPERIMENT_ARGUMENT_INVALID"
    assert not state.applications
    assert not state.sessions


def test_experiment_run_forces_sim_wires_selected_contract_and_returns_probe(
    tmp_path,
    monkeypatch,
    capsys,
):
    state, definition, context, manifest_payload, result = (
        _install_fake_experiment_runtime(monkeypatch, tmp_path)
    )
    monkeypatch.setenv("ROBOT_BACKEND", "real")
    monkeypatch.setenv("VISION_BACKEND", "hik")

    code = main(
        [
            "experiment-run",
            "--experiment",
            "R1-05",
            "--config",
            "config/custom.json",
            "--host",
            "127.0.0.9",
            "--port",
            "23111",
            "--output",
            str(tmp_path / "output with spaces"),
        ]
    )
    payload, stderr = _one_json_line(capsys)

    assert stderr == ""
    assert code == 0
    assert payload == {
        "status": "PASS",
        "experiment_id": "R1-05",
        "scene_probe_status": "PASS",
        "summary": str(result.summary_path),
        "evidence": str(result.evidence_dir),
        "error": None,
        "hardware_status": "PENDING_HARDWARE",
    }
    load_call = state.load_config_calls[0]
    assert load_call["config_path"] == "config/custom.json"
    assert load_call["project_root"] == cli_module.PROJECT_ROOT
    assert load_call["environ"]["ROBOT_BACKEND"] == "sim"
    assert load_call["environ"]["VISION_BACKEND"] == "sim"
    assert load_call["environ"]["COPPELIA_HOST"] == "127.0.0.9"
    assert load_call["environ"]["COPPELIA_PORT"] == "23111"
    controller = state.controllers[0]
    assert controller.kwargs["experiment_context"] is context
    assert controller.kwargs["experiment_definition"] is definition
    assert controller.kwargs["scene_manifest"] == manifest_payload
    assert controller.kwargs["output_root"] == str(
        tmp_path / "output with spaces"
    )
    assert (
        "controller.load",
        definition.student_template,
    ) in state.operations
    assert state.sessions[0].close_calls == 1
    assert state.sessions[0].close_quarantined_calls == 0
    assert state.sessions[0].closed
    assert state.applications[0].closed
    assert not controller.process_is_alive


def test_experiment_run_routes_runtime_diagnostics_away_from_stdout(
    tmp_path,
    monkeypatch,
    capsys,
):
    _install_fake_experiment_runtime(
        monkeypatch,
        tmp_path,
        runtime_stdout="[SIM] selected formal scene",
    )

    code = main(["experiment-run", "--experiment", "R1-05"])
    payload, stderr = _one_json_line(capsys)

    assert code == 0
    assert payload["status"] == "PASS"
    assert stderr.splitlines() == ["[SIM] selected formal scene"]


def test_experiment_run_validation_failure_returns_two_without_start(
    tmp_path,
    monkeypatch,
    capsys,
):
    issue = ValidationIssue("MAIN_MISSING", "必须定义 main(ctx)")
    validation = ValidationResult(
        (tmp_path / "invalid.py").resolve(),
        False,
        (issue,),
    )
    state, _, _, _, _ = _install_fake_experiment_runtime(
        monkeypatch,
        tmp_path,
        validation=validation,
    )

    code = main(["experiment-run", "--experiment", "R1-05"])
    payload, _ = _one_json_line(capsys)

    assert code == 2
    assert payload["status"] == "FAIL"
    assert payload["issues"] == [
        {
            "code": "MAIN_MISSING",
            "message": "必须定义 main(ctx)",
            "line": None,
            "column": None,
        }
    ]
    assert "controller.start" not in state.operations
    assert state.sessions[0].closed
    assert state.applications[0].closed


def test_experiment_run_validation_failure_keeps_json_when_cleanup_exits(
    tmp_path,
    monkeypatch,
    capsys,
):
    issue = ValidationIssue("MAIN_MISSING", "必须定义 main(ctx)")
    validation = ValidationResult(
        (tmp_path / "invalid.py").resolve(),
        False,
        (issue,),
    )
    state, _, _, _, _ = _install_fake_experiment_runtime(
        monkeypatch,
        tmp_path,
        validation=validation,
    )
    cleanup_error = SystemExit(94)
    session_class = session_module.VisionLabSession
    base_close = session_class.close

    def interrupting_close(self):
        base_close(self)
        raise cleanup_error

    monkeypatch.setattr(session_class, "close", interrupting_close)

    code = main(["experiment-run", "--experiment", "R1-05"])
    payload, stderr = _one_json_line(capsys)

    assert code == 2
    assert payload["status"] == "FAIL"
    assert payload["issues"] == [
        {
            "code": "MAIN_MISSING",
            "message": "必须定义 main(ctx)",
            "line": None,
            "column": None,
        }
    ]
    assert payload["error"]["code"] == "EXPERIMENT_SESSION_CLOSE_FAILED"
    assert payload["error"]["type"] == "builtins.SystemExit"
    assert "EXPERIMENT_SESSION_CLOSE_FAILED" in stderr
    assert "SystemExit" in stderr
    assert "controller.start" not in state.operations
    assert state.sessions[0].close_calls == 1
    assert state.sessions[0].closed
    assert state.applications[0].closed


@pytest.mark.parametrize(
    "fail_stage",
    [
        "config",
        "application",
        "load_scene",
        "load",
        "validate",
        "start",
        "wait",
        "timeout",
    ],
)
def test_experiment_run_stage_failure_is_json_and_releases_resources(
    tmp_path,
    monkeypatch,
    capsys,
    fail_stage,
):
    state, _, _, _, _ = _install_fake_experiment_runtime(
        monkeypatch,
        tmp_path,
        fail_stage=fail_stage,
    )

    code = main(["experiment-run", "--experiment", "R1-05"])
    payload, _ = _one_json_line(capsys)

    assert code == 1
    assert payload["status"] == "FAIL"
    assert payload["error"]["code"] == "EXPERIMENT_RUN_FAILED"
    assert payload["hardware_status"] == "PENDING_HARDWARE"
    assert all(session.closed for session in state.sessions)
    assert all(application.closed for application in state.applications)
    assert all(not controller.process_is_alive for controller in state.controllers)
    if fail_stage in {"start", "wait", "timeout"}:
        assert state.controllers[0].cancel_calls == 1


def test_experiment_run_student_failure_returns_one_without_relabeling(
    tmp_path,
    monkeypatch,
    capsys,
):
    summary = tmp_path / "failed-summary.json"
    summary.write_text(
        json.dumps({"scene_probe_status": "FAIL"}),
        encoding="utf-8",
    )
    result = SimpleNamespace(
        status="FAILED",
        summary_path=summary,
        evidence_dir=tmp_path / "failed-evidence",
        error={"code": "STUDENT_PROGRAM_FAILED", "message": "student failed"},
    )
    state, _, _, _, _ = _install_fake_experiment_runtime(
        monkeypatch,
        tmp_path,
        result=result,
    )

    code = main(["experiment-run", "--experiment", "R1-05"])
    payload, _ = _one_json_line(capsys)

    assert code == 1
    assert payload["status"] == "FAILED"
    assert payload["scene_probe_status"] == "FAIL"
    assert payload["error"] == result.error
    assert state.sessions[0].closed


@pytest.mark.parametrize(
    "cleanup_error",
    (
        pytest.param(SystemExit(95), id="system-exit"),
        pytest.param(
            KeyboardInterrupt("cleanup operator stop after student failure"),
            id="keyboard-interrupt",
        ),
    ),
)
def test_experiment_run_student_failure_keeps_json_when_cleanup_interrupts(
    tmp_path,
    monkeypatch,
    capsys,
    cleanup_error,
):
    summary = tmp_path / "failed-summary.json"
    summary.write_text(
        json.dumps({"scene_probe_status": "FAIL"}),
        encoding="utf-8",
    )
    student_error = {
        "code": "STUDENT_PROGRAM_FAILED",
        "message": "student failed",
    }
    result = runner_module.StudentRunResult(
        status="FAILED",
        summary_path=summary,
        evidence_dir=tmp_path / "failed-evidence",
        error=student_error,
    )
    state, _, _, _, _ = _install_fake_experiment_runtime(
        monkeypatch,
        tmp_path,
        result=result,
    )
    session_class = session_module.VisionLabSession
    base_close = session_class.close

    def interrupting_close(self):
        base_close(self)
        raise cleanup_error

    monkeypatch.setattr(session_class, "close", interrupting_close)

    code = main(["experiment-run", "--experiment", "R1-05"])
    payload, stderr = _one_json_line(capsys)

    assert code == 1
    assert payload["status"] == "FAILED"
    assert payload["scene_probe_status"] == "FAIL"
    assert payload["error"]["code"] == student_error["code"]
    assert payload["error"]["message"] == student_error["message"]
    cleanup_payload = payload["error"]["details"]["cleanup_errors"][0]
    assert cleanup_payload["code"] == "EXPERIMENT_SESSION_CLOSE_FAILED"
    assert cleanup_payload["type"] == (
        f"{type(cleanup_error).__module__}."
        f"{type(cleanup_error).__qualname__}"
    )
    assert cleanup_payload["message"] == str(cleanup_error)
    assert "EXPERIMENT_SESSION_CLOSE_FAILED" in stderr
    assert type(cleanup_error).__name__ in stderr
    assert state.sessions[0].close_calls == 1
    assert state.sessions[0].closed
    assert state.applications[0].closed
    assert not state.controllers[0].process_is_alive


def test_experiment_run_cleanup_failure_does_not_replace_primary_error(
    tmp_path,
    monkeypatch,
    capsys,
):
    state, _, _, _, _ = _install_fake_experiment_runtime(
        monkeypatch,
        tmp_path,
        fail_stage="wait",
    )
    # Change only final close behavior after the primary wait failure is set.
    session_class = session_module.VisionLabSession
    base_close = session_class.close

    def failing_close(self):
        base_close(self)
        raise RuntimeError("close exploded")

    monkeypatch.setattr(session_class, "close", failing_close)

    code = main(["experiment-run", "--experiment", "R1-05"])
    payload, _ = _one_json_line(capsys)

    assert code == 1
    assert payload["error"]["code"] == "EXPERIMENT_RUN_FAILED"
    assert payload["error"]["message"] == "wait exploded"
    assert payload["error"]["details"]["cleanup_errors"][0]["code"] == (
        "EXPERIMENT_SESSION_CLOSE_FAILED"
    )
    assert state.sessions[0].closed
    assert state.applications[0].closed
    assert not state.controllers[0].process_is_alive


@pytest.mark.parametrize(
    "cleanup_error",
    (
        pytest.param(SystemExit(93), id="system-exit"),
        pytest.param(
            KeyboardInterrupt("cleanup operator stop"),
            id="keyboard-interrupt",
        ),
    ),
)
def test_experiment_run_cleanup_base_exception_does_not_replace_run_error_json(
    tmp_path,
    monkeypatch,
    capsys,
    cleanup_error,
):
    primary_error = RuntimeError("wait exploded before cleanup")
    state, _, _, _, _ = _install_fake_experiment_runtime(
        monkeypatch,
        tmp_path,
    )
    controller_class = runner_module.StudentProgramController
    session_class = session_module.VisionLabSession
    base_close = session_class.close

    def failing_wait(self, *, timeout_s):
        state.operations.append(("controller.wait", timeout_s))
        raise primary_error

    def interrupting_close(self):
        base_close(self)
        raise cleanup_error

    monkeypatch.setattr(controller_class, "wait", failing_wait)
    monkeypatch.setattr(session_class, "close", interrupting_close)

    code = main(["experiment-run", "--experiment", "R1-05"])
    payload, stderr = _one_json_line(capsys)

    assert code == 1
    assert payload["error"]["code"] == "EXPERIMENT_RUN_FAILED"
    assert payload["error"]["message"] == str(primary_error)
    cleanup_payload = payload["error"]["details"]["cleanup_errors"][0]
    assert cleanup_payload["code"] == "EXPERIMENT_SESSION_CLOSE_FAILED"
    assert cleanup_payload["type"] == (
        f"{type(cleanup_error).__module__}."
        f"{type(cleanup_error).__qualname__}"
    )
    assert cleanup_payload["message"] == str(cleanup_error)
    diagnostics = "\n".join(getattr(primary_error, "__notes__", ()))
    assert "EXPERIMENT_SESSION_CLOSE_FAILED" in diagnostics
    assert type(cleanup_error).__name__ in diagnostics
    assert "EXPERIMENT_SESSION_CLOSE_FAILED" in stderr
    assert type(cleanup_error).__name__ in stderr
    controller = state.controllers[0]
    assert controller.cancel_calls == 1
    assert controller.quiescence_calls == [3.25]
    assert not controller.process_is_alive
    assert state.sessions[0].close_calls == 1
    assert state.sessions[0].closed
    assert state.applications[0].closed
    assert state.operations.index(
        ("controller.wait_for_quiescence", 3.25)
    ) < state.operations.index("session.close")


def test_experiment_run_cancel_failure_is_cleanup_detail_not_primary(
    tmp_path,
    monkeypatch,
    capsys,
):
    state, _, _, _, _ = _install_fake_experiment_runtime(
        monkeypatch,
        tmp_path,
        fail_stage="wait_cancel",
    )

    code = main(["experiment-run", "--experiment", "R1-05"])
    payload, _ = _one_json_line(capsys)

    assert code == 1
    assert payload["error"]["code"] == "EXPERIMENT_RUN_FAILED"
    assert payload["error"]["message"] == "wait exploded"
    assert payload["error"]["details"]["cleanup_errors"][0]["code"] == (
        "EXPERIMENT_CANCEL_FAILED"
    )
    assert state.sessions[0].close_quarantined_calls == 1
    assert state.sessions[0].closed
    assert state.applications[0].closed
    assert not state.controllers[0].process_is_alive


@pytest.mark.parametrize(
    "raised",
    [
        pytest.param(KeyboardInterrupt("operator stop"), id="keyboard"),
        pytest.param(SystemExit(23), id="system-exit"),
    ],
)
def test_experiment_run_base_exception_cleans_resources_and_reraises_same(
    tmp_path,
    monkeypatch,
    capsys,
    raised,
):
    state, _, _, _, _ = _install_fake_experiment_runtime(
        monkeypatch,
        tmp_path,
    )
    controller_class = runner_module.StudentProgramController

    def interrupting_wait(self, *, timeout_s):
        state.operations.append(("controller.wait", timeout_s))
        self.process_is_alive = True
        raise raised

    monkeypatch.setattr(controller_class, "wait", interrupting_wait)

    with pytest.raises(BaseException) as caught:
        main(["experiment-run", "--experiment", "R1-05"])

    captured = capsys.readouterr()
    controller = state.controllers[0]
    assert caught.value is raised
    assert captured.out == ""
    assert controller.cancel_calls == 1
    assert controller.quiescence_calls == [3.25]
    assert state.sessions[0].close_calls == 1
    assert state.sessions[0].closed
    assert state.applications[0].closed
    assert not controller.process_is_alive
    assert state.operations.index("controller.cancel") < state.operations.index(
        ("controller.wait_for_quiescence", 3.25)
    )
    assert state.operations.index(
        ("controller.wait_for_quiescence", 3.25)
    ) < state.operations.index("session.close")


def test_experiment_run_cleanup_error_does_not_replace_base_exception(
    tmp_path,
    monkeypatch,
    capsys,
):
    raised = KeyboardInterrupt("operator stop")
    state, _, _, _, _ = _install_fake_experiment_runtime(
        monkeypatch,
        tmp_path,
    )
    controller_class = runner_module.StudentProgramController
    session_class = session_module.VisionLabSession
    base_close = session_class.close

    def interrupting_wait(self, *, timeout_s):
        state.operations.append(("controller.wait", timeout_s))
        self.process_is_alive = True
        raise raised

    def failing_close(self):
        base_close(self)
        raise RuntimeError("close exploded during interrupt")

    monkeypatch.setattr(controller_class, "wait", interrupting_wait)
    monkeypatch.setattr(session_class, "close", failing_close)

    with pytest.raises(KeyboardInterrupt) as caught:
        main(["experiment-run", "--experiment", "R1-05"])

    captured = capsys.readouterr()
    diagnostics = "\n".join(getattr(raised, "__notes__", ()))
    diagnostics += captured.err
    assert caught.value is raised
    assert "close exploded during interrupt" in diagnostics
    assert state.sessions[0].closed
    assert state.applications[0].closed
    assert not state.controllers[0].process_is_alive


def test_experiment_run_cleanup_base_exception_does_not_replace_primary(
    tmp_path,
    monkeypatch,
    capsys,
):
    primary = KeyboardInterrupt("primary operator stop")
    cleanup = SystemExit(91)
    state, _, _, _, _ = _install_fake_experiment_runtime(
        monkeypatch,
        tmp_path,
    )
    controller_class = runner_module.StudentProgramController

    def interrupting_wait(self, *, timeout_s):
        state.operations.append(("controller.wait", timeout_s))
        self.process_is_alive = True
        raise primary

    def interrupting_cancel(self):
        self.cancel_calls += 1
        state.operations.append("controller.cancel")
        self.process_is_alive = False
        raise cleanup

    monkeypatch.setattr(controller_class, "wait", interrupting_wait)
    monkeypatch.setattr(controller_class, "cancel", interrupting_cancel)

    with pytest.raises(KeyboardInterrupt) as caught:
        main(["experiment-run", "--experiment", "R1-05"])

    captured = capsys.readouterr()
    diagnostics = "\n".join(getattr(primary, "__notes__", ()))
    diagnostics += captured.err
    assert caught.value is primary
    assert "EXPERIMENT_CANCEL_FAILED" in diagnostics
    assert "SystemExit" in diagnostics
    assert state.sessions[0].closed
    assert state.applications[0].closed
    assert not state.controllers[0].process_is_alive


def test_experiment_run_cleanup_base_exception_propagates_without_primary(
    tmp_path,
    monkeypatch,
    capsys,
):
    cleanup = SystemExit(92)
    state, _, _, _, _ = _install_fake_experiment_runtime(
        monkeypatch,
        tmp_path,
    )
    session_class = session_module.VisionLabSession
    base_close = session_class.close

    def interrupting_close(self):
        base_close(self)
        raise cleanup

    monkeypatch.setattr(session_class, "close", interrupting_close)

    with pytest.raises(SystemExit) as caught:
        main(["experiment-run", "--experiment", "R1-05"])

    captured = capsys.readouterr()
    assert caught.value is cleanup
    assert captured.out == ""
    assert state.sessions[0].closed
    assert state.applications[0].closed
    assert not state.controllers[0].process_is_alive


@pytest.mark.parametrize(
    ("main_mode", "cleanup_base_kind"),
    (
        pytest.param("pass", "system-exit", id="pass-system-exit"),
        pytest.param(
            "pass",
            "keyboard-interrupt",
            id="pass-keyboard-interrupt",
        ),
        pytest.param("failed", "system-exit", id="failed-system-exit"),
        pytest.param(
            "validation",
            "system-exit",
            id="validation-system-exit",
        ),
        pytest.param("raised", "system-exit", id="raised-system-exit"),
        pytest.param(
            "interrupt",
            "system-exit",
            id="interrupt-system-exit",
        ),
    ),
)
def test_experiment_run_sequential_cleanup_error_priority_matrix(
    tmp_path,
    monkeypatch,
    capsys,
    main_mode,
    cleanup_base_kind,
):
    validation = None
    result = None
    if main_mode == "validation":
        validation = ValidationResult(
            (tmp_path / "invalid.py").resolve(),
            False,
            (ValidationIssue("MAIN_MISSING", "必须定义 main(ctx)"),),
        )
    elif main_mode == "failed":
        failed_summary = tmp_path / "failed-summary.json"
        failed_summary.write_text(
            json.dumps({"scene_probe_status": "FAIL"}),
            encoding="utf-8",
        )
        result = runner_module.StudentRunResult(
            status="FAILED",
            summary_path=failed_summary,
            evidence_dir=tmp_path / "failed-evidence",
            error={
                "code": "STUDENT_PROGRAM_FAILED",
                "message": "student failed",
            },
        )

    state, _, _, _, _ = _install_fake_experiment_runtime(
        monkeypatch,
        tmp_path,
        validation=validation,
        result=result,
    )
    cleanup_base_error = (
        SystemExit(96)
        if cleanup_base_kind == "system-exit"
        else KeyboardInterrupt("later cleanup interrupt")
    )
    first_cleanup_error = _install_sequential_cleanup_failures(
        monkeypatch,
        state,
        cleanup_base_error=cleanup_base_error,
    )
    primary_error = None
    if main_mode in {"raised", "interrupt"}:
        primary_error = (
            RuntimeError("main run failed")
            if main_mode == "raised"
            else KeyboardInterrupt("main run interrupted")
        )
        controller_class = runner_module.StudentProgramController

        def failing_wait(self, *, timeout_s):
            state.operations.append(("controller.wait", timeout_s))
            raise primary_error

        monkeypatch.setattr(controller_class, "wait", failing_wait)

    if main_mode == "interrupt":
        with pytest.raises(KeyboardInterrupt) as caught:
            main(["experiment-run", "--experiment", "R1-05"])
        captured = capsys.readouterr()
        assert caught.value is primary_error
        assert captured.out == ""
        stderr = captured.err
        diagnostics = "\n".join(
            getattr(primary_error, "__notes__", ())
        )
        assert "EXPERIMENT_PROCESS_LIVENESS_FAILED" in diagnostics
        assert "EXPERIMENT_SESSION_CLOSE_FAILED" in diagnostics
    else:
        unexpected_error = None
        try:
            code = main(["experiment-run", "--experiment", "R1-05"])
        except BaseException as error:
            unexpected_error = error
        assert unexpected_error is None
        payload, stderr = _one_json_line(capsys)

        expected_exit = 2 if main_mode == "validation" else 1
        assert code == expected_exit
        if main_mode == "failed":
            assert payload["status"] == "FAILED"
            assert payload["error"]["code"] == "STUDENT_PROGRAM_FAILED"
            cleanup_errors = payload["error"]["details"][
                "cleanup_errors"
            ]
            assert [error["code"] for error in cleanup_errors] == [
                "EXPERIMENT_PROCESS_LIVENESS_FAILED",
                "EXPERIMENT_SESSION_CLOSE_FAILED",
            ]
        elif main_mode == "raised":
            assert payload["status"] == "FAIL"
            assert payload["error"]["code"] == "EXPERIMENT_RUN_FAILED"
            cleanup_errors = payload["error"]["details"][
                "cleanup_errors"
            ]
            assert [error["code"] for error in cleanup_errors] == [
                "EXPERIMENT_PROCESS_LIVENESS_FAILED",
                "EXPERIMENT_SESSION_CLOSE_FAILED",
            ]
            diagnostics = "\n".join(
                getattr(primary_error, "__notes__", ())
            )
            assert "EXPERIMENT_PROCESS_LIVENESS_FAILED" in diagnostics
            assert "EXPERIMENT_SESSION_CLOSE_FAILED" in diagnostics
        else:
            assert payload["status"] == "FAIL"
            assert payload["error"]["code"] == (
                "EXPERIMENT_PROCESS_LIVENESS_FAILED"
            )
            cleanup_errors = payload["error"]["details"][
                "cleanup_errors"
            ]
            assert [error["code"] for error in cleanup_errors] == [
                "EXPERIMENT_SESSION_CLOSE_FAILED"
            ]
            if main_mode == "validation":
                assert payload["issues"][0]["code"] == "MAIN_MISSING"

    assert "EXPERIMENT_SESSION_CLOSE_FAILED" in stderr
    assert type(cleanup_base_error).__name__ in stderr
    controller = state.controllers[0]
    assert controller.quiescence_calls == [3.25]
    assert not controller.process_is_alive
    assert state.sessions[0].close_quarantined_calls == 1
    assert state.sessions[0].closed
    assert state.applications[0].closed
    assert state.operations.index(
        "controller.process_is_alive.cleanup"
    ) < state.operations.index(("controller.wait_for_quiescence", 3.25))
    assert state.operations.index(
        ("controller.wait_for_quiescence", 3.25)
    ) < state.operations.index("session.close_quarantined")
