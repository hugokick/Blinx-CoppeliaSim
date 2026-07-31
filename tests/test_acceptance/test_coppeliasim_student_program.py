from __future__ import annotations

import json
from math import dist
from pathlib import Path

import pytest

from vision_platform.student.protocol import RunState
from vision_platform.student.runner import StudentProgramController
from vision_platform.student.safety import StudentExecutionPolicy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROGRAM = (
    PROJECT_ROOT
    / "student_programs"
    / "templates"
    / "pick_and_place.py"
)
_ACTIVE_STATES = frozenset({RunState.RUNNING, RunState.PAUSED})


class BoundSession:
    def __init__(self, application):
        self.application = application
        self.reset_calls = 0

    def reset_simulation(self):
        self.reset_calls += 1
        raise AssertionError(
            "online test must not reset before assertions"
        )


def _configured_policy(application) -> StudentExecutionPolicy:
    student = application.config.student
    speed_range = student["speed_range"]
    return StudentExecutionPolicy(
        min_speed=float(speed_range[0]),
        max_speed=float(speed_range[1]),
        max_runtime_s=float(student["max_runtime_s"]),
        max_commands=int(student["max_commands"]),
        command_timeout_s=float(student["command_timeout_s"]),
        max_sleep_s=float(student["max_sleep_s"]),
        tool_on_max_z_mm=float(student["tool_on_max_z_mm"]),
    )


def _fail_safe_quiesce(controller: StudentProgramController) -> None:
    if controller.wait_for_quiescence(0):
        return
    cancel_error = None
    try:
        if controller.state in _ACTIVE_STATES:
            controller.cancel()
    except BaseException as error:
        cancel_error = error
    if controller.wait_for_quiescence(10):
        return
    if cancel_error is not None:
        raise AssertionError(
            "student controller did not quiesce after cancel failed"
        ) from cancel_error
    raise AssertionError(
        "student controller did not quiesce before scene teardown"
    )


@pytest.mark.coppeliasim
def test_student_program_moves_live_openr6_and_records_evidence(
    running_vision_scene,
    tmp_path,
):
    application = running_vision_scene
    application.open()
    session = BoundSession(application)
    controller = StudentProgramController(
        session=session,
        execution_policy=_configured_policy(application),
        output_root=tmp_path / "student-runs",
    )
    measured_pose_results: list[tuple[float, float, float]] = []
    original_command_pose = controller._command_pose

    def record_command_pose(args):
        value = original_command_pose(args)
        measured_pose_results.append(tuple(value))
        return value

    controller._command_pose = record_command_pose

    try:
        controller.load(PROGRAM)
        assert controller.validate().ok is True

        controller.start()
        result = controller.wait(timeout_s=30)

        assert result.status == "PASS"
        assert controller.process_is_alive is False
        assert controller.wait_for_quiescence(5) is True
        assert result.evidence_dir is not None
        assert result.summary_path is not None
        evidence_dir = result.evidence_dir
        expected_files = (
            "source.py",
            "source.sha256",
            "manifest.json",
            "commands.jsonl",
            "events.jsonl",
            "summary.json",
        )
        for name in expected_files:
            assert (evidence_dir / name).is_file()

        assert (evidence_dir / "source.py").read_bytes() == (
            PROGRAM.read_bytes()
        )
        source_sha256 = (
            evidence_dir / "source.sha256"
        ).read_text(encoding="utf-8").strip()
        manifest = json.loads(
            (evidence_dir / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["source_sha256"] == source_sha256
        assert manifest["program_path"] == str(PROGRAM.resolve())
        assert manifest["hardware_status"] == "PENDING_HARDWARE"
        commands = [
            json.loads(line)
            for line in (
                evidence_dir / "commands.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(commands) == controller.command_count
        critical_commands = [
            command
            for command in commands
            if command["name"] != "context.log"
        ]
        assert [command["name"] for command in critical_commands] == [
            "robot.home",
            "robot.move_world",
            "robot.pose",
            "robot.move_world",
            "tool.on",
            "robot.pose",
            "robot.move_world",
            "robot.move_world",
            "robot.pose",
            "robot.move_world",
            "tool.off",
            "robot.pose",
            "robot.move_world",
            "robot.home",
        ]
        move_commands = [
            command
            for command in critical_commands
            if command["name"] == "robot.move_world"
        ]
        assert len(move_commands) == 6
        assert len(measured_pose_results) == 4
        assert move_commands[0]["args"] == {
            "x_mm": 55,
            "y_mm": -55,
            "z_mm": 110,
            "speed": 15,
        }
        assert move_commands[1]["args"]["x_mm"] == pytest.approx(
            55,
            abs=0.5,
        )
        assert move_commands[1]["args"]["y_mm"] == pytest.approx(
            -55,
            abs=0.5,
        )
        assert move_commands[1]["args"]["z_mm"] == 20
        assert move_commands[1]["args"]["speed"] == 8
        assert move_commands[1]["args"]["x_mm"] == pytest.approx(
            measured_pose_results[0][0],
            abs=1e-9,
        )
        assert move_commands[1]["args"]["y_mm"] == pytest.approx(
            measured_pose_results[0][1],
            abs=1e-9,
        )
        assert measured_pose_results[0][2] >= 100
        assert move_commands[2]["args"]["x_mm"] == pytest.approx(
            55,
            abs=0.5,
        )
        assert move_commands[2]["args"]["y_mm"] == pytest.approx(
            -55,
            abs=0.5,
        )
        assert move_commands[2]["args"]["z_mm"] == 110
        assert move_commands[2]["args"]["speed"] == 12
        assert move_commands[3]["args"] == {
            "x_mm": 122,
            "y_mm": -66,
            "z_mm": 110,
            "speed": 15,
        }
        assert move_commands[4]["args"]["x_mm"] == pytest.approx(
            122,
            abs=0.5,
        )
        assert move_commands[4]["args"]["y_mm"] == pytest.approx(
            -66,
            abs=0.5,
        )
        assert move_commands[4]["args"]["z_mm"] == 20
        assert move_commands[4]["args"]["speed"] == 8
        assert move_commands[4]["args"]["x_mm"] == pytest.approx(
            measured_pose_results[2][0],
            abs=1e-9,
        )
        assert move_commands[4]["args"]["y_mm"] == pytest.approx(
            measured_pose_results[2][1],
            abs=1e-9,
        )
        assert measured_pose_results[2][2] >= 100
        assert move_commands[5]["args"]["x_mm"] == pytest.approx(
            122,
            abs=0.5,
        )
        assert move_commands[5]["args"]["y_mm"] == pytest.approx(
            -66,
            abs=0.5,
        )
        assert move_commands[5]["args"]["z_mm"] == 110
        assert move_commands[5]["args"]["speed"] == 12
        assert (evidence_dir / "events.jsonl").stat().st_size > 0

        assert result.summary_path == evidence_dir / "summary.json"
        summary = json.loads(
            result.summary_path.read_text(encoding="utf-8")
        )
        assert summary["status"] == "PASS"
        assert summary["hardware_status"] == "PENDING_HARDWARE"
        assert summary["source_sha256"] == source_sha256
        assert summary["command_count"] == controller.command_count

        expected = tuple(
            float(value)
            for value in application.scene_spec["robot_visuals"][
                "teaching_ready_pose"
            ]["tcp_expected_mm"]
        )
        actual = application.robot.current_world_pose()
        assert dist(actual, expected) <= 2.0

        object_path = "/VisionLab/Pickables/object_01_red_square"
        object_handle = application.sim.getObject(object_path)
        object_position_mm = tuple(
            float(value) * 1000.0
            for value in application.sim.getObjectPosition(
                object_handle,
                application.sim.handle_world,
            )
        )
        red_zone = application.scene_spec["zones"]["red"]
        zone_x, zone_y, _ = (
            float(value) for value in red_zone["center_mm"]
        )
        size_x, size_y, _ = (
            float(value) for value in red_zone["size_mm"]
        )
        tolerance_mm = 2.0
        assert (
            zone_x - size_x / 2.0 - tolerance_mm
            <= object_position_mm[0]
            <= zone_x + size_x / 2.0 + tolerance_mm
        ), (
            f"object XY={object_position_mm[:2]} mm did not reach "
            f"red zone center=({zone_x}, {zone_y}) mm "
            f"size=({size_x}, {size_y}) mm"
        )
        assert (
            zone_y - size_y / 2.0 - tolerance_mm
            <= object_position_mm[1]
            <= zone_y + size_y / 2.0 + tolerance_mm
        ), (
            f"object XY={object_position_mm[:2]} mm did not reach "
            f"red zone center=({zone_x}, {zone_y}) mm "
            f"size=({size_x}, {size_y}) mm"
        )
        assert session.reset_calls == 0
    finally:
        _fail_safe_quiesce(controller)
