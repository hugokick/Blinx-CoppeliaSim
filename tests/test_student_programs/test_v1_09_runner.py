from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest

from vision_platform.errors import VisionPlatformError
from vision_platform.student.protocol import CommandMessage, RunState
from vision_platform.student.runner import StudentProgramController
from vision_platform.student.safety import StudentExecutionPolicy
from vision_platform.robot.safety import WorkspacePolicy
from vision_platform.experiments.defect_sorting import defect_sort_plan_to_dict
from tests.test_experiments.test_defect_sorting import _build


class Robot:
    def __init__(self) -> None:
        self.moves: list[tuple] = []
        self.pose = [80.0, 0.0, 110.0]

    def current_world_pose(self):
        return tuple(self.pose)

    def move_world(self, x, y, z, *, speed):
        self.moves.append((x, y, z, speed))
        self.pose[:] = [x, y, z]

    def move_home(self):
        self.moves.append(("home",))


class Tool:
    def __init__(self) -> None:
        self.events: list[str] = []

    def on(self):
        self.events.append("on")

    def off(self):
        self.events.append("off")


class Evidence:
    run_id = "run-001"
    directory = None

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def record_event(self, *args, **kwargs):
        self.events.append((args, kwargs))


def _controller(tmp_path: Path) -> tuple[StudentProgramController, Robot, Tool]:
    robot = Robot()
    tool = Tool()
    app = SimpleNamespace(
        config=SimpleNamespace(robot_backend="sim", camera_backend="sim"),
        workspace=WorkspacePolicy(x_mm=(20, 155), y_mm=(-95, 95), z_mm=(10, 140), safe_z_mm=110),
        robot=robot,
        tool=tool,
        camera=None,
        sim=None,
    )
    policy = StudentExecutionPolicy(
        min_speed=1, max_speed=30, max_runtime_s=60, max_commands=200,
        command_timeout_s=10, max_sleep_s=5, tool_on_max_z_mm=35,
    )
    controller = StudentProgramController(
        session=SimpleNamespace(application=app),
        execution_policy=policy,
        output_root=tmp_path / "runs",
    )
    controller._experiment_definition = SimpleNamespace(capabilities=("vision2d.surface_defects",))
    return controller, robot, tool


def _surface_command(controller: StudentProgramController):
    return controller._dispatch(
        CommandMessage("000001", "vision2d.surface_defects", {})
    )


def test_v1_09_surface_analysis_begins_before_gateway_activation(tmp_path: Path) -> None:
    controller, _, _ = _controller(tmp_path)
    calls: list[str] = []
    plan = _build()

    class Gateway:
        def dispatch(self, name: str, args: dict):
            calls.append(name)
            assert controller._defect_guard.state == "ANALYZING"
            controller._defect_guard.activate(plan)
            return {"status": "PASS", "plan_id": plan.plan_id}

    controller._experiment_gateway = Gateway()

    result = _surface_command(controller)

    assert result == {"status": "PASS", "plan_id": plan.plan_id}
    assert calls == ["vision2d.surface_defects"]
    assert controller._defect_guard.state == "ACTIVE"
    assert controller._defect_guard.plan_id == plan.plan_id


def test_v1_09_surface_analysis_failure_fails_guard_and_preserves_primary_error(tmp_path: Path) -> None:
    controller, _, _ = _controller(tmp_path)
    calls: list[str] = []

    class Gateway:
        def dispatch(self, name: str, args: dict):
            calls.append(name)
            assert controller._defect_guard.state == "ANALYZING"
            raise VisionPlatformError("DEFECT_SORT_ANALYSIS_FAILED", "analysis rejected")

    controller._experiment_gateway = Gateway()

    with pytest.raises(VisionPlatformError) as captured:
        _surface_command(controller)

    assert captured.value.code == "DEFECT_SORT_ANALYSIS_FAILED"
    assert calls == ["vision2d.surface_defects"]
    assert controller._defect_guard.state == "FAILED"
    assert controller._defect_guard.plan_id is None
    assert controller._defect_guard.actions == ()
    assert controller._defect_guard.error["code"] == "DEFECT_SORT_ANALYSIS_FAILED"


def test_v1_09_surface_response_failure_invalidates_activated_plan(tmp_path: Path) -> None:
    controller, _, _ = _controller(tmp_path)
    plan = _build()
    calls: list[str] = []

    class Gateway:
        def dispatch(self, name: str, args: dict):
            calls.append(name)
            assert controller._defect_guard.state == "ANALYZING"
            controller._defect_guard.activate(plan)
            raise VisionPlatformError("DEFECT_SORT_RESPONSE_INVALID", "response rejected")

    controller._experiment_gateway = Gateway()

    with pytest.raises(VisionPlatformError) as captured:
        _surface_command(controller)

    assert captured.value.code == "DEFECT_SORT_RESPONSE_INVALID"
    assert calls == ["vision2d.surface_defects"]
    assert controller._defect_guard.state == "FAILED"
    assert controller._defect_guard.plan_id is None
    assert controller._defect_guard.actions == ()
    assert controller._defect_guard.error["code"] == "DEFECT_SORT_RESPONSE_INVALID"


def test_v1_09_surface_analysis_without_activation_fails_closed(tmp_path: Path) -> None:
    controller, _, _ = _controller(tmp_path)
    calls: list[str] = []

    class Gateway:
        def dispatch(self, name: str, args: dict):
            calls.append(name)
            assert controller._defect_guard.state == "ANALYZING"
            return {"status": "PASS", "plan_id": "not-activated"}

    controller._experiment_gateway = Gateway()

    with pytest.raises(VisionPlatformError) as captured:
        _surface_command(controller)

    assert captured.value.code == "DEFECT_SORT_PLAN_NOT_ACTIVE"
    assert calls == ["vision2d.surface_defects"]
    assert controller._defect_guard.state == "FAILED"
    assert controller._defect_guard.plan_id is None
    assert controller._defect_guard.actions == ()
    assert controller._defect_guard.error["code"] == "DEFECT_SORT_PLAN_NOT_ACTIVE"


def test_v1_09_repeated_surface_analysis_is_rejected_before_gateway_reentry(tmp_path: Path) -> None:
    controller, _, _ = _controller(tmp_path)
    plan = _build()
    calls: list[str] = []

    class Gateway:
        def dispatch(self, name: str, args: dict):
            calls.append(name)
            assert controller._defect_guard.state == "ANALYZING"
            controller._defect_guard.activate(plan)
            return {"status": "PASS", "plan_id": plan.plan_id}

    controller._experiment_gateway = Gateway()

    first = _surface_command(controller)
    with pytest.raises(VisionPlatformError) as captured:
        _surface_command(controller)

    assert first == {"status": "PASS", "plan_id": plan.plan_id}
    assert captured.value.code == "DEFECT_SORT_ANALYSIS_ALREADY_ACTIVE"
    assert calls == ["vision2d.surface_defects"]
    assert controller._defect_guard.state == "ACTIVE"
    assert controller._defect_guard.plan_id == plan.plan_id


@pytest.mark.parametrize("name", ["robot.home", "robot.pose", "robot.move_world", "tool.on", "tool.off"])
def test_v1_09_raw_device_commands_are_rejected_before_device_calls(tmp_path: Path, name: str) -> None:
    controller, robot, tool = _controller(tmp_path)
    controller._state = RunState.RUNNING
    args = {
        "robot.home": {},
        "robot.pose": {},
        "robot.move_world": {"x_mm": 50, "y_mm": 0, "z_mm": 110, "speed": 10},
        "tool.on": {},
        "tool.off": {},
    }[name]
    with pytest.raises(VisionPlatformError) as captured:
        controller._dispatch(CommandMessage("000001", name, args))
    assert captured.value.code == "COMMAND_NOT_ALLOWED"
    assert robot.moves == []
    assert tool.events == []


def test_v1_09_entry_command_is_private_and_does_not_reenter_gateway(tmp_path: Path) -> None:
    controller, robot, tool = _controller(tmp_path)
    controller._state = RunState.RUNNING
    with pytest.raises(VisionPlatformError):
        controller._dispatch(CommandMessage("000001", "vision2d.defect_sort_entry", {"entry_id": "entry_a"}))
    assert robot.moves == []
    assert tool.events == []


def test_v1_09_controller_owns_a_separate_defect_guard(tmp_path: Path) -> None:
    controller, _, _ = _controller(tmp_path)
    assert controller._defect_guard.state == "EMPTY"
    assert controller._ocr_guard.state == "EMPTY"


def _active_defect_controller(tmp_path: Path):
    controller, robot, tool = _controller(tmp_path)
    plan = _build()
    controller._defect_guard.begin_analysis()
    controller._defect_guard.activate(plan)
    evidence = Evidence()

    class Gateway:
        _defect_evidence = {"plan": plan}

        def collect_defect_entry_pre_probe(self, entry_id: str, *, run_id: str):
            entry = next(item for item in plan.entries if item.entry_id == entry_id)
            return {
                "run_id": run_id,
                "scene_sha256": plan.scene_sha256,
                "frame_id": plan.frame_id,
                "plan_id": plan.plan_id,
                "entry_id": entry.entry_id,
                "part_id": entry.part_id,
                "slot_id": entry.slot_id,
                "evidence_id": f"pre-{entry_id}",
            }

        def collect_defect_entry_post_probe(self, entry_id: str, *, run_id: str):
            entry = next(item for item in plan.entries if item.entry_id == entry_id)
            return {
                "run_id": run_id,
                "scene_sha256": plan.scene_sha256,
                "frame_id": plan.frame_id,
                "plan_id": plan.plan_id,
                "entry_id": entry.entry_id,
                "part_id": entry.part_id,
                "slot_id": entry.slot_id,
                "evidence_id": f"post-{entry_id}",
            }

    controller._experiment_gateway = Gateway()
    controller._evidence = evidence
    controller._state = RunState.RUNNING
    return controller, robot, tool, evidence


def test_v1_09_private_runner_completes_only_after_post_probe(tmp_path: Path) -> None:
    controller, robot, tool, _ = _active_defect_controller(tmp_path)
    receipt = controller._command_defect_sort_entry({"entry_id": "entry_a"})
    assert receipt["status"] == "COMPLETED"
    assert receipt["decision"] == "qualified"
    assert len(robot.moves) == 6  # one safe pick/drop transfer plus the repeated guard actions
    assert tool.events == ["on", "off"]
    assert controller._defect_guard.consumed_entry_ids == ("entry_a",)


def test_v1_09_complete_guard_captures_same_run_final_probe_context(
    tmp_path: Path,
) -> None:
    controller, _, _, evidence = _active_defect_controller(tmp_path)
    gateway = controller._experiment_gateway
    plan = gateway._defect_evidence["plan"]
    controller._experiment_context = SimpleNamespace(
        scene_sha256=plan.scene_sha256,
    )

    for entry in plan.entries:
        receipt = controller._command_defect_sort_entry(
            {"entry_id": entry.entry_id}
        )
        assert receipt["entry_id"] == entry.entry_id
        assert receipt["status"] == "COMPLETED"

    assert controller._defect_guard.state == "COMPLETE"
    context = controller._defect_probe_context()

    assert context is not None
    expected_ids = [f"entry_{letter}" for letter in "abcdef"]
    assert context["run_id"] == evidence.run_id
    assert context["frame_id"] == plan.frame_id
    assert context["scene_hash"] == plan.scene_sha256
    assert context["plan_id"] == plan.plan_id
    assert context["plan_public"] == defect_sort_plan_to_dict(plan)
    assert context["consumed_entry_ids"] == expected_ids
    assert [item["entry_id"] for item in context["entry_evidence"]] == expected_ids
    assert context["guard"]["state"] == "COMPLETE"
    assert context["guard"]["consumed_entry_ids"] == expected_ids
    assert context["robot_home"] is False
    assert context["tool_on"] is False
    json.dumps(
        context,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )


def test_v1_09_single_step_permits_one_device_call(tmp_path: Path) -> None:
    controller, robot, _, _ = _active_defect_controller(tmp_path)
    controller._state = RunState.PAUSED
    action = controller._defect_guard.begin_entry("entry_a")[0]
    errors: list[BaseException] = []

    def execute() -> None:
        try:
            controller._execute_defect_action(action)
        except BaseException as error:  # pragma: no cover - diagnostic only
            errors.append(error)

    worker = threading.Thread(target=execute)
    worker.start()
    deadline = time.monotonic() + 1.0
    controller.step()
    while worker.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(robot.moves) == 1
    assert not worker.is_alive()
    worker.join(timeout=1.0)
    assert len(robot.moves) == 1
    assert errors == []
