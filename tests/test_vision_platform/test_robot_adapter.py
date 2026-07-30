import pytest

from vision_platform.errors import MotionSafetyError
from vision_platform.robot.adapter import RobotAdapter
from vision_platform.robot.safety import WorkspacePolicy
from vision_platform.robot.tool import BackendPumpTool


class RecordingBackend:
    def __init__(self):
        self.coordinate_calls = []
        self.home_calls = 0
        self.move_home_calls = 0
        self.pump_on_calls = 0
        self.pump_off_calls = 0
        self.close_calls = 0

    def home(self):
        self.home_calls += 1

    def move_home(self):
        self.move_home_calls += 1

    def move_coordinate_all(self, x, y, z, rx, ry, rz, speed):
        self.coordinate_calls.append((x, y, z, rx, ry, rz, speed))

    def positive_solution(self):
        return [80.0, -20.0, 50.0, 1.0, 2.0, 3.0]

    def pump_on(self):
        self.pump_on_calls += 1

    def pump_off(self):
        self.pump_off_calls += 1

    def close(self):
        self.close_calls += 1


def _policy():
    return WorkspacePolicy(
        x_mm=(20, 140),
        y_mm=(-90, 90),
        z_mm=(10, 140),
        safe_z_mm=100,
    )


def test_move_world_preserves_position_only_backend_contract():
    backend = RecordingBackend()
    robot = RobotAdapter(backend, policy=_policy())

    robot.move_world(80, -20, 50, speed=15)

    assert backend.coordinate_calls == [(80, -20, 50, 0, 0, 0, 15)]
    assert robot.current_world_pose() == (80.0, -20.0, 50.0)


def test_move_world_rejects_target_outside_configured_workspace():
    backend = RecordingBackend()
    robot = RobotAdapter(backend, policy=_policy())

    with pytest.raises(MotionSafetyError) as exc:
        robot.move_world(141, 0, 50, speed=15)

    assert exc.value.code == "TARGET_OUT_OF_WORKSPACE"
    assert backend.coordinate_calls == []


def test_robot_lifecycle_delegates_to_existing_backend():
    backend = RecordingBackend()
    robot = RobotAdapter(backend, policy=_policy())

    robot.initialize()
    robot.move_home()
    robot.close()

    assert (backend.home_calls, backend.move_home_calls, backend.close_calls) == (
        1,
        1,
        1,
    )


def test_backend_pump_tool_exposes_idempotent_command_state():
    backend = RecordingBackend()
    tool = BackendPumpTool(backend)

    first = tool.on()
    second = tool.on()
    assert tool.is_attached() is True
    tool.off()
    tool.off()

    assert first.attached is True
    assert second.attached is True
    assert backend.pump_on_calls == 1
    assert backend.pump_off_calls == 1
    assert tool.is_attached() is False
