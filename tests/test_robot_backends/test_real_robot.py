import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_backends.real_robot import RealRobotBackend


class FakeSdkRobot:
    def __init__(self):
        self.calls = []

    def blinx_home(self):
        self.calls.append(("home",))

    def blinx_move_home(self):
        self.calls.append(("move_home",))

    def blinx_move_angle(self, joint_id, speed, value):
        self.calls.append(("move_angle", joint_id, speed, value))

    def blinx_move_angle_all(self, v1, v2, v3, v4, v5, v6, speed):
        self.calls.append(("move_angle_all", v1, v2, v3, v4, v5, v6, speed))

    def blinx_move_coordinate_all(self, v1, v2, v3, v4, v5, v6, speed):
        self.calls.append(("move_coordinate_all", v1, v2, v3, v4, v5, v6, speed))

    def blinx_pump_on(self):
        self.calls.append(("pump_on",))

    def blinx_pump_off(self):
        self.calls.append(("pump_off",))

    def blinx_positive_solution(self):
        self.calls.append(("positive_solution",))
        return [0, 0, 0, 0, 0, 0]

    def blinx_close(self):
        self.calls.append(("close",))


def test_real_backend_forwards_calls():
    sdk_robot = FakeSdkRobot()
    backend = RealRobotBackend(sdk_robot=sdk_robot)
    backend.home()
    backend.move_home()
    backend.move_angle(2, 50, 10)
    assert sdk_robot.calls == [
        ("home",),
        ("move_home",),
        ("move_angle", 2, 50, 10),
    ]


def test_real_backend_pump():
    sdk_robot = FakeSdkRobot()
    backend = RealRobotBackend(sdk_robot=sdk_robot)
    backend.pump_on()
    backend.pump_off()
    assert sdk_robot.calls == [("pump_on",), ("pump_off",)]


def test_real_backend_move_angle_all():
    sdk_robot = FakeSdkRobot()
    backend = RealRobotBackend(sdk_robot=sdk_robot)
    backend.move_angle_all(10, 20, 30, 40, 50, 60, 50)
    assert sdk_robot.calls == [("move_angle_all", 10, 20, 30, 40, 50, 60, 50)]
