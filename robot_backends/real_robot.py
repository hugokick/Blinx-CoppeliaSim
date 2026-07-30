import importlib.util
from pathlib import Path

from robot_backends.base import RobotBackend


class RealRobotBackend(RobotBackend):
    def __init__(self, sdk_robot=None, source_path=None):
        self._sdk_robot = sdk_robot or self._load_default_robot(source_path)

    def _load_default_robot(self, source_path=None):
        source_path = source_path or Path(
            r"h:\智能机械与机器人基础\实训课\6.基于视觉的机器人应用\1.机械臂认知和基础操作\Six_Robot_Control.py"
        )
        spec = importlib.util.spec_from_file_location("real_six_robot_control", source_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.Blinx_Six_Robot_Control()

    def home(self):
        return self._sdk_robot.blinx_home()

    def move_home(self):
        return self._sdk_robot.blinx_move_home()

    def move_angle(self, joint_id, speed, value):
        return self._sdk_robot.blinx_move_angle(joint_id, speed, value)

    def move_angle_all(self, value1, value2, value3, value4, value5, value6, speed):
        return self._sdk_robot.blinx_move_angle_all(value1, value2, value3, value4, value5, value6, speed)

    def move_coordinate_all(self, value1, value2, value3, value4, value5, value6, speed):
        return self._sdk_robot.blinx_move_coordinate_all(value1, value2, value3, value4, value5, value6, speed)

    def pump_on(self):
        return self._sdk_robot.blinx_pump_on()

    def pump_off(self):
        return self._sdk_robot.blinx_pump_off()

    def positive_solution(self):
        return self._sdk_robot.blinx_positive_solution()

    def close(self):
        return self._sdk_robot.blinx_close()
