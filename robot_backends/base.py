from abc import ABC, abstractmethod


class RobotBackend(ABC):
    @abstractmethod
    def home(self):
        """Initialize / reset the robot arm (equivalent to blinx_home / set_robot_arm_init)."""
        raise NotImplementedError

    @abstractmethod
    def move_home(self):
        """Move the robot arm to its home position (equivalent to blinx_move_home / set_robot_arm_home)."""
        raise NotImplementedError

    @abstractmethod
    def move_angle(self, joint_id, speed, value):
        raise NotImplementedError

    @abstractmethod
    def move_angle_all(self, value1, value2, value3, value4, value5, value6, speed):
        raise NotImplementedError

    @abstractmethod
    def move_coordinate_all(self, value1, value2, value3, value4, value5, value6, speed):
        raise NotImplementedError

    @abstractmethod
    def pump_on(self):
        raise NotImplementedError

    @abstractmethod
    def pump_off(self):
        raise NotImplementedError

    @abstractmethod
    def positive_solution(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError
