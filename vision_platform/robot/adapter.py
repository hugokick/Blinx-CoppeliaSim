from __future__ import annotations

from typing import Any, Iterable

from vision_platform.robot.safety import Point3, WorkspacePolicy


class RobotAdapter:
    """Position-only teaching contract over the existing BLX backend."""

    def __init__(self, backend: Any, *, policy: WorkspacePolicy | None = None):
        self.backend = backend
        self.policy = policy
        self._closed = False

    def initialize(self) -> None:
        self._ensure_open()
        self.backend.home()

    def move_home(self) -> None:
        self._ensure_open()
        self.backend.move_home()

    def move_world(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        *,
        speed: float,
    ) -> None:
        self._ensure_open()
        if speed <= 0:
            raise ValueError("speed must be positive")
        point = (float(x_mm), float(y_mm), float(z_mm))
        if self.policy is not None:
            self.policy.validate(point)
        self.backend.move_coordinate_all(
            x_mm,
            y_mm,
            z_mm,
            0,
            0,
            0,
            speed,
        )

    def execute_path(self, points: Iterable[Point3], *, speed: float) -> None:
        for x_mm, y_mm, z_mm in points:
            self.move_world(x_mm, y_mm, z_mm, speed=speed)

    def current_world_pose(self) -> Point3:
        self._ensure_open()
        pose = self.backend.positive_solution()
        if len(pose) < 3:
            raise RuntimeError("Robot backend returned an incomplete Cartesian pose")
        return (float(pose[0]), float(pose[1]), float(pose[2]))

    def close(self) -> None:
        if self._closed:
            return
        self.backend.close()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Robot adapter is closed")
