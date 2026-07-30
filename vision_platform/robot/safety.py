from __future__ import annotations

from dataclasses import dataclass
from math import ceil, dist, isfinite
from typing import Iterable

from vision_platform.errors import MotionSafetyError


Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class WorkspacePolicy:
    x_mm: tuple[float, float]
    y_mm: tuple[float, float]
    z_mm: tuple[float, float]
    safe_z_mm: float
    max_step_mm: float = 250.0

    def __post_init__(self) -> None:
        for axis, bounds in (
            ("x", self.x_mm),
            ("y", self.y_mm),
            ("z", self.z_mm),
        ):
            if len(bounds) != 2 or bounds[0] >= bounds[1]:
                raise ValueError(f"{axis}_mm must contain increasing bounds")
            if not all(isfinite(float(value)) for value in bounds):
                raise ValueError(f"{axis}_mm bounds must be finite")
        if not self.z_mm[0] <= self.safe_z_mm <= self.z_mm[1]:
            raise ValueError("safe_z_mm must be inside z_mm bounds")
        if not isfinite(self.max_step_mm) or self.max_step_mm <= 0:
            raise ValueError("max_step_mm must be positive and finite")

    def validate(self, point: Iterable[float]) -> Point3:
        values = tuple(float(value) for value in point)
        if len(values) != 3:
            raise MotionSafetyError(
                "Cartesian target must contain X, Y and Z",
                target=list(values),
            )
        target = (values[0], values[1], values[2])
        if not all(isfinite(value) for value in target):
            raise MotionSafetyError(
                "Cartesian target must contain finite values",
                target=list(target),
            )
        bounds = (self.x_mm, self.y_mm, self.z_mm)
        outside = [
            axis
            for axis, value, (minimum, maximum) in zip("xyz", target, bounds)
            if not minimum <= value <= maximum
        ]
        if outside:
            raise MotionSafetyError(
                f"Target is outside workspace on axis: {', '.join(outside)}",
                target=list(target),
                axes=outside,
                workspace={
                    "x_mm": list(self.x_mm),
                    "y_mm": list(self.y_mm),
                    "z_mm": list(self.z_mm),
                },
            )
        return target


def _subdivide(start: Point3, end: Point3, max_step_mm: float) -> list[Point3]:
    segment_length = dist(start, end)
    count = max(1, ceil(segment_length / max_step_mm))
    return [
        tuple(
            start[axis] + (end[axis] - start[axis]) * index / count
            for axis in range(3)
        )
        for index in range(1, count + 1)
    ]


def plan_gate_path(
    *,
    pick: Iterable[float],
    drop: Iterable[float],
    policy: WorkspacePolicy,
) -> tuple[Point3, ...]:
    """Plan an approach/descend/lift/transfer/release gate-shaped path."""
    pick_point = policy.validate(pick)
    drop_point = policy.validate(drop)
    above_pick = (pick_point[0], pick_point[1], float(policy.safe_z_mm))
    above_drop = (drop_point[0], drop_point[1], float(policy.safe_z_mm))
    anchors = (
        policy.validate(above_pick),
        pick_point,
        policy.validate(above_pick),
        policy.validate(above_drop),
        drop_point,
        policy.validate(above_drop),
    )

    result: list[Point3] = [anchors[0]]
    for start, end in zip(anchors, anchors[1:]):
        result.extend(_subdivide(start, end, policy.max_step_mm))
    return tuple(result)
