"""Explicit finite rigid transforms for metric 3D points."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import RgbdContractError
from .models import Point3M


def _validated_matrix(matrix: np.ndarray) -> np.ndarray:
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.shape != (4, 4)
        or matrix.dtype.kind not in {"f", "i", "u"}
        or not np.all(np.isfinite(matrix))
    ):
        raise RgbdContractError(
            "RGBD_TRANSFORM_INVALID", "matrix must be finite numeric 4x4"
        )
    value = np.array(matrix, dtype=np.float64, copy=True, order="C")
    if not np.allclose(
        value[3], np.array([0.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=1e-6
    ):
        raise RgbdContractError(
            "RGBD_TRANSFORM_INVALID", "homogeneous row is invalid"
        )
    rotation = value[:3, :3]
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6
    ) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise RgbdContractError(
            "RGBD_TRANSFORM_INVALID", "rotation must be proper and orthonormal"
        )
    backing = value.tobytes(order="C")
    return np.frombuffer(backing, dtype=np.float64, count=16).reshape(
        (4, 4), order="C"
    )


@dataclass(frozen=True, eq=False)
class RigidTransform:
    matrix: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", _validated_matrix(self.matrix))

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "RigidTransform":
        return cls(matrix)


def transform_point(point: Point3M, transform: RigidTransform) -> Point3M:
    if not isinstance(point, Point3M) or not isinstance(transform, RigidTransform):
        raise RgbdContractError(
            "RGBD_TRANSFORM_INVALID", "point or transform has invalid type"
        )
    homogeneous = np.array(
        [point.x_m, point.y_m, point.z_m, 1.0], dtype=np.float64
    )
    result = transform.matrix @ homogeneous
    if not np.all(np.isfinite(result)) or not math.isclose(
        float(result[3]), 1.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise RgbdContractError(
            "RGBD_TRANSFORM_INVALID", "transformed point is invalid"
        )
    return Point3M(float(result[0]), float(result[1]), float(result[2]))


__all__ = ["RigidTransform", "transform_point"]
