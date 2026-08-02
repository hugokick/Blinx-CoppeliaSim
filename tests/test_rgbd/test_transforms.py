from __future__ import annotations

import numpy as np
import pytest

from vision_platform.rgbd.errors import RgbdContractError
from vision_platform.rgbd.models import Point3M
from vision_platform.rgbd.transforms import RigidTransform, transform_point


def test_identity_and_translation_are_applied_without_aliasing() -> None:
    matrix = np.array(
        [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, -0.2],
         [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transform = RigidTransform.from_matrix(matrix)
    matrix[0, 3] = 99.0
    result = transform_point(Point3M(1.0, 2.0, 3.0), transform)
    assert (result.x_m, result.y_m, result.z_m) == pytest.approx((1.5, 1.8, 4.0))
    assert not transform.matrix.flags.writeable


def test_z_rotation_matches_analytic_point() -> None:
    matrix = np.array(
        [[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    result = transform_point(
        Point3M(1.0, 0.0, 2.0), RigidTransform.from_matrix(matrix)
    )
    assert (result.x_m, result.y_m, result.z_m) == pytest.approx((0.0, 1.0, 2.0))


def _invalid_matrices() -> list[np.ndarray]:
    wrong_shape = np.eye(3, dtype=np.float64)
    non_finite = np.eye(4, dtype=np.float64)
    non_finite[0, 0] = np.nan
    bad_row = np.eye(4, dtype=np.float64)
    bad_row[3, 0] = 1.0
    scaled = np.eye(4, dtype=np.float64)
    scaled[0, 0] = 2.0
    sheared = np.eye(4, dtype=np.float64)
    sheared[0, 1] = 0.2
    reflected = np.eye(4, dtype=np.float64)
    reflected[0, 0] = -1.0
    return [wrong_shape, non_finite, bad_row, scaled, sheared, reflected]


@pytest.mark.parametrize("matrix", _invalid_matrices())
def test_rigid_transform_rejects_non_rigid_matrix(matrix: np.ndarray) -> None:
    with pytest.raises(RgbdContractError) as captured:
        RigidTransform.from_matrix(matrix)
    assert captured.value.code == "RGBD_TRANSFORM_INVALID"


def test_rigid_transform_rejects_object_dtype() -> None:
    with pytest.raises(RgbdContractError):
        RigidTransform.from_matrix(np.eye(4, dtype=object))
