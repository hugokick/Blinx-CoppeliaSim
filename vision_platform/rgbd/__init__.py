"""Public API for the pure in-memory D1 RGB-D kernel."""

from .errors import RgbdContractError
from .geometry import deproject_pixel
from .measurement import measure_pixel
from .models import (
    CameraIntrinsics,
    DepthSample,
    Point3M,
    RgbdFrame,
    RgbdMeasurement,
)
from .sampling import sample_depth
from .serialization import measurement_to_dict
from .transforms import RigidTransform, transform_point

__all__ = [
    "CameraIntrinsics",
    "DepthSample",
    "Point3M",
    "RgbdContractError",
    "RgbdFrame",
    "RgbdMeasurement",
    "RigidTransform",
    "deproject_pixel",
    "measure_pixel",
    "measurement_to_dict",
    "sample_depth",
    "transform_point",
]
