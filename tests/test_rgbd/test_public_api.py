from __future__ import annotations

import ast
from pathlib import Path

import vision_platform.rgbd as rgbd


EXPECTED_PUBLIC_API = {
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
}


def test_public_api_is_exact_and_resolvable() -> None:
    assert set(rgbd.__all__) == EXPECTED_PUBLIC_API
    assert len(rgbd.__all__) == len(EXPECTED_PUBLIC_API)
    for name in rgbd.__all__:
        assert getattr(rgbd, name) is not None


def test_rgbd_sources_have_no_forbidden_imports() -> None:
    root = Path(rgbd.__file__).resolve().parent
    forbidden = {"cv2", "open3d", "rclpy", "zmq", "PyQt5", "socket", "subprocess"}
    for source in root.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & forbidden, f"{source.name}: {sorted(imported & forbidden)}"
