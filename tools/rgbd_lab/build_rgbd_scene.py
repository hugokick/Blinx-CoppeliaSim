from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from vision_platform.coppelia_scene import stage_scene_for_coppeliasim


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RGBD_DIR = PROJECT_ROOT / "simulation" / "rgbd_lab"
SPEC_PATH = RGBD_DIR / "scene_spec.json"
OUTPUT_PATH = RGBD_DIR / "BL23_rgbd_lab.ttt"
MANIFEST_PATH = RGBD_DIR / "scene_manifest.json"
D1_PORT = 23009


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def validate_build_arguments(*, spec_path: Path, port: int) -> None:
    if int(port) != D1_PORT:
        raise ValueError("D1-01 scene builder is bound to port 23009")
    if Path(spec_path).resolve() != SPEC_PATH.resolve():
        raise ValueError("builder accepts only its dedicated RGB-D scene spec")


def _set_alias(sim: Any, handle: int, alias: str) -> None:
    sim.setObjectAlias(int(handle), alias)


def _set_parent(sim: Any, handle: int, parent: int) -> None:
    sim.setObjectParent(int(handle), int(parent), True)


def _set_color(sim: Any, handle: int, color: list[float]) -> None:
    sim.setShapeColor(
        int(handle), "", sim.colorcomponent_ambient_diffuse, [float(v) for v in color]
    )


def _make_dummy(sim: Any, alias: str, parent: int | None = None) -> int:
    handle = int(sim.createDummy(0.01))
    _set_alias(sim, handle, alias)
    if parent is not None:
        _set_parent(sim, handle, parent)
    return handle


def _make_shape(
    sim: Any,
    *,
    alias: str,
    parent: int,
    size_m: list[float],
    position_m: list[float],
    color: list[float],
) -> int:
    handle = int(sim.createPrimitiveShape(sim.primitiveshape_cuboid, size_m, 0))
    _set_alias(sim, handle, alias)
    _set_color(sim, handle, color)
    sim.setObjectPosition(handle, [float(v) for v in position_m], sim.handle_world)
    _set_parent(sim, handle, parent)
    return handle


def _set_int(sim: Any, handle: int, name: str, value: int) -> None:
    sim.setObjectInt32Param(handle, getattr(sim, name), int(value))


def _set_float(sim: Any, handle: int, name: str, value: float) -> None:
    sim.setObjectFloatParam(handle, getattr(sim, name), float(value))


def _build_sensor(sim: Any, spec: dict[str, Any], rig: int) -> int:
    sensor = spec["sensor"]
    width, height = [int(v) for v in sensor["resolution"]]
    angle = math.radians(float(sensor["perspective_angle_deg"]))
    handle = int(
        sim.createVisionSensor(
            2 | 4 | 64 | 128,
            [width, height, 0, 0],
            [
                float(sensor["near_clip_m"]),
                float(sensor["far_clip_m"]),
                angle,
                0.02,
                0.0,
                0.0,
                0.08,
                0.08,
                0.1,
                0.0,
                0.0,
            ],
        )
    )
    _set_alias(sim, handle, "RgbdSensor")
    sim.setObjectPose(
        handle,
        [
            *[float(v) for v in sensor["position_m"]],
            *[float(v) for v in sensor["orientation_quaternion"]],
        ],
        sim.handle_world,
    )
    _set_parent(sim, handle, rig)
    _set_int(sim, handle, "visionintparam_resolution_x", width)
    _set_int(sim, handle, "visionintparam_resolution_y", height)
    _set_int(sim, handle, "visionintparam_perspective_operation", 1)
    sim.setExplicitHandling(handle, 1)
    if int(sim.getExplicitHandling(handle)) != 1:
        raise RuntimeError("RGB-D sensor was not created with explicit handling")
    _set_int(sim, handle, "visionintparam_rgbignored", 0)
    _set_int(sim, handle, "visionintparam_depthignored", 0)
    _set_float(sim, handle, "visionfloatparam_perspective_angle", angle)
    _set_float(sim, handle, "visionfloatparam_near_clipping", float(sensor["near_clip_m"]))
    _set_float(sim, handle, "visionfloatparam_far_clipping", float(sensor["far_clip_m"]))
    return handle


def build_scene(*, output: Path = OUTPUT_PATH, host: str = "127.0.0.1", port: int = D1_PORT) -> dict[str, Any]:
    validate_build_arguments(spec_path=SPEC_PATH, port=port)
    spec = _load(SPEC_PATH)
    template = (PROJECT_ROOT / spec["formal_scene_template"]).resolve()
    template_before = _sha256(template)
    client = RemoteAPIClient(host=host, port=port)
    try:
        sim = client.require("sim")
        if sim.getSimulationState() != sim.simulation_stopped:
            sim.stopSimulation()
            client.step()
        sim.loadScene(stage_scene_for_coppeliasim(template).as_posix())
        try:
            sim.removeObject(sim.getObject("/VisionLab"), sim.handle_tree)
        except Exception:
            pass
        root = _make_dummy(sim, "RgbdLab")
        rig = _make_dummy(sim, "CameraRig", root)
        _build_sensor(sim, spec, rig)
        plane = spec["reference_plane"]
        _make_shape(
            sim,
            alias="ReferencePlane",
            parent=root,
            size_m=[float(v) for v in plane["size_m"]],
            position_m=[0.0, 0.0, float(plane["top_z_m"]) - float(plane["size_m"][2]) / 2.0],
            color=plane["color"],
        )
        targets = _make_dummy(sim, "Targets", root)
        for target in spec["targets"]:
            _make_shape(
                sim,
                alias=target["path"].rsplit("/", 1)[-1],
                parent=targets,
                size_m=[float(v) for v in target["size_m"]],
                position_m=[float(v) for v in target["position_m"]],
                color=target["color"],
            )
        anchors = _make_dummy(sim, "ProbeAnchors", root)
        for anchor in spec["probe_anchors"]:
            handle = _make_dummy(sim, anchor["path"].rsplit("/", 1)[-1], anchors)
            sim.setObjectPosition(handle, [float(v) for v in anchor["position_m"]], sim.handle_world)
        for required in spec["required_paths"]:
            sim.getObject(required)
        output = Path(output).resolve()
        if output != OUTPUT_PATH.resolve():
            raise ValueError("scene output must be the dedicated RGB-D scene path")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        sim.saveScene(output.as_posix())
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError("CoppeliaSim did not write the RGB-D scene")
    finally:
        try:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
    template_after = _sha256(template)
    if template_before != template_after:
        raise RuntimeError("protected template changed during RGB-D scene build")
    scene_hash = _sha256(OUTPUT_PATH)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "tools.rgbd_lab.build_rgbd_scene",
        "port": D1_PORT,
        "protected_assets_unchanged": True,
        "template": {"path": spec["formal_scene_template"], "sha256": template_before},
        "scene": {"path": OUTPUT_PATH.name, "sha256": scene_hash, "size_bytes": OUTPUT_PATH.stat().st_size},
        "sensor": spec["sensor"],
        "source_depth_model": spec["source_depth_model"],
        "required_paths": spec["required_paths"],
        "probe_anchors": spec["probe_anchors"],
        "validation_rois": spec["validation_rois"],
        "depth_order": spec["depth_order"],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the D1-01 RGB-D CoppeliaSim scene")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=D1_PORT)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    manifest = build_scene(output=args.output, host=args.host, port=args.port)
    print(json.dumps({"status": "PASS", "scene": manifest["scene"], "port": D1_PORT}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
