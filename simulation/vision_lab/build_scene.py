from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from simulation.vision_lab.hashing import asset_sha256
from vision_platform.coppelia_scene import stage_scene_for_coppeliasim


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VISION_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = VISION_DIR / "BL23_vision_lab.ttt"

FORMAL_VISUAL_PATHS = {
    "base": "/BLX_base_link",
    "link1": "/BLX_link1",
    "link2": "/BLX_link2",
    "link3": "/BLX_link3",
    "link4": "/BLX_link4",
    "link5": "/BLX_link5",
    "link6": "/BLX_link6",
}

ROBOT_COLORS = {
    "base": [0.18, 0.22, 0.27],
    "link1": [0.90, 0.92, 0.94],
    "link2": [0.16, 0.43, 0.64],
    "link3": [0.90, 0.92, 0.94],
    "link4": [0.16, 0.43, 0.64],
    "link5": [0.90, 0.92, 0.94],
    "link6": [0.16, 0.43, 0.64],
    "tool": [0.10, 0.55, 0.58],
}

OBJECT_COLORS = {
    "red": [0.88, 0.08, 0.08],
    "green": [0.06, 0.72, 0.18],
    "blue": [0.05, 0.28, 0.92],
    "yellow": [0.98, 0.80, 0.04],
    "white": [0.94, 0.94, 0.94],
    "magenta": [0.90, 0.05, 0.78],
}

ZONE_COLORS = {
    "red": [1.0, 0.87, 0.87],
    "green": [0.86, 1.0, 0.88],
    "blue": [0.86, 0.91, 1.0],
    "yellow": [1.0, 0.97, 0.82],
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_protected(source: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for entry in source["protected_formal_assets"]:
        path = PROJECT_ROOT / entry["path"]
        actual = asset_sha256(
            path,
            mode=entry.get("hash_mode", "bytes"),
        )
        if actual.lower() != entry["sha256"].lower():
            raise RuntimeError(
                f"Protected formal asset hash mismatch: {entry['path']}"
            )
        hashes[entry["path"]] = actual
    return hashes


def _set_int_parameter(sim: Any, handle: int, parameter: int, value: int) -> None:
    setter = getattr(sim, "setObjectInt32Param", None)
    if not callable(setter):
        setter = getattr(sim, "setObjectInt32Parameter")
    setter(handle, parameter, int(value))


def _set_alias(sim: Any, handle: int, alias: str) -> None:
    sim.setObjectAlias(handle, alias)


def _set_parent(sim: Any, child: int, parent: int) -> None:
    sim.setObjectParent(child, parent, True)


def _set_shape_color(sim: Any, handle: int, rgb: list[float]) -> None:
    sim.setShapeColor(
        handle,
        "",
        sim.colorcomponent_ambient_diffuse,
        [float(value) for value in rgb],
    )


def _make_dummy(sim: Any, alias: str, parent: int | None = None) -> int:
    handle = int(sim.createDummy(0.005))
    _set_alias(sim, handle, alias)
    if parent is not None:
        _set_parent(sim, handle, parent)
    return handle


def _configure_static_shape(
    sim: Any,
    handle: int,
    *,
    respondable: bool,
) -> None:
    _set_int_parameter(sim, handle, sim.shapeintparam_static, 1)
    _set_int_parameter(
        sim,
        handle,
        sim.shapeintparam_respondable,
        int(respondable),
    )


def _set_vision_renderable(sim: Any, handle: int, value: bool) -> None:
    current = int(sim.getObjectSpecialProperty(handle))
    flag = int(sim.objectspecialproperty_renderable)
    updated = current | flag if value else current & ~flag
    sim.setObjectSpecialProperty(handle, updated)


def _create_primitive(
    sim: Any,
    *,
    primitive: int,
    size_m: list[float],
    alias: str,
    parent: int,
    position_m: list[float],
    color: list[float],
    respondable: bool = False,
) -> int:
    handle = int(sim.createPrimitiveShape(primitive, size_m, 0))
    _set_alias(sim, handle, alias)
    _set_shape_color(sim, handle, color)
    _configure_static_shape(sim, handle, respondable=respondable)
    sim.setObjectPosition(handle, position_m, sim.handle_world)
    _set_parent(sim, handle, parent)
    return handle


def _reference_mesh_entry(
    source: dict[str, Any],
    segment: str,
) -> dict[str, Any]:
    filename = "base_link.STL" if segment == "base" else f"{segment}.STL"
    expected = f"robot_backends/models/meshes_blx/{filename}"
    for entry in source["protected_formal_assets"]:
        if entry["path"] == expected:
            return entry
    raise RuntimeError(f"Reference OpenR6 mesh is not pinned: {expected}")


def _build_suction_visuals(
    sim: Any,
    *,
    visual_spec: dict[str, Any],
) -> dict[str, Any]:
    end_effector = visual_spec["end_effector"]
    tcp_path = end_effector["tcp_path"]
    tcp_handle = int(sim.getObject(tcp_path))
    component_paths: list[str] = []
    component_rows: list[dict[str, Any]] = []
    primitive_by_name = {
        "cylinder": sim.primitiveshape_cylinder,
    }
    for component in end_effector["components"]:
        primitive_name = component["primitive"]
        if primitive_name not in primitive_by_name:
            raise ValueError(
                f"Unsupported suction primitive: {primitive_name}"
            )
        handle = int(
            sim.createPrimitiveShape(
                primitive_by_name[primitive_name],
                [float(value) for value in component["size_m"]],
                0,
            )
        )
        _set_alias(sim, handle, component["alias"])
        _set_shape_color(
            sim,
            handle,
            [float(value) for value in component["color"]],
        )
        _configure_static_shape(sim, handle, respondable=False)
        sim.setObjectParent(handle, tcp_handle, False)
        sim.setObjectPose(
            handle,
            [
                float(value)
                for value in component["pose_relative_tcp"]
            ],
            tcp_handle,
        )
        _set_vision_renderable(sim, handle, True)
        runtime_full_path = sim.getObjectAlias(handle, 2)
        path = f"{tcp_path}/{component['alias']}"
        if int(sim.getObject(path)) != handle:
            raise RuntimeError(
                f"Suction component contract path did not resolve: {path}"
            )
        component_paths.append(path)
        component_rows.append(
            {
                "path": path,
                "runtime_full_path": runtime_full_path,
                "primitive": primitive_name,
                "size_m": [
                    float(value) for value in component["size_m"]
                ],
                "pose_relative_tcp": [
                    float(value)
                    for value in sim.getObjectPose(handle, tcp_handle)
                ],
            }
        )
    return {
        "segment": "tool",
        "path": tcp_path,
        "parent_path": FORMAL_VISUAL_PATHS["link6"],
        "tcp_path": tcp_path,
        "component_paths": component_paths,
        "components": component_rows,
        "original_end_effector_reused": bool(
            end_effector["original_end_effector_reused"]
        ),
        "alignment_basis": {
            "mode": "custom_suction_primitives_at_tcp",
        },
        "usage": "runtime_custom_suction",
        "collision_strategy": "non_respondable_visuals",
    }


def _apply_teaching_ready_pose(
    sim: Any,
    *,
    visual_spec: dict[str, Any],
) -> dict[str, Any]:
    pose_spec = visual_spec["teaching_ready_pose"]
    angles_deg = [
        float(value) for value in pose_spec["joint_angles_deg"]
    ]
    if len(angles_deg) != 6:
        raise ValueError("Teaching-ready pose must define six joint angles")
    for index, angle_deg in enumerate(angles_deg, start=1):
        handle = int(sim.getObject(f"/BLX_joint{index}"))
        angle_rad = math.radians(angle_deg)
        sim.setJointPosition(handle, angle_rad)
        sim.setJointTargetPosition(handle, angle_rad)
    actual_angles_deg = [
        math.degrees(
            float(
                sim.getJointPosition(
                    sim.getObject(f"/BLX_joint{index}")
                )
            )
        )
        for index in range(1, 7)
    ]
    tcp_position_mm = [
        float(value) * 1000.0
        for value in sim.getObjectPosition(
            sim.getObject("/BLX_tool_suction"),
            sim.handle_world,
        )
    ]
    expected_tcp_mm = [
        float(value) for value in pose_spec["tcp_expected_mm"]
    ]
    tcp_error_mm = math.dist(tcp_position_mm, expected_tcp_mm)
    if tcp_error_mm > 1.0:
        raise RuntimeError(
            "Teaching-ready pose TCP error exceeds 1 mm: "
            f"{tcp_error_mm:.3f} mm"
        )
    return {
        "joint_angles_deg": angles_deg,
        "actual_joint_angles_deg": actual_angles_deg,
        "tcp_expected_mm": pose_spec["tcp_expected_mm"],
        "tcp_actual_mm": tcp_position_mm,
        "tcp_error_mm": tcp_error_mm,
        "applied": True,
        "selection": pose_spec["selection"],
    }


def _build_robot_visuals(
    sim: Any,
    *,
    spec: dict[str, Any],
    source: dict[str, Any],
) -> list[dict[str, Any]]:
    visual_spec = spec["robot_visuals"]
    visibility_layer = int(visual_spec["visibility_layer"])
    scale_corrections = {
        str(path): float(value)
        for path, value in visual_spec["scale_corrections"].items()
    }
    reports: list[dict[str, Any]] = []
    for segment, path in FORMAL_VISUAL_PATHS.items():
        handle = int(sim.getObject(path))
        scale_correction = scale_corrections.get(path)
        if scale_correction is not None:
            sim.scaleObject(
                handle,
                scale_correction,
                scale_correction,
                scale_correction,
            )
        _set_int_parameter(
            sim,
            handle,
            sim.objintparam_visibility_layer,
            visibility_layer,
        )
        _set_vision_renderable(sim, handle, True)
        _set_shape_color(sim, handle, ROBOT_COLORS[segment])
        bbox_size, _bbox_pose = sim.getShapeBB(handle)
        mesh_entry = _reference_mesh_entry(source, segment)
        parent_handle = int(sim.getObjectParent(handle))
        alignment_basis = {"mode": "openr6_embedded_reference"}
        if scale_correction is not None:
            alignment_basis = {
                "mode": (
                    "openr6_embedded_reference_with_scale_correction"
                ),
                "scale_correction": scale_correction,
            }
        reports.append(
            {
                "segment": segment,
                "path": path,
                "parent_path": (
                    "/"
                    if parent_handle < 0
                    else sim.getObjectAlias(parent_handle, 2)
                ),
                "bbox_size_m": [float(value) for value in bbox_size],
                "shape_pose_world": [
                    float(value)
                    for value in sim.getObjectPose(
                        handle,
                        sim.handle_world,
                    )
                ],
                "pose_relative_parent": [
                    float(value)
                    for value in sim.getObjectPose(
                        handle,
                        parent_handle,
                    )
                ],
                "visibility_layer": int(
                    sim.getObjectInt32Param(
                        handle,
                        sim.objintparam_visibility_layer,
                    )
                ),
                "alignment_basis": alignment_basis,
                "source_mesh_path": mesh_entry["path"],
                "source_mesh_sha256": mesh_entry["sha256"],
                "usage": "runtime_openr6_reference_body",
                "collision_strategy": (
                    "embedded_formal_shape_and_collision_chain"
                ),
            }
        )
    reports.append(
        _build_suction_visuals(
            sim,
            visual_spec=visual_spec,
        )
    )
    return reports


def _build_workspace(
    sim: Any,
    *,
    spec: dict[str, Any],
    root: int,
) -> dict[str, int]:
    workspace = spec["workspace"]
    workspace_handle = _create_primitive(
        sim,
        primitive=sim.primitiveshape_cuboid,
        size_m=[float(value) for value in workspace["size_m"]],
        alias="Workspace",
        parent=root,
        position_m=[float(value) for value in workspace["center_m"]],
        color=[0.30, 0.32, 0.34],
        respondable=True,
    )
    return {"workspace": workspace_handle}


def _build_pickables(
    sim: Any,
    *,
    spec: dict[str, Any],
    root: int,
) -> tuple[int, list[dict[str, Any]]]:
    group = _make_dummy(sim, "Pickables", root)
    rows: list[dict[str, Any]] = []
    for item in spec["objects"]:
        size_m = [float(value) / 1000.0 for value in item["size_mm"]]
        primitive = (
            sim.primitiveshape_cylinder
            if item["shape"] == "circle"
            else sim.primitiveshape_cuboid
        )
        alias = item["path"].rsplit("/", 1)[-1]
        handle = _create_primitive(
            sim,
            primitive=primitive,
            size_m=size_m,
            alias=alias,
            parent=group,
            position_m=[
                float(value) / 1000.0 for value in item["position_mm"]
            ],
            color=OBJECT_COLORS[item["color"]],
            respondable=True,
        )
        rows.append(
            {
                "path": sim.getObjectAlias(handle, 2),
                "handle_at_build": handle,
                "color": item["color"],
                "shape": item["shape"],
            }
        )
    return group, rows


def _build_zones(
    sim: Any,
    *,
    spec: dict[str, Any],
    root: int,
) -> tuple[int, list[dict[str, Any]]]:
    group = _make_dummy(sim, "Zones", root)
    rows: list[dict[str, Any]] = []
    for name, item in spec["zones"].items():
        handle = _create_primitive(
            sim,
            primitive=sim.primitiveshape_cuboid,
            size_m=[float(value) / 1000.0 for value in item["size_mm"]],
            alias=name,
            parent=group,
            position_m=[
                float(value) / 1000.0 for value in item["center_mm"]
            ],
            color=ZONE_COLORS[name],
            respondable=False,
        )
        transparency = getattr(sim, "colorcomponent_transparency", None)
        if transparency is not None:
            sim.setShapeColor(handle, "", transparency, [0.72])
        rows.append(
            {
                "name": name,
                "path": sim.getObjectAlias(handle, 2),
                "handle_at_build": handle,
            }
        )
    return group, rows


def _build_calibration(
    sim: Any,
    *,
    spec: dict[str, Any],
    root: int,
) -> tuple[int, list[dict[str, Any]]]:
    group = _make_dummy(sim, "Calibration", root)
    rows: list[dict[str, Any]] = []
    for marker in spec["calibration_markers"]:
        handle = _create_primitive(
            sim,
            primitive=sim.primitiveshape_cylinder,
            size_m=[0.007, 0.007, 0.002],
            alias=marker["path"].rsplit("/", 1)[-1],
            parent=group,
            position_m=[
                float(value) / 1000.0 for value in marker["world_mm"]
            ],
            color=OBJECT_COLORS[marker["color"]],
            respondable=False,
        )
        rows.append(
            {
                "path": sim.getObjectAlias(handle, 2),
                "role": marker["role"],
                "world_mm": marker["world_mm"],
            }
        )
    return group, rows


def _build_camera(
    sim: Any,
    *,
    spec: dict[str, Any],
    root: int,
) -> int:
    camera = spec["camera"]
    resolution = [int(value) for value in camera["resolution"]]
    angle = math.radians(float(camera["perspective_angle_deg"]))
    options = 2 | 4 | 64 | 128
    handle = int(
        sim.createVisionSensor(
            options,
            [resolution[0], resolution[1], 0, 0],
            [
                float(camera["near_clip_m"]),
                float(camera["far_clip_m"]),
                angle,
                0.02,
                0.0,
                0.0,
                0.08,
                0.08,
                0.10,
                0.0,
                0.0,
            ],
        )
    )
    _set_alias(sim, handle, "Camera")
    sim.setObjectPose(
        handle,
        [
            *[float(value) for value in camera["position_m"]],
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        sim.handle_world,
    )
    _set_parent(sim, handle, root)
    return handle


def _attach_camera_render_scope_script(sim: Any, *, root: int) -> int:
    script_text = """
function sysCall_init()
    teachingCollection = sim.createCollection(1)
    sim.addItemToCollection(
        teachingCollection,
        sim.handle_single,
        sim.getObject('/VisionLab/Workspace'),
        0
    )
    sim.addItemToCollection(
        teachingCollection,
        sim.handle_tree,
        sim.getObject('/VisionLab/Pickables'),
        0
    )
    sim.addItemToCollection(
        teachingCollection,
        sim.handle_tree,
        sim.getObject('/VisionLab/Zones'),
        0
    )
    sim.addItemToCollection(
        teachingCollection,
        sim.handle_tree,
        sim.getObject('/VisionLab/Calibration'),
        0
    )
    local camera = sim.getObject('/VisionLab/Camera')
    sim.setObjectInt32Param(
        camera,
        sim.visionintparam_entity_to_render,
        teachingCollection
    )
end

function sysCall_cleanup()
    if teachingCollection then
        sim.destroyCollection(teachingCollection)
        teachingCollection = nil
    end
end
""".strip()
    handle = int(
        sim.createScript(
            sim.scripttype_simulation,
            script_text,
            0,
            "lua",
        )
    )
    _set_alias(sim, handle, "CameraRenderScope")
    _set_parent(sim, handle, root)
    return handle


def build_scene(
    *,
    output: Path,
    host: str,
    port: int,
) -> dict[str, Any]:
    spec_path = VISION_DIR / "scene_spec.json"
    source_path = VISION_DIR / "source_manifest.json"
    assets_path = VISION_DIR / "robot_assets_manifest.json"
    spec = _load_json(spec_path)
    source = _load_json(source_path)
    assets = _load_json(assets_path)
    if assets["source_step_sha256"] != source["source_step"]["sha256"]:
        raise RuntimeError("Robot assets are not bound to the selected STEP")
    before = _hash_protected(source)

    client = RemoteAPIClient(host=host, port=port)
    sim = client.require("sim")
    if sim.getSimulationState() != sim.simulation_stopped:
        sim.stopSimulation()
        client.step()
    template = (PROJECT_ROOT / spec["formal_scene_template"]).resolve()
    staged_template = stage_scene_for_coppeliasim(template)
    sim.loadScene(staged_template.as_posix())

    teaching_ready_pose = _apply_teaching_ready_pose(
        sim,
        visual_spec=spec["robot_visuals"],
    )
    robot_reports = _build_robot_visuals(
        sim,
        spec=spec,
        source=source,
    )
    root = _make_dummy(sim, "VisionLab")
    _build_workspace(sim, spec=spec, root=root)
    pickables, pickable_rows = _build_pickables(
        sim,
        spec=spec,
        root=root,
    )
    zones, zone_rows = _build_zones(sim, spec=spec, root=root)
    calibration, marker_rows = _build_calibration(
        sim,
        spec=spec,
        root=root,
    )
    camera_handle = _build_camera(
        sim,
        spec=spec,
        root=root,
    )
    camera_scope_script = _attach_camera_render_scope_script(
        sim,
        root=root,
    )

    for required_path in spec["required_paths"]:
        sim.getObject(required_path)
    camera_resolution = [
        int(sim.getObjectInt32Param(camera_handle, sim.visionintparam_resolution_x)),
        int(sim.getObjectInt32Param(camera_handle, sim.visionintparam_resolution_y)),
    ]
    if camera_resolution != spec["camera"]["resolution"]:
        raise RuntimeError(
            f"Camera resolution mismatch: {camera_resolution}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    sim.saveScene(output.resolve().as_posix())
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError(f"CoppeliaSim did not write the scene: {output}")

    after = _hash_protected(source)
    if after != before:
        raise RuntimeError("A protected formal BLX asset changed during scene build")
    scene_hash = _sha256(output)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "simulation.vision_lab.build_scene",
        "source_step_sha256": source["source_step"]["sha256"],
        "source_step_runtime_usage": (
            spec["robot_visuals"]["bl23_step_runtime_usage"]
        ),
        "source_archive_sha256": source["source_archive"]["sha256"],
        "reference_assembly": {
            "archive_filename": source["reference_assembly"][
                "source_archive"
            ]["filename"],
            "archive_sha256": source["reference_assembly"][
                "source_archive"
            ]["sha256"],
            "runtime_body": source["reference_assembly"]["runtime_body"],
            "mesh_equivalence": source["reference_assembly"][
                "mesh_equivalence"
            ],
            "link6_scale_correction": source["reference_assembly"][
                "link6_scale_correction"
            ],
        },
        "robot_visual_strategy": {
            "body": spec["robot_contract"]["visual_body"],
            "end_effector": spec["robot_contract"]["end_effector"],
            "bl23_step_runtime_usage": spec["robot_visuals"][
                "bl23_step_runtime_usage"
            ],
        },
        "teaching_ready_pose": teaching_ready_pose,
        "scene_spec_sha256": _sha256(spec_path),
        "robot_assets_manifest_sha256": _sha256(assets_path),
        "formal_scene_template_sha256": before[spec["formal_scene_template"]],
        "protected_assets_unchanged": True,
        "scene": {
            "path": output.relative_to(VISION_DIR).as_posix(),
            "sha256": scene_hash,
            "size_bytes": output.stat().st_size,
        },
        "contract": spec["robot_contract"],
        "camera": {
            "path": spec["camera"]["path"],
            "resolution": camera_resolution,
            "position_m": spec["camera"]["position_m"],
            "render_scope": "runtime_teaching_workspace_collection",
            "robot_visuals_renderable": False,
            "runtime_scope_script": sim.getObjectAlias(
                camera_scope_script,
                2,
            ),
            "render_members": [
                spec["workspace"]["path"],
                "/VisionLab/Pickables/*",
                "/VisionLab/Zones/*",
                "/VisionLab/Calibration/*",
            ],
        },
        "robot_visuals": robot_reports,
        "pickables": pickable_rows,
        "zones": zone_rows,
        "calibration_markers": marker_rows,
        "required_paths": spec["required_paths"],
    }
    _write_json(VISION_DIR / "scene_manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the BL23 visual teaching scene in CoppeliaSim",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    vision_root = VISION_DIR.resolve()
    if output != vision_root and vision_root not in output.parents:
        raise ValueError("Scene output must stay inside simulation/vision_lab")
    manifest = build_scene(
        output=output,
        host=args.host,
        port=args.port,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "scene": manifest["scene"],
                "robot_visuals": len(manifest["robot_visuals"]),
                "pickables": len(manifest["pickables"]),
                "protected_assets_unchanged": True,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
