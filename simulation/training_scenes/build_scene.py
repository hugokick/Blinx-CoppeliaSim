from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from simulation.vision_lab.hashing import asset_sha256
from vision_platform.coppelia_scene import stage_scene_for_coppeliasim


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = (
    PROJECT_ROOT / "simulation" / "vision_lab" / "source_manifest.json"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_hashes() -> dict[str, str]:
    source = _load(SOURCE_MANIFEST)
    result = {}
    for item in source["protected_formal_assets"]:
        path = PROJECT_ROOT / item["path"]
        actual = asset_sha256(
            path,
            mode=item.get("hash_mode", "bytes"),
        )
        if actual.lower() != item["sha256"].lower():
            raise RuntimeError(
                f"protected asset hash mismatch: {item['path']}"
            )
        result[item["path"]] = actual
    return result


def _alias(sim: Any, handle: int, name: str) -> int:
    sim.setObjectAlias(handle, name)
    return int(handle)


def _dummy(sim: Any, name: str, parent: int | None = None) -> int:
    handle = _alias(sim, int(sim.createDummy(0.005)), name)
    if parent is not None:
        sim.setObjectParent(handle, parent, True)
    return handle


def _set_int_parameter(
    sim: Any,
    handle: int,
    parameter: int,
    value: int,
) -> None:
    setter = getattr(sim, "setObjectInt32Param", None)
    if not callable(setter):
        setter = getattr(sim, "setObjectInt32Parameter")
    setter(handle, parameter, int(value))


def _shape(
    sim: Any,
    *,
    name: str,
    shape: str,
    size_mm: list[float],
    position_mm: list[float],
    color: list[float],
    parent: int,
    respondable: bool,
) -> int:
    primitive = (
        sim.primitiveshape_cylinder
        if shape == "cylinder"
        else sim.primitiveshape_cuboid
    )
    handle = _alias(
        sim,
        int(
            sim.createPrimitiveShape(
                primitive,
                [float(value) / 1000.0 for value in size_mm],
                0,
            )
        ),
        name,
    )
    sim.setShapeColor(
        handle,
        "",
        sim.colorcomponent_ambient_diffuse,
        [float(value) for value in color],
    )
    _set_int_parameter(sim, handle, sim.shapeintparam_static, 1)
    _set_int_parameter(
        sim,
        handle,
        sim.shapeintparam_respondable,
        int(respondable),
    )
    sim.setObjectParent(handle, parent, False)
    sim.setObjectPosition(
        handle,
        [float(value) / 1000.0 for value in position_mm],
        parent,
    )
    return handle


def _camera(sim: Any, spec: dict[str, Any], parent: int) -> int:
    resolution = [int(value) for value in spec["resolution"]]
    options = 2 | 4 | 64 | 128
    handle = int(
        sim.createVisionSensor(
            options,
            [resolution[0], resolution[1], 0, 0],
            [
                float(spec["near_clip_m"]),
                float(spec["far_clip_m"]),
                math.radians(float(spec["perspective_angle_deg"])),
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
    _alias(sim, handle, spec["alias"])
    sim.setObjectPose(
        handle,
        [
            *[float(value) for value in spec["position_m"]],
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        sim.handle_world,
    )
    sim.setObjectParent(handle, parent, True)
    return handle


_SEGMENTS = {
    1: ("b", "c"),
    2: ("a", "b", "g", "e", "d"),
    3: ("a", "b", "g", "c", "d"),
}
_SEGMENT_POSES = {
    "a": ([0, 5, 10], [10, 2, 2]),
    "b": ([5, 0, 10], [2, 10, 2]),
    "c": ([5, -10, 10], [2, 10, 2]),
    "d": ([0, -15, 10], [10, 2, 2]),
    "e": ([-5, -10, 10], [2, 10, 2]),
    "g": ([0, -5, 10], [10, 2, 2]),
}


def _digit(sim: Any, item: dict[str, Any], parent: int) -> int:
    center = [float(value) for value in item["position_mm"]]
    parts = [
        _shape(
            sim,
            name=f"{item['alias']}_plate",
            shape="cuboid",
            size_mm=[20, 30, 8],
            position_mm=center,
            color=[0.92, 0.92, 0.92],
            parent=parent,
            respondable=True,
        )
    ]
    for segment in _SEGMENTS[int(item["digit"])]:
        offset, size = _SEGMENT_POSES[segment]
        parts.append(
            _shape(
                sim,
                name=f"{item['alias']}_{segment}",
                shape="cuboid",
                size_mm=size,
                position_mm=[
                    center[0] + offset[0],
                    center[1] + offset[1],
                    center[2] + offset[2],
                ],
                color=[0.04, 0.04, 0.04],
                parent=parent,
                respondable=False,
            )
        )
    compound = int(sim.groupShapes(parts, False))
    _alias(sim, compound, item["alias"])
    sim.setObjectParent(compound, parent, True)
    return compound


def _build_workspace(sim: Any, spec: dict[str, Any], root: int) -> None:
    _shape(
        sim,
        name=spec["alias"],
        shape="cuboid",
        size_mm=spec["size_mm"],
        position_mm=spec["center_mm"],
        color=[0.30, 0.32, 0.34],
        parent=root,
        respondable=True,
    )


def _build_basics(sim: Any, spec: dict[str, Any], root: int) -> None:
    points = _dummy(sim, "TeachPoints", root)
    for marker in spec["markers"]:
        _shape(
            sim,
            name=marker["alias"],
            shape="cylinder",
            size_mm=[8, 8, 2],
            position_mm=marker["position_mm"],
            color=[0.12, 0.75, 0.85],
            parent=points,
            respondable=False,
        )


def _build_logistics(sim: Any, spec: dict[str, Any], root: int) -> None:
    tasks = _dummy(sim, "Tasks", root)
    for index, (name, task) in enumerate(spec["tasks"].items()):
        group = _dummy(sim, name, tasks)
        sim.setObjectPosition(
            group,
            [0.0, 0.0, 0.0 if index == 0 else -float(index + 2)],
            sim.handle_world,
        )
        pickables = _dummy(sim, "Pickables", group)
        targets = _dummy(sim, "Targets", group)
        for item in task["objects"]:
            if "digit" in item:
                _digit(sim, item, pickables)
            else:
                _shape(
                    sim,
                    name=item["alias"],
                    shape=item["shape"],
                    size_mm=item["size_mm"],
                    position_mm=item["position_mm"],
                    color=item["color"],
                    parent=pickables,
                    respondable=True,
                )
        for target in task["targets"]:
            handle = _shape(
                sim,
                name=target["alias"],
                shape="cuboid",
                size_mm=[24, 24, 2],
                position_mm=target["position_mm"],
                color=[0.35, 0.75, 0.95],
                parent=targets,
                respondable=False,
            )
            transparency = getattr(
                sim,
                "colorcomponent_transparency",
                None,
            )
            if transparency is not None:
                sim.setShapeColor(handle, "", transparency, [0.70])


def _stop_simulation(sim: Any) -> None:
    if int(sim.getSimulationState()) == int(sim.simulation_stopped):
        return
    sim.stopSimulation()
    deadline = time.monotonic() + 10.0
    while (
        int(sim.getSimulationState()) != int(sim.simulation_stopped)
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    if int(sim.getSimulationState()) != int(sim.simulation_stopped):
        raise RuntimeError("CoppeliaSim did not stop before scene build")


def build_scene(
    *,
    spec_path: Path,
    host: str,
    port: int,
) -> dict[str, Any]:
    spec_path = spec_path.expanduser().resolve()
    spec = _load(spec_path)
    if spec.get("schema_version") != 1:
        raise ValueError("scene schema_version must be 1")
    template = (PROJECT_ROOT / spec["template"]).resolve()
    output = (PROJECT_ROOT / spec["output"]).resolve()
    if output == template:
        raise ValueError("training scene output must not overwrite template")
    if PROJECT_ROOT not in output.parents:
        raise ValueError("training scene output must stay inside project")
    before = _protected_hashes()
    template_hash = _sha256(template)
    client = RemoteAPIClient(host=host, port=port)
    sim = client.require("sim")
    _stop_simulation(sim)
    sim.loadScene(stage_scene_for_coppeliasim(template).as_posix())
    for path in spec["remove_paths"]:
        handle = int(sim.getObject(path))
        sim.removeObjects([handle], False)
    root_name = spec["root_path"].rsplit("/", 1)[-1]
    root = _dummy(sim, root_name)
    _build_workspace(sim, spec["workspace"], root)
    _camera(sim, spec["camera"], root)
    if spec["scene_id"] == "robot-basics":
        _build_basics(sim, spec, root)
    elif spec["scene_id"] == "logistics-lab":
        _build_logistics(sim, spec, root)
    else:
        raise ValueError(f"unsupported scene_id: {spec['scene_id']}")
    for path in spec["required_paths"]:
        sim.getObject(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    sim.saveScene(output.as_posix())
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"CoppeliaSim did not write {output}")
    after = _protected_hashes()
    if after != before or _sha256(template) != template_hash:
        raise RuntimeError("protected assets changed during scene build")
    manifest = {
        "schema_version": 1,
        "scene_id": spec["scene_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "simulation.training_scenes.build_scene",
        "template": {
            "path": Path(spec["template"]).as_posix(),
            "sha256": template_hash,
        },
        "scene": {
            "path": Path(spec["output"]).as_posix(),
            "sha256": _sha256(output),
            "size_bytes": output.stat().st_size,
        },
        "required_paths": spec["required_paths"],
        "task_contracts": spec.get("tasks", {}),
        "protected_assets_unchanged": True,
    }
    _write(spec_path.parent / "scene_manifest.json", manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    args = parser.parse_args(argv)
    manifest = build_scene(
        spec_path=args.spec,
        host=args.host,
        port=args.port,
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
