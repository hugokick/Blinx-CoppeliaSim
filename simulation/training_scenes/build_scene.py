from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import threading
import time
import uuid
import warnings
from dataclasses import dataclass
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
TEMPLATE_RELATIVE = "simulation/vision_lab/BL23_vision_lab.ttt"
PROTECTED_ROOTS = (
    TEMPLATE_RELATIVE,
    "robot_backends/models/BLX_openr6.ttt",
    "robot_backends/models/openr6_arm_coppeliasim.urdf",
    "robot_backends/models/meshes_blx",
    "simulation/vision_lab/assets/robot",
)
DECLARED_PROTECTED_ROOTS = PROTECTED_ROOTS[1:4]
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.Lock] = {}

_BASICS_REQUIRED_PATHS = (
    "/BLX_base_link",
    "/BLX_joint1",
    "/BLX_joint2",
    "/BLX_joint3",
    "/BLX_joint4",
    "/BLX_joint5",
    "/BLX_joint6",
    "/BLX_tool_suction",
    "/RobotBasics",
    "/RobotBasics/Workspace",
    "/RobotBasics/Camera",
    "/RobotBasics/TeachPoints/TeachPoint1",
    "/RobotBasics/TeachPoints/TeachPoint2",
    "/RobotBasics/TeachPoints/TeachPoint3",
    "/RobotBasics/TeachPoints/TeachPoint4",
)
_LOGISTICS_REQUIRED_PATHS = (
    "/BLX_base_link",
    "/BLX_joint1",
    "/BLX_joint2",
    "/BLX_joint3",
    "/BLX_joint4",
    "/BLX_joint5",
    "/BLX_joint6",
    "/BLX_tool_suction",
    "/LogisticsLab",
    "/LogisticsLab/Workspace",
    "/LogisticsLab/Camera",
    "/LogisticsLab/Tasks/Stack",
    "/LogisticsLab/Tasks/Stack/Pickables",
    "/LogisticsLab/Tasks/Stack/Targets",
    "/LogisticsLab/Tasks/Digits",
    "/LogisticsLab/Tasks/Digits/Pickables",
    "/LogisticsLab/Tasks/Digits/Targets",
    "/LogisticsLab/Tasks/Classes",
    "/LogisticsLab/Tasks/Classes/Pickables",
    "/LogisticsLab/Tasks/Classes/Targets",
)


@dataclass(frozen=True)
class _FormalScene:
    spec_relative: str
    scene_id: str
    root_path: str
    output_relative: str
    required_paths: tuple[str, ...]


@dataclass(frozen=True)
class _RecoverableRelease:
    scene_sha256: str
    scene_size: int


class _BuildOutputLock:
    """Fail-fast thread/process lock for one normalized formal output."""

    def __init__(self, output: Path):
        normalized = Path(os.path.abspath(output)).resolve(strict=False)
        self._key = os.path.normcase(str(normalized))
        digest = hashlib.sha256(self._key.encode("utf-8")).hexdigest()
        self._path = (
            Path(tempfile.gettempdir())
            / "robot-sim-scene-build-locks"
            / f"{digest}.lock"
        )
        self._thread_lock: threading.Lock | None = None
        self._stream = None
        self._os_lock_acquired = False

    def acquire(self) -> None:
        with _THREAD_LOCKS_GUARD:
            thread_lock = _THREAD_LOCKS.setdefault(
                self._key,
                threading.Lock(),
            )
        if not thread_lock.acquire(blocking=False):
            raise RuntimeError("build already in progress")
        self._thread_lock = thread_lock
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self._path.open("a+b")
            self._stream.seek(0, os.SEEK_END)
            if self._stream.tell() == 0:
                self._stream.write(b"\0")
                self._stream.flush()
                os.fsync(self._stream.fileno())
            self._stream.seek(0)
            try:
                self._acquire_os_lock()
            except OSError as exc:
                raise RuntimeError("build already in progress") from exc
            self._os_lock_acquired = True
        except BaseException as acquire_error:
            acquire_cleanup_errors = []
            if self._stream is not None:
                try:
                    self._stream.close()
                except BaseException as exc:
                    acquire_cleanup_errors.append(
                        _cleanup_diagnostic(
                            "output lock acquire close failed",
                            exc,
                        )
                    )
                self._stream = None
            try:
                thread_lock.release()
            except BaseException as exc:
                acquire_cleanup_errors.append(
                    _cleanup_diagnostic(
                        "thread lock acquire rollback failed",
                        exc,
                    )
                )
            self._thread_lock = None
            _note_cleanup_errors(acquire_error, acquire_cleanup_errors)
            raise

    def _acquire_os_lock(self) -> None:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(
                self._stream.fileno(),
                msvcrt.LK_NBLCK,
                1,
            )
            return
        import fcntl

        fcntl.flock(
            self._stream.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )

    def _release_os_lock(self) -> None:
        if os.name == "nt":
            import msvcrt

            self._stream.seek(0)
            msvcrt.locking(
                self._stream.fileno(),
                msvcrt.LK_UNLCK,
                1,
            )
            return
        import fcntl

        fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)

    def release(self) -> list[str]:
        errors: list[str] = []
        if self._stream is not None:
            if self._os_lock_acquired:
                try:
                    self._release_os_lock()
                except BaseException as exc:
                    errors.append(
                        "output lock release failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                self._os_lock_acquired = False
            try:
                self._stream.close()
            except BaseException as exc:
                errors.append(
                    "output lock close failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            self._stream = None
        if self._thread_lock is not None:
            try:
                self._thread_lock.release()
            except BaseException as exc:
                errors.append(
                    "thread build lock release failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            self._thread_lock = None
        return errors


_FORMAL_SCENES = {
    "simulation/robot_basics/scene_spec.json": _FormalScene(
        spec_relative="simulation/robot_basics/scene_spec.json",
        scene_id="robot-basics",
        root_path="/RobotBasics",
        output_relative="simulation/robot_basics/BL23_robot_basics.ttt",
        required_paths=_BASICS_REQUIRED_PATHS,
    ),
    "simulation/logistics_lab/scene_spec.json": _FormalScene(
        spec_relative="simulation/logistics_lab/scene_spec.json",
        scene_id="logistics-lab",
        root_path="/LogisticsLab",
        output_relative="simulation/logistics_lab/BL23_logistics_lab.ttt",
        required_paths=_LOGISTICS_REQUIRED_PATHS,
    ),
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_reparse_or_symlink(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError as exc:
        raise ValueError(f"could not inspect path identity: {path}") from exc
    return bool(attributes & _REPARSE_POINT)


def _lexical_project_path(path: str | Path, *, label: str) -> Path:
    root = PROJECT_ROOT.expanduser().resolve(strict=True)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside PROJECT_ROOT") from exc
    return candidate


def _assert_no_alias(
    path: Path,
    *,
    label: str,
    must_exist: bool,
) -> None:
    root = PROJECT_ROOT.expanduser().resolve(strict=True)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside PROJECT_ROOT") from exc
    current = root
    for component in relative.parts:
        current = current / component
        if not os.path.lexists(current):
            if current == path and not must_exist:
                return
            raise ValueError(f"{label} does not exist: {current}")
        if _is_reparse_or_symlink(current):
            raise ValueError(
                f"{label} must not use a symlink, junction, or alias: {current}"
            )
    if path.is_file() and path.stat().st_nlink != 1:
        raise ValueError(f"{label} must not use a hardlink alias: {path}")


def _project_file_from_relative(
    raw: Any,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    if (
        not isinstance(raw, str)
        or not raw
        or raw != raw.strip()
        or Path(raw).is_absolute()
        or bool(Path(raw).anchor)
        or "\\" in raw
        or any(part in {".", ".."} for part in Path(raw).parts)
    ):
        raise ValueError(f"{label} must be a canonical project-relative path")
    path = _lexical_project_path(raw, label=label)
    _assert_no_alias(path.parent, label=f"{label} parent", must_exist=True)
    _assert_no_alias(path, label=label, must_exist=must_exist)
    resolved = path.resolve(strict=must_exist)
    root = PROJECT_ROOT.expanduser().resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} must stay inside PROJECT_ROOT")
    return path


def _exact_keys(payload: Any, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{label} must contain exactly {sorted(expected)}")
    return payload


def _vector(
    value: Any,
    *,
    label: str,
    length: int,
    positive: bool = False,
) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must be a {length}-item numeric list")
    result = []
    for item in value:
        if type(item) not in (int, float) or not math.isfinite(float(item)):
            raise ValueError(f"{label} must be a {length}-item numeric list")
        if positive and float(item) <= 0:
            raise ValueError(f"{label} values must be positive")
        result.append(float(item))
    return result


def _alias_value(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value) is None
    ):
        raise ValueError(f"{label} must be a safe object alias")
    return value


def _validate_workspace(payload: Any) -> None:
    workspace = _exact_keys(
        payload,
        {"alias", "size_mm", "center_mm"},
        label="workspace",
    )
    if workspace["alias"] != "Workspace":
        raise ValueError("workspace alias must be Workspace")
    _vector(workspace["size_mm"], label="workspace size_mm", length=3, positive=True)
    _vector(workspace["center_mm"], label="workspace center_mm", length=3)


def _validate_camera(payload: Any, *, root_path: str) -> None:
    camera = _exact_keys(
        payload,
        {
            "alias",
            "path",
            "resolution",
            "position_m",
            "near_clip_m",
            "far_clip_m",
            "perspective_angle_deg",
        },
        label="camera",
    )
    if camera["alias"] != "Camera" or camera["path"] != f"{root_path}/Camera":
        raise ValueError("camera alias/path must match the formal scene root")
    resolution = camera["resolution"]
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(type(item) is not int or item <= 0 for item in resolution)
    ):
        raise ValueError("camera resolution must be two positive integers")
    _vector(camera["position_m"], label="camera position_m", length=3)
    near = _vector(
        [camera["near_clip_m"]],
        label="camera near_clip_m",
        length=1,
        positive=True,
    )[0]
    far = _vector(
        [camera["far_clip_m"]],
        label="camera far_clip_m",
        length=1,
        positive=True,
    )[0]
    angle = _vector(
        [camera["perspective_angle_deg"]],
        label="camera perspective_angle_deg",
        length=1,
        positive=True,
    )[0]
    if near >= far or angle >= 180.0:
        raise ValueError("camera clipping range or perspective angle is invalid")


def _validate_markers(payload: Any) -> None:
    if not isinstance(payload, list) or len(payload) != 4:
        raise ValueError("markers must contain four teaching points")
    aliases = []
    for item in payload:
        marker = _exact_keys(item, {"alias", "position_mm"}, label="marker")
        aliases.append(_alias_value(marker["alias"], label="marker alias"))
        _vector(marker["position_mm"], label="marker position_mm", length=3)
    if aliases != [f"TeachPoint{index}" for index in range(1, 5)]:
        raise ValueError("markers must be TeachPoint1 through TeachPoint4")


def _validate_shape_item(item: Any, *, label: str) -> None:
    shape = _exact_keys(
        item,
        {"alias", "shape", "size_mm", "position_mm", "color"},
        label=label,
    )
    _alias_value(shape["alias"], label=f"{label} alias")
    if shape["shape"] not in {"cuboid", "cylinder"}:
        raise ValueError(f"{label} shape is unsupported")
    _vector(shape["size_mm"], label=f"{label} size_mm", length=3, positive=True)
    _vector(shape["position_mm"], label=f"{label} position_mm", length=3)
    color = _vector(shape["color"], label=f"{label} color", length=3)
    if any(component < 0.0 or component > 1.0 for component in color):
        raise ValueError(f"{label} color values must be between 0 and 1")


def _validate_target(item: Any, *, label: str) -> None:
    target = _exact_keys(item, {"alias", "position_mm"}, label=label)
    _alias_value(target["alias"], label=f"{label} alias")
    _vector(target["position_mm"], label=f"{label} position_mm", length=3)


def _validate_tasks(payload: Any) -> None:
    if not isinstance(payload, dict) or list(payload) != ["Stack", "Digits", "Classes"]:
        raise ValueError("tasks must be ordered Stack, Digits, Classes")
    expected_counts = {"Stack": (6, 6), "Digits": (3, 3), "Classes": (4, 4)}
    for name, (object_count, target_count) in expected_counts.items():
        task = _exact_keys(payload[name], {"objects", "targets"}, label=f"task {name}")
        objects = task["objects"]
        targets = task["targets"]
        if not isinstance(objects, list) or len(objects) != object_count:
            raise ValueError(f"task {name} objects count is invalid")
        if not isinstance(targets, list) or len(targets) != target_count:
            raise ValueError(f"task {name} targets count is invalid")
        object_aliases = []
        for item in objects:
            if name == "Digits":
                digit = _exact_keys(
                    item,
                    {"alias", "digit", "position_mm"},
                    label="digit object",
                )
                object_aliases.append(
                    _alias_value(digit["alias"], label="digit object alias")
                )
                if type(digit["digit"]) is not int or digit["digit"] not in (1, 2, 3):
                    raise ValueError("digit object digit must be integer 1, 2, or 3")
                _vector(
                    digit["position_mm"],
                    label="digit object position_mm",
                    length=3,
                )
            else:
                _validate_shape_item(item, label=f"{name} object")
                object_aliases.append(item["alias"])
        if len(object_aliases) != len(set(object_aliases)):
            raise ValueError(f"task {name} object aliases must be unique")
        target_aliases = []
        for item in targets:
            _validate_target(item, label=f"{name} target")
            target_aliases.append(item["alias"])
        if len(target_aliases) != len(set(target_aliases)):
            raise ValueError(f"task {name} target aliases must be unique")
    digits = [item["digit"] for item in payload["Digits"]["objects"]]
    if digits != [1, 2, 3]:
        raise ValueError("Digits objects must be ordered 1, 2, 3")
    classes = {item["alias"]: item for item in payload["Classes"]["objects"]}
    if set(classes) != {"red_block", "blue_block", "green_cylinder", "yellow_cylinder"}:
        raise ValueError("Classes objects must declare the four formal classes")
    if classes["yellow_cylinder"]["position_mm"][1] != -25:
        raise ValueError("yellow_cylinder y position must be -25 mm")
    class_items = list(classes.values())
    for index, first in enumerate(class_items):
        for second in class_items[index + 1 :]:
            separated = any(
                abs(
                    float(first["position_mm"][axis])
                    - float(second["position_mm"][axis])
                )
                >= (
                    float(first["size_mm"][axis])
                    + float(second["size_mm"][axis])
                )
                / 2.0
                for axis in (0, 1)
            )
            if not separated:
                raise ValueError("Classes object footprints must not overlap")


def _validate_spec(spec: dict[str, Any], formal: _FormalScene) -> None:
    detail_field = "markers" if formal.scene_id == "robot-basics" else "tasks"
    _exact_keys(
        spec,
        {
            "schema_version",
            "scene_id",
            "template",
            "output",
            "remove_paths",
            "root_path",
            "workspace",
            "camera",
            detail_field,
            "required_paths",
        },
        label="scene spec",
    )
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("scene schema_version must be integer 1")
    bindings = {
        "scene_id": formal.scene_id,
        "root_path": formal.root_path,
        "template": TEMPLATE_RELATIVE,
        "output": formal.output_relative,
    }
    for field, expected in bindings.items():
        if spec[field] != expected:
            raise ValueError(f"{field} must be {expected}")
    if spec["remove_paths"] != ["/VisionLab"]:
        raise ValueError("remove_paths must be exactly ['/VisionLab']")
    if spec["required_paths"] != list(formal.required_paths):
        raise ValueError("required_paths must match the formal scene contract")
    _validate_workspace(spec["workspace"])
    _validate_camera(spec["camera"], root_path=formal.root_path)
    if formal.scene_id == "robot-basics":
        _validate_markers(spec["markers"])
    else:
        _validate_tasks(spec["tasks"])


def _formal_spec_path(spec_path: str | Path) -> tuple[Path, _FormalScene]:
    candidate = _lexical_project_path(spec_path, label="spec_path")
    root = PROJECT_ROOT.expanduser().resolve(strict=True)
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("spec_path must stay inside PROJECT_ROOT") from exc
    formal = _FORMAL_SCENES.get(relative)
    if formal is None:
        raise ValueError("spec_path must be an approved formal spec")
    expected = root / formal.spec_relative
    if candidate != expected:
        raise ValueError("spec_path must be the canonical approved formal spec")
    _assert_no_alias(candidate, label="spec_path", must_exist=True)
    if candidate.resolve(strict=True) != expected:
        raise ValueError("spec_path must not use an alias")
    return candidate, formal


def _files_under(relative: str) -> list[Path]:
    path = _project_file_from_relative(
        relative,
        label=f"protected path {relative}",
        must_exist=True,
    )
    if path.is_file():
        return [path]
    items = sorted(path.rglob("*"))
    for item in items:
        _assert_no_alias(item, label=f"protected entry {item}", must_exist=True)
    files = [item for item in items if item.is_file()]
    if not files:
        raise ValueError(f"protected directory is empty: {relative}")
    return files


def _protected_hashes() -> dict[str, str]:
    expected_manifest = (
        PROJECT_ROOT.expanduser().resolve(strict=True)
        / "simulation/vision_lab/source_manifest.json"
    )
    manifest_path = _lexical_project_path(
        SOURCE_MANIFEST,
        label="SOURCE_MANIFEST",
    )
    if manifest_path != expected_manifest:
        raise ValueError("SOURCE_MANIFEST must use the formal project path")
    _assert_no_alias(
        manifest_path,
        label="SOURCE_MANIFEST",
        must_exist=True,
    )
    source = _load(manifest_path)
    if type(source.get("schema_version")) is not int or source["schema_version"] != 1:
        raise ValueError("SOURCE_MANIFEST schema_version must be integer 1")
    entries = source.get("protected_formal_assets")
    if not isinstance(entries, list) or not entries:
        raise ValueError("SOURCE_MANIFEST protected_formal_assets must be non-empty")

    expected_declared = {
        path.relative_to(PROJECT_ROOT.expanduser().resolve(strict=True)).as_posix()
        for relative in DECLARED_PROTECTED_ROOTS
        for path in _files_under(relative)
    }
    declared: set[str] = set()
    result: dict[str, str] = {}
    for item in entries:
        if not isinstance(item, dict) or set(item) not in (
            {"path", "sha256"},
            {"path", "sha256", "hash_mode"},
        ):
            raise ValueError("protected asset entry has invalid fields")
        raw = item["path"]
        path = _project_file_from_relative(
            raw,
            label="protected asset path",
            must_exist=True,
        )
        relative = path.relative_to(
            PROJECT_ROOT.expanduser().resolve(strict=True)
        ).as_posix()
        if relative not in expected_declared or relative in declared:
            raise ValueError(
                "protected asset entries must uniquely cover formal model assets"
            )
        declared.add(relative)
        expected_hash = item["sha256"]
        if (
            not isinstance(expected_hash, str)
            or _LOWER_SHA256.fullmatch(expected_hash) is None
        ):
            raise ValueError("protected asset sha256 must be lowercase hexadecimal")
        mode = item.get("hash_mode", "bytes")
        if mode not in {"bytes", "text_lf"}:
            raise ValueError("protected asset hash_mode is unsupported")
        actual = asset_sha256(path, mode=mode)
        if actual != expected_hash:
            raise RuntimeError(f"protected asset hash mismatch: {relative}")
        result[f"declared:{relative}"] = actual
    if declared != expected_declared:
        raise ValueError(
            "SOURCE_MANIFEST must hash every protected formal model asset"
        )

    root = PROJECT_ROOT.expanduser().resolve(strict=True)
    for relative in PROTECTED_ROOTS:
        for path in _files_under(relative):
            item_relative = path.relative_to(root).as_posix()
            result[f"snapshot:{item_relative}"] = _sha256(path)
    return result


def _alias(sim: Any, handle: int, name: str) -> int:
    sim.setObjectAlias(handle, name)
    return int(handle)


def _dummy(sim: Any, name: str, parent: int | None = None) -> int:
    handle = _alias(sim, int(sim.createDummy(0.005)), name)
    if parent is not None:
        sim.setObjectParent(handle, parent, False)
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
    parts = []
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
    # CoppeliaSim uses the final grouped shape as the compound reference.
    parts.append(
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


def _attach_logistics_camera_scope(sim: Any, root: int) -> int:
    script_text = """
function sysCall_init()
    trainingCollection = sim.createCollection(1)
    sim.addItemToCollection(
        trainingCollection,
        sim.handle_tree,
        sim.getObject('/LogisticsLab'),
        0
    )
    local camera = sim.getObject('/LogisticsLab/Camera')
    sim.setObjectInt32Param(
        camera,
        sim.visionintparam_entity_to_render,
        trainingCollection
    )
end

function sysCall_cleanup()
    if trainingCollection then
        sim.destroyCollection(trainingCollection)
        trainingCollection = nil
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
    _alias(sim, handle, "CameraRenderScope")
    sim.setObjectParent(handle, root, True)
    return handle


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


def _recoverable_release(
    output: Path,
    manifest_path: Path,
    formal: _FormalScene,
    template_hash: str,
) -> _RecoverableRelease | None:
    if not output.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = _load(manifest_path)
        template = manifest.get("template")
        scene = manifest.get("scene")
        if (
            type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1
            or manifest.get("scene_id") != formal.scene_id
            or manifest.get("protected_assets_unchanged") is not True
            or manifest.get("required_paths")
            != list(formal.required_paths)
            or not isinstance(template, dict)
            or template.get("path") != TEMPLATE_RELATIVE
            or template.get("sha256") != template_hash
            or not isinstance(scene, dict)
            or scene.get("path") != formal.output_relative
        ):
            return None
        sha256 = scene.get("sha256")
        size = scene.get("size_bytes")
        if (
            not isinstance(sha256, str)
            or _LOWER_SHA256.fullmatch(sha256) is None
            or type(size) is not int
            or size <= 0
        ):
            return None
        release = _RecoverableRelease(
            scene_sha256=sha256,
            scene_size=size,
        )
        if not _release_still_matches(output, release):
            return None
        return release
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _release_still_matches(
    output: Path,
    release: _RecoverableRelease,
) -> bool:
    try:
        return (
            output.is_file()
            and output.stat().st_size == release.scene_size
            and _sha256(output) == release.scene_sha256
        )
    except OSError:
        return False


def _cleanup_diagnostic(action: str, exc: BaseException) -> str:
    return f"{action}: {type(exc).__name__}: {exc}"


def _note_cleanup_errors(
    primary: BaseException,
    cleanup_errors: list[str],
) -> None:
    add_note = getattr(primary, "add_note", None)
    if not callable(add_note):
        return
    for diagnostic in cleanup_errors:
        add_note(f"cleanup error: {diagnostic}")


def build_scene(
    *,
    spec_path: Path,
    host: str,
    port: int,
) -> dict[str, Any]:
    spec_path, formal = _formal_spec_path(spec_path)
    spec = _load(spec_path)
    _validate_spec(spec, formal)
    if not isinstance(host, str) or not host.strip() or host != host.strip():
        raise ValueError("host must be a non-empty string")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be an integer from 1 to 65535")

    template = _project_file_from_relative(
        TEMPLATE_RELATIVE,
        label="template",
        must_exist=True,
    )
    output_candidate = _lexical_project_path(
        formal.output_relative,
        label="output",
    )
    output = _project_file_from_relative(
        formal.output_relative,
        label="output",
        must_exist=os.path.lexists(output_candidate),
    )
    manifest_path = spec_path.parent / "scene_manifest.json"
    _assert_no_alias(
        manifest_path.parent,
        label="manifest parent",
        must_exist=True,
    )
    _assert_no_alias(
        manifest_path,
        label="manifest",
        must_exist=os.path.lexists(manifest_path),
    )
    if output.exists() and output.samefile(template):
        raise ValueError("output must not be a template or protected alias")

    build_lock = _BuildOutputLock(output)
    build_lock.acquire()
    client = None
    temporaries: list[Path] = []
    cleanup_errors: list[str] = []
    primary_error: BaseException | None = None
    primary_traceback = None
    result: dict[str, Any] | None = None
    try:
        before = _protected_hashes()
        template_hash = _sha256(template)
        token = uuid.uuid4().hex
        staged_scene = output.parent / (
            f".{output.stem}.staged-{token}.ttt"
        )
        staged_manifest = output.parent / (
            f".scene_manifest.staged-{token}.json"
        )
        backup_manifest = output.parent / (
            f".scene_manifest.backup-{token}.json"
        )
        candidates = [staged_scene, staged_manifest, backup_manifest]
        if any(os.path.lexists(path) for path in candidates):
            raise RuntimeError("unique scene staging path already exists")
        temporaries.extend(candidates)

        client = RemoteAPIClient(host=host, port=port)
        sim = client.require("sim")
        _stop_simulation(sim)
        sim.loadScene(stage_scene_for_coppeliasim(template).as_posix())
        for path in spec["remove_paths"]:
            handle = int(sim.getObject(path))
            sim.removeObjects([handle], False)
        root_name = formal.root_path.rsplit("/", 1)[-1]
        root = _dummy(sim, root_name)
        _build_workspace(sim, spec["workspace"], root)
        _camera(sim, spec["camera"], root)
        if formal.scene_id == "robot-basics":
            _build_basics(sim, spec, root)
        else:
            _build_logistics(sim, spec, root)
            _attach_logistics_camera_scope(sim, root)
        for path in formal.required_paths:
            sim.getObject(path)

        sim.saveScene(staged_scene.as_posix())
        if not staged_scene.is_file() or staged_scene.stat().st_size == 0:
            raise RuntimeError(
                f"CoppeliaSim did not write staged scene: {staged_scene}"
            )
        _assert_no_alias(
            staged_scene,
            label="staged scene",
            must_exist=True,
        )
        scene_hash = _sha256(staged_scene)
        scene_size = staged_scene.stat().st_size
        after = _protected_hashes()
        if after != before or _sha256(template) != template_hash:
            raise RuntimeError("protected assets changed during scene build")

        manifest = {
            "schema_version": 1,
            "scene_id": formal.scene_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "simulation.training_scenes.build_scene",
            "template": {
                "path": TEMPLATE_RELATIVE,
                "sha256": template_hash,
            },
            "scene": {
                "path": formal.output_relative,
                "sha256": scene_hash,
                "size_bytes": scene_size,
            },
            "required_paths": list(formal.required_paths),
            "task_contracts": spec.get("tasks", {}),
            "protected_assets_unchanged": True,
        }
        _write_exclusive(staged_manifest, manifest)

        old_release = _recoverable_release(
            output,
            manifest_path,
            formal,
            template_hash,
        )
        if os.path.lexists(manifest_path):
            os.replace(manifest_path, backup_manifest)
        try:
            os.replace(staged_scene, output)
        except BaseException as exc:
            if (
                old_release is not None
                and os.path.lexists(backup_manifest)
                and _release_still_matches(output, old_release)
            ):
                try:
                    os.replace(backup_manifest, manifest_path)
                except BaseException as restore_exc:
                    cleanup_errors.append(
                        _cleanup_diagnostic(
                            "old manifest restore failed",
                            restore_exc,
                        )
                    )
            raise
        if (
            not output.is_file()
            or output.stat().st_size != scene_size
            or _sha256(output) != scene_hash
        ):
            raise RuntimeError("published training scene verification failed")
        if _protected_hashes() != before:
            raise RuntimeError("protected assets changed during scene publish")
        os.replace(staged_manifest, manifest_path)
        result = manifest
    except BaseException as exc:
        primary_error = exc
        primary_traceback = exc.__traceback__
    finally:
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as exc:
                    cleanup_errors.append(
                        _cleanup_diagnostic(
                            "client.close cleanup failed",
                            exc,
                        )
                    )
        for temporary in temporaries:
            try:
                if os.path.lexists(temporary):
                    temporary.unlink()
            except BaseException as exc:
                cleanup_errors.append(
                    _cleanup_diagnostic(
                        f"temporary cleanup failed for {temporary.name}",
                        exc,
                    )
                )
        cleanup_errors.extend(build_lock.release())

    if primary_error is not None:
        _note_cleanup_errors(primary_error, cleanup_errors)
        raise primary_error.with_traceback(primary_traceback)
    if result is None:
        raise RuntimeError("scene build ended without a result")
    if cleanup_errors:
        response = dict(result)
        response["cleanup_errors"] = list(cleanup_errors)
        warning_message = (
            "scene build committed with cleanup_errors: "
            + "; ".join(cleanup_errors)
        )
        try:
            warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
        except Warning:
            pass
        return response
    return result


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
