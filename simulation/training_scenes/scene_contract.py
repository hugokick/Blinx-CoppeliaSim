from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_one(payload: dict[str, Any]) -> None:
    value = payload.get("schema_version")
    if type(value) is not int or value != 1:
        raise ValueError("scene schema_version must be integer 1")


def _scene_id(payload: dict[str, Any]) -> str:
    value = payload.get("scene_id")
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
    ):
        raise ValueError("scene_id must be a non-empty string")
    return value


def _project_relative_path(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
    ):
        raise ValueError(
            f"{label} must be a non-empty project-relative path"
        )
    path = Path(value)
    if path.is_absolute() or bool(path.anchor):
        raise ValueError(
            f"{label} must be a non-empty project-relative path"
        )
    return value


def _manifest_file(
    manifest: dict[str, Any],
    field: str,
) -> tuple[str, str]:
    payload = manifest.get(field)
    if not isinstance(payload, dict):
        raise ValueError(f"manifest {field} must be an object")
    path = _project_relative_path(
        payload.get("path"),
        label=f"manifest {field} path",
    )
    sha256 = payload.get("sha256")
    if not isinstance(sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", sha256
    ) is None:
        raise ValueError(
            f"manifest {field} sha256 must be 64 lowercase hexadecimal "
            "characters"
        )
    return path, sha256


def _required_paths(payload: dict[str, Any], *, label: str) -> list[str]:
    paths = payload.get("required_paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"{label} required_paths must be a non-empty list")
    if any(
        not isinstance(path, str)
        or path != path.strip()
        or not path.startswith("/")
        for path in paths
    ):
        raise ValueError(
            f"{label} required_paths must contain absolute object paths"
        )
    if len(paths) != len(set(paths)):
        raise ValueError(f"{label} required_paths must be unique")
    return paths


def _inside(root: Path, raw: str) -> Path:
    path = (root / raw).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"path must stay inside project: {raw}")
    if not path.is_file():
        raise ValueError(f"file does not exist: {raw}")
    return path


def _same_file(first: Path, second: Path) -> bool:
    try:
        return first.samefile(second)
    except OSError as exc:
        raise ValueError(
            "could not verify training scene file identity"
        ) from exc


def validate_scene_contract(
    spec_path: str | Path,
    manifest_path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    spec = _load(Path(spec_path).expanduser().resolve())
    manifest = _load(Path(manifest_path).expanduser().resolve())
    _schema_one(spec)
    _schema_one(manifest)
    spec_scene_id = _scene_id(spec)
    manifest_scene_id = _scene_id(manifest)
    if spec_scene_id != manifest_scene_id:
        raise ValueError("scene_id mismatch")
    spec_template = _project_relative_path(
        spec.get("template"),
        label="spec template",
    )
    spec_output = _project_relative_path(
        spec.get("output"),
        label="spec output",
    )
    if spec_template == spec_output:
        raise ValueError("training scene output must not overwrite template")
    manifest_template, template_sha256 = _manifest_file(
        manifest,
        "template",
    )
    manifest_scene, scene_sha256 = _manifest_file(manifest, "scene")
    if manifest_template != spec_template:
        raise ValueError("template path mismatch")
    if manifest_scene != spec_output:
        raise ValueError("scene path mismatch")
    spec_required_paths = _required_paths(spec, label="spec")
    manifest_required_paths = _required_paths(manifest, label="manifest")
    if spec_required_paths != manifest_required_paths:
        raise ValueError("required_paths mismatch")
    if manifest.get("protected_assets_unchanged") is not True:
        raise ValueError("protected assets were not verified")
    template = _inside(root, manifest_template)
    scene = _inside(root, manifest_scene)
    if _same_file(template, scene):
        raise ValueError("training scene output must not overwrite template")
    if _sha256(template) != template_sha256:
        raise ValueError("template sha256 mismatch")
    if _sha256(scene) != scene_sha256:
        raise ValueError("scene sha256 mismatch")
    report = {
        "status": "PASS",
        "scene_id": spec_scene_id,
        "scene_sha256": scene_sha256,
        "required_path_count": len(spec_required_paths),
    }
    if spec_scene_id == "vision-quality-lab":
        profile_path = _project_relative_path(
            spec.get("profiles"),
            label="spec profiles",
        )
        if profile_path != "simulation/vision_quality_lab/profiles.json":
            raise ValueError("spec profiles must be the canonical vision-quality catalog")
        manifest_profile_path, profile_sha256 = _manifest_file(
            manifest,
            "profile_catalog",
        )
        expected_profile = {"path": profile_path, "sha256": _sha256(_inside(root, profile_path))}
        if manifest.get("profile_catalog") != expected_profile:
            raise ValueError("profile_catalog must match the formal profile catalog")
        if manifest_profile_path != profile_path:
            raise ValueError("profile_catalog path mismatch")
        if _sha256(_inside(root, manifest_profile_path)) != profile_sha256:
            raise ValueError("profile_catalog sha256 mismatch")
        report["profile_sha256"] = profile_sha256
    return report
