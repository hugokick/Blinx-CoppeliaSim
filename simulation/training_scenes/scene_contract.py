from __future__ import annotations

import hashlib
import json
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


def _inside(root: Path, raw: str) -> Path:
    path = (root / raw).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"path must stay inside project: {raw}")
    if not path.is_file():
        raise ValueError(f"file does not exist: {raw}")
    return path


def validate_scene_contract(
    spec_path: str | Path,
    manifest_path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    spec = _load(Path(spec_path).expanduser().resolve())
    manifest = _load(Path(manifest_path).expanduser().resolve())
    if spec.get("schema_version") != 1 or manifest.get("schema_version") != 1:
        raise ValueError("scene schema_version must be 1")
    if spec["scene_id"] != manifest["scene_id"]:
        raise ValueError("scene_id mismatch")
    if spec["template"] == spec["output"]:
        raise ValueError("training scene output must not overwrite template")
    if manifest["template"]["path"] != spec["template"]:
        raise ValueError("template path mismatch")
    if manifest["scene"]["path"] != spec["output"]:
        raise ValueError("scene path mismatch")
    template = _inside(root, manifest["template"]["path"])
    scene = _inside(root, manifest["scene"]["path"])
    if _sha256(template) != manifest["template"]["sha256"]:
        raise ValueError("template sha256 mismatch")
    if _sha256(scene) != manifest["scene"]["sha256"]:
        raise ValueError("scene sha256 mismatch")
    if manifest.get("protected_assets_unchanged") is not True:
        raise ValueError("protected assets were not verified")
    if spec["required_paths"] != manifest["required_paths"]:
        raise ValueError("required_paths mismatch")
    return {
        "status": "PASS",
        "scene_id": spec["scene_id"],
        "scene_sha256": manifest["scene"]["sha256"],
        "required_path_count": len(spec["required_paths"]),
    }
