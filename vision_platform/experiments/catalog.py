from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from vision_platform.experiments.models import (
    ExperimentAcceptance,
    ExperimentDefinition,
)


_EXPERIMENT_ID = re.compile(r"^[A-Z][A-Z0-9]*-[0-9]{2}$")
_REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "pack_id",
    "title",
    "version",
    "scene",
    "scene_manifest",
    "student_template",
    "guide",
    "capabilities",
    "workspace",
    "public_parameters",
    "acceptance",
    "hardware_status",
}
_CATALOG_FIELDS = {"schema_version", "experiments"}
_ACCEPTANCE_FIELDS = {"probe_kind", "automated_checks", "human_checks"}
_WORKSPACE_FIELDS = {"x_mm", "y_mm", "z_mm", "safe_z_mm"}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _exact_fields(payload: dict[str, Any], expected: set[str], field: str) -> None:
    missing = sorted(expected - payload.keys())
    extra = sorted(payload.keys() - expected)
    if missing or extra:
        raise ValueError(f"{field} fields mismatch: missing={missing}, extra={extra}")


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _inside(
    root: Path,
    raw: Any,
    field: str,
    *,
    base: Path | None = None,
    must_exist: bool = True,
) -> Path:
    text = _non_empty_string(raw, field)
    path = ((base or root) / text).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"{field} must stay inside project")
    if must_exist and not path.is_file():
        raise ValueError(f"{field} does not exist: {raw}")
    return path


def _strings(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a non-empty string list")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} contains duplicate values")
    return result


def _number(value: Any, field: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field} must be a finite number")
    return value


def _interval(value: Any, field: str) -> tuple[int | float, int | float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must be a two-number interval")
    lower = _number(value[0], field)
    upper = _number(value[1], field)
    if lower >= upper:
        raise ValueError(f"{field} lower bound must be less than upper bound")
    return lower, upper


def _workspace(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("workspace must be an object")
    _exact_fields(value, _WORKSPACE_FIELDS, "workspace")
    validated = dict(value)
    for field in ("x_mm", "y_mm", "z_mm"):
        validated[field] = _interval(value[field], f"workspace.{field}")
    safe_z = _number(value["safe_z_mm"], "workspace.safe_z_mm")
    z_min, z_max = validated["z_mm"]
    if not z_min <= safe_z <= z_max:
        raise ValueError("workspace.safe_z_mm must stay inside workspace.z_mm")
    validated["safe_z_mm"] = safe_z
    return validated


def _definition(path: Path, root: Path) -> ExperimentDefinition:
    payload = _load_json(path)
    _exact_fields(payload, _REQUIRED_FIELDS, "experiment")
    if payload["schema_version"] != 1:
        raise ValueError("experiment schema_version must be 1")
    experiment_id = _non_empty_string(payload["experiment_id"], "experiment_id")
    if not _EXPERIMENT_ID.fullmatch(experiment_id):
        raise ValueError(f"invalid experiment_id: {experiment_id}")
    if payload["hardware_status"] != "PENDING_HARDWARE":
        raise ValueError("hardware_status must remain PENDING_HARDWARE")
    capabilities = _strings(payload["capabilities"], "capabilities")
    acceptance_payload = payload["acceptance"]
    if not isinstance(acceptance_payload, dict):
        raise ValueError("acceptance must be an object")
    _exact_fields(acceptance_payload, _ACCEPTANCE_FIELDS, "acceptance")
    acceptance = ExperimentAcceptance(
        probe_kind=_non_empty_string(
            acceptance_payload["probe_kind"],
            "acceptance.probe_kind",
        ),
        automated_checks=_strings(
            acceptance_payload["automated_checks"],
            "acceptance.automated_checks",
        ),
        human_checks=_strings(
            acceptance_payload["human_checks"],
            "acceptance.human_checks",
        ),
    )
    workspace = _workspace(payload["workspace"])
    public_parameters = payload["public_parameters"]
    if not isinstance(public_parameters, dict):
        raise ValueError("public_parameters must be an object")
    return ExperimentDefinition(
        experiment_id=experiment_id,
        pack_id=_non_empty_string(payload["pack_id"], "pack_id"),
        title=_non_empty_string(payload["title"], "title"),
        version=_non_empty_string(payload["version"], "version"),
        scene=_inside(root, payload["scene"], "scene"),
        scene_manifest=_inside(
            root,
            payload["scene_manifest"],
            "scene_manifest",
        ),
        student_template=_inside(
            root,
            payload["student_template"],
            "student_template",
        ),
        guide=_inside(root, payload["guide"], "guide"),
        capabilities=capabilities,
        workspace=workspace,
        public_parameters=public_parameters,
        acceptance=acceptance,
        hardware_status="PENDING_HARDWARE",
    )


class ExperimentCatalog:
    def __init__(self, definitions: tuple[ExperimentDefinition, ...]) -> None:
        by_id = {item.experiment_id: item for item in definitions}
        if len(by_id) != len(definitions):
            raise ValueError("catalog contains duplicate experiment_id")
        self._definitions = definitions
        self._by_id = by_id

    @classmethod
    def load(cls, path: str | Path, *, project_root: str | Path) -> "ExperimentCatalog":
        root = Path(project_root).expanduser().resolve()
        selected_path = Path(path).expanduser().resolve()
        if selected_path != root and root not in selected_path.parents:
            raise ValueError("catalog must stay inside project")
        selected = selected_path
        payload = _load_json(selected)
        _exact_fields(payload, _CATALOG_FIELDS, "catalog")
        if payload.get("schema_version") != 1:
            raise ValueError("catalog schema_version must be 1")
        files = _strings(payload.get("experiments"), "experiments")
        definitions = tuple(
            _definition(
                _inside(root, name, "experiment", base=selected.parent),
                root,
            )
            for name in files
        )
        return cls(definitions)

    @property
    def definitions(self) -> tuple[ExperimentDefinition, ...]:
        return self._definitions

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(item.experiment_id for item in self._definitions)

    def require(self, experiment_id: str) -> ExperimentDefinition:
        try:
            return self._by_id[experiment_id]
        except KeyError as error:
            raise KeyError(f"Unknown experiment: {experiment_id}") from error
