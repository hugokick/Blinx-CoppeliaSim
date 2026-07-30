from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from simulation.vision_lab.hashing import asset_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VISION_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = VISION_DIR / "assets" / "robot"
DEFAULT_ARCHIVE = Path(
    "H:/智能机械与机器人基础/实训课/六轴机器臂模型/"
    "LC-YT1119-BL23-A000(机械臂总组装）-ZQSZXYx-V1.1.2-SJ.zip"
)
DEFAULT_PROVENANCE = VISION_DIR / "provenance"
DEFAULT_VOLUMES = DEFAULT_PROVENANCE / "step_volumes_2026-06-08.json"
DEFAULT_MAPPING = DEFAULT_PROVENANCE / "segment_volume_mapping_2026-06-08.json"


def _sha256(path: Path) -> str:
    return asset_sha256(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_hash(
    path: Path,
    expected: str,
    label: str,
    *,
    hash_mode: str = "bytes",
) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = asset_sha256(path, mode=hash_mode)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected}, got {actual} ({path})"
        )
    return actual


def _protected_hashes(source: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in source["protected_formal_assets"]:
        path = PROJECT_ROOT / entry["path"]
        actual = _assert_hash(
            path,
            entry["sha256"],
            "Protected formal asset",
            hash_mode=entry.get("hash_mode", "bytes"),
        )
        result[entry["path"]] = actual
    return result


def _select_step_entry(entries: Iterable[Any], source: dict[str, Any]) -> Any:
    entries = list(entries)
    expected_name = source["source_step"]["archive_entry"]
    expected_size = int(source["source_step"]["size_bytes"])
    exact = [
        entry
        for entry in entries
        if not entry.is_dir()
        and entry.filename == expected_name
        and int(entry.file_size) == expected_size
    ]
    if len(exact) == 1:
        return exact[0]
    candidates = [
        entry
        for entry in entries
        if not entry.is_dir()
        and str(entry.filename).upper().endswith((".STEP", ".STP"))
        and int(entry.file_size) == expected_size
    ]
    if len(candidates) != 1:
        names = [str(entry.filename) for entry in entries if not entry.is_dir()]
        raise RuntimeError(
            (
                "Selected archive must contain exactly one pinned-size STEP "
                f"entry; found {len(candidates)} candidates in {names!r}"
            )
        )
    return candidates[0]


def _extract_step(archive: Path, source: dict[str, Any], staging: Path) -> Path:
    with zipfile.ZipFile(archive) as package:
        info = _select_step_entry(package.infolist(), source)
        if info.file_size != source["source_step"]["size_bytes"]:
            raise RuntimeError(
                f"STEP entry size changed: {info.file_size} bytes"
            )
        staging.parent.mkdir(parents=True, exist_ok=True)
        with package.open(info) as input_stream, staging.open("wb") as output:
            shutil.copyfileobj(input_stream, output, 1024 * 1024)
    _assert_hash(staging, source["source_step"]["sha256"], "Extracted STEP")
    return staging


def _bbox(shape: Any) -> tuple[list[float], list[float]]:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    minimum_x, minimum_y, minimum_z, maximum_x, maximum_y, maximum_z = (
        box.Get()
    )
    return (
        [float(minimum_x), float(minimum_y), float(minimum_z)],
        [float(maximum_x), float(maximum_y), float(maximum_z)],
    )


def _read_solids(step_path: Path) -> list[Any]:
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"OCP could not read STEP (status={status})")
    transferred = reader.TransferRoots()
    if transferred <= 0:
        raise RuntimeError("OCP did not transfer any STEP roots")
    explorer = TopExp_Explorer(reader.OneShape(), TopAbs_SOLID)
    solids: list[Any] = []
    while explorer.More():
        solids.append(explorer.Current())
        explorer.Next()
    return solids


def _validate_solid_order(
    solids: list[Any],
    volume_report: dict[str, Any],
    *,
    tolerance_mm: float = 0.02,
) -> float:
    records = volume_report["volumes"]
    if len(solids) != volume_report["total_volumes"] or len(solids) != len(
        records
    ):
        raise RuntimeError(
            f"STEP solid count mismatch: OCP={len(solids)}, report={len(records)}"
        )
    maximum_error = 0.0
    for index, (solid, record) in enumerate(zip(solids, records), start=1):
        if record["tag"] != index:
            raise RuntimeError(
                f"Volume report tag order changed at {index}: {record['tag']}"
            )
        actual_min, actual_max = _bbox(solid)
        expected = [*record["bbox_min"], *record["bbox_max"]]
        actual = [*actual_min, *actual_max]
        error = max(
            abs(float(left) - float(right))
            for left, right in zip(actual, expected)
        )
        maximum_error = max(maximum_error, error)
        if error > tolerance_mm:
            raise RuntimeError(
                (
                    f"OCP solid order does not match volume tag {index}: "
                    f"bbox error={error:.6f} mm"
                )
            )
    return maximum_error


def _mapping_tags(mapping: dict[str, Any]) -> set[int]:
    tags: set[int] = set()
    for segment in mapping["segments"].values():
        for key in (
            "required_tags",
            "recommended_tags",
            "uncertain_tags",
            "exclude_tags",
        ):
            tags.update(int(tag) for tag in segment.get(key, []))
    return tags


def _semantic_partition_tags(source: dict[str, Any]) -> list[int]:
    return [
        int(tag)
        for segment in source["segments"]
        for tag in source["semantic_partition_141"][segment]
    ]


def _build_compound(solids: list[Any], tags: Iterable[int]) -> Any:
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for tag in tags:
        builder.Add(compound, solids[int(tag) - 1])
    return compound


def _export_segment(
    solids: list[Any],
    tags: list[int],
    output: Path,
    *,
    mesh_deflection_mm: float,
) -> None:
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer

    compound = _build_compound(solids, tags)
    BRepMesh_IncrementalMesh(
        compound,
        mesh_deflection_mm,
        True,
        mesh_deflection_mm * 0.5,
        True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    writer = StlAPI_Writer()
    writer.ASCIIMode = False
    if not writer.Write(compound, str(output)):
        raise RuntimeError(f"OCP failed to write {output}")


def _mesh_audit(path: Path) -> dict[str, Any]:
    import numpy as np
    import trimesh

    loaded = trimesh.load(path, force="mesh", process=True)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_mesh()
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise RuntimeError(f"Generated STL contains no triangles: {path}")
    bounds = np.asarray(loaded.bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise RuntimeError(f"Generated STL has invalid bounds: {path}")
    return {
        "triangle_count": int(len(loaded.faces)),
        "vertex_count": int(len(loaded.vertices)),
        "bounds_mm": bounds.tolist(),
        "extents_mm": np.asarray(loaded.extents, dtype=float).tolist(),
        "watertight": bool(loaded.is_watertight),
        "winding_consistent": bool(loaded.is_winding_consistent),
        "graph_component_audit": "omitted_no_optional_graph_dependency",
    }


def import_assets(
    *,
    archive: Path,
    volumes_path: Path,
    mapping_path: Path,
    output_dir: Path,
    mesh_deflection_mm: float,
) -> dict[str, Any]:
    source = _load_json(VISION_DIR / "source_manifest.json")
    before = _protected_hashes(source)
    _assert_hash(archive, source["source_archive"]["sha256"], "Selected archive")
    _assert_hash(
        volumes_path,
        source["provenance_snapshots"]["step_volumes"]["sha256"],
        "141-volume report",
    )
    _assert_hash(
        mapping_path,
        source["provenance_snapshots"]["segment_mapping"]["sha256"],
        "Segment mapping",
    )
    volume_report = _load_json(volumes_path)
    mapping = _load_json(mapping_path)
    if int(volume_report["total_volumes"]) != 141:
        raise RuntimeError("Volume report no longer contains 141 entities")
    if _mapping_tags(mapping) != set(range(1, 142)):
        raise RuntimeError("Segment mapping no longer accounts for all 141 tags")
    semantic_partition = _semantic_partition_tags(source)
    if (
        sorted(semantic_partition) != list(range(1, 142))
        or len(semantic_partition) != len(set(semantic_partition))
    ):
        raise RuntimeError(
            "Semantic 141-entity partition must account for each tag once"
        )

    provenance_dir = VISION_DIR / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        volumes_path,
        provenance_dir / "step_volumes_2026-06-08.json",
    )
    shutil.copyfile(
        mapping_path,
        provenance_dir / "segment_volume_mapping_2026-06-08.json",
    )

    staging = (
        PROJECT_ROOT
        / "artifacts"
        / "vision_lab"
        / "cad-staging"
        / "BL23-source.step"
    )
    _extract_step(archive, source, staging)
    started = time.perf_counter()
    solids = _read_solids(staging)
    bbox_error = _validate_solid_order(solids, volume_report)
    records_by_tag = {
        int(record["tag"]): record for record in volume_report["volumes"]
    }

    assets: dict[str, Any] = {}
    for segment in source["segments"]:
        visual_source = source["visual_segment_sources"][segment]
        if visual_source["kind"] != "ocp_entity_tags":
            raise RuntimeError(
                f"Only current STEP OCP entity exports are allowed: {segment}"
            )
        if (
            visual_source["source_step_sha256"].lower()
            != source["source_step"]["sha256"].lower()
        ):
            raise RuntimeError(
                f"Visual source {segment} is bound to a different STEP"
            )
        output = output_dir / f"{segment}.STL"
        tags = [int(tag) for tag in visual_source["tags"]]
        _export_segment(
            solids,
            tags,
            output,
            mesh_deflection_mm=mesh_deflection_mm,
        )
        source_details = {
            "source_kind": "vendor_step_ocp_entity_export",
            "source_tags": tags,
            "source_solid_count": len(tags),
            "source_names": [
                records_by_tag[tag].get("short_name", "") for tag in tags
            ],
        }
        audit = _mesh_audit(output)
        watertight = audit["watertight"]
        anchor_entry = None
        anchor_tag = visual_source.get("alignment_anchor_tag")
        if anchor_tag is not None:
            anchor_tag = int(anchor_tag)
            if anchor_tag not in tags:
                raise RuntimeError(
                    f"Alignment anchor {anchor_tag} is not in {segment}"
                )
            anchor_path = output_dir / "anchors" / f"{segment}_anchor.STL"
            _export_segment(
                solids,
                [anchor_tag],
                anchor_path,
                mesh_deflection_mm=mesh_deflection_mm,
            )
            anchor_audit = _mesh_audit(anchor_path)
            anchor_entry = {
                "path": anchor_path.relative_to(VISION_DIR).as_posix(),
                "sha256": _sha256(anchor_path),
                "size_bytes": anchor_path.stat().st_size,
                "source_tag": anchor_tag,
                "source_name": records_by_tag[anchor_tag].get(
                    "short_name",
                    "",
                ),
                **anchor_audit,
            }
        assets[segment] = {
            "path": output.relative_to(VISION_DIR).as_posix(),
            "sha256": _sha256(output),
            "size_bytes": output.stat().st_size,
            "source_step_sha256": source["source_step"]["sha256"],
            "source_unit": "millimeter",
            "mesh_deflection_mm": mesh_deflection_mm,
            "coppeliasim_import_scale": 0.001,
            "usage": "collision_and_visual" if watertight else "visual_only",
            "collision_strategy": "mesh" if watertight else "simplified_bbox",
            "alignment_anchor": anchor_entry,
            "alignment_group": visual_source.get("alignment_group"),
            **source_details,
            **audit,
        }
        print(
            (
                f"[asset] {segment}: tags={len(tags)} "
                f"triangles={audit['triangle_count']} "
                f"watertight={watertight}"
            ),
            flush=True,
        )

    after = _protected_hashes(source)
    if after != before:
        raise RuntimeError("A protected formal BLX asset changed during import")
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "simulation.vision_lab.import_robot_assets",
        "source_step_sha256": source["source_step"]["sha256"],
        "source_archive_sha256": source["source_archive"]["sha256"],
        "source_solid_count": len(solids),
        "solid_bbox_verification_max_error_mm": bbox_error,
        "mesh_deflection_mm": mesh_deflection_mm,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "protected_assets_unchanged": True,
        "assets": assets,
    }
    _write_json(VISION_DIR / "robot_assets_manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export eight audited BL23 segment meshes from the pinned STEP",
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--volumes", type=Path, default=DEFAULT_VOLUMES)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mesh-deflection-mm", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mesh_deflection_mm <= 0:
        raise ValueError("--mesh-deflection-mm must be positive")
    output = args.output_dir.expanduser().resolve()
    vision_root = VISION_DIR.resolve()
    if vision_root not in output.parents and output != vision_root:
        raise ValueError("Output directory must stay inside simulation/vision_lab")
    manifest = import_assets(
        archive=args.archive.expanduser().resolve(),
        volumes_path=args.volumes.expanduser().resolve(),
        mapping_path=args.mapping.expanduser().resolve(),
        output_dir=output,
        mesh_deflection_mm=float(args.mesh_deflection_mm),
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "assets": len(manifest["assets"]),
                "source_step_sha256": manifest["source_step_sha256"],
                "protected_assets_unchanged": True,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
