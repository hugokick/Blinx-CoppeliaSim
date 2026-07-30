import hashlib
import json
from pathlib import Path

import numpy as np

from simulation.vision_lab.hashing import asset_sha256


ROOT = Path(__file__).resolve().parents[2]
VISION_SCENE = ROOT / "simulation" / "vision_lab"
EXPECTED_STEP_SHA256 = (
    "f3f493626792ef25b99e6b1e79f791da96f2526cee7161c90194443e9c34a9a2"
)
SEGMENTS = [
    "base",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "tool",
]


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_source_manifest_pins_step_and_separates_semantic_audit_from_visuals():
    manifest = _json(VISION_SCENE / "source_manifest.json")

    assert manifest["source_step"]["sha256"] == EXPECTED_STEP_SHA256
    assert manifest["segments"] == SEGMENTS
    assert manifest["source_step"]["solid_count"] == 141
    flattened = [
        tag
        for segment in SEGMENTS
        for tag in manifest["semantic_partition_141"][segment]
    ]
    assert sorted(flattened) == list(range(1, 142))
    assert len(flattened) == len(set(flattened))
    visual_sources = manifest["visual_segment_sources"]
    assert list(visual_sources) == SEGMENTS
    assert all(
        visual_sources[segment]["kind"] == "ocp_entity_tags"
        for segment in SEGMENTS
    )
    assert all(
        visual_sources[segment]["source_step_sha256"] == EXPECTED_STEP_SHA256
        for segment in SEGMENTS
    )
    visual_tags = [
        tag for segment in SEGMENTS for tag in visual_sources[segment]["tags"]
    ]
    assert len(visual_tags) == len(set(visual_tags))
    assert set(visual_tags) < set(range(1, 142))
    assert manifest["source_of_truth"] == [
        "vendor_step",
        "step_volumes_2026-06-08.json",
        "segment_volume_mapping_2026-06-08.json",
        "ocp_entity_export",
    ]


def test_provenance_snapshots_match_manifest_hashes():
    manifest = _json(VISION_SCENE / "source_manifest.json")

    for name, entry in manifest["provenance_snapshots"].items():
        path = VISION_SCENE / entry["path"]
        assert path.is_file(), name
        assert _sha256(path) == entry["sha256"], name


def test_each_step_derived_segment_has_audited_geometry_and_hash():
    source = _json(VISION_SCENE / "source_manifest.json")
    assets = _json(VISION_SCENE / "robot_assets_manifest.json")

    assert assets["source_step_sha256"] == EXPECTED_STEP_SHA256
    assert list(assets["assets"]) == SEGMENTS
    for segment in SEGMENTS:
        entry = assets["assets"][segment]
        path = VISION_SCENE / entry["path"]
        assert path.is_file(), segment
        assert _sha256(path) == entry["sha256"], segment
        assert entry["triangle_count"] > 0
        bounds = np.asarray(entry["bounds_mm"], dtype=float)
        assert bounds.shape == (2, 3)
        assert np.isfinite(bounds).all()
        assert (bounds[1] > bounds[0]).all()
        visual_source = source["visual_segment_sources"][segment]
        assert entry["source_step_sha256"] == EXPECTED_STEP_SHA256
        assert entry["source_tags"] == visual_source["tags"]
        assert entry["source_kind"] == "vendor_step_ocp_entity_export"
        assert entry["source_unit"] == "millimeter"
        assert entry["coppeliasim_import_scale"] == 0.001
        anchor = entry.get("alignment_anchor")
        if "alignment_anchor_tag" in visual_source:
            assert anchor["source_tag"] == visual_source["alignment_anchor_tag"]
            anchor_path = VISION_SCENE / anchor["path"]
            assert anchor_path.is_file()
            assert _sha256(anchor_path) == anchor["sha256"]
        if not entry["watertight"]:
            assert entry["usage"] == "visual_only"
            assert entry["collision_strategy"] == "simplified_bbox"


def test_formal_blx_assets_still_match_recorded_baseline():
    manifest = _json(VISION_SCENE / "source_manifest.json")

    for entry in manifest["protected_formal_assets"]:
        path = ROOT / entry["path"]
        assert path.is_file()
        assert (
            asset_sha256(path, mode=entry.get("hash_mode", "bytes"))
            == entry["sha256"]
        )


def test_text_lf_hash_mode_ignores_checkout_line_endings(tmp_path):
    lf_path = tmp_path / "model-lf.urdf"
    crlf_path = tmp_path / "model-crlf.urdf"
    lf_path.write_bytes(b"<robot>\n<link/>\n</robot>\n")
    crlf_path.write_bytes(b"<robot>\r\n<link/>\r\n</robot>\r\n")

    assert asset_sha256(lf_path, mode="text_lf") == asset_sha256(
        crlf_path,
        mode="text_lf",
    )
    assert asset_sha256(lf_path) != asset_sha256(crlf_path)


def test_zip_entry_selection_survives_legacy_chinese_filename_encoding():
    from types import SimpleNamespace

    from simulation.vision_lab.import_robot_assets import _select_step_entry

    source = _json(VISION_SCENE / "source_manifest.json")
    entries = [
        SimpleNamespace(
            filename="LC-YT1119-BL23-A000(��е������װ��-ZQSZXYx-V1.1.2-SJ.STEP",
            file_size=source["source_step"]["size_bytes"],
            is_dir=lambda: False,
        )
    ]

    selected = _select_step_entry(entries, source)

    assert selected is entries[0]


def test_mesh_audit_does_not_require_optional_graph_engine(tmp_path, monkeypatch):
    import trimesh

    from simulation.vision_lab.import_robot_assets import _mesh_audit

    path = tmp_path / "box.STL"
    trimesh.creation.box().export(path)

    def unavailable(*_args, **_kwargs):
        raise ImportError("no graph engines available")

    monkeypatch.setattr(trimesh.Trimesh, "split", unavailable)

    audit = _mesh_audit(path)

    assert audit["triangle_count"] == 12
    assert audit["watertight"] is True
