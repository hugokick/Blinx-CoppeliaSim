from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

import simulation.training_scenes.build_scene as scene_builder
from tools.vision_lab import build_v1_08_scene


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "simulation" / "vision_ocr_sorting_lab" / "scene_spec.json"
CONFIG = ROOT / "config" / "experiments" / "V1-08.json"


def test_python_wrapper_passes_only_canonical_v1_08_spec_and_port(monkeypatch):
    calls = []

    def fake_build_scene(**kwargs):
        calls.append(kwargs)
        return {"status": "PENDING_COPPELIASIM"}

    monkeypatch.setattr(build_v1_08_scene, "build_scene", fake_build_scene)
    assert build_v1_08_scene.main() == 0
    assert calls == [
        {
            "spec_path": SPEC,
            "host": "127.0.0.1",
            "port": 23008,
        }
    ]


def test_python_wrapper_rejects_port_override(monkeypatch):
    source = (ROOT / "tools" / "vision_lab" / "build_v1_08_scene.py").read_text(
        encoding="utf-8"
    )
    assert "23008" in source
    assert "sys.argv" not in source
    assert "--port" not in source


def test_powershell_wrapper_composes_owned_launch_and_cleanup_contract():
    path = ROOT / "tools" / "vision_lab" / "build_v1_08_scene.ps1"
    source = path.read_text(encoding="utf-8")
    assert "launch_coppeliasim.ps1" in source
    assert "process_ownership.ps1" in source
    assert "build_v1_08_scene.py" in source
    assert "23008" in source
    assert "Stop-ExactOwnedProcess" in source
    assert "finally" in source
    assert "-Code" not in source
    assert "Invoke-Expression" not in source
    assert "-Output" not in source


def test_v1_08_spec_is_strictly_validated_before_remote_connection(monkeypatch):
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    formal = scene_builder._FORMAL_SCENES[
        "simulation/vision_ocr_sorting_lab/scene_spec.json"
    ]
    scene_builder._validate_spec(spec, formal)

    class ForbiddenClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("remote connection must follow spec validation")

    monkeypatch.setattr(scene_builder, "RemoteAPIClient", ForbiddenClient)
    with pytest.raises(AssertionError, match="remote connection"):
        scene_builder.build_scene(spec_path=SPEC, host="127.0.0.1", port=23008)


def test_existing_v1_08_release_recovery_does_not_require_code_assets_manifest():
    manifest_path = SPEC.parent / "scene_manifest.json"
    scene_path = SPEC.parent / "BL23_vision_ocr_sorting_lab.ttt"
    template_path = ROOT / "simulation" / "vision_lab" / "BL23_vision_lab.ttt"
    profile_path = SPEC.parent / "profiles.json"
    assets_path = SPEC.parent / "ocr_assets_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    formal = scene_builder._FORMAL_SCENES[
        "simulation/vision_ocr_sorting_lab/scene_spec.json"
    ]
    release = scene_builder._recoverable_release(
        scene_path,
        manifest_path,
        formal,
        hashlib.sha256(template_path.read_bytes()).hexdigest(),
        (
            "simulation/vision_ocr_sorting_lab/profiles.json",
            profile_path,
            hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        ),
        None,
        (
            "simulation/vision_ocr_sorting_lab/ocr_assets_manifest.json",
            assets_path,
            hashlib.sha256(assets_path.read_bytes()).hexdigest(),
        ),
    )
    assert release is not None
    assert release.scene_sha256 == manifest["scene"]["sha256"]


def test_v1_08_bitmap_cells_are_contiguous_for_connected_strokes():
    """The rendered 10x7 label must not break adjacent bitmap pixels apart."""

    geometry = scene_builder._ocr_bitmap_geometry(face_width=30.0, face_height=30.0)

    assert geometry["cell_width_mm"] >= geometry["pitch_width_mm"]
    assert geometry["cell_height_mm"] >= geometry["pitch_height_mm"]


def test_v1_08_bitmap_footprint_fits_the_fixed_roi():
    geometry = scene_builder._ocr_bitmap_geometry(face_width=30.0, face_height=30.0)

    assert 11 * geometry["cell_width_mm"] <= 18.0
    assert 7 * geometry["cell_height_mm"] <= 18.0


def test_v1_08_bitmap_cells_keep_the_training_scale():
    geometry = scene_builder._ocr_bitmap_geometry(face_width=30.0, face_height=30.0)

    assert 1.0 <= geometry["pitch_width_mm"] <= 1.15
    # The training generator uses square bitmap pixels.  A vertically
    # stretched scene glyph changes the normalized 20x20 feature shape and
    # lowers the raw KNN confidence before the published 0.90 gate.
    assert geometry["pitch_height_mm"] == pytest.approx(geometry["pitch_width_mm"])
    # Training variants include the deterministic one-pixel dilation case;
    # scene blocks must preserve that stroke thickness after rasterisation.
    assert geometry["cell_width_mm"] >= 1.15 * geometry["pitch_width_mm"]


def test_v1_08_bitmap_keeps_a_gap_between_identifier_glyphs():
    geometry = scene_builder._ocr_bitmap_geometry(face_width=30.0, face_height=30.0)

    assert geometry["glyph_gap_pitch_mm"] >= geometry["pitch_width_mm"]


def test_v1_08_layout_is_expanded_away_from_robot_base():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    positions = [tuple(part["position_mm"]) for part in spec["parts"]]
    slots = [
        tuple(slot["position_mm"])
        for route in spec["routes"]
        for slot in route["slots"]
    ]

    assert positions == [
        (35, -55, 18),
        (75, -55, 18),
        (35, 25, 18),
        (75, 25, 18),
    ]
    assert slots == [
        (116, -75, 22),
        (128, -75, 22),
        (116, 75, 22),
        (128, 75, 22),
    ]
    assert min(abs(y) for _, y, _ in positions) >= 25
    assert min(abs(y) for _, y, _ in slots) >= 70


def test_v1_08_expanded_layout_keeps_geometry_and_rois_inside_contract():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    workspace = spec["workspace"]
    workspace_x = (
        float(workspace["center_mm"][0]) - float(workspace["size_mm"][0]) / 2.0,
        float(workspace["center_mm"][0]) + float(workspace["size_mm"][0]) / 2.0,
    )
    workspace_y = (
        float(workspace["center_mm"][1]) - float(workspace["size_mm"][1]) / 2.0,
        float(workspace["center_mm"][1]) + float(workspace["size_mm"][1]) / 2.0,
    )
    for part in spec["parts"]:
        x, y, _z = (float(value) for value in part["position_mm"])
        width, height, _depth = (float(value) for value in part["size_mm"])
        assert workspace_x[0] <= x - width / 2.0
        assert x + width / 2.0 <= workspace_x[1]
        assert workspace_y[0] <= y - height / 2.0
        assert y + height / 2.0 <= workspace_y[1]
    for route in spec["routes"]:
        for slot in route["slots"]:
            x, y, z = (float(value) for value in slot["position_mm"])
            assert workspace_x[0] <= x <= workspace_x[1]
            assert workspace_y[0] <= y <= workspace_y[1]
            assert 10.0 <= z <= 140.0

    rois = config["public_parameters"]["ocr_sorting"]["fixed_rois_px"]
    assert rois == {
        "A1": [760, 120, 110, 120],
        "A2": [515, 120, 110, 120],
        "B1": [760, 610, 110, 120],
        "B2": [515, 610, 110, 120],
    }
    for x, y, width, height in rois.values():
        assert x >= 24 and y >= 24
        assert x + width + 24 <= 1024
        assert y + height + 24 <= 1024


def test_v1_08_ocr_part_uses_contiguous_cell_size(monkeypatch):
    calls = []

    def fake_shape(_sim, **kwargs):
        calls.append(kwargs)
        return len(calls)

    class FakeSim:
        shapeintparam_static = 1
        shapeintparam_respondable = 2

        def groupShapes(self, pieces, _merge):
            return 99

        def setObjectParent(self, *_args):
            return None

    monkeypatch.setattr(scene_builder, "_shape", fake_shape)
    monkeypatch.setattr(scene_builder, "_alias", lambda _sim, handle, _name: handle)
    monkeypatch.setattr(scene_builder, "_set_int_parameter", lambda *_args: None)
    monkeypatch.setattr(scene_builder, "_dummy", lambda *_args: 100)
    # Two adjacent pixels force the generated cuboids to meet in both axes.
    np = __import__("numpy")
    label = np.full((96, 64), 255, dtype="uint8")
    for row, column in ((0, 0), (0, 1), (1, 0)):
        label[30 + row * 5 : 35 + row * 5, 3 + column * 5 : 8 + column * 5] = 0
    monkeypatch.setattr(scene_builder.cv2, "imread", lambda *_args: label)
    scene_builder._ocr_part(
        FakeSim(),
        {"alias": "part_a", "position_mm": [40, -45, 18], "size_mm": [34, 34, 16]},
        Path("label.png"),
        100,
    )

    glyphs = calls[2:]
    assert len(glyphs) == 3
    assert glyphs[0]["size_mm"][0] >= glyphs[1]["position_mm"][0] - glyphs[0]["position_mm"][0]
    assert glyphs[0]["size_mm"][1] >= glyphs[2]["position_mm"][1] - glyphs[0]["position_mm"][1]
    assert glyphs[0]["position_mm"][0] > glyphs[1]["position_mm"][0]


def test_v1_08_label_bitmap_preserves_the_two_source_glyphs():
    label = scene_builder.cv2.imread(
        str(SPEC.parent / "labels" / "A1.png"), scene_builder.cv2.IMREAD_GRAYSCALE
    )

    bitmap = scene_builder._ocr_label_bitmap(label)

    assert [
        "".join("#" if value < 160 else "." for value in row) for row in bitmap
    ] == [
        ".###...#..",
        "#...#.##..",
        "#...#..#..",
        "#####..#..",
        "#...#..#..",
        "#...#..#..",
        "#...#.###.",
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("baseline_profile_id", "not_standard"),
        ("profile.perspective_angle_deg", float("inf")),
        ("profile.resolution", [1024.0, 1024]),
        ("profile.key_diffuse_rgb", [1.5, 0.8, 0.8]),
        ("profile.camera_rig_z_m", "0.5"),
    ],
)
def test_ocr_profile_catalog_rejects_invalid_fixed_profile_values(
    tmp_path, field, value
):
    payload = json.loads(
        (ROOT / "simulation" / "vision_ocr_sorting_lab" / "profiles.json").read_text(
            encoding="utf-8"
        )
    )
    if field.startswith("profile."):
        payload["profiles"][0][field.split(".", 1)[1]] = value
    else:
        payload[field] = value
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    with pytest.raises(ValueError):
        scene_builder._load_ocr_profile_catalog(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.__setitem__("near_clip_m", 10**1000),
        lambda payload: payload["profiles"][0].__setitem__(
            "perspective_angle_deg", 10**1000
        ),
        lambda payload: payload["profiles"][0].__setitem__(
            "key_diffuse_rgb", [10**1000, 0.8, 0.8]
        ),
    ],
)
def test_ocr_profile_catalog_rejects_unbounded_integer_values(tmp_path, mutate):
    payload = json.loads(
        (ROOT / "simulation" / "vision_ocr_sorting_lab" / "profiles.json").read_text(
            encoding="utf-8"
        )
    )
    mutate(payload)
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        scene_builder._load_ocr_profile_catalog(path)


@pytest.mark.parametrize("tamper", ["hash", "path", "size_type"])
def test_ocr_assets_manifest_rejects_tampered_label_binding(tmp_path, tamper):
    payload = json.loads(
        (ROOT / "simulation" / "vision_ocr_sorting_lab" / "ocr_assets_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if tamper == "hash":
        payload["labels"][0]["sha256"] = "0" * 64
    elif tamper == "path":
        payload["labels"][0]["path"] = "../labels/A1.png"
    else:
        payload["labels"][0]["size_px"] = [64.0, 96]
    path = tmp_path / "ocr_assets_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        scene_builder._validate_ocr_assets_manifest(path)
