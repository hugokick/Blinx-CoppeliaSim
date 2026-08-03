"""Generate the original deterministic surface-defect assets for V1-09.

The seven images are made from a small, repository-owned geometric subject.
No installed fonts, external images, network data, timestamps, or machine
paths participate in the render.  The committed manifest binds every PNG to
the exact bytes produced by this generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_DIR = PROJECT_ROOT / "simulation" / "vision_defect_sorting_lab"
DEFAULT_OUTPUT = SCENE_DIR
MANIFEST_NAME = "defect_assets_manifest.json"
GENERATOR = "tools.vision_lab.generate_v1_09_defect_assets"
GENERATOR_VERSION = "1.0.0"
SEED = 20260803
IMAGE_SIZE = (256, 256)  # width, height
SCENE_ID = "V1-09"
SOURCE = "project-original-generated"

EXPECTED_ASSETS: dict[str, str] = {
    "reference": "assets/reference.png",
    "entry_a": "assets/candidate_a.png",
    "entry_b": "assets/candidate_b.png",
    "entry_c": "assets/candidate_c.png",
    "entry_d": "assets/candidate_d.png",
    "entry_e": "assets/candidate_e.png",
    "entry_f": "assets/candidate_f.png",
}

PURPOSES = {
    "reference": "fixed reference surface",
    "entry_a": "qualified candidate surface",
    "entry_b": "edge-removal candidate surface",
    "entry_c": "enclosed-void candidate surface",
    "entry_d": "separated-component candidate surface",
    "entry_e": "split-subject candidate surface",
    "entry_f": "principal-dimension candidate surface",
}


def _png_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(
        ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )
    if not ok:
        raise RuntimeError("V1-09 PNG encoding failed")
    return bytes(encoded)


def _geometry() -> dict[str, int]:
    """Return one deterministic subject geometry selected from the fixed seed."""

    rng = np.random.default_rng(SEED)
    width = int(rng.integers(138, 144))
    height = int(rng.integers(144, 152))
    background = int(rng.integers(232, 241))
    foreground = int(rng.integers(40, 57))
    hole_radius = int(rng.integers(16, 20))
    missing_width = int(rng.integers(8, 11))
    missing_height = int(rng.integers(28, 34))
    split_gap = int(rng.integers(8, 13))
    foreign_radius = int(rng.integers(9, 12))
    left = (IMAGE_SIZE[0] - width) // 2
    top = (IMAGE_SIZE[1] - height) // 2
    return {
        "left": left,
        "top": top,
        "right": left + width - 1,
        "bottom": top + height - 1,
        "center_x": left + width // 2,
        "center_y": top + height // 2,
        "background": background,
        "foreground": foreground,
        "hole_radius": hole_radius,
        "missing_width": missing_width,
        "missing_height": missing_height,
        "split_gap": split_gap,
        "foreign_radius": foreign_radius,
    }


def _subject_image(geometry: dict[str, int]) -> np.ndarray:
    width, height = IMAGE_SIZE
    background = geometry["background"]
    foreground = geometry["foreground"]
    image = np.full((height, width, 3), background, dtype=np.uint8)
    cv2.rectangle(
        image,
        (geometry["left"], geometry["top"]),
        (geometry["right"], geometry["bottom"]),
        (foreground, foreground, foreground),
        thickness=-1,
    )
    return image


def _asset_images() -> dict[str, np.ndarray]:
    geometry = _geometry()
    background = geometry["background"]
    foreground = geometry["foreground"]
    reference = _subject_image(geometry)
    images = {
        "reference": reference,
        "entry_a": reference.copy(),
    }

    missing = reference.copy()
    missing_y = geometry["center_y"] - geometry["missing_height"] // 2
    cv2.rectangle(
        missing,
        (geometry["left"], missing_y),
        (
            geometry["left"] + geometry["missing_width"] - 1,
            missing_y + geometry["missing_height"] - 1,
        ),
        (background, background, background),
        thickness=-1,
    )
    images["entry_b"] = missing

    hole = reference.copy()
    cv2.circle(
        hole,
        (geometry["center_x"], geometry["center_y"]),
        geometry["hole_radius"],
        (background, background, background),
        thickness=-1,
    )
    images["entry_c"] = hole

    foreign = reference.copy()
    foreign_center = (
        IMAGE_SIZE[0] - geometry["foreign_radius"] - 14,
        IMAGE_SIZE[1] - geometry["foreign_radius"] - 14,
    )
    cv2.circle(
        foreign,
        foreign_center,
        geometry["foreign_radius"],
        (foreground, foreground, foreground),
        thickness=-1,
    )
    images["entry_d"] = foreign

    broken = reference.copy()
    split_top = geometry["center_y"] - geometry["split_gap"] // 2
    cv2.rectangle(
        broken,
        (geometry["left"], split_top),
        (geometry["right"], split_top + geometry["split_gap"] - 1),
        (background, background, background),
        thickness=-1,
    )
    images["entry_e"] = broken

    dimension = np.full(
        (IMAGE_SIZE[1], IMAGE_SIZE[0], 3), background, dtype=np.uint8
    )
    enlarged_width = int(round((geometry["right"] - geometry["left"] + 1) * 1.22))
    enlarged_height = geometry["bottom"] - geometry["top"] + 1
    enlarged_left = (IMAGE_SIZE[0] - enlarged_width) // 2
    enlarged_top = (IMAGE_SIZE[1] - enlarged_height) // 2
    cv2.rectangle(
        dimension,
        (enlarged_left, enlarged_top),
        (enlarged_left + enlarged_width - 1, enlarged_top + enlarged_height - 1),
        (foreground, foreground, foreground),
        thickness=-1,
    )
    images["entry_f"] = dimension
    return images


def _asset_record(asset_id: str, image: np.ndarray) -> tuple[dict[str, object], bytes]:
    payload = _png_bytes(image)
    return (
        {
            "asset_id": asset_id,
            "generator": GENERATOR,
            "generator_version": GENERATOR_VERSION,
            "path": EXPECTED_ASSETS[asset_id],
            "purpose": PURPOSES[asset_id],
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_px": [int(image.shape[1]), int(image.shape[0])],
            "source": SOURCE,
        },
        payload,
    )


def _expected_output() -> tuple[dict[str, object], dict[str, bytes]]:
    images = _asset_images()
    records: list[dict[str, object]] = []
    files: dict[str, bytes] = {}
    for asset_id in EXPECTED_ASSETS:
        record, payload = _asset_record(asset_id, images[asset_id])
        records.append(record)
        files[str(record["path"])] = payload
    manifest: dict[str, object] = {
        "assets": records,
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "schema_version": 1,
        "scene_id": SCENE_ID,
        "seed": SEED,
        "source": SOURCE,
    }
    return manifest, files


def _manifest_bytes(manifest: dict[str, object]) -> bytes:
    return (
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _asset_paths(output_dir: Path, files: dict[str, bytes]) -> list[Path]:
    return [output_dir / relative for relative in files]


def _assert_no_extra_assets(output_dir: Path, expected_files: dict[str, bytes]) -> None:
    assets_dir = output_dir / "assets"
    if not assets_dir.is_dir() or assets_dir.is_symlink():
        raise RuntimeError("V1-09 assets directory is missing or not a directory")
    expected_names = {Path(relative).name for relative in expected_files}
    actual_names = {path.name for path in assets_dir.iterdir()}
    if actual_names != expected_names:
        raise RuntimeError(
            f"V1-09 asset set drift: expected={sorted(expected_names)} actual={sorted(actual_names)}"
        )


def _check(
    output_dir: Path,
    manifest_path: Path,
    expected_manifest: bytes,
    expected_files: dict[str, bytes],
) -> Path:
    _assert_no_extra_assets(output_dir, expected_files)
    for relative, expected in expected_files.items():
        path = output_dir / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"V1-09 asset missing: {relative}")
        if path.read_bytes() != expected:
            raise RuntimeError(f"V1-09 asset drift: {relative}")
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError("V1-09 manifest missing")
    if manifest_path.read_bytes() != expected_manifest:
        raise RuntimeError("V1-09 manifest drift")
    return manifest_path


def generate(
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    manifest_path: Path | None = None,
    check: bool = False,
) -> Path:
    """Generate or byte-check the V1-09 asset directory."""

    output_dir = Path(output_dir)
    manifest_path = (
        output_dir / MANIFEST_NAME if manifest_path is None else Path(manifest_path)
    )
    manifest, expected_files = _expected_output()
    expected_manifest = _manifest_bytes(manifest)
    if check:
        return _check(output_dir, manifest_path, expected_manifest, expected_files)

    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    for relative, payload in expected_files.items():
        path = output_dir / relative
        if path.exists() and path.is_symlink():
            raise RuntimeError(f"V1-09 asset is a symlink: {relative}")
        path.write_bytes(payload)
    manifest_path.write_bytes(expected_manifest)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate V1-09 defect assets")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    generate(arguments.output, check=arguments.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
