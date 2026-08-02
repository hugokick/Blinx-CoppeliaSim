"""Generate the original deterministic OCR assets for V1-08.

The glyphs are repository-owned 5x7 bitmaps.  The generator deliberately
keeps the render pipeline small and deterministic so that a checkout can
recreate the committed bytes without a host-specific data source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_NAME = "vision_ocr_sorting_lab"
DEFAULT_OUTPUT = PROJECT_ROOT / "simulation" / SCENE_NAME
GENERATOR = "tools.vision_lab.generate_ocr_assets"
GENERATOR_VERSION = "1.0.0"
SEED = 20260802
ALPHABET = ("1", "2", "A", "B")
VARIANTS_PER_GLYPH = 12
TRAINING_SIZE = (64, 96)  # width, height
LABEL_SIZE = (64, 96)  # labels use the same fixed ROI footprint as glyphs

# Five columns by seven rows.  Each row is intentionally literal so the
# semantic source is reviewable and is independent of any installed assets.
BITMAPS: dict[str, tuple[str, ...]] = {
    "1": (
        "00100",
        "01100",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110",
    ),
    "2": (
        "01110",
        "10001",
        "00001",
        "00010",
        "00100",
        "01000",
        "11111",
    ),
    "A": (
        "01110",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ),
    "B": (
        "11110",
        "10001",
        "10001",
        "11110",
        "10001",
        "10001",
        "11110",
    ),
}


def _png_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(
        ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )
    if not ok:
        raise RuntimeError("V1-08 PNG encoding failed")
    return bytes(encoded)


def _draw_bitmap(
    canvas: np.ndarray,
    glyph: str,
    *,
    x: int,
    y: int,
    scale: int,
    line_width: int,
) -> None:
    """Draw one literal bitmap into a BGR canvas."""

    bitmap = BITMAPS[glyph]
    mask = np.zeros((len(bitmap) * scale, len(bitmap[0]) * scale), np.uint8)
    for row, bits in enumerate(bitmap):
        for column, bit in enumerate(bits):
            if bit == "1":
                left = column * scale
                top = row * scale
                cv2.rectangle(
                    mask,
                    (left, top),
                    (left + scale - 1, top + scale - 1),
                    255,
                    thickness=-1,
                )
    if line_width > 1:
        kernel = np.ones((line_width, line_width), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    height, width = mask.shape
    end_x = x + width
    end_y = y + height
    if x < 0 or y < 0 or end_x > canvas.shape[1] or end_y > canvas.shape[0]:
        raise ValueError("bitmap placement is outside the image")
    canvas[y:end_y, x:end_x, :] = np.where(
        mask[..., None] > 0,
        np.zeros((height, width, 3), dtype=np.uint8),
        canvas[y:end_y, x:end_x, :],
    )


def _render_training_variant(glyph: str, index: int) -> np.ndarray:
    width, height = TRAINING_SIZE
    rng = random.Random(SEED + (ALPHABET.index(glyph) + 1) * 1000 + index)
    scale = 7 + rng.randrange(3)
    line_width = 1 + rng.randrange(2)
    bitmap_width = 5 * scale + (line_width - 1)
    bitmap_height = 7 * scale + (line_width - 1)
    x = 7 + rng.randrange(max(1, width - bitmap_width - 13))
    y = 12 + rng.randrange(max(1, height - bitmap_height - 19))
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    _draw_bitmap(
        canvas,
        glyph,
        x=x,
        y=y,
        scale=scale,
        line_width=line_width,
    )
    return canvas


def _render_label(identifier: str) -> np.ndarray:
    width, height = LABEL_SIZE
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    scale = 5
    line_width = 1
    # The two glyphs share one fixed ROI.  The four identifiers therefore
    # exercise the same bitmap table as the training samples.
    _draw_bitmap(canvas, identifier[0], x=3, y=30, scale=scale, line_width=line_width)
    _draw_bitmap(canvas, identifier[1], x=31, y=30, scale=scale, line_width=line_width)
    return canvas


def _asset_record(
    *,
    relative_path: str,
    image: np.ndarray,
    purpose: str,
    glyph: str | None = None,
    identifier: str | None = None,
) -> tuple[dict[str, object], bytes]:
    payload = _png_bytes(image)
    record: dict[str, object] = {
        "path": relative_path,
        "purpose": purpose,
        "size_px": [int(image.shape[1]), int(image.shape[0])],
        "channels": int(image.shape[2]) if image.ndim == 3 else 1,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if glyph is not None:
        record = {"glyph": glyph, **record}
    if identifier is not None:
        record = {"identifier": identifier, **record}
    return record, payload


def _expected_assets() -> tuple[dict[str, object], dict[str, bytes]]:
    training: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    files: dict[str, bytes] = {}
    digests: set[str] = set()
    for glyph in ALPHABET:
        for index in range(1, VARIANTS_PER_GLYPH + 1):
            image = _render_training_variant(glyph, index)
            record, payload = _asset_record(
                relative_path=f"training/{glyph}/{index:02d}.png",
                image=image,
                purpose="training",
                glyph=glyph,
            )
            digest = str(record["sha256"])
            if digest in digests:
                raise RuntimeError("deterministic glyph variants must be byte-unique")
            digests.add(digest)
            training.append(record)
            files[str(record["path"])] = payload
    for identifier in ("A1", "A2", "B1", "B2"):
        image = _render_label(identifier)
        record, payload = _asset_record(
            relative_path=f"labels/{identifier}.png",
            image=image,
            purpose="scene_label",
            identifier=identifier,
        )
        digest = str(record["sha256"])
        if digest in digests:
            raise RuntimeError("scene labels must not duplicate training bytes")
        digests.add(digest)
        labels.append(record)
        files[str(record["path"])] = payload
    training.sort(key=lambda item: str(item["path"]))
    labels.sort(key=lambda item: str(item["path"]))
    manifest: dict[str, object] = {
        "schema_version": 1,
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "seed": SEED,
        "alphabet": list(ALPHABET),
        "training_parameters": {
            "method": "knn",
            "test_fraction": 0.25,
            "seed": SEED,
            "variants_per_glyph": VARIANTS_PER_GLYPH,
            "bitmap_size_px": [5, 7],
            "image_size_px": list(TRAINING_SIZE),
            "channels": 3,
        },
        "allowed_scene_ids": ["V1-08"],
        "training": training,
        "labels": labels,
    }
    return manifest, files


def _manifest_bytes(manifest: dict[str, object]) -> bytes:
    return (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False)
        + "\n"
    ).encode("utf-8")


def generate(
    output_dir: Path,
    *,
    manifest_path: Path | None = None,
    check: bool = False,
) -> Path:
    """Generate or byte-check an OCR asset directory."""

    output_dir = Path(output_dir)
    manifest_path = (
        output_dir / "ocr_assets_manifest.json"
        if manifest_path is None
        else Path(manifest_path)
    )
    manifest, expected_files = _expected_assets()
    expected_manifest = _manifest_bytes(manifest)
    if check:
        for relative_path, expected in expected_files.items():
            path = output_dir / relative_path
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"V1-08 asset missing: {relative_path}")
            if path.read_bytes() != expected:
                raise RuntimeError(f"V1-08 asset drift: {relative_path}")
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise RuntimeError("V1-08 manifest missing")
        if manifest_path.read_bytes() != expected_manifest:
            raise RuntimeError("V1-08 manifest drift")
        return manifest_path

    output_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, payload in expected_files.items():
        path = output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(expected_manifest)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate V1-08 OCR assets")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    generate(arguments.output, check=arguments.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
