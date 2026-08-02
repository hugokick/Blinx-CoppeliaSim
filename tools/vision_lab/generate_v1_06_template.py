from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


TEMPLATE_ID = "v1_06_red_rectangle"
TEMPLATE_VERSION = "1.0.0"
ASSET_NAME = "v1_06_reference.png"
MANIFEST_NAME = "manifest.json"
ASSET_PATH = "simulation/vision_quality_lab/templates/v1_06_reference.png"
SEARCH_ROI_PX = [0, 0, 512, 512]


@dataclass(frozen=True)
class TemplateGeneration:
    asset_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def _reference_image() -> np.ndarray:
    """Build the small reference without random or machine-dependent input."""

    height, width = 40, 67
    # The quality scene's ReferenceRectangle is a solid red sample on a
    # white inspection board.  Recreate that contract deterministically;
    # this is not a camera capture or a user-provided file.
    image = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
    image[5:34, 5:61] = (71, 71, 255)
    return image


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(output_dir: Path | None = None) -> TemplateGeneration:
    root = Path(__file__).resolve().parents[2]
    target = output_dir or root / "simulation/vision_quality_lab/templates"
    target.mkdir(parents=True, exist_ok=True)
    asset_path = target / ASSET_NAME
    manifest_path = target / MANIFEST_NAME
    image = _reference_image()
    if not cv2.imwrite(str(asset_path), image):
        raise RuntimeError(f"failed to write template asset: {asset_path}")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "template_id": TEMPLATE_ID,
        "template_version": TEMPLATE_VERSION,
        "asset_path": (
            ASSET_PATH
            if target.resolve()
            == (root / "simulation/vision_quality_lab/templates").resolve()
            else ASSET_NAME
        ),
        "sha256": _sha256(asset_path),
        "size_px": [67, 40],
        "channels": 3,
        "generator": "tools/vision_lab/generate_v1_06_template.py",
        "source": "deterministic synthetic reference; no camera capture",
        "method": "TM_CCOEFF_NORMED",
        "threshold": 0.72,
        "search_roi_px": SEARCH_ROI_PX,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return TemplateGeneration(asset_path, manifest_path, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the V1-06 template asset")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = generate(args.output_dir)
    print(json.dumps(result.manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
