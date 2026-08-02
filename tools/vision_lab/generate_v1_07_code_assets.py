"""Generate the deterministic original code faces used by V1-07."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2

# Keep the documented standalone invocation independent of the caller's
# working-directory import path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision_platform.vision2d.code_recognition import encode_ean13_payload


ASSETS = (
    ("qr_v1_07_a", "part_a", "qr", "V1-07-A", "qr_v1_07_a.png"),
    ("qr_v1_07_b", "part_b", "qr", "V1-07-B", "qr_v1_07_b.png"),
    ("ean_6901234567892", "part_c", "ean13", "6901234567892", "ean_6901234567892.png"),
    ("ean_6901234567809", "part_d", "ean13", "6901234567809", "ean_6901234567809.png"),
)


def _qr(payload: str):
    encoded = cv2.QRCodeEncoder_create().encode(payload)
    scaled = cv2.resize(encoded, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
    bordered = cv2.copyMakeBorder(
        scaled, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255
    )
    return cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)


def _png_bytes(image) -> bytes:
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise RuntimeError("V1-07 PNG encoding failed")
    return bytes(encoded)


def generate(
    output_dir: Path,
    *,
    manifest_path: Path | None = None,
    check: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    manifest_path = (
        output_dir / "code_assets_manifest.json"
        if manifest_path is None
        else Path(manifest_path)
    )
    if not check:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    expected_files: dict[str, bytes] = {}
    for asset_id, part_id, code_type, payload, filename in ASSETS:
        image = _qr(payload) if code_type == "qr" else encode_ean13_payload(payload)
        png = _png_bytes(image)
        expected_files[filename] = png
        entries.append(
            {
                "asset_id": asset_id,
                "part_id": part_id,
                "code_type": code_type,
                "payload": payload,
                "path": f"simulation/vision_code_routing_lab/code_assets/{filename}",
                "sha256": hashlib.sha256(png).hexdigest(),
                "size_px": [int(image.shape[1]), int(image.shape[0])],
                "channels": 3,
                "generator": "robot-sim-opencv-v1",
            }
        )
    manifest_bytes = (
        json.dumps(
            {"schema_version": 1, "entries": entries},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    for filename, expected in expected_files.items():
        path = output_dir / filename
        if check:
            if not path.is_file() or path.read_bytes() != expected:
                raise RuntimeError(f"V1-07 asset drift: {filename}")
        else:
            path.write_bytes(expected)
    if check:
        if not manifest_path.is_file() or manifest_path.read_bytes() != manifest_bytes:
            raise RuntimeError("V1-07 asset drift: code_assets_manifest.json")
    else:
        manifest_path.write_bytes(manifest_bytes)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    output = PROJECT_ROOT / "simulation" / "vision_code_routing_lab" / "code_assets"
    generate(
        output,
        manifest_path=output.parent / "code_assets_manifest.json",
        check=arguments.check,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
