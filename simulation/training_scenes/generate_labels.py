from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_digit_labels(output_dir: str | Path) -> dict:
    root = Path(output_dir).expanduser().resolve()
    digits = root / "digits"
    digits.mkdir(parents=True, exist_ok=True)
    patterns = {
        1: ("b", "c"),
        2: ("a", "b", "g", "e", "d"),
        3: ("a", "b", "g", "c", "d"),
    }
    segments = {
        "a": ((20, 12), (44, 18)),
        "b": ((43, 16), (49, 46)),
        "c": ((43, 50), (49, 80)),
        "d": ((20, 78), (44, 84)),
        "e": ((15, 50), (21, 80)),
        "g": ((20, 45), (44, 51)),
    }
    files = []
    for value in (1, 2, 3):
        image = np.full((96, 64), 255, dtype=np.uint8)
        for segment in patterns[value]:
            cv2.rectangle(
                image,
                segments[segment][0],
                segments[segment][1],
                0,
                thickness=-1,
            )
        path = digits / f"{value}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write {path}")
        files.append(
            {
                "digit": value,
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "size": [64, 96],
            }
        )
    manifest = {
        "schema_version": 1,
        "generator": "simulation.training_scenes.generate_labels",
        "license": "project-original-generated",
        "rendering": "project-original seven-segment geometry",
        "files": files,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = generate_digit_labels(args.output)
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
