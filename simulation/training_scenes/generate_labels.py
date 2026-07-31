from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_staged_png(path: Path, expected: np.ndarray) -> None:
    actual = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if (
        actual is None
        or actual.dtype != np.uint8
        or actual.shape != expected.shape
        or not np.array_equal(actual, expected)
    ):
        raise RuntimeError(f"Could not verify staged digit label {path}")


def _cleanup_staged(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def generate_digit_labels(output_dir: str | Path) -> dict:
    root = Path(output_dir).expanduser().resolve()
    digits = root / "digits"
    digits.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    staged_paths: list[Path] = []
    staged_labels: list[tuple[Path, Path]] = []
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
    try:
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
            target = digits / f"{value}.png"
            staged = digits / f".label-stage-{token}-{value}.png"
            staged_paths.append(staged)
            if not cv2.imwrite(str(staged), image):
                raise RuntimeError(f"Could not stage digit label {value}")
            _verify_staged_png(staged, image)
            staged_labels.append((staged, target))
            files.append(
                {
                    "digit": value,
                    "path": target.relative_to(root).as_posix(),
                    "sha256": _sha256(staged),
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
        staged_manifest = root / f".manifest-stage-{token}.json"
        staged_paths.append(staged_manifest)
        staged_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if json.loads(
            staged_manifest.read_text(encoding="utf-8")
        ) != manifest:
            raise RuntimeError("Could not verify staged label manifest")

        manifest_path = root / "manifest.json"
        try:
            manifest_path.unlink(missing_ok=True)
            for staged, target in staged_labels:
                staged.replace(target)
            staged_manifest.replace(manifest_path)
        except OSError as exc:
            raise RuntimeError("Could not publish digit labels") from exc
        return manifest
    finally:
        _cleanup_staged(staged_paths)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = generate_digit_labels(args.output)
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
