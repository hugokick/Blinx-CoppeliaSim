from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_staging_root() -> Path:
    configured = os.environ.get("VISION_ASCII_STAGING_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    candidate = (Path(tempfile.gettempdir()) / "vision-lab-coppeliasim").resolve()
    if str(candidate).isascii():
        return candidate
    return (
        Path(os.environ.get("PROGRAMDATA", "C:/ProgramData"))
        / "vision-lab-coppeliasim"
    ).resolve()


def stage_scene_for_coppeliasim(
    scene_path: str | Path,
    *,
    staging_root: str | Path | None = None,
) -> Path:
    """Return a scene path that CoppeliaSim's remote API can load safely."""

    source = Path(scene_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"CoppeliaSim scene does not exist: {source}")
    if str(source).isascii():
        return source

    root = (
        Path(staging_root).expanduser().resolve()
        if staging_root is not None
        else _default_staging_root()
    )
    if not str(root).isascii():
        raise ValueError(
            "CoppeliaSim ASCII staging directory must contain only ASCII "
            f"characters: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)

    source_hash = _sha256(source)
    staged = root / f"{source_hash}.ttt"
    if not staged.is_file() or _sha256(staged) != source_hash:
        temporary = root / f".{source_hash}.tmp"
        shutil.copy2(source, temporary)
        os.replace(temporary, staged)
    if _sha256(staged) != source_hash:
        raise RuntimeError(
            f"Staged CoppeliaSim scene hash mismatch: {staged}"
        )
    return staged
