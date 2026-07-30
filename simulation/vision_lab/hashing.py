from __future__ import annotations

import hashlib
from pathlib import Path


def asset_sha256(path: str | Path, *, mode: str = "bytes") -> str:
    """Hash an asset using its declared cross-checkout representation."""

    asset_path = Path(path)
    digest = hashlib.sha256()
    if mode == "bytes":
        with asset_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if mode == "text_lf":
        normalized = (
            asset_path.read_bytes()
            .replace(b"\r\n", b"\n")
            .replace(b"\r", b"\n")
        )
        digest.update(normalized)
        return digest.hexdigest()
    raise ValueError(f"Unsupported asset hash mode: {mode}")
