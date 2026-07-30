from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_bgr(path: str | Path) -> np.ndarray | None:
    """Read a BGR image without relying on OpenCV's Windows path handling."""
    source = Path(path).expanduser().resolve()
    try:
        encoded = np.frombuffer(source.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def write_image(path: str | Path, image: np.ndarray) -> bool:
    """Write an image through Python file I/O so Unicode paths remain portable."""
    target = Path(path).expanduser().resolve()
    extension = target.suffix or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        return False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded.tobytes())
    except OSError:
        return False
    return True
