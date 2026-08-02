from __future__ import annotations

from pathlib import Path


def test_all_rgbd_delivery_files_are_retained() -> None:
    root = Path(__file__).resolve().parents[2]
    retained = {
        line.strip()
        for line in (root / "RETAINED_FILES.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    expected = {
        path.relative_to(root).as_posix()
        for base in (root / "vision_platform" / "rgbd", root / "tests" / "test_rgbd")
        for path in base.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert expected <= retained
    assert not any("__pycache__" in path or path.endswith(".pyc") for path in retained)
