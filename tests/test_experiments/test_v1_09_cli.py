from __future__ import annotations

from pathlib import Path

import pytest

from vision_platform.cli import _resolve_experiment_port


def test_v1_09_uses_dedicated_port_23010() -> None:
    assert _resolve_experiment_port("V1-08", None) == 23008
    assert _resolve_experiment_port("V1-09", None) == 23010
    assert _resolve_experiment_port("V1-09", 23010) == 23010
    with pytest.raises(ValueError):
        _resolve_experiment_port("V1-09", 23000)


def test_v1_09_wrapper_uses_exact_process_ownership_helpers() -> None:
    source = Path("tools/vision_lab/run_v1_09_surface_defects.ps1").read_text(encoding="utf-8")
    assert "launch_coppeliasim.ps1" in source
    assert "process_ownership.ps1" in source
    assert "Stop-ExactOwnedProcess" in source
    assert "23010" in source
    assert "Stop-Process -Name" not in source
