from __future__ import annotations

import pytest


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "coppeliasim: requires the opt-in D1-01 CoppeliaSim RGB-D probe",
    )


def pytest_collection_modifyitems(config, items) -> None:
    marker_expression = config.getoption("-m") or ""
    if "coppeliasim" in marker_expression:
        return
    skip = pytest.mark.skip(
        reason=(
            "D1-01 live CoppeliaSim acceptance is opt-in; rerun with "
            "'-m coppeliasim' (a skip is not an online PASS)"
        )
    )
    for item in items:
        if item.get_closest_marker("coppeliasim"):
            item.add_marker(skip)
