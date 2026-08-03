from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "student_programs/templates/v1_09_surface_defects.py"


def test_v1_09_template_uses_only_controlled_surface_defect_api() -> None:
    source = TEMPLATE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "surface_defects" in calls
    assert "defect_sort_entry" in calls
    assert "move_world" not in source
    assert "tool" not in source
    assert "open(" not in source
    assert "threshold" not in source
    assert "roi" not in source
    assert "route" not in source
    assert "slot" not in source


def test_v1_09_template_iterates_the_host_frozen_entries() -> None:
    source = TEMPLATE.read_text(encoding="utf-8")
    assert "analysis = ctx.vision2d.surface_defects()" in source
    assert "for entry in analysis.entries:" in source
    assert "ctx.vision2d.defect_sort_entry(entry.entry_id)" in source
