from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_v1_07_definition_publishes_only_the_fixed_closed_loop() -> None:
    definition = json.loads(
        (ROOT / "config" / "experiments" / "V1-07.json").read_text(encoding="utf-8")
    )
    assert definition["experiment_id"] == "V1-07"
    assert definition["version"] == "2.2.0"
    assert definition["scene"] == "simulation/vision_code_routing_lab/BL23_vision_code_routing_lab.ttt"
    assert definition["hardware_status"] == "PENDING_HARDWARE"
    routing = definition["public_parameters"]["code_routing"]
    manifest = json.loads(
        (ROOT / definition["scene_manifest"]).read_text(encoding="utf-8")
    )
    assert routing["expected_count"] == 4
    assert {route["code_type"] for route in routing["routes"]} == {"qr", "ean13"}
    assert len({route["payload"] for route in routing["routes"]}) == 4
    assert len({tuple(route["drop_xyz_mm"]) for route in routing["routes"]}) == 4
    assert routing["calibration_matrix"] == manifest["code_routing"]["calibration_matrix"]
    assert not any(key in routing for key in ("path", "roi_px", "threshold", "command"))


def test_v1_07_guide_preserves_simulation_human_and_hardware_boundaries() -> None:
    guide = (ROOT / "docs" / "experiments" / "V1-07.md").read_text(encoding="utf-8")
    for heading in (
        "## 实验目标", "## 安全边界", "## 操作步骤", "## 证据说明",
        "## 错误解释", "## 自动检查", "## 人工验收", "## 真机迁移边界",
    ):
        assert heading in guide
    assert "PENDING_HUMAN_ACCEPTANCE" in guide
    assert "PENDING_HARDWARE" in guide
    assert "仿真通过不代表" in guide


def test_v1_07_template_contains_no_private_or_host_escape_api() -> None:
    source = (ROOT / "student_programs" / "templates" / "v1_07_code_routing.py").read_text(encoding="utf-8")
    assert "ctx.vision2d.code_routes()" in source
    assert "pick_and_place(" in source
    for forbidden in ("subprocess", "os.system", "cv2", "numpy", "sim.", "Path(", "open("):
        assert forbidden not in source
