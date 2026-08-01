from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_v1_01_delivery_is_complete_retained_and_isolated():
    retained_lines = [
        line.strip()
        for line in (ROOT / "RETAINED_FILES.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.startswith("#")
    ]
    retained = set(retained_lines)
    required = {
        ".gitattributes",
        "config/experiments/V1-01.json",
        "docs/experiments/V1-01.md",
        "docs/superpowers/plans/2026-08-01-vision-quality-platform-v1-01-plan.md",
        "docs/superpowers/specs/2026-08-01-vision-quality-platform-v1-01-design.md",
        "simulation/vision_quality_lab/__init__.py",
        "simulation/vision_quality_lab/BL23_vision_quality_lab.ttt",
        "simulation/vision_quality_lab/profiles.json",
        "simulation/vision_quality_lab/scene_manifest.json",
        "simulation/vision_quality_lab/scene_spec.json",
        "student_programs/templates/v1_01_virtual_vision.py",
        "tools/vision_lab/run_vision_quality_acceptance.ps1",
        "vision_platform/ui/vision_result_panel.py",
        "vision_platform/vision_quality/__init__.py",
        "vision_platform/vision_quality/catalog.py",
        "vision_platform/vision_quality/controller.py",
        "vision_platform/vision_quality/evidence.py",
        "vision_platform/vision_quality/models.py",
        "vision_platform/vision_quality/results.py",
        "tests/test_acceptance/test_coppeliasim_v1_01.py",
        "tests/test_acceptance/test_coppeliasim_vision_quality_scene.py",
        "tests/test_simulation/test_vision_quality_scene_contract.py",
        "tests/test_vision_platform/test_vision_result_panel.py",
        "tests/test_vision_quality/__init__.py",
        "tests/test_vision_quality/test_catalog.py",
        "tests/test_vision_quality/test_controller.py",
        "tests/test_vision_quality/test_delivery.py",
        "tests/test_vision_quality/test_evidence.py",
        "tests/test_vision_quality/test_results.py",
        "tests/test_vision_quality/test_v1_01_materials.py",
    }
    assert required <= retained
    assert len(retained_lines) == len(retained)
    assert all((ROOT / path).is_file() for path in required)
    assert not any(path.startswith("artifacts/") for path in retained)
