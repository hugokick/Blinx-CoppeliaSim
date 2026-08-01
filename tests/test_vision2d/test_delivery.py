from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_retained_files_contains_complete_vision2d_kernel_delivery():
    retained = {
        line.strip()
        for line in (ROOT / "RETAINED_FILES.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    required = {
        "docs/experiments/V1-02.md",
        "config/experiments/V1-02.json",
        "config/experiments/V1-03.json",
        "config/experiments/V1-04.json",
        "student_programs/templates/v1_02_size_measurement.py",
        "student_programs/templates/v1_03_pose_measurement.py",
        "student_programs/templates/v1_04_geometry_measurement.py",
        "tests/test_vision_quality/test_v1_02_materials.py",
        "tests/test_vision_quality/test_v1_03_materials.py",
        "tests/test_vision_quality/test_v1_04_materials.py",
        "docs/experiments/V1-03.md",
        "docs/experiments/V1-04.md",
        "docs/experiments/V1-05.md",
        "docs/superpowers/plans/2026-07-31-vision2d-algorithm-kernel-plan.md",
        "docs/superpowers/specs/2026-07-31-vision2d-algorithm-kernel-design.md",
        "vision_platform/vision2d/__init__.py",
        "vision_platform/vision2d/curriculum.py",
        "vision_platform/vision2d/models.py",
        "vision_platform/vision2d/preprocessing.py",
        "vision_platform/vision2d/segmentation.py",
        "vision_platform/vision2d/geometry.py",
        "vision_platform/vision2d/appearance.py",
        "vision_platform/vision2d/pipeline.py",
        "vision_platform/vision2d/serialization.py",
        "tests/test_vision2d/__init__.py",
        "tests/test_vision2d/synthetic_factory.py",
        "tests/test_vision2d/test_models.py",
        "tests/test_vision2d/test_synthetic_factory.py",
        "tests/test_vision2d/test_preprocessing.py",
        "tests/test_vision2d/test_segmentation.py",
        "tests/test_vision2d/test_geometry.py",
        "tests/test_vision2d/test_appearance.py",
        "tests/test_vision2d/test_pipeline.py",
        "tests/test_vision2d/test_serialization.py",
        "tests/test_vision2d/test_compatibility.py",
        "tests/test_vision2d/test_curriculum.py",
        "tests/test_vision2d/test_guides.py",
        "tests/test_vision2d/test_delivery.py",
    }
    assert required <= retained
    assert all((ROOT / path).is_file() for path in required)
    assert len(retained) == len(
        [
            line.strip()
            for line in (ROOT / "RETAINED_FILES.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    )
    assert not any(path.startswith("artifacts/") for path in retained)
