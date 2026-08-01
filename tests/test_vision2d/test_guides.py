from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_four_guides_define_algorithm_inputs_outputs_and_boundaries():
    required = {
        "V1-02": ("像素尺寸", "毫米"),
        "V1-03": ("中心", "旋转角"),
        "V1-04": ("面积", "周长"),
        "V1-05": ("颜色", "形状"),
    }
    for experiment_id, terms in required.items():
        path = ROOT / "docs" / "experiments" / f"{experiment_id}.md"
        text = path.read_text(encoding="utf-8")
        assert experiment_id in text
        assert all(term in text for term in terms)
        assert "原创合成图" in text
        assert "不是课程成绩" in text
        assert "PENDING_HARDWARE" in text
        assert "CoppeliaSim 在线验收不在本分支范围" in text
