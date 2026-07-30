from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUIDES = tuple(
    ROOT / "docs" / "experiments" / f"R1-{index:02d}.md"
    for index in (1, 2, 5, 6, 7)
)
SECTIONS = (
    "## 实验任务",
    "## 实验入口",
    "## 原理",
    "## 操作步骤",
    "## 完成条件",
    "## 常见错误",
    "## 仿真与真机差异",
    "## 结果保存",
    "## 人工验收点",
)
HARDWARE_BOUNDARY = (
    "本实验当前只完成软件与 CoppeliaSim 仿真验证，硬件状态为 "
    "PENDING_HARDWARE。仿真 PASS 不代表真机验收通过；真实相机噪声、"
    "机械臂误差、急停、气路和物理抓取需在 V2.3 单独验收。"
)


def test_first_batch_guides_have_complete_teaching_contract():
    for path in GUIDES:
        text = path.read_text(encoding="utf-8")
        for section in SECTIONS:
            assert section in text, f"{path.name} missing {section}"
        assert HARDWARE_BOUNDARY in text
        assert "自动评分" not in text
