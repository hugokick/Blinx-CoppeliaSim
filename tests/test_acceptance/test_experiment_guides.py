import re
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
EVIDENCE_FILES = (
    "source.py",
    "manifest.json",
    "snapshots.jsonl",
    "scene-initial.json",
    "scene-final.json",
    "summary.json",
)
CONTRACTS = {
    "R1-01.md": {
        "实验任务": ("六个关节", "TCP", "工作空间", "回零", "观察位"),
        "原理": ("关节链", "世界坐标", "position-only"),
        "完成条件": ("PASS", "初态", "末态", "home"),
        "人工验收点": ("六个关节", "TCP", "PENDING_HARDWARE"),
    },
    "R1-02.md": {
        "实验任务": ("四个示教点", "暂停", "继续", "单步"),
        "原理": ("示教点", "安全高度", "命令边界"),
        "完成条件": ("四个 checkpoint", "无越界", "安全"),
        "人工验收点": ("低空", "横移"),
    },
    "R1-05.md": {
        "实验任务": ("快照", "六个", "两列三层"),
        "原理": ("HSV", "像素", "世界坐标", "分层"),
        "完成条件": ("PASS", "matched=6"),
        "人工验收点": ("参数", "码垛顺序"),
    },
    "R1-06.md": {
        "实验任务": ("1、2、3", "升序", "三个目标槽位"),
        "原理": ("归一化", "模板匹配", "排序规则"),
        "完成条件": ("三个数字", "matched=3"),
        "人工验收点": ("倒序", "规则"),
    },
    "R1-07.md": {
        "实验任务": ("红方块", "蓝方块", "绿圆柱", "黄圆柱", "分类"),
        "原理": ("HSV", "轮廓圆度", "映射"),
        "完成条件": ("四类", "matched=4"),
        "人工验收点": ("空结果", "低置信度", "停止"),
    },
}
FORBIDDEN_CLAIMS = (
    "自动评分",
    "真机已通过",
    "真机验收已通过",
    "硬件验收已通过",
    "教学效果已通过",
    "TODO",
    "TBD",
    "FIXME",
    "待补",
    "待实现",
)


def _section(text, title):
    match = re.search(
        rf"^## {re.escape(title)}\s*$\n(.*?)(?=^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing section {title}"
    return match.group(1)


def test_first_batch_guides_have_complete_teaching_contract():
    for path in GUIDES:
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")
        text = path.read_text(encoding="utf-8")
        assert re.findall(r"^## .+$", text, flags=re.MULTILINE) == list(
            SECTIONS
        )
        for title, required in CONTRACTS[path.name].items():
            content = _section(text, title)
            for phrase in required:
                assert phrase in content, (
                    f"{path.name} {title} missing {phrase}"
                )
        entry = _section(text, "实验入口")
        assert "PyQt" in entry and "实验目录" in entry
        assert "experiment-run" in entry
        assert "在线验收" in entry and "不构成" in entry
        results = _section(text, "结果保存")
        for name in EVIDENCE_FILES:
            assert name in results, f"{path.name} missing evidence {name}"
        assert HARDWARE_BOUNDARY in text
        for forbidden in FORBIDDEN_CLAIMS:
            assert forbidden not in text, f"{path.name} contains {forbidden}"
        if path.name in {"R1-05.md", "R1-06.md", "R1-07.md"}:
            principle = _section(text, "原理")
            assert "confidence" in principle
            assert "0.5" in principle
            assert "停止" in principle
        if path.name == "R1-06.md":
            principle = _section(text, "原理")
            assert "深色前景" in principle
            assert "紧致包围框" in principle
            assert "黑底" in principle
            assert "模板匹配" in principle
