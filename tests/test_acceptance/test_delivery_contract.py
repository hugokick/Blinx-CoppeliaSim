from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_delivery_has_one_command_for_each_student_workflow():
    required = [
        ROOT / "tools" / "vision_lab" / "run_pyqt.ps1",
        ROOT / "tools" / "vision_lab" / "run_replay_demo.ps1",
        ROOT / "tools" / "vision_lab" / "run_acceptance.ps1",
        ROOT / "tools" / "vision_lab" / "launch_coppeliasim.ps1",
    ]
    assert all(path.is_file() for path in required)


def test_usage_guide_includes_first_run_environment_bootstrap():
    text = (
        ROOT / "docs" / "视觉仿真实训平台使用说明.md"
    ).read_text(encoding="utf-8")

    assert "tools\\vision_lab\\bootstrap.ps1" in text
    assert "-SkipCad" in text
    assert "完整自动验收环境" in text


def test_hardware_checklist_does_not_claim_camera_acceptance():
    path = ROOT / "docs" / "仿真转真机验证清单.md"
    text = path.read_text(encoding="utf-8")

    assert "PENDING_HARDWARE" in text
    assert "海康相机" in text
    assert "MVS" in text
    assert "低速空跑" in text
    assert "单物体抓取" in text
    assert "多物体分类" in text
    assert "海康相机实机已通过" not in text
    assert "真实机械臂验收通过" not in text


def test_student_guides_contain_complete_teaching_contract():
    guides = [
        ROOT / "docs" / "实验三-CoppeliaSim视觉标定.md",
        ROOT / "docs" / "实验四-CoppeliaSim物体分类.md",
    ]
    required_sections = (
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
    for guide in guides:
        text = guide.read_text(encoding="utf-8")
        for section in required_sections:
            assert section in text, f"{guide.name} is missing {section}"


def test_scene_readme_pins_selected_vendor_model_and_position_only_contract():
    text = (
        ROOT / "simulation" / "vision_lab" / "README.md"
    ).read_text(encoding="utf-8")

    assert (
        "9ea431ee1b87417c19f1a0f579209dfe20d0c8d1e381b660010900718cb2b4dc"
        in text
    )
    assert (
        "f3f493626792ef25b99e6b1e79f791da96f2526cee7161c90194443e9c34a9a2"
        in text
    )
    assert "position-only" in text
    assert "RX/RY/RZ" in text
    assert "不翻转" in text


def test_acceptance_script_runs_live_gates_and_marks_hardware_pending():
    text = (
        ROOT / "tools" / "vision_lab" / "run_acceptance.ps1"
    ).read_text(encoding="utf-8")

    assert "simulation.vision_lab.verify_scene" in text
    assert "vision_platform.cli accept" in text
    assert "vision_platform.cli simui-smoke" in text
    assert "-m coppeliasim" in text
    assert "PENDING_HARDWARE" in text
    assert "acceptance-summary.json" in text


def test_coppeliasim_launcher_quotes_scene_paths_with_spaces():
    text = (
        ROOT / "tools" / "vision_lab" / "launch_coppeliasim.ps1"
    ).read_text(encoding="utf-8")

    assert """$QuotedScene = '"' + $Scene + '"'""" in text
    argument_block = text.split("$StartArguments = @(", 1)[1].split(")", 1)[0]
    assert "$QuotedScene" in argument_block


def test_student_launcher_has_required_contract():
    launcher = ROOT / "tools" / "vision_lab" / "run_student_program.ps1"
    source = launcher.read_text(encoding="utf-8")

    assert "[Parameter(Mandatory = $true)]" in source
    assert "[string]$Program" in source
    assert "[string]$OutputDir" in source
    assert "launch_coppeliasim.ps1" in source
    assert "vision_platform.cli student-run" in source
    assert "--robot sim" in source
    assert "--scene $Scene" in source
    assert "Resolve-Path -LiteralPath $Program" in source
    assert "GetExtension($Program)" in source
    assert "exit $StudentExitCode" in source


def test_student_launcher_does_not_kill_unowned_simulator():
    launcher = ROOT / "tools" / "vision_lab" / "run_student_program.ps1"
    source = launcher.read_text(encoding="utf-8")

    assert "Stop-Process -Name" not in source
    assert "taskkill" not in source.lower()


def test_student_program_guide_has_complete_simulation_contract():
    guide = ROOT / "student_programs" / "README.md"
    source = guide.read_text(encoding="utf-8")

    required = (
        "ctx.robot.home()",
        "ctx.robot.move_world",
        "ctx.robot.pose()",
        "ctx.tool.on()",
        "ctx.tool.off()",
        "ctx.log",
        "ctx.sleep",
        'ctx.checkpoint("阶段名")',
        "checkpoint(label)",
        "mm",
        "世界坐标",
        "安全高度",
        "safe_z",
        "student-validate",
        "run_student_program.ps1",
        "artifacts/vision_lab/student-runs",
        "RX/RY/RZ",
        "sim",
        "PENDING_HARDWARE",
    )
    for item in required:
        assert item in source
