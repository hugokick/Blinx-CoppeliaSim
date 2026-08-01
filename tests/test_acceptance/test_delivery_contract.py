from __future__ import annotations

from pathlib import Path

from vision_platform.student.validator import BLOCKED_IMPORTS


ROOT = Path(__file__).resolve().parents[2]

V21_REQUIRED_RELEASE_PATHS = frozenset(
    {
        "README.md",
        "RETAINED_FILES.txt",
        "config/vision_lab.default.json",
        "docs/superpowers/plans/"
        "2026-07-30-student-program-runner-v2-plan.md",
        "docs/superpowers/specs/"
        "2026-07-30-student-program-runner-v2-design.md",
        "docs/学生自编程实验说明.md",
        "docs/视觉仿真实训平台V2-开发端执行Prompt.md",
        "docs/视觉仿真实训平台使用说明.md",
        "docs/视觉仿真实训平台自动验收报告.md",
        "student_programs/README.md",
        "student_programs/templates/basic_motion.py",
        "student_programs/templates/pick_and_place.py",
        "tests/fixtures/student_programs/infinite_loop.py",
        "tests/test_acceptance/conftest.py",
        "tests/test_acceptance/test_coppeliasim_fixture_helpers.py",
        "tests/test_acceptance/test_coppeliasim_readiness.py",
        "tests/test_acceptance/test_coppeliasim_student_program.py",
        "tests/test_acceptance/test_delivery_contract.py",
        "tests/test_acceptance/test_powershell_process_ownership.py",
        "tests/test_student_programs/test_cli.py",
        "tests/test_student_programs/test_evidence.py",
        "tests/test_student_programs/test_protocol.py",
        "tests/test_student_programs/test_runner.py",
        "tests/test_student_programs/test_sdk.py",
        "tests/test_student_programs/test_student_safety.py",
        "tests/test_student_programs/test_validator.py",
        "tests/test_student_programs/test_worker.py",
        "tests/test_vision_platform/test_application_close.py",
        "tests/test_vision_platform/test_config.py",
        "tests/test_vision_platform/test_pyqt_smoke.py",
        "tests/test_vision_platform/test_session.py",
        "tests/test_vision_platform/test_student_program_panel.py",
        "tools/vision_lab/launch_coppeliasim.ps1",
        "tools/vision_lab/process_ownership.ps1",
        "tools/vision_lab/run_acceptance.ps1",
        "tools/vision_lab/run_pyqt.ps1",
        "tools/vision_lab/run_student_program.ps1",
        "vision_platform/application.py",
        "vision_platform/cli.py",
        "vision_platform/config.py",
        "vision_platform/coppeliasim_readiness.py",
        "vision_platform/session.py",
        "vision_platform/student/__init__.py",
        "vision_platform/student/evidence.py",
        "vision_platform/student/protocol.py",
        "vision_platform/student/runner.py",
        "vision_platform/student/safety.py",
        "vision_platform/student/sdk.py",
        "vision_platform/student/validator.py",
        "vision_platform/student/worker.py",
        "vision_platform/ui/pyqt_app.py",
        "vision_platform/ui/student_program_panel.py",
    }
)

V22_FIRST_BATCH_REQUIRED_RELEASE_PATHS = frozenset(
    {
        "config/experiments/R1-01.json",
        "config/experiments/R1-02.json",
        "config/experiments/R1-05.json",
        "config/experiments/R1-06.json",
        "config/experiments/R1-07.json",
        "config/experiments/catalog.json",
        "docs/experiments/R1-01.md",
        "docs/experiments/R1-02.md",
        "docs/experiments/R1-05.md",
        "docs/experiments/R1-06.md",
        "docs/experiments/R1-07.md",
        "docs/superpowers/plans/"
        "2026-07-31-robot-curriculum-v2-2-first-batch-plan.md",
        "docs/superpowers/specs/"
        "2026-07-31-robot-curriculum-coppeliasim-roadmap-design.md",
        "simulation/logistics_lab/__init__.py",
        "simulation/logistics_lab/BL23_logistics_lab.ttt",
        "simulation/logistics_lab/assets/labels/digits/1.png",
        "simulation/logistics_lab/assets/labels/digits/2.png",
        "simulation/logistics_lab/assets/labels/digits/3.png",
        "simulation/logistics_lab/assets/labels/manifest.json",
        "simulation/logistics_lab/scene_manifest.json",
        "simulation/logistics_lab/scene_spec.json",
        "simulation/robot_basics/__init__.py",
        "simulation/robot_basics/BL23_robot_basics.ttt",
        "simulation/robot_basics/scene_manifest.json",
        "simulation/robot_basics/scene_spec.json",
        "simulation/training_scenes/__init__.py",
        "simulation/training_scenes/build_scene.py",
        "simulation/training_scenes/generate_labels.py",
        "simulation/training_scenes/scene_contract.py",
        "simulation/training_scenes/verify_scene.py",
        "student_programs/__init__.py",
        "student_programs/templates/__init__.py",
        "student_programs/templates/r1_01_robot_basics.py",
        "student_programs/templates/r1_02_teach_points.py",
        "student_programs/templates/r1_05_visual_stacking.py",
        "student_programs/templates/r1_06_digit_sort.py",
        "student_programs/templates/r1_07_component_sort.py",
        "student_programs/templates/r1_common.py",
        "tests/test_acceptance/test_coppeliasim_r1_experiments.py",
        "tests/test_acceptance/test_coppeliasim_training_scenes.py",
        "tests/test_acceptance/test_experiment_guides.py",
        "tests/test_experiments/__init__.py",
        "tests/test_experiments/test_capabilities.py",
        "tests/test_experiments/test_catalog.py",
        "tests/test_experiments/test_cli.py",
        "tests/test_experiments/test_formal_catalog.py",
        "tests/test_experiments/test_probes.py",
        "tests/test_experiments/test_scene_setup.py",
        "tests/test_experiments/test_session.py",
        "tests/test_experiments/test_student_templates.py",
        "tests/test_simulation/test_formal_training_scenes.py",
        "tests/test_simulation/test_training_labels.py",
        "tests/test_simulation/test_training_scene_contract.py",
        "tests/test_student_programs/test_experiment_evidence.py",
        "tests/test_student_programs/test_experiment_gateway.py",
        "tests/test_vision_platform/test_experiment_catalog_panel.py",
        "tools/vision_lab/run_experiment.ps1",
        "vision_platform/experiments/__init__.py",
        "vision_platform/experiments/capabilities.py",
        "vision_platform/experiments/catalog.py",
        "vision_platform/experiments/models.py",
        "vision_platform/experiments/probes.py",
        "vision_platform/experiments/scene_setup.py",
        "vision_platform/experiments/session.py",
        "vision_platform/student/experiment_gateway.py",
        "vision_platform/ui/experiment_catalog_panel.py",
    }
)


def _retained_release_paths() -> set[str]:
    return {
        line.strip().replace("\\", "/")
        for line in (ROOT / "RETAINED_FILES.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


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
    workflow_definition = text.index(
        "function Invoke-AcceptanceWorkflow"
    )
    offscreen_position = text.index(
        '$env:QT_QPA_PLATFORM = "offscreen"'
    )
    pyqt_position = text.index(
        'Invoke-CheckedPython -Name "pyqt_offscreen_smoke"'
    )
    full_static_position = text.index(
        'Invoke-CheckedPython -Name "full_static_test_suite"'
    )
    scene_position = text.index(
        'Invoke-CheckedPython -Name "scene_runtime_verification"'
    )
    clear_position = text.rindex("$env:QT_QPA_PLATFORM = $null")
    workflow_call = text.rindex("Invoke-AcceptanceWorkflow")
    restore_position = text.rindex(
        "$env:QT_QPA_PLATFORM = $CallerQtPlatform"
    )
    assert "$CallerQtPlatform = $env:QT_QPA_PLATFORM" in text
    assert (
        workflow_definition
        < offscreen_position
        < pyqt_position
        < full_static_position
        < scene_position
    )
    assert clear_position < workflow_call < restore_position
    assert (
        "$env:QT_QPA_PLATFORM = $null"
        in text[offscreen_position:scene_position]
    )
    assert "PostReadinessSettleMilliseconds" not in text


def test_acceptance_tracks_structured_launch_and_exact_final_cleanup():
    source = (
        ROOT / "tools" / "vision_lab" / "run_acceptance.ps1"
    ).read_text(encoding="utf-8")
    compact = " ".join(source.replace("`", "").split())

    assert "$Launch = &" in source
    assert "if ($Launch.StartedByScript)" in source
    assert "$OwnedProcessId = [int]$Launch.ProcessId" in source
    assert "$OwnedProcessPath = [string]$Launch.ProcessPath" in source
    assert (
        "$OwnedProcessStartTimeUtcTicks = "
        "[long]$Launch.ProcessStartTimeUtcTicks"
    ) in compact
    assert "process_ownership.ps1" in source
    assert "Stop-ExactOwnedProcess" in source
    assert "-ProcessId $OwnedProcessId" in compact
    assert "-ProcessPath $OwnedProcessPath" in compact
    assert (
        "-ProcessStartTimeUtcTicks $OwnedProcessStartTimeUtcTicks"
        in compact
    )
    assert "$ListenerBefore" not in source
    assert "Stop-Process -Name" not in source


def test_coppeliasim_launcher_quotes_scene_paths_with_spaces():
    text = (
        ROOT / "tools" / "vision_lab" / "launch_coppeliasim.ps1"
    ).read_text(encoding="utf-8")

    assert """$QuotedScene = '"' + $Scene + '"'""" in text
    argument_block = text.split("$StartArguments = @(", 1)[1].split(")", 1)[0]
    assert "$QuotedScene" in argument_block


def test_launcher_uses_rpc_scene_readiness_helper_before_returning():
    helper = ROOT / "vision_platform" / "coppeliasim_readiness.py"
    launcher = ROOT / "tools" / "vision_lab" / "launch_coppeliasim.ps1"
    source = launcher.read_text(encoding="utf-8")

    assert helper.is_file()
    assert "& $Python -m vision_platform.coppeliasim_readiness" in source
    assert "--host $HostAddress" in source
    assert "--port $Port" in source
    assert "--scene $Scene" in source
    assert "--timeout $TimeoutSeconds" in source
    assert "$ReadyExitCode = $LASTEXITCODE" in source
    assert "readiness failed" in source


def test_launcher_keeps_readiness_text_out_of_structured_return_pipeline():
    source = (
        ROOT / "tools" / "vision_lab" / "launch_coppeliasim.ps1"
    ).read_text(encoding="utf-8")
    compact = " ".join(source.replace("`", "").split())

    assert (
        "$ReadinessOutput = & $Python "
        "-m vision_platform.coppeliasim_readiness"
    ) in compact
    assert "$ReadyExitCode = $LASTEXITCODE" in source
    assert "$ReadinessOutput | Out-Host" in source
    assert "2>&1" not in source


def test_launcher_readiness_failure_cleans_only_exact_started_process():
    source = (
        ROOT / "tools" / "vision_lab" / "launch_coppeliasim.ps1"
    ).read_text(encoding="utf-8")
    compact = " ".join(source.replace("`", "").split())

    assert "$StartedProcess = $null" in source
    assert "if ($null -ne $StartedProcess)" in source
    assert "process_ownership.ps1" in source
    assert "Stop-ExactOwnedProcess" in source
    assert "-ProcessId $LaunchProcessId" in compact
    assert "-ProcessPath $LaunchProcessPath" in compact
    assert (
        "-ProcessStartTimeUtcTicks $LaunchProcessStartTimeUtcTicks"
        in compact
    )
    assert "ProcessStartTimeUtcTicks = $LaunchProcessStartTimeUtcTicks" in source
    assert "$HasCapturedLaunchIdentity" in source
    assert "Stop-StartedProcessObject" in source
    assert "-StartedProcess $StartedProcess" in compact
    assert "Stop-Process -Name" not in source
    assert "Stop-Process -Id" not in source
    assert "taskkill" not in source.lower()


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


def test_student_launcher_finally_cleans_only_its_owned_launch():
    source = (
        ROOT / "tools" / "vision_lab" / "run_student_program.ps1"
    ).read_text(encoding="utf-8")
    compact = " ".join(source.replace("`", "").split())

    assert "$Launch = &" in source
    assert "$Launch.StartedByScript" in source
    assert "$OwnedProcessId = [int]$Launch.ProcessId" in source
    assert "$OwnedProcessPath = [string]$Launch.ProcessPath" in source
    assert (
        "$OwnedProcessStartTimeUtcTicks = "
        "[long]$Launch.ProcessStartTimeUtcTicks"
    ) in compact
    assert "finally {" in source
    assert "process_ownership.ps1" in source
    assert "Stop-ExactOwnedProcess" in source
    assert "-ProcessId $OwnedProcessId" in compact
    assert "-ProcessPath $OwnedProcessPath" in compact
    assert (
        "-ProcessStartTimeUtcTicks $OwnedProcessStartTimeUtcTicks"
        in compact
    )


def test_pyqt_launcher_finally_cleans_only_its_owned_launch():
    source = (
        ROOT / "tools" / "vision_lab" / "run_pyqt.ps1"
    ).read_text(encoding="utf-8")
    compact = " ".join(source.replace("`", "").split())

    assert "$Launch = &" in source
    assert "$Launch.StartedByScript" in source
    assert "$OwnedProcessId = [int]$Launch.ProcessId" in source
    assert "$OwnedProcessPath = [string]$Launch.ProcessPath" in source
    assert (
        "$OwnedProcessStartTimeUtcTicks = "
        "[long]$Launch.ProcessStartTimeUtcTicks"
    ) in compact
    assert "finally {" in source
    assert "process_ownership.ps1" in source
    assert "Stop-ExactOwnedProcess" in source
    assert "-ProcessId $OwnedProcessId" in compact
    assert "-ProcessPath $OwnedProcessPath" in compact
    assert (
        "-ProcessStartTimeUtcTicks $OwnedProcessStartTimeUtcTicks"
        in compact
    )
    assert "Stop-Process -Name" not in source


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


def test_live_student_program_acceptance_has_required_contract():
    test_path = (
        ROOT
        / "tests"
        / "test_acceptance"
        / "test_coppeliasim_student_program.py"
    )

    assert test_path.is_file()
    source = test_path.read_text(encoding="utf-8")
    required = (
        "@pytest.mark.coppeliasim",
        "running_vision_scene",
        "StudentProgramController",
        "StudentExecutionPolicy",
        "student_programs",
        "pick_and_place.py",
            'result.status == "PASS"',
            "critical_commands",
            "len(move_commands) == 6",
            "measured_pose_results",
            'move_commands[5]["args"]',
            "pytest.approx(",
            '"robot.pose"',
        '"/VisionLab/Pickables/object_01_red_square"',
        "application.sim.getObjectPosition",
        'application.scene_spec["zones"]["red"]',
        "controller.wait_for_quiescence",
        "controller.process_is_alive is False",
        '"source.py"',
        '"source.sha256"',
        '"manifest.json"',
        '"summary.json"',
        '"hardware_status"] == "PENDING_HARDWARE"',
        "teaching_ready_pose",
        "dist(actual, expected) <= 2.0",
        "online test must not reset before assertions",
        "finally:",
        "controller.cancel()",
    )
    for item in required:
        assert item in source
    assert "pytest.skip" not in source


def test_acceptance_script_has_explicit_live_student_program_gate():
    source = (
        ROOT / "tools" / "vision_lab" / "run_acceptance.ps1"
    ).read_text(encoding="utf-8")

    required = (
        'Invoke-CheckedPython -Name "student_program_online"',
        '"tests/test_acceptance/test_coppeliasim_student_program.py"',
        '"-m", "coppeliasim"',
        "student-program.xml",
        "Assert-JUnitNoSkips",
        "student_program_online",
    )
    for item in required:
        assert item in source


def test_acceptance_live_pytest_groups_receive_selected_endpoint():
    source = (
        ROOT / "tools" / "vision_lab" / "run_acceptance.ps1"
    ).read_text(encoding="utf-8")

    for step_name in (
        "live_coppeliasim_tests",
        "student_program_online",
    ):
        block = source.split(
            f'Invoke-CheckedPython -Name "{step_name}"',
            1,
        )[1].split(")", 1)[0]
        assert '"--coppelia-host", $HostAddress' in block
        assert '"--coppelia-port", [string]$Port' in block


def test_live_student_acceptance_checks_pick_place_command_semantics():
    source = (
        ROOT
        / "tests"
        / "test_acceptance"
        / "test_coppeliasim_student_program.py"
    ).read_text(encoding="utf-8")

    required = (
        "pick_and_place.py",
        'command["name"]',
        "critical_commands",
        "len(move_commands) == 6",
        "measured_pose_results[0][0]",
        "measured_pose_results[2][1]",
        'move_commands[5]["args"]',
        "pytest.approx(",
        '"robot.pose"',
        '"/VisionLab/Pickables/object_01_red_square"',
        "application.sim.getObjectPosition",
        'application.scene_spec["zones"]["red"]',
        "tolerance_mm = 2.0",
    )
    for item in required:
        assert item in source


def test_student_self_programming_lab_guide_is_complete():
    guide = ROOT / "docs" / "学生自编程实验说明.md"

    assert guide.is_file()
    source = guide.read_text(encoding="utf-8")
    required = (
        "# 学生自编程实验说明",
        "## 学习目标",
        "## 从模板复制程序",
        "## 教学 SDK",
        "ctx.robot.home() -> None",
        "ctx.robot.move_world(x_mm, y_mm, z_mm, *, speed) -> None",
        "ctx.robot.pose() -> tuple[float, float, float]",
        "ctx.tool.on() -> None",
        "ctx.tool.off() -> None",
        "ctx.log(message) -> None",
        "ctx.sleep(seconds) -> None",
        "ctx.checkpoint(label) -> None",
        "mm",
        "safe_z",
        "35 mm",
        "## PyQt 操作流程",
        "打开程序",
        "保存",
        "检查代码",
        "运行",
        "暂停",
        "继续",
        "下一步",
        "停止",
        "复位场景",
        "## CLI 与 PowerShell 操作",
        "Set-Location -LiteralPath",
        "student-validate",
        "run_student_program.ps1",
        "## 运行证据",
        "source.py",
        "source.sha256",
        "manifest.json",
        "commands.jsonl",
        "events.jsonl",
        "summary.json",
        "## 故意失败练习",
        "MAIN_MISSING",
        "IMPORT_NOT_ALLOWED",
        "TARGET_OUT_OF_WORKSPACE",
        "STUDENT_SPEED_INVALID",
        "STUDENT_TOOL_HEIGHT_INVALID",
        "STUDENT_SLEEP_INVALID",
        "## 仿真与真机边界",
        "不是恶意代码安全沙箱",
        "海康相机",
        "PENDING_HARDWARE",
        "## 学生提交清单",
        "## 教师验收清单",
    )
    for item in required:
        assert item in source


def test_student_guide_import_exercise_matches_validator_blocklist():
    source = (
        ROOT / "docs" / "学生自编程实验说明.md"
    ).read_text(encoding="utf-8")
    exercise = source.split("## 故意失败练习", 1)[1].split(
        "## 仿真与真机边界",
        1,
    )[0]

    assert "socket" in BLOCKED_IMPORTS
    assert "os" not in BLOCKED_IMPORTS
    assert "`import socket`" in exercise
    assert "`import os`" not in exercise
    assert "危险模块" in source
    assert "不是恶意代码安全沙箱" in source


def test_task11_file_manifest_matches_actual_online_acceptance_scope():
    plan = (
        ROOT
        / "docs"
        / "superpowers"
        / "plans"
        / "2026-07-30-student-program-runner-v2-plan.md"
    ).read_text(encoding="utf-8")
    task11 = plan.split(
        "### Task 11: 增加真实 CoppeliaSim 在线验收",
        1,
    )[1].split("### Task 12:", 1)[0]

    required = (
        "- Create: `vision_platform/coppeliasim_readiness.py`",
        "- Create: `tools/vision_lab/process_ownership.ps1`",
        "- Create: `tests/test_acceptance/test_coppeliasim_readiness.py`",
        "- Create: `tests/test_acceptance/test_coppeliasim_fixture_helpers.py`",
        "- Create: `tests/test_acceptance/test_coppeliasim_student_program.py`",
        "- Create: `tests/test_acceptance/test_powershell_process_ownership.py`",
        "- Modify: `student_programs/templates/pick_and_place.py`",
        "- Modify: `tests/test_acceptance/test_delivery_contract.py`",
        "- Modify: `tests/test_acceptance/conftest.py`",
        "- Modify: `tests/test_student_programs/test_student_safety.py`",
    )
    for item in required:
        assert item in task11
    assert (
        'git commit -m "test(student): verify live student program workflow"'
        in task11
    )


def test_main_usage_guide_links_student_self_programming_workflow():
    source = (
        ROOT / "docs" / "视觉仿真实训平台使用说明.md"
    ).read_text(encoding="utf-8")

    required = (
        "## 6. 学生自编程",
        "[学生自编程实验说明](学生自编程实验说明.md)",
        "student_programs\\templates\\basic_motion.py",
        "run_student_program.ps1",
        "student-validate",
        "PyQt",
        "artifacts/vision_lab/student-runs",
        "safe_z",
        "PENDING_HARDWARE",
    )
    for item in required:
        assert item in source


def test_v21_docs_define_prelaunched_listener_ownership_contract():
    architecture_docs = (
        ROOT
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-07-30-student-program-runner-v2-design.md",
        ROOT
        / "docs"
        / "superpowers"
        / "plans"
        / "2026-07-30-student-program-runner-v2-plan.md",
        ROOT / "docs" / "视觉仿真实训平台V2-开发端执行Prompt.md",
    )
    for path in architecture_docs:
        source = path.read_text(encoding="utf-8")
        assert "vision_platform/coppeliasim_readiness.py" in source
        assert "launch_coppeliasim.ps1" in source
        assert "StartedByScript" in source
        assert "fixture 不启动或终止 CoppeliaSim" in source


def test_v21_docs_define_hardened_readiness_and_physical_pick_drop_gate():
    architecture_docs = (
        ROOT
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-07-30-student-program-runner-v2-design.md",
        ROOT
        / "docs"
        / "superpowers"
        / "plans"
        / "2026-07-30-student-program-runner-v2-plan.md",
        ROOT / "docs" / "视觉仿真实训平台V2-开发端执行Prompt.md",
    )
    for path in architecture_docs:
        source = path.read_text(encoding="utf-8")
        assert "timeout_s" in source
        assert "READY" in source
        assert "Resolve-Path" in source
        assert "Stop-Process -InputObject" in source
        assert "ProcessStartTimeUtcTicks" in source
        assert "pick_approach_pose" in source
        assert "drop_approach_pose" in source
        assert "object_01_red_square" in source
        assert "red zone" in source


def test_pick_place_template_uses_measured_vertical_transitions():
    source = (
        ROOT / "student_programs" / "templates" / "pick_and_place.py"
    ).read_text(encoding="utf-8")
    compact = " ".join(source.split())

    assert "safe_z = 110" in source
    assert "pick_approach_pose = ctx.robot.pose()" in source
    assert "pick_pose = ctx.robot.pose()" in source
    assert "drop_approach_pose = ctx.robot.pose()" in source
    assert "drop_pose = ctx.robot.pose()" in source
    assert (
        "ctx.robot.move_world( "
        "pick_approach_pose[0], pick_approach_pose[1], "
        "pick[2], speed=8 )"
    ) in compact
    assert (
        "ctx.robot.move_world("
        "pick_pose[0], pick_pose[1], safe_z, speed=12)"
    ) in source
    assert (
        "ctx.robot.move_world( "
        "drop_approach_pose[0], drop_approach_pose[1], "
        "drop[2], speed=8 )"
    ) in compact
    assert (
        "ctx.robot.move_world("
        "drop_pose[0], drop_pose[1], safe_z, speed=12)"
    ) in source


def test_online_guides_launch_coppeliasim_before_direct_pytest():
    guides = (
        ROOT / "docs" / "视觉仿真实训平台使用说明.md",
        ROOT / "docs" / "学生自编程实验说明.md",
    )
    for path in guides:
        source = path.read_text(encoding="utf-8")
        launch_position = source.index("launch_coppeliasim.ps1")
        pytest_position = source.index("-m pytest")
        assert launch_position < pytest_position
        assert "fixture 不会启动或终止 CoppeliaSim" in source


def test_clean_release_whitelist_contains_complete_v21_delivery():
    retained = _retained_release_paths()
    missing = sorted(V21_REQUIRED_RELEASE_PATHS - retained)

    assert not missing, (
        f"RETAINED_FILES.txt is missing {len(missing)} V2.1 paths: "
        + ", ".join(missing)
    )
    assert all(
        (ROOT / path).is_file()
        for path in V21_REQUIRED_RELEASE_PATHS
    )
    assert all((ROOT / path).is_file() for path in retained)


def test_formal_experiment_catalog_resolves_every_delivery_path():
    from vision_platform.experiments.catalog import ExperimentCatalog

    catalog = ExperimentCatalog.load(
        ROOT / "config" / "experiments" / "catalog.json",
        project_root=ROOT,
    )

    assert catalog.ids == (
        "R1-01",
        "R1-02",
        "R1-05",
        "R1-06",
        "R1-07",
    )
    for item in catalog.definitions:
        assert item.scene.is_file()
        assert item.scene_manifest.is_file()
        assert item.student_template.is_file()
        assert item.guide.is_file()
        assert item.hardware_status == "PENDING_HARDWARE"


def test_retained_files_contains_every_first_batch_delivery():
    retained = _retained_release_paths()
    missing = sorted(V22_FIRST_BATCH_REQUIRED_RELEASE_PATHS - retained)

    assert not missing, (
        f"RETAINED_FILES.txt is missing {len(missing)} V2.2 paths: "
        + ", ".join(missing)
    )
    assert all(
        (ROOT / path).is_file()
        for path in V22_FIRST_BATCH_REQUIRED_RELEASE_PATHS
    )
    assert all((ROOT / path).is_file() for path in retained)
    assert not any(path.startswith("artifacts/") for path in retained)


def test_first_batch_plan_release_list_uses_real_digit_label_paths():
    source = (
        ROOT
        / "docs"
        / "superpowers"
        / "plans"
        / "2026-07-31-robot-curriculum-v2-2-first-batch-plan.md"
    ).read_text(encoding="utf-8")
    release_section = source.split(
        "**Step 2: 登记全部新增正式文件**",
        1,
    )[1].split("**Step 3: 更新使用说明和 README**", 1)[0]
    listed_paths = {
        line.strip().replace("\\", "/")
        for line in release_section.splitlines()
    }
    required = {
        f"simulation/logistics_lab/assets/labels/digits/{digit}.png"
        for digit in (1, 2, 3)
    }
    obsolete = {
        f"simulation/logistics_lab/assets/labels/{digit}.png"
        for digit in (1, 2, 3)
    }
    missing = sorted(required - listed_paths)
    obsolete_present = sorted(obsolete & listed_paths)

    assert not missing and not obsolete_present, (
        f"missing real digit paths: {missing}; "
        f"obsolete paths still listed: {obsolete_present}"
    )


def test_first_batch_plan_final_commit_includes_the_plan_itself():
    plan_path = (
        "docs/superpowers/plans/"
        "2026-07-31-robot-curriculum-v2-2-first-batch-plan.md"
    )
    source = (ROOT / plan_path).read_text(encoding="utf-8")
    commit_section = source.split(
        "**Step 10: 提交最终交付**",
        1,
    )[1].split("## 完成定义", 1)[0]

    assert plan_path in commit_section


def test_first_batch_plan_compares_protected_assets_to_v21_baseline():
    source = (
        ROOT
        / "docs"
        / "superpowers"
        / "plans"
        / "2026-07-31-robot-curriculum-v2-2-first-batch-plan.md"
    ).read_text(encoding="utf-8")
    audit_section = source.split(
        "**Step 8: 检查文档、编码、占位符和受保护资产**",
        1,
    )[1].split("**Step 9: 更新自动验收报告**", 1)[0]

    assert (
        "git diff --exit-code "
        "4bc638f50fd590ca700615da46741a86c4146b5f..HEAD --"
    ) in audit_section
    assert "git diff --exit-code --" in audit_section


def test_clean_release_whitelist_excludes_generated_or_personal_data():
    retained = _retained_release_paths()
    forbidden_roots = (
        "artifacts/",
        ".venv-vision/",
        "submissions/",
        "student_submissions/",
        "student_programs/submissions/",
    )
    forbidden_parts = {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "cache",
    }

    for path in retained:
        normalized = path.replace("\\", "/").lower()
        assert not normalized.startswith(forbidden_roots)
        assert forbidden_parts.isdisjoint(Path(normalized).parts)


def test_entry_docs_cover_complete_student_programming_workflow():
    entry_docs = (
        ROOT / "README.md",
        ROOT / "docs" / "视觉仿真实训平台使用说明.md",
    )
    required = (
        "student_programs",
        "templates",
        "basic_motion.py",
        "pick_and_place.py",
        "学生编程",
        "student-validate",
        "run_student_program.ps1",
        "暂停",
        "单步",
        "停止",
        "复位",
        "artifacts/vision_lab/student-runs",
        "不是恶意代码安全沙箱",
        "PENDING_HARDWARE",
    )

    for path in entry_docs:
        source = path.read_text(encoding="utf-8")
        missing = [item for item in required if item not in source]
        assert not missing, f"{path.name} is missing: {missing}"


def test_entry_docs_cover_first_batch_experiment_workflow_and_boundaries():
    entry_docs = (
        ROOT / "README.md",
        ROOT / "docs" / "视觉仿真实训平台使用说明.md",
    )
    required = (
        "tools\\vision_lab\\python.ps1",
        "-m vision_platform.cli experiment-list",
        "tools\\vision_lab\\run_experiment.ps1",
        "-Experiment R1-05",
        "实验目录",
        "先选择实验",
        "终止旧会话",
        "重新加载确定场景",
        "R1-05",
        "R1-06",
        "R1-07",
        "artifacts/vision_lab/experiment-runs",
        "snapshots.jsonl",
        "scene-final.json",
        "终态探针不是正式成绩",
        "教师人工验收",
        "海康 MVS",
        "急停",
        "气路",
        "物理抓取",
        "PENDING_HARDWARE",
    )

    for path in entry_docs:
        source = path.read_text(encoding="utf-8")
        missing = [item for item in required if item not in source]
        assert not missing, f"{path.name} is missing: {missing}"


def test_v21_acceptance_report_preserves_automated_and_manual_boundaries():
    source = (
        ROOT / "docs" / "视觉仿真实训平台自动验收报告.md"
    ).read_text(encoding="utf-8")
    required = (
        "V2.1 Task 12 自动验收增补",
        "artifacts/vision_lab/student-program-v2-final/",
        "624 passed, 3 skipped",
        "2 passed, 0 skipped",
        "1 passed, 0 skipped",
        "21 passed",
        "`update_count` 以 `simui-smoke.json` 为准",
        "39 passed",
        "RETAINED_FILES.txt",
        "人工可见检查：PASS",
        "real Windows Qt window",
        "offscreen: false",
        "manual-ui-evidence/manual-ui-summary.json",
        "08-stopped.png",
        "09-reset.png",
        "PENDING_HUMAN_ACCEPTANCE",
        "PENDING_HARDWARE",
        "一期历史验收记录",
        "不是恶意代码安全沙箱",
    )

    for item in required:
        assert item in source
    assert "待主代理真实检查" not in source
    assert "V2.1 全部完成：PASS" not in source


def test_v22_acceptance_report_records_exact_gates_and_boundaries():
    source = (
        ROOT / "docs" / "视觉仿真实训平台自动验收报告.md"
    ).read_text(encoding="utf-8")
    required = (
        "V2.2 首批课程自动验收增补",
        "4bc638f50fd590ca700615da46741a86c4146b5f",
        "cf45df5049f1f2544434279108bc5d0112d4eaf0",
        "codex/v2-2-first-batch-integration",
        "1174 passed, 10 skipped",
        "1177 passed, 10 skipped",
        "最终静态回归比一键验收内多 3 项",
        "Step 10 自包含提交清单合同",
        "Step 8 基线资产审计合同",
        "pytest-static-final.txt",
        "7 passed, 0 skipped",
        "2 个场景 + 5 个实验",
        "task15-live-junit-verification.json",
        "test_training_scene_loads_path0/robot_basics.json",
        "test_training_scene_loads_path1/logistics_lab.json",
        "20260801-061317-r1_01_robot_basics-b26752ee",
        "20260801-061319-r1_02_teach_points-27b57e2a",
        "20260801-061321-r1_05_visual_stacking-7d4da6e6",
        "20260801-061324-r1_06_digit_sort-61329be8",
        "20260801-061327-r1_07_component_sort-aa8c9db0",
        "场景探针 `PASS`",
        "受保护资产差异路径数为 0",
        "Codex developer-side visible inspection",
        "2026-08-01",
        "artifacts/vision_lab/v2-2-first-batch-final/ui/manual-ui-summary.json",
        "artifacts/vision_lab/v2-2-first-batch-final/ui/scale-100/04-all-five-experiments.png",
        "artifacts/vision_lab/v2-2-first-batch-final/ui/scale-125/05-r1-07-long-name-details.png",
        "PENDING_HUMAN_ACCEPTANCE",
        "海康 MVS",
        "急停",
        "气路",
        "物理抓取",
        "PENDING_HARDWARE",
    )

    for item in required:
        assert item in source
    assert "教师教学效果验收：PASS" not in source
    assert "真机验收：PASS" not in source


def test_delivery_has_formal_experiment_cli_and_powershell_entry():
    script = ROOT / "tools" / "vision_lab" / "run_experiment.ps1"
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "experiment-run" in text
    assert "R1-01" in text
    assert "R1-07" in text
    assert "ValidateSet" in text
    assert "HostName" in text
    assert "python.ps1" in text
    assert "--program" in text
    assert "--host" in text
    assert "--port" in text
    assert "--output" in text
    assert "$LASTEXITCODE" in text
    assert "--robot" not in text
    assert "--scene" not in text

    retained = _retained_release_paths()
    assert "tests/test_experiments/test_cli.py" in retained
    assert "tools/vision_lab/run_experiment.ps1" in retained
