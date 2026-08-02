# V2.2 V1-07 Validation Report

## Scope and claim boundary

本报告只记录 V1-07 在 CoppeliaSim 中的代码识别、白名单路线、受控运动、虚拟仓位和同次运行证据。教学效果保持 PENDING_HUMAN_ACCEPTANCE；海康相机、真实机械臂、急停、气路和物理抓取保持 PENDING_HARDWARE。

仿真通过不代表教学效果、海康相机、真实机械臂、急停、气路或物理抓取验收通过；前者保持 PENDING_HUMAN_ACCEPTANCE，后者保持 PENDING_HARDWARE。

## Exact baseline and branch tip

- 工作树：`C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-07-code-routing`
- 分支：`codex/v2-2-v1-07-code-routing`
- 基线：`origin/main` / `a8d7b98054f0740b65d0d718b34be97c0efaa667`
- Task 11 完成时分支 tip：`908f916af8029359c6d622d34940f0feb3c0e381`
- Task 12 报告提交后会重新记录最终 tip；本报告只在推送前提交。

## Task commits

| Task | 提交 |
| --- | --- |
| 1 | `a190e65` |
| 2 | `1f5a440` |
| 3 | `d612e40` |
| 4 | `8b8b205` |
| 5 | `aa9ad09` |
| 6 | `258c670` |
| 7 | `ba38f58` |
| 8 | `610cede` |
| 10 | `7858a56` |
| 11 | `908f916` |
| 12 | `38f6b38` |

## RED and GREEN evidence

- Task 10 在线测试的静态 RED/聚焦 GREEN 已执行；最终聚焦组合为 `544 passed, 1 skipped`，唯一 skip 是未显式启用的 CoppeliaSim 标记。
- Task 10 显式在线 RED 门禁随后以专用端口执行并 GREEN：`tests/test_acceptance/test_coppeliasim_v1_07.py -m coppeliasim --coppelia-host 127.0.0.1 --coppelia-port 23007`，结果 `1 passed, 0 skipped`。
- Task 11 发布合同 RED：旧目录合同因实际目录已含 `V1-07` 而失败，结果 `46 passed, 1 failed`；失败原因是旧 `V1-06` 终点断言，不是实现运行时错误。
- Task 11 发布合同 GREEN：`48 passed`。
- Task 11 相关回归首次暴露 2 个旧目录顺序断言；更新为追加 `V1-07` 后 GREEN：`1729 passed, 1 skipped`。

## Static regression

- 基线完整静态：`1839 passed, 18 skipped`。
- Task 11 相关静态套件：`1729 passed, 1 skipped`；skip 是显式 CoppeliaSim 在线测试，未计作在线 PASS。
- Task 11/12 最终完整静态套件：`1916 passed, 19 skipped`。
- 发布合同、V1-06 模板交付合同和 vision2d 交付合同：`48 passed`。
- `git diff --check`：通过。

## Explicitly enabled CoppeliaSim acceptance

wrapper 输出：`C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-07-code-routing\artifacts\vision_lab\v2-2-v1-07\acceptance-summary.json`。

- wrapper 总状态：`PASS`。
- `vision-quality-online.xml`：`tests=7, skipped=0, failures=0, errors=0`。
- `v1-07-online.xml`：`tests=1, skipped=0, failures=0, errors=0`。
- `v1_07_scene_load`：`PASS`，加载的正式场景为 `simulation/vision_code_routing_lab/BL23_vision_code_routing_lab.ttt`。
- `v1_01_experiment_run` 至 `v1_07_experiment_run`：全部为 `PASS`。
- V1-07 最终运行证据目录：`C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-07-code-routing\artifacts\vision_lab\v2-2-v1-07\experiment-runs\20260802-125033-v1_07_code_routing-b4db4da7`。
- wrapper 使用专用端口 `23005`；V1-07 场景加载与在线测试使用同一端口和同一进程所有权链。
- wrapper 摘要明确保留 `hardware_status=PENDING_HARDWARE`、`teaching_effect=PENDING_HUMAN_ACCEPTANCE`。

## PyQt 100% and 125% checks

自动化结果摘要：`C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-07-code-routing\artifacts\vision_lab\ui-checks\v1-07\ui-check-summary.json`。

- 100%：`C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-07-code-routing\artifacts\vision_lab\ui-checks\v1-07\v1-07-ui-100.png`
- 125%：`C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-07-code-routing\artifacts\vision_lab\ui-checks\v1-07\v1-07-ui-125.png`
- 摘要中的真实检查：路线文本只读、payload 以普通文本呈现、无裁剪或重叠、错误与验收状态可见，均为 `true`。
- UI 证据仍标记 `PENDING_HUMAN_ACCEPTANCE`，截图属于运行时证据，不加入 Git 发布清单。

## Protected assets and retained files

- 基线审计命令：`git diff --name-only $(git merge-base HEAD origin/main)..HEAD -- "*.urdf" "*.stl" "*.dae" "*.obj" simulation/vision_lab/BL23_vision_lab.ttt simulation/vision_quality_lab/BL23_vision_quality_lab.ttt`。
- 实际结果：`PROTECTED_ASSET_DIFF_EMPTY`。
- `simulation/vision_lab/BL23_vision_lab.ttt`、`simulation/vision_quality_lab/BL23_vision_quality_lab.ttt`、URDF/STL/DAE/OBJ 和正式机器人资产相对 `a8d7b98` 未变化。
- `RETAINED_FILES.txt` 已登记 V1-07 配置、课程、计划/spec、原始代码 PNG、资产清单、profiles、scene manifest/spec、模板、测试和工具；运行时 `artifacts/` 未登记。
- V1-07 发布合同测试确认 V1-07 全部 29 项正式路径存在且 retained；没有重复 retained 项或缓存/个人数据路径。

## Process ownership and cleanup

- 在线 wrapper 只停止本次启动并由 ownership token 识别的 CoppeliaSim 进程；未按进程名批量结束。
- 最终清理复核：无 `coppelia`/`python` 残留进程；专用端口 `23005`、`23007` 无监听。
- 普通静态测试中的 CoppeliaSim skip 未写成在线 PASS；在线 JUnit 是显式启用并保存的真实输出。

## Independent review findings and fixes

独立审查按设计文档检查了 payload 惰性、计划原子性、运动前门禁、route guard 绕过、错误优先级、cleanup、场景/资产 provenance、在线进程所有权和同次运行证据链。审查结论为 `P0=0, P1=0, P2=1, P3=0`；未发现 P0/P1，且审查代理建议接受本轮实现。

唯一 P2 是 `docs/experiments/V1-07.md` 错误解释列出了未由实现发出的旧错误码。已按 RED→GREEN 修复为实际接口错误码：`CODE_ROUTE_RECOGNITION_INVALID`、`CODE_ROUTE_PLAN_INVALID`、`CODE_ROUTE_SEQUENCE_INVALID`、`TARGET_OUT_OF_WORKSPACE` 和 `STUDENT_TOOL_HEIGHT_INVALID`；新增材料测试先以缺失实际错误码失败，随后 `50 passed` 通过。PNG provenance 观察与批准计划参考实现一致，不作为缺陷。

## Remaining limitations

- 教学效果：`PENDING_HUMAN_ACCEPTANCE`，尚未以教师和学生现场观察替代自动化证据。
- 海康相机、真实机械臂、急停、气路、物理抓取和真实光学精度：`PENDING_HARDWARE`；本机未连接海康相机，本轮不宣称真机通过。
- 不合入、不 cherry-pick 并行算法分支 `f5bef91`；V1-02～V1-05 后续集成应单独执行协议兼容性、场景证据、静态回归和在线验收门禁。
