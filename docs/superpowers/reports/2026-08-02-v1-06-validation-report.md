# V1-06 模板匹配视觉实验验证记录

日期：2026-08-02  
工作树：`C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-06-template-matching`  
分支：`codex/v2-2-v1-06-template-matching`  
基线：`9af91087d29cf3186ef84f3528c5f4c495b1ff71`

## Task 提交

| Task | 提交 |
| --- | --- |
| 1 纯模板匹配内核 | `f5fe1a4` |
| 2 可复现模板资产 | `ff69625` |
| 3 SDK 与协议 | `dbf0598` |
| 4 能力、网关与回滚安全 | `a85fc46` |
| 5 V1-06 配置与课程入口 | `be6dfca` |
| 6 场景清单与发布合同 | `3100c5a`, `78d26a0` |
| 7 CoppeliaSim 在线验收 | `606f910` |
| 8 结果面板证据 | `96d6455` |
| 9 课程包与完整验证 | `ef8c1f1`（本记录） |
| 10 独立审查与推送 | 审查与推送门禁，无代码提交 |

## 自动化验证

- V1-06 及相关回归聚焦套件：`408 passed, 0 skipped`，记录于 `artifacts/vision_lab/v2-2-v1-06/focused-final.txt`。
- 完整静态套件：`1795 passed, 18 skipped`，记录于 `artifacts/vision_lab/v2-2-v1-06/full-static-final.txt`。静态套件中的 CoppeliaSim skip 未计为在线通过。
- 发布合同、RETAINED 清单、场景绑定和保护资产检查均包含在聚焦套件中；RETAINED 清单共 348 项，无重复、缺失项或 `artifacts/` 项。

## 显式 CoppeliaSim 在线验收

最终 wrapper 输出目录：

`C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-06-template-matching\artifacts\vision_lab\v2-2-v1-06-final-online`

真实在线结果：`7 passed, 7 deselected`；V1-01～V1-06 的 scene probe 和实验运行均为 `PASS`。V1-06 证据包位于：

`C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-06-template-matching\artifacts\vision_lab\v2-2-v1-06-final-online\experiment-runs\20260802-041238-v1_06_template_matching-9b87bc64`

V1-06 实际结果：`matched=true`，`score=0.9767017960548401`，`threshold=0.72`，ROI 为 `[0, 0, 512, 512]`，bundle 含 `raw`、`template`、`annotated` 三层，`scene_probe_status=PASS`。专用端口 `23005` 在 wrapper 清理后无监听进程；只使用本次启动并由 ownership token 识别的 CoppeliaSim 进程。

早期校准运行曾因模板与场景图案不一致得到 `score=0.241850...`；该运行未计为最终通过，模板资产随后按场景绑定重新生成，并以本记录中的最终在线证据复验通过。

## PyQt 结果面板证据

自动化截图测试在逻辑尺寸 `1180×760` 下分别以 DPR 1.0 和 1.25 通过：

- 100%：`C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-06-template-matching\artifacts\vision_lab\v2-2-v1-06\ui-100\vision-result-annotated.png`
- 125%：`C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-06-template-matching\artifacts\vision_lab\v2-2-v1-06\ui-125\vision-result-annotated.png`

两张图均来自上述真实 V1-06 证据包，显示三层选择器、模板匹配标注和只读结果区域；结果面板继续明确显示 `PENDING_HARDWARE`，不提供课程评分。

## 保护边界与未完成门禁

- `simulation/vision_lab/BL23_vision_lab.ttt`、`simulation/vision_lab/robot_assets_manifest.json` 与基线无差异。
- 所有 URDF、STL、OBJ 及机器人网格路径与基线无差异。
- 受保护质量场景 SHA-256：`ce4c0189bb486caa2bfbd557d0b5148775d846a80e07983423f1b29d8d9eb310`，与预期一致。
- 教学效果仍为 `PENDING_HUMAN_ACCEPTANCE`。
- 海康相机、真实机械臂、急停、气路、物理抓取和真实光学精度仍为 `PENDING_HARDWARE`。
- 本轮未修改、复制、合并或 cherry-pick `f5bef91` 的 V1-02～V1-05 算法端内容；后续应在独立集成任务中以协议兼容性、场景证据和回归测试为门禁，再决定是否接入。
