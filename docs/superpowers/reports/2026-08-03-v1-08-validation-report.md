# V1-08 OCR 字符训练、编号识别与机器人分拣验证报告

日期：2026-08-03

工作树：`C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-08-ocr-sorting`

分支：`codex/v2-2-v1-08-ocr-sorting`

精确基线：`5c09695b764fa35915dd5ae486b5f13f9a4d8631`

## 范围与边界

本轮完成 V1-08 Task 5～Task 12，并在安全门禁清零后实施了受控几何布局改版。没有合入 `main`，没有修改 D1/rgbd_sim、`vision_platform/vision2d/**`、`tests/test_vision2d/**`、既有正式 `.ttt`、URDF、STL、网格或机器人资产。V1-08 仍只发布独立场景 `simulation/vision_ocr_sorting_lab/BL23_vision_ocr_sorting_lab.ttt`。

## Task 提交

| Task | 提交号（完整 SHA） | 内容 |
|---|---|---|
| Task 1 | `5c09695b764fa35915dd5ae486b5f13f9a4d8631` | 精确基线、专用 worktree/端口与环境核对 |
| Task 2～4 | `36408a4b22f8b47fc18abe4538417987ba727e4a`、`56cbe81a2f2b9932e959fe9a0636fbe4aacd33f1`、`1cbe4ab05de3f156e3a738ac16459b66cfd5b12a`、`d8bf1945dcbdbde924668fe239985b93e3e1c448`、`d39bcf6a5c3fc4e947b01b45f46bbaaa43f361ae`、`f3828d219842c9c9a772cbd7c461106acc482e27`、`e02ee1617dcbe01955340485b5aa00d7d55147ff`、`472a2ef0087191d11c290de5e4c06ba6074a89ef` | V1-08 设计、资产与独立场景基础及安全校验 |
| Task 5 | `ff6c118b5f91053378c9d7af6b3e968248449f40`、`e6a3174b3fc6f35806f1475886328661779834e2`、`a331bd1e77ab9bc0298b7f91e7e6a954d60673af`、`3849fc7f59f04f0f8c35fd4fc1e1e364c7741e30` | OCR 训练服务、输入与证据合同 |
| Task 6 | `f8e9764bccc60250dda589e517b21aa67a822ee8`、`061d5c3ce950ee0d244208f9e1e5c8abaa87af9c` | 原子分拣 guard 与失效安全 |
| Task 7 | `c9be32cb7fb7ee0b10fdfa7b14ba046d54b0d183`、`693f18145e5beeb89dc0fa7948b91c480e536da6`、`9048678b8ea2a1bd0139013fc860c4c3698a6edb`、`b70f1db6138a612efee4e79c95678ea79671619e` | 受控 OCR 入口、严格 ID、结果校验与停止失效 |
| Task 8 | `8e91e3a9ccb299c547404dc0910c56bef7e8cdaa`、`8cb9e7880a6771cc434be2d6116ed8fc8091d99b`、`58c41d46ae5a64c992789f3e3933f2231cf772a4`、`17e7a92ba9e0a4a1f6cca458d4f7e2885d4c0cc0`、`1c34c8e366e5a4388b4a4724d26ffae2c95404f1`、`94b9ae2dfb252c3009b1831c7edf4f146935a124` | 课程材料、模板、SDK 对齐与 retained 清单 |
| Task 9 | `205ce3d17c233319c7cc93c4d0c9732e65346f08`、`57014b6c88d865a2ed8bd7d8d0a236997f96eba7`、`7d39d8a7241ec663a0406fb4a3a3fe24b94c4192`、`005997d2df81590d0c6cfb732b16eb5531078588`、`c9d1485f2a369a6e909710bb1db82196c7ae77c6`、`ed45774b6e2e6118bd96d991434631341426c2c7` | 同次运行证据、逐条目探针与最终占位 |
| Task 10 | `257677c16917f37cdf92196991fefea5c33488bd` | PyQt 只读 OCR 结果面板与目录标签 |
| Task 11 | `3dd652066579deef9a8de11b15789c5b34eb16e6`、`dc876d284bfd3dd4628f21d40c1b0a2039c33fcb` | 在线闭环、安全抬升修复、独立场景与受控几何布局 |
| Task 12 | `55b8ad954c8821c61d7db132647416fe08716270`、`327dc66116f6087db4db68fe51867abe29c197f7`、`c58ecd989b3c5848c723757a5770a3b7007ba875`、`41ca41c37cd4ff7ae3c479a1ffff34d3b8996b2d`、`b96c0ee56477f7e76530aac4b3b5633bc35f1613`、本报告所在提交 | 独立复审、严格默认门禁、每设备调用暂停/单步门禁、严格置信度响应合同、完整回归、发布合同与交付记录 |

## TDD 与安全门禁证据

- 受控布局 RED：旧场景位置仍为 `(40,-45)/(80,-45)/(40,5)/(80,5)`、槽位为 `y=±60` 时，`test_v1_08_layout_is_expanded_away_from_robot_base` 失败；GREEN 后位置为取料 `y=-55/+25`、槽位 `y=-75/+75`。
- 置信度默认门禁 RED：`test_release_default_confidence_gate_is_strict` 先观测到内部默认值 `0.4` 而预期为 `0.90`；GREEN 后服务和网关缺省也使用 `0.90`。
- 未保留任何固定置信度抬升函数；原始 KNN 分数和 `confidence_method` 直接参与门禁。网关负向测试证明 raw `<0.90` 时返回 `OCR_SORT_RESULT_INVALID`，guard 不激活且 robot/tool 调用数为零。
- runner 在第一次水平动作前执行安全抬升；所有动作仍由私有执行器逐步复用停止、暂停/单步、运动校验、探针和清理流程。
- 本次复审 RED 证据保存在 `artifacts/vision_lab/v1-08-review-repair/red-sdk.txt` 和 `red-runner.txt`：SDK 的 `0.41/0.899999`、抬升后水平移动和暂停工具调用均按预期失败。
- GREEN 后，`test_private_ocr_runner_requires_next_step_between_safety_lift_and_horizontal` 证明一次许可只执行抬升，下一次 `step()` 才执行水平移动；工具 `on/off` 均在各自实际调用前等待许可。SDK 字符级和条目级 `0.41/0.899999/0.90` 边界测试全部通过。
- `test_private_ocr_runner_stop_releases_waiting_call_and_runs_cleanup` 证明停止会立即唤醒等待中的设备调用、使 guard 失效，并执行既有 `tool.off`、安全抬升和回零清理。

## 测试结果

- 本次复审后的可复制 focused 命令（`.venv-vision`，按计划 Task 8 文件集执行，并用 `-rs` 显示 skip 原因）：
  ```powershell
  .\.venv-vision\Scripts\python.exe -m pytest -q -rs `
    tests/test_vision2d/test_ocr.py `
    tests/test_experiments/test_ocr_sorting.py `
    tests/test_experiments/test_ocr_assets.py `
    tests/test_experiments/test_ocr_service.py `
    tests/test_student_programs/test_v1_08_sort_guard.py `
    tests/test_student_programs/test_v1_08_protocol_sdk.py `
    tests/test_student_programs/test_v1_08_gateway.py `
    tests/test_simulation/test_v1_08_ocr_assets.py `
    tests/test_simulation/test_v1_08_scene_contract.py `
    tests/test_vision_quality/test_v1_08_materials.py `
    tests/test_experiments/test_v1_08_cli.py `
    tests/test_experiments/test_v1_08_probe.py `
    tests/test_vision_platform/test_v1_08_result_panel.py `
    tests/test_acceptance/test_delivery_contract.py
  ```
  新鲜结果：`224 passed, 1 skipped`（无 deselected；完整输出保存在 `artifacts/vision_lab/v1-08-review-repair/focused-repair-final.txt`）。唯一 skip 为 `tests/test_experiments/test_ocr_assets.py:202` 的 Windows worker 不支持符号链接；该 skip 不计为在线 PASS。此前无法复现的 `232 passed, 1 skipped, 1 deselected` 口径已删除。
- 完整静态回归（`.venv-vision`）：`2220 passed, 21 skipped`，无失败。静态 skip 仍按 skip 记录，不改写为在线 PASS。
- 发布合同：`47 passed`（`tests/test_acceptance/test_delivery_contract.py`）。
- UI 自动化：`5 passed`。Windows Qt 后端已检查 100% 和 125% 缩放截图，结果摘要、四行 OCR、只读证据、`PENDING_HARDWARE` 均可读。

## CoppeliaSim 在线验收

仅将下面这次由专用 wrapper 启动并显式 `-m coppeliasim` 的运行记为在线证据；未启动监听器时直接运行的失败尝试不计为通过：

- wrapper 摘要：`artifacts/vision_lab/v1-08-online-review-repair/v1-08-online-summary.json`
- JUnit：`artifacts/vision_lab/v1-08-online-review-repair/v1-08-online.xml`
- 真实运行目录：`C:\Users\yqzhe\AppData\Local\Temp\pytest-of-yqzhe\pytest-2594\test_v1_08_online_ocr_sorting_0\runs\20260803-045143-v1_08_ocr_sorting-a614819f`
- 结果：`1 passed, 0 skipped, 0 failures, 0 errors`，端口 `23008`，场景哈希 `3bd1cb20d78be81f5f1358d8601e384a5cd07ab346062d55e120e031c3b993c1`。
- wrapper `launch` 由本次脚本创建并准确回收 PID `7760`；四个字符均 PASS，原始字符级最低置信度约 `0.9082`，训练留出准确率 `1.0`，未做置信度映射；命令记录只有一个 `vision2d.ocr_sorting`、四个严格 `vision2d.ocr_sort_entry` 和 context 证据命令，没有 raw `robot.*`/`tool.*`。
- `scene-final.json`：`matched=4/4`、同次运行证据为真、四槽位占位正确、`robot_home=true`、`tool_off=true`、`safety_violation_count=0`。

## UI 证据

- 100%：`artifacts/vision_lab/ui/v1-08/vision_result_panel_100.png`
- 125%：`artifacts/vision_lab/ui/v1-08/vision_result_panel_125.png`

两张截图均由 Windows Qt 后端生成并实际目视检查；没有把 offscreen 空白图当作视觉通过。

## 受保护资产核对

相对精确基线执行 `git diff --name-only 5c09695b764fa35915dd5ae486b5f13f9a4d8631..HEAD`：无 `simulation/vision_lab/BL23_vision_lab.ttt`、URDF、STL、机器人网格、`vision_platform/vision2d/**` 或 `tests/test_vision2d/**` 变化。旧正式场景哈希保持不变；新场景模板绑定哈希为 `5154dfc3132cd29aafcbe5de83c1a6b61960595681ac5c90d2d92a5b642bda55`。

## 未完成验收与后续建议

- `PENDING_HUMAN_ACCEPTANCE`：教师仍需观察学生是否理解训练/测试划分、字符分割、置信度、白名单、失败关闭和证据含义。
- `PENDING_HARDWARE`：海康相机、真实机械臂、急停、气路、物理抓取和真实光学精度均未验收，仿真 PASS 不替代真机验收。
- 本轮没有 cherry-pick 或合入并行算法分支，也没有实施 V1-02～V1-05 的任何新集成。后续若整合，应先冻结当前 V1-08 的 raw-confidence、ROI、场景哈希和端口合同，再按实验逐项建立 adapter/回归/在线证据；不要把独立二维算法测试结果直接当成 CoppeliaSim 或硬件结果。
