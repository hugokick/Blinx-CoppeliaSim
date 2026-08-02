# V1-06 模板匹配视觉实验实施计划

## 执行规则

- 工作树：`C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-06-template-matching`。
- 分支：`codex/v2-2-v1-06-template-matching`；基线：`9af91087d29cf3186ef84f3528c5f4c495b1ff71`。
- 每个 Task 必须按 RED（先写失败测试并确认失败原因）→GREEN（最小实现）→聚焦回归→独立提交执行。
- 使用目标工作树 `.venv-vision\Scripts\python.exe`；不得把 skip 写成 PASS。
- 不修改 `simulation/vision_lab/BL23_vision_lab.ttt`、`simulation/vision_lab/robot_assets_manifest.json`、任何 URDF/STL/机器人网格或正式机器人资产；不修改 V1-02～V1-05 独占目录/文档/分支。
- 新视觉场景若需引用，只能使用 `simulation/vision_quality_lab/BL23_vision_quality_lab.ttt`。
- 每个新增正式交付文件加入 `RETAINED_FILES.txt`；每个 Task 验证实际执行后才勾选。

## Task 1 — 纯模板匹配内核

先新增 `tests/test_vision2d/test_template_matching.py`，覆盖正匹配、低分未匹配、ROI 坐标换算、确定性并列、输入/尺寸/ROI/非有限分数错误合同；确认测试因缺少模块失败。再新增 `vision_platform/vision2d/template_matching.py`，实现 `TemplateMatchConfig`、`TemplateMatchResult`、`match_template()`、`annotate_template_match()` 和字典序列化，保持现有 V1-02～V1-05 文件不变。

验证：聚焦模板测试全通过；现有 `tests/test_vision2d` 全通过；提交 `test/feat: add deterministic template matching kernel`。

## Task 2 — 可复现模板资产

先写资产/清单测试，确认 `generate_v1_06_template.py` 和资产不存在而失败。再新增确定性生成器、`simulation/vision_quality_lab/templates/v1_06_reference.png`、`manifest.json`，manifest 包含 ID、版本、路径、尺寸、生成器和 SHA-256。连续生成并比较哈希，验证路径在仓库内且内容不是用户采集数据。

验证：资产测试、生成器两次哈希一致、JSON/PNG 可读；提交 `feat: add reproducible V1-06 template asset`。

## Task 3 — SDK 与协议

先为 `vision2d.template_match`、结果字段、`ctx.vision2d.template_match()`/`match_template()` 写失败测试。再在允许命令、协议模型和 student SDK 中添加最小接口；拒绝路径、阈值、ROI 和任意命令注入字段，保持未知字段拒绝/向后兼容规则。

验证：协议/SDK 聚焦测试和现有 SDK 回归；提交 `feat: expose V1-06 template match SDK contract`。

## Task 4 — 能力、网关与回滚安全

先写 capability、gateway、runner 回滚测试，确认新命令尚未路由而失败。再扩展 capability registry、`experiment.info` 安全公开字段、网关固定资产校验、一次采集三层 bundle、错误映射和 runner 的 gateway-only profile 回滚。不得添加机器人运动。

验证：网关/runner/capability 聚焦测试、相关回归；提交 `feat: route template matching through safe gateway`。

## Task 5 — V1-06 配置与课程入口

先写配置、catalog、课程文件、CLI/PowerShell 入口合同测试，确认 V1-06 不存在而失败。再新增 `config/experiments/v1_06.json`、`docs/experiments/V1-06.md`、模板代码、catalog 条目和与现有进程所有权一致的入口；固定专用端口 `23005`，默认只读查询。

验证：配置/课程/CLI/PowerShell 聚焦测试、旧实验回归；提交 `feat: publish V1-06 experiment entry points`。

## Task 6 — 场景清单绑定与发布合同

先写 scene manifest、RETAINED 和 delivery contract 失败测试。再在 `simulation/vision_quality_lab/scene_manifest.json` 增加 template catalog 绑定，补充发布合同和 retained 文件；不改 `.ttt` 和资产 manifest。验证受保护路径相对基线无差异。

验证：scene/delivery contract/保护资产测试；提交 `test: lock V1-06 scene and delivery contract`。

## Task 7 — 显式 CoppeliaSim 在线验收

先扩展在线测试和 PowerShell wrapper，确认缺少 V1-06 在线样本时失败；再使用专用端口 `23005` 启动一次 CoppeliaSim，只记录本次启动且由 ownership token 识别的进程，执行 scene probe、camera profile、template match 和 bundle 检查，结束时仅停止该 PID。环境不可用时保留真实 `PENDING_ONLINE`，不得伪造通过。

验证：在线测试保存真实 JSON/日志；提交 `test: verify V1-06 CoppeliaSim online contract`。

## Task 8 — 结果面板证据

先为模板 ID、分数、框、中心和三层图像写 PyQt/面板测试，确认缺少字段而失败。再以只读方式扩展已有结果面板，生成 100% 与 125% 缩放截图到 `artifacts/vision_lab/v2-2-v1-06/ui-100/` 和 `ui-125/`；无头环境只能如实记录未执行视觉检查。

验证：面板自动化测试、两档截图和尺寸/可读性检查；提交 `feat: show V1-06 template evidence in result panel`。

## Task 9 — 课程包与完整验证

运行 V1-06 聚焦测试、完整静态套件、发布合同、retained 清单、保护资产差异和在线 wrapper；更新证据索引/限制说明，保留 `PENDING_HUMAN_ACCEPTANCE` 与 `PENDING_HARDWARE`。修复仅限本计划边界，每项修复单独验证后提交 `chore: complete V1-06 validation evidence`。

## Task 10 — 独立审查与推送

请求独立代码审查，处理真实阻断项并重新验证。检查工作树干净、各 Task 有独立提交、没有算法分支或受保护资产变更；只在所有门禁通过后推送 `codex/v2-2-v1-06-template-matching`，不合入 `main`。

## 完成报告

最终报告列出 Task 提交号、主要文件、聚焦/完整静态 passed 与 skipped、CoppeliaSim 在线真实状态、100%/125% UI 证据路径、保护资产核对、GitHub 推送状态、人工/硬件待验收项，以及未来与 `f5bef91` 集成 V1-02～V1-05 的建议（本轮不集成）。

