# D1-01 CoppeliaSim RGB-D Simulation Foundation Validation

## Scope and claim boundary

本报告只验证独立的 D1-01 CoppeliaSim RGB-D 仿真底座：独立场景、一次显式处理的 RGB-D 采集、source-depth 模型观测、optical-Z 归一化、内参、预览、ROI/深度顺序证据、受限 probe CLI 和真实在线探针。

本报告不宣称正式 D1-01 学生课程、D1-02/D1-03、学生 SDK、实验 gateway/catalog、UI、正式课程 CLI、机器人动作、机械臂、真实深度相机、急停、气路、抓取或物理硬件通过。教学保持 `PENDING_HUMAN_ACCEPTANCE`；硬件保持 `PENDING_HARDWARE`。

## Exact baseline, branch and worktree

- 权威基线：`origin/main = 5c09695b764fa35915dd5ae486b5f13f9a4d8631`。
- 分支：`codex/v2-2-d1-01-rgbd-sim-foundation`。
- worktree：`C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-d1-01-rgbd-sim-foundation`。
- 本轮修复代码验证 tip（文档更新前）：`b41aaf5d87e954dc827a2fa08d96a31f3f272635`。
- 固定 D1-01 端口：`23009`；没有占用或清理 V1-08 的 `23008`。
- `.venv-vision` 是指向已验证主线环境的 worktree-local junction，不在 Git 中。

## Task commits

1. `4bc0dd5a8ae9c98f1d8f2d49be014bef2126e037` — `feat(rgbd-sim): define capture and intrinsics contracts`
2. `e601a815887827a49f239095156cece57bfc7332` — `feat(rgbd-sim): capture aligned metric RGB-D frames`
3. `1c067badc6323f0d5a21a494cfd9280a6fa3be26` — `feat(rgbd-sim): add independent RGB-D lab scene`
4. `808b43046414be1e7a05e72726ec335f48e8f963` — `feat(rgbd-sim): render deterministic depth previews`
5. `8ee1e4a8b5f0479a9a92338f6a6a764b501f9ec1` — `feat(rgbd-sim): bind captures to scene evidence`
6. `b336dca755b895e4c65b742a1b2d063395342406` — `feat(rgbd-sim): add bounded RGB-D probe tool`
7. `60d16f0a9cc43ab044651b706b7d7e6cc98f9f2f` — `test(rgbd-sim): verify online metric RGB-D capture`
8. `75a4f51f5ebe6e3bc5916a6510eeb677cd915584` — `fix(rgbd-sim): tighten manifest and process review contracts`
9. `8c593b8d96277109843f3b04c64bb4e0811bf384` — `docs(rgbd-sim): publish foundation validation report`
10. `418012b7430df6b22c2b1972f308ebfeb05c3204` — `fix(rgbd-sim): bind source-depth observations to measurements`
11. `7f66e7b30782a0ac5dc3832e7a2209a32187d7d8` — `fix(rgbd-sim): verify owned listener before scene build`
12. `92e857d0cbf161d0c91177f25d558d14f9c6cdbe` — `fix(rgbd-sim): enforce template and sensor manifest contracts`
13. `8b0062a6164dc86f35599affa83d7d93e375568f` — `chore(rgbd-sim): retain scene contract and listener tests`
14. `b41aaf5d87e954dc827a2fa08d96a31f3f272635` — `test(rgbd-sim): cover template binding in online tamper case`

## TDD RED/GREEN evidence

- Task 2：模型/内参测试先因 `vision_platform.rgbd_sim` 模块缺失 RED；实现后相关模型、内参及既有 RGB-D 回归 `70 passed`。
- Task 3：capture/depth-model 测试先因模块缺失 RED；实现一次 `getExplicitHandling==1`、一次 `handleVisionSensor`、原始米制 source-depth、模型观测和 optical-Z 归一化后相关测试 `25 passed`。
- Task 4：场景合同先因 `tools.rgbd_lab` 缺失 RED；构建器初次真实运行暴露 CoppeliaSim 没有 `sim.visionintparam_explicit_handling`，改用 `sim.setExplicitHandling(handle, 1)` 后重建成功，场景/交付回归 `55 passed`。
- Task 5：预览测试先因 preview 模块缺失 RED；GREEN 后 `11 passed`，覆盖复制 BGR、零深度洋红、全零/单值/窄范围/极值、输入不变和 JSON-native 摘要。
- Task 6：scene binding/probe 测试先因模块缺失 RED；严格 manifest/ROI/hash/路径校验及 probe 后 `20 passed, 1 skipped`。该 skip 是 Windows 无 symlink 特权，不写作 PASS。
- Task 7：CLI 测试先因模块缺失 RED；GREEN 后 `5 passed`，覆盖越界/仓内输出、固定端口、非数值、覆盖保护、错误 JSON 和成功三个输出文件。
- Task 8：未加 marker 的在线文件明确 `3 skipped`，消息写明 skip 不等于在线 PASS。首次真实 opt-in 运行为 `1 failed, 2 passed`：发现 RGB/depth ignored 参数在本机 CoppeliaSim 返回 `None`；新增回归后仅在已取得两个 buffer 的同一事务中将该“旧参数不支持”映射为 enabled。第二次运行发现原 anchor 像素落入目标投影；按真实深度连通域改正 anchor/ROI 几何，而非放宽容差。最终真实运行 `3 passed`。
- Task 9 独立复审先新增三项 manifest 安全断言，旧实现真实 `3 failed`；GREEN 后 `10 passed, 1 skipped`，补齐 template 额外键、固定传感器路径和非有限 anchor position 拒绝，并收紧在线进程监听 PID 必须等于自身 PID。
- 本轮 P1-1：新增 source-depth 观测证明测试在旧实现上真实 `4 failed, 4 passed`；GREEN 后深度模型 `8 passed`，含相关 probe/CLI 回归 `15 passed`。证明绑定 scene SHA、source digest、sequence、anchor digest，并拒绝裸字符串、复制/伪造对象、跨帧/跨 source/跨 anchor 重放和 NaN/Inf 误差。
- 本轮 P1-2：监听身份替换测试在旧 builder 合同上真实 `3 failed, 3 passed`；GREEN 后 `test_scene_builder.py` `6 passed`。PowerShell 在 readiness 之后、builder 之前重新核对 listener PID、可执行路径和 UTC 启动时间；替换 listener 场景实际以 identity mismatch 失败，清理复用精确归属合同。
- 本轮 P1-3/P2：严格 template/sensor manifest 负向测试在旧实现上真实 `8 failed, 10 passed, 1 skipped`；GREEN 后 `test_scene_binding.py` `18 passed, 1 skipped`。template 路径、存在性、SHA、传感器有限位姿及单位四元数均绑定校验；symlink skip 保持单独计数。

## Focused, online and full results

- `tests/test_rgbd_sim`（默认静态）：`87 passed, 4 skipped`。
- `tests/test_rgbd`、`tests/test_vision_platform/test_coppeliasim_camera.py`：`108 passed`；`tests/test_acceptance/test_delivery_contract.py`：`46 passed`。
- 实际在线命令：`pytest -q -s -m coppeliasim tests/test_rgbd_sim/test_coppeliasim_rgbd_online.py` → `3 passed in 11.99s`。在线进程只启动并终止自身 PID，清理后 `23009` 无监听。
- 全仓静态：`2108 passed, 23 skipped in 77.01s`。
- 全仓 23 个 skip 中，19 个为基线已有环境/静态门；本分支新增 3 个未启用在线测试 skip 和 1 个 Windows symlink 权限 skip。所有 skip 均单独计数，未解释为 PASS。
- `git diff --check`：通过。
- 512×512 CPU 纯内存测量/序列化回归：`0.022522s`，低于计划的 `2.0s` 上限；`tests/test_rgbd/test_performance.py` `1 passed`。不代表仿真、相机或硬件吞吐。

## Real CoppeliaSim evidence

- CoppeliaSim Remote API `sim.intparam_program_version=41000`（本机 API 返回的原始版本值）；`23009` 专用监听。
- 场景：`simulation/rgbd_lab/BL23_rgbd_lab.ttt`。
- 最终场景 SHA-256：`dd7e9957fc34a463f9602f1796ea8d5e302a0ed89e0593319aaef355e75e3369`。
- 受保护模板 SHA-256：`5154dfc3132cd29aafcbe5de83c1a6b61960595681ac5c90d2d92a5b642bda55`。
- 传感器：`/RgbdLab/CameraRig/RgbdSensor`，`256×256`，60°，near/far `0.05/5.0 m`，显式处理 `1`，perspective/RGB/depth 合同通过。
- 三个真实采集周期均恰好一次 `handleVisionSensor`，颜色/source-depth 尺寸一致，source-depth 为有限非负 `float32` 米制数组，sequence 为 `0,1,2`，内参稳定。
- source model：声明 `optical_z`，观测 `optical_z`；anchor 最大 optical 误差 `2.384185791015625e-06 m`，ray-range 误差 `0.3130023841857912 m`，因此没有把 ray-range 静默当作 optical-Z。
- optical-Z 输出：`output_depth_model=optical_z`；一周期摘要 `valid=65536`、`invalid=0`、`min=1.6999925374984741 m`、`median=2.499997615814209 m`、`max=2.499997615814209 m`。
- ROI 顺序：`near_block=1.6999925374984741 m < far_block=2.1000022888183594 m`；`step_high=1.7999969720840454 m < step_low=2.1999993324279785 m`，均超过配置 `0.05 m` margin。

## Ownership, dependency and retained audit

- 允许新增/修改仅在 `vision_platform/rgbd_sim/**`、`tests/test_rgbd_sim/**`、`simulation/rgbd_lab/**`、`tools/rgbd_lab/**`、本报告、`.gitattributes` 和 `RETAINED_FILES.txt`。
- `vision_platform/rgbd/**`、`vision_platform/cameras/coppeliasim.py`、V1-07/V1-08、正式场景、student/experiments/UI/SDK、机器人资产和依赖文件均未修改。
- sim adapter 只复用只读的 `CoppeliaClientResolver`；OpenCV 只在 preview/CLI 边界使用；测试进程辅助中的 `socket/subprocess` 不进入共享 RGB-D 算法内核。
- 集成树 `RETAINED_FILES.txt` 审计：`530 raw lines`（2 comment/blank + 528 valid paths）、`528 valid unique`、`0 missing`；`origin/main` 为 `497 raw lines`（2 + 495 valid paths），主线条目全部保留。已补入的 D1-01 合同测试/脚本均在清单内；本分支未新增 runtime artifact 路径，现有的 4 条 `simulation/vision_lab/evidence/**` 条目均已存在于 `origin/main`，未被本分支新增或修改。
- ownership/dependency diff 审计通过：相对精确基线的修改均落在 D1-01 owned 路径；无 `vision_platform/rgbd`、`vision_platform/cameras/coppeliasim.py`、V1-07/V1-08、正式机器人资产、student/experiments/UI/SDK 或依赖文件改动。
- 未注册 `config/experiments/D1-01.json`，未接入正式课程、SDK/UI 或机器人动作。

## 新主线受控集成验证

本节记录 V1-08 已进入新主线后的 D1-01 受控集成，不替代上文原 D1-01 独立开发基线证据。

- 新主线基线：`origin/main = 1c11302811a070d62c4d440a012a0399027a1683`。
- 原 D1-01 输入：`origin/codex/v2-2-d1-01-rgbd-sim-foundation = dd17f74a6bc34bc4a90dc9202692a5ba69f239d2`；共同基线仍为 `5c09695b764fa35915dd5ae486b5f13f9a4d8631`。
- 集成分支/worktree：`codex/v2-2-d1-01-main-integration` / `C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-d1-01-main-integration`。
- 修复前候选 tip：`9ab5655bd40ac6279726ae02658738b811c926da`；本轮开发端修复提交依次为 `4de7e37`（观测证明门）、`762a334`（CLI 有界客户端超时）、`3537f29`（替换监听者下清理自有进程）、`c820187`（真实 RemoteAPIClient 生命周期与 ZMQ 超时）；本报告提交前 tip 为 `c820187`。
- 可复制重放命令：`git fetch origin --prune`；`git worktree add C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-d1-01-main-integration -b codex/v2-2-d1-01-main-integration origin/main`；按 `git rev-list --reverse 5c09695b764fa35915dd5ae486b5f13f9a4d8631..dd17f74a6bc34bc4a90dc9202692a5ba69f239d2` 输出的顺序逐个 `git cherry-pick`，共 17 个提交。
- 冲突记录：仅 `.gitattributes` 发生内容冲突；已并集保留 V1-08 的 4 条 OCR 规则与 D1-01 的 `simulation/rgbd_lab/*.json` 规则。`RETAINED_FILES.txt` 无冲突，未删除、重排或覆盖任一端条目。
- 保护范围审计：相对新 `origin/main` 的差异仅为 D1-01 owned 路径、`.gitattributes`、`RETAINED_FILES.txt` 和本报告；V1-08、student/experiments/UI/SDK、正式 `.ttt`、URDF/STL/网格/机器人资产及 `vision_platform/rgbd/**`、`vision_platform/cameras/coppeliasim.py` 均无差异。
- 本轮定向修复 RED/GREEN：旧实现观测门回归真实 `3 failed`，替换监听者清理回归真实失败；本次新增真实 RemoteAPIClient 无服务端口子进程回归也真实失败（2 秒内未正常退出）。GREEN 后聚焦 `48 passed`：wrapper 以整数毫秒设置 send/receive timeout、`IMMEDIATE=1`、`LINGER=0`，绕过外部客户端的浮点初连路径，并在 `require` 失败/正常关闭时幂等清理 socket/context；错误 JSON 仍有界。
- 集成专项：`pytest -q -rs tests/test_rgbd_sim` → `95 passed, 4 skipped`；`pytest -q tests/test_rgbd tests/test_vision_platform/test_coppeliasim_camera.py` → `108 passed`；`pytest -q tests/test_acceptance/test_delivery_contract.py` → `47 passed`。
- 全仓静态：`pytest -q` → `2315 passed, 25 skipped in 91.06s`。D1-01 默认在线测试 3 项保持 skip，另有 1 项 Windows symlink 权限 skip；其余 skip 为既有环境/静态门，均单独计数，未写作 PASS。
- 显式在线：先确认 `23008`、`23009` 均无 listener；`pytest -q -s -m coppeliasim tests/test_rgbd_sim/test_coppeliasim_rgbd_online.py` → `3 passed, 0 skipped in 15.87s`；固定端口 `23009`，进程精确归属并清理，结束复核 `23008`、`23009` 均无 listener。在线合同验证 source model=`optical_z`、显式处理和 optical-Z 输出。
- 真实 CLI 成功路径：在自有 CoppeliaSim 进程/23009 上运行 `run_rgbd_probe`（`--timeout-s 5 --overwrite`）返回 `0`，`report.json` 为 `PASS`，生成 `rgb.png`、`depth.png`、`report.json`；退出后 `23008`、`23009` 均无 listener。
- 集成清单审计原始输出：`RETAINED_FILES.txt` 共 `530 raw lines`（2 comment/blank + 528 valid paths）、`528 valid unique`、`0 missing`；`origin/main` 为 `497 raw lines`（2 + 495 valid paths），所有主线条目保留。`git diff --check` 通过；相对 `origin/main` 的 changed paths `35`、unexpected `0`、protected `0`。
- 开发端已完成定向修复与自测，协调端最终独立复审待定。

## Review conclusion and remaining PENDING items

开发端已完成本轮 P1-1 source-depth 观测门、P1-2 CLI 有界超时、P1-3 replacement-listener 清理及本次真实 RemoteAPIClient 生命周期定向修复与自测；协调端最终独立复审待定。现有测试/在线证据仅覆盖“仿真 RGB-D source capture → source model 观测 → optical-Z frame → ROI/JSON evidence”范围；真实在线结果不能外推为真实深度相机精度、课程教学效果、机器人闭环或硬件验收。

- Teaching: `PENDING_HUMAN_ACCEPTANCE`。
- Hardware: `PENDING_HARDWARE`（真实深度相机、机械臂、急停、气路、抓取均未验收）。
