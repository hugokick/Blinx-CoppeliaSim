# D1-01 CoppeliaSim RGB-D Simulation Foundation Validation

## Scope and claim boundary

本报告只验证独立的 D1-01 CoppeliaSim RGB-D 仿真底座：独立场景、一次显式处理的 RGB-D 采集、source-depth 模型观测、optical-Z 归一化、内参、预览、ROI/深度顺序证据、受限 probe CLI 和真实在线探针。

本报告不宣称正式 D1-01 学生课程、D1-02/D1-03、学生 SDK、实验 gateway/catalog、UI、正式课程 CLI、机器人动作、机械臂、真实深度相机、急停、气路、抓取或物理硬件通过。教学保持 `PENDING_HUMAN_ACCEPTANCE`；硬件保持 `PENDING_HARDWARE`。

## Exact baseline, branch and worktree

- 权威基线：`origin/main = 5c09695b764fa35915dd5ae486b5f13f9a4d8631`。
- 分支：`codex/v2-2-d1-01-rgbd-sim-foundation`。
- worktree：`C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-d1-01-rgbd-sim-foundation`。
- 报告提交前 tip：`75a4f51f5ebe6e3bc5916a6510eeb677cd915584`。
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

## TDD RED/GREEN evidence

- Task 2：模型/内参测试先因 `vision_platform.rgbd_sim` 模块缺失 RED；实现后相关模型、内参及既有 RGB-D 回归 `70 passed`。
- Task 3：capture/depth-model 测试先因模块缺失 RED；实现一次 `getExplicitHandling==1`、一次 `handleVisionSensor`、原始米制 source-depth、模型观测和 optical-Z 归一化后相关测试 `25 passed`。
- Task 4：场景合同先因 `tools.rgbd_lab` 缺失 RED；构建器初次真实运行暴露 CoppeliaSim 没有 `sim.visionintparam_explicit_handling`，改用 `sim.setExplicitHandling(handle, 1)` 后重建成功，场景/交付回归 `55 passed`。
- Task 5：预览测试先因 preview 模块缺失 RED；GREEN 后 `11 passed`，覆盖复制 BGR、零深度洋红、全零/单值/窄范围/极值、输入不变和 JSON-native 摘要。
- Task 6：scene binding/probe 测试先因模块缺失 RED；严格 manifest/ROI/hash/路径校验及 probe 后 `20 passed, 1 skipped`。该 skip 是 Windows 无 symlink 特权，不写作 PASS。
- Task 7：CLI 测试先因模块缺失 RED；GREEN 后 `5 passed`，覆盖越界/仓内输出、固定端口、非数值、覆盖保护、错误 JSON 和成功三个输出文件。
- Task 8：未加 marker 的在线文件明确 `3 skipped`，消息写明 skip 不等于在线 PASS。首次真实 opt-in 运行为 `1 failed, 2 passed`：发现 RGB/depth ignored 参数在本机 CoppeliaSim 返回 `None`；新增回归后仅在已取得两个 buffer 的同一事务中将该“旧参数不支持”映射为 enabled。第二次运行发现原 anchor 像素落入目标投影；按真实深度连通域改正 anchor/ROI 几何，而非放宽容差。最终真实运行 `3 passed`。
- Task 9 独立复审先新增三项 manifest 安全断言，旧实现真实 `3 failed`；GREEN 后 `10 passed, 1 skipped`，补齐 template 额外键、固定传感器路径和非有限 anchor position 拒绝，并收紧在线进程监听 PID 必须等于自身 PID。

## Focused, online and full results

- `tests/test_rgbd_sim`（默认静态）：`69 passed, 4 skipped`。
- `tests/test_rgbd`、`tests/test_vision_platform/test_coppeliasim_camera.py`、`tests/test_acceptance/test_delivery_contract.py`：`154 passed`。
- 实际在线命令：`pytest -q -s -m coppeliasim tests/test_rgbd_sim/test_coppeliasim_rgbd_online.py` → `3 passed in 11.68s`。在线进程只启动并终止自身 PID，清理后 `23009` 无监听。
- 全仓静态：`2093 passed, 23 skipped in 76.24s`。
- 全仓 23 个 skip 中，19 个为基线已有环境/静态门；本分支新增 3 个未启用在线测试 skip 和 1 个 Windows symlink 权限 skip。所有 skip 均单独计数，未解释为 PASS。
- `git diff --check`：通过。
- 512×512 CPU 纯内存测量/序列化回归：`0.023041s`，低于计划的 `2.0s` 上限；不代表仿真、相机或硬件吞吐。

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
- `RETAINED_FILES.txt` 当前 442 条、无重复；所有 D1-01 正式新增文件已列入。本分支未新增 runtime artifact 路径；现有的 4 条 `simulation/vision_lab/evidence/**` 条目均已存在于 `origin/main`，未被本分支新增或修改。
- 未注册 `config/experiments/D1-01.json`，未接入正式课程、SDK/UI 或机器人动作。

## Review conclusion and remaining PENDING items

独立复审未发现尚未修复的 P0/P1/P2。修复后的底座已验证“仿真 RGB-D source capture → source model 观测 → optical-Z frame → ROI/JSON evidence”这一范围；真实在线结果不能外推为真实深度相机精度、课程教学效果、机器人闭环或硬件验收。

- Teaching: `PENDING_HUMAN_ACCEPTANCE`。
- Hardware: `PENDING_HARDWARE`（真实深度相机、机械臂、急停、气路、抓取均未验收）。
