# D1 RGB-D Pure Kernel Validation Report

## Scope and claim boundary

本报告只验证 D1 纯内存 RGB-D 数据合同、深度采样、针孔反投影、刚体变换、JSON 序列化和原创合成测试。CoppeliaSim RGB-D、正式 D1 课程、学生 SDK、PyQt、机器人闭环、真实深度相机和物理硬件均未验收。

## Exact baseline and branch tip

- 权威基线：`origin/main` = `a8d7b98054f0740b65d0d718b34be97c0efaa667`。
- 分支：`codex/v2-2-d1-rgbd-kernel`。
- worktree：`C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-d1-rgbd-kernel`。
- 实现阶段 tip（报告提交前）：`d48ac57b6fd973960b4d4b848cac12ebaef3199f`。
- 本分支只从上述基线增加 D1 RGB-D 内核、D1 专项测试、验证报告和 retained 记录。

## Task commits

1. `b7402fbee6d2da17acacd2c5f62b3e82de5b88f9` — `feat(rgbd): define immutable frame contracts`
2. `5a20d472467a43a2ed49e27b138e8773a20f724f` — `feat(rgbd): add bounded metric depth sampling`
3. `9a132eb0f20b300069160b29170b29eab13c1f5a` — `feat(rgbd): deproject pixels into metric points`
4. `cd4bb350f4ba8e973810d6cb27a92c860922d016` — `feat(rgbd): add explicit rigid point transforms`
5. `52296f8666ddc8db28752bee3ed541e5ffe3f593` — `feat(rgbd): compose serializable 3D measurements`
6. `e32d98a512afe279ed943e7d7b87fada71a0d3ad` — `test(rgbd): prove synthetic 3D measurement contracts`
7. `d48ac57b6fd973960b4d4b848cac12ebaef3199f` — `feat(rgbd): publish the pure kernel API`

## RED and GREEN evidence

- Task 2：先以缺少 `vision_platform.rgbd.models` / `errors` 的 RED 测试启动；实现后模型合同测试 GREEN，累计 `23 passed`。
- Task 3：先以缺少 `sampling` 的 RED 测试启动；实现有界奇数窗口、中位数和结构化 `NO_VALID_DEPTH` 后累计 `36 passed`。
- Task 4：先以缺少 `geometry` 的 RED 测试启动；实现无半像素偏移的针孔反投影后累计 `47 passed`。
- Task 5：先以缺少 `transforms` 的 RED 测试启动；实现有限、正交、`det=+1` 的刚体矩阵校验后几何/变换测试 `20 passed`。
- Task 6：先以缺少测量组合模块的 RED 测试启动；实现相机点、目标点和 JSON 合同后累计 `67 passed`。
- Task 7：先以缺少原创合成工厂的 RED 测试启动；补齐平面、台阶、盒体、倾斜平面、孔洞和确定性噪声后累计 `75 passed`。
- Task 8：先以缺少公开命名空间和交付审计测试的 RED 测试启动；公开 API/交付边界测试 `3 passed`，RGB-D 专项累计 `78 passed`。

## RGB-D focused tests

使用已验证的 Python 环境运行 `tests/test_rgbd`：`78 passed in 0.26s`，`0 failed`，`0 skipped`。

覆盖数据类型和只读合同、尺寸/有限值校验、边界像素、奇数窗口裁剪、零深度排除、中位数采样、无有效深度结构化状态、针孔反投影、显式刚体变换、测量配对、JSON 原生类型、确定性合成工厂、孔洞和全链路序列化。

## Full static regression and skips

- 基线门禁（`origin/main`）：`1839 passed, 18 skipped`，`git diff --check` 通过。
- D1 最终全仓回归：`1917 passed, 18 skipped in 72.29s`。
- 18 个 skip 为仓库既有静态/环境门，不被本分支重新解释为 PASS；本分支未新增 skip。
- 全仓测试过程中 worktree 缺少本地 `.venv-vision` 路径链接，首次运行仅触发 2 个既有 PowerShell 进程测试的环境前置失败；建立指向已验证主线环境的临时 junction 后，同一全仓命令完整通过。该 junction 未纳入 Git、未修改源码，交付前移除。
- `git diff --check` 通过。

## 512x512 CPU timing

在 512x512 合成平面、3x3 深度窗口、CPU-only 条件下完成 1024 次测量和 JSON 序列化：`0.022192s`，低于计划中的 `2.0s` 上限。该数据只代表当前本地环境的纯内存内核微基准，不代表相机、仿真或硬件吞吐。

## Dependency, ownership, and retained-file audit

- D1 生产代码依赖仅为 Python 标准库、NumPy 和 `vision_platform.rgbd` 内部模块；未导入 `cv2`、Open3D、ROS/rclpy、ZMQ、PyQt、socket、subprocess、相机/深度 SDK 或 CoppeliaSim。
- `git diff --name-only origin/main...HEAD` 仅包含 `vision_platform/rgbd/**`、`tests/test_rgbd/**`、本报告和 `RETAINED_FILES.txt`。
- 对 `vision_platform/vision2d`、`vision_platform/student`、`vision_platform/experiments`、`vision_platform/ui`、`config`、`tools`、`simulation` 的 ownership diff 为空。
- `RETAINED_FILES.txt` 已覆盖全部 D1 交付文件；无重复条目、无 `__pycache__`/`.pyc` 条目，报告路径将与本报告同一提交追加。

## Independent review findings and fixes

完成独立 P0/P1/P2 审核：逐文件复核模型、采样、几何、变换、测量、序列化和公开 API；复核异常路径、有限值/矩阵合同、导入边界、所有权 diff、retained 审计和最终测试结果。未发现需要阻断交付的 P0、P1 或 P2 问题，因此没有额外修复提交。

## Remaining limitations

D1 仍是纯内存 RGB-D 软件内核：不包含 RGB-D 相机采集、帧同步、标定文件读写、点云/SLAM、CoppeliaSim 场景、正式课程、学生 SDK、UI/CLI、机器人闭环、真实深度相机或物理硬件验收。上述能力保留为后续集成边界，不能由本报告推断已完成。
