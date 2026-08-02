# D1 RGB-D Pure Kernel Validation Report

## Scope and claim boundary

本报告只验证 D1 纯内存 RGB-D 数据合同、深度采样、针孔反投影、刚体变换、JSON 序列化和原创合成测试。CoppeliaSim RGB-D、正式 D1 课程、学生 SDK、PyQt、机器人闭环、真实深度相机和物理硬件均未验收。

## Exact baseline and branch tip

- 权威基线：`origin/main` = `a8d7b98054f0740b65d0d718b34be97c0efaa667`。
- 分支：`codex/v2-2-d1-rgbd-kernel`。
- worktree：`C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-d1-rgbd-kernel`。
- 原始交付 tip：`d48ac57b6fd973960b4d4b848cac12ebaef3199f`。
- 本次返修验证 tip（报告提交前）：`cdbc86cdd4442cb9aaf1fff24a658ccf27244e7a`。
- 本分支只从上述基线增加 D1 RGB-D 内核、D1 专项测试、验证报告和 retained 记录。

## Task commits

1. `b7402fbee6d2da17acacd2c5f62b3e82de5b88f9` — `feat(rgbd): define immutable frame contracts`
2. `5a20d472467a43a2ed49e27b138e8773a20f724f` — `feat(rgbd): add bounded metric depth sampling`
3. `9a132eb0f20b300069160b29170b29eab13c1f5a` — `feat(rgbd): deproject pixels into metric points`
4. `cd4bb350f4ba8e973810d6cb27a92c860922d016` — `feat(rgbd): add explicit rigid point transforms`
5. `52296f8666ddc8db28752bee3ed541e5ffe3f593` — `feat(rgbd): compose serializable 3D measurements`
6. `e32d98a512afe279ed943e7d7b87fada71a0d3ad` — `test(rgbd): prove synthetic 3D measurement contracts`
7. `d48ac57b6fd973960b4d4b848cac12ebaef3199f` — `feat(rgbd): publish the pure kernel API`
8. `b15a5b44509c92da4c4f501a2ec2994e6b0392d3` — `test(rgbd): add D1 review evidence regressions`
9. `cdbc86cdd4442cb9aaf1fff24a658ccf27244e7a` — `fix(rgbd): seal immutable array backing`

## RED and GREEN evidence

- Task 2：先以缺少 `vision_platform.rgbd.models` / `errors` 的 RED 测试启动；实现后模型合同测试 GREEN，累计 `23 passed`。
- Task 3：先以缺少 `sampling` 的 RED 测试启动；实现有界奇数窗口、中位数和结构化 `NO_VALID_DEPTH` 后累计 `36 passed`。
- Task 4：先以缺少 `geometry` 的 RED 测试启动；实现无半像素偏移的针孔反投影后累计 `47 passed`。
- Task 5：先以缺少 `transforms` 的 RED 测试启动；实现有限、正交、`det=+1` 的刚体矩阵校验后几何/变换测试 `20 passed`。
- Task 6：先以缺少测量组合模块的 RED 测试启动；实现相机点、目标点和 JSON 合同后累计 `67 passed`。
- Task 7：先以缺少原创合成工厂的 RED 测试启动；补齐平面、台阶、盒体、倾斜平面、孔洞和确定性噪声后累计 `75 passed`。
- Task 8：先以缺少公开命名空间和交付审计测试的 RED 测试启动；公开 API/交付边界测试 `3 passed`，RGB-D 专项累计 `78 passed`。
- 本次 P2 特征证据：新增的 X/Y 旋转、组合变换、角点反投影、采样边界、平面/方块 3D 解析值等测试均直接 PASS；未伪造 RED。
- 本次 P2 真实 RED：在旧实现 `8ea629e` 上，新增模型/变换不可变 backing 测试得到 `2 failed, 37 passed`，失败均为 `setflags(write=True)` 未抛出 `ValueError`。
- 本次 P2 GREEN：将 `RgbdFrame` 的 image/depth 和 `RigidTransform.matrix` 改为独立 `bytes` backing 重建后，模型/变换聚焦测试 `39 passed`；调用方数组修改不影响内部值，内部数组不能重新开启写权限。

## RGB-D focused tests

使用已验证的 Python 环境运行 `tests/test_rgbd`：`98 passed in 0.28s`，`0 failed`，`0 skipped`。

覆盖数据类型和不可重新解锁的只读合同、尺寸/有限值校验、边界像素、四角解析反投影、奇数窗口裁剪、零深度排除、中位数采样、`min_valid_count` 失败、非法窗口/浮点像素、帧内容不变、针孔反投影、显式 X/Y/Z 旋转和旋转平移组合、溢出失败关闭、输入点不变、测量配对、平面/方块完整 3D 解析值、JSON 原生类型、确定性合成工厂、孔洞和全链路序列化。

## Full static regression and skips

- 基线门禁（`origin/main`）：`1839 passed, 18 skipped`，`git diff --check` 通过。
- D1 返修后全仓回归：`1937 passed, 18 skipped in 68.43s`。
- `tests/test_acceptance/test_delivery_contract.py`：`45 passed in 0.25s`。
- 18 个 skip 为仓库既有静态/环境门，不被本分支重新解释为 PASS；本分支未新增 skip。
- 本轮全仓测试使用 worktree 已存在、指向已验证环境的 `.venv-vision` junction；该环境链接未纳入 Git，未修改源码。
- `git diff --check` 通过。

## 512x512 CPU timing

在 512x512 合成平面、3x3 深度窗口、CPU-only 条件下完成 1024 次测量和 JSON 序列化：`0.023036s`，低于计划中的 `2.0s` 上限。该数据只代表当前本地环境的纯内存内核微基准，不代表相机、仿真或硬件吞吐。

## Dependency, ownership, and retained-file audit

- D1 生产代码依赖仅为 Python 标准库、NumPy 和 `vision_platform.rgbd` 内部模块；未导入 `cv2`、Open3D、ROS/rclpy、ZMQ、PyQt、socket、subprocess、相机/深度 SDK 或 CoppeliaSim。
- 本次返修明确修改的生产文件仅为 `vision_platform/rgbd/models.py` 与 `vision_platform/rgbd/transforms.py`；其余返修代码均为 `tests/test_rgbd/**` 和本报告。
- `git diff --name-only origin/main...HEAD` 仅包含 `vision_platform/rgbd/**`、`tests/test_rgbd/**`、本报告和 `RETAINED_FILES.txt`。
- 对 `vision_platform/vision2d`、`vision_platform/student`、`vision_platform/experiments`、`vision_platform/ui`、`config`、`tools`、`simulation` 的 ownership diff 为空。
- `RETAINED_FILES.txt` 已覆盖全部 D1 交付文件；无重复条目、无 `__pycache__`/`.pyc` 条目，报告路径将与本报告同一提交追加。

## Independent review findings and fixes

完成独立 P0/P1/P2 审核：本轮补测覆盖计划遗漏的数值合同证据；同时修复一个真实阻塞性 P2——此前 `frame.image_bgr`、`frame.depth_m` 和 `transform.matrix` 可通过 `setflags(write=True)` 解锁。修复后它们均基于独立不可变 `bytes` backing，调用方改写隔离且重新开启写权限会抛出 `ValueError`。除该 P2 外未发现需要阻断交付的 P0/P1/P2 问题。

## Remaining limitations

D1 仍是纯内存 RGB-D 软件内核：不包含 RGB-D 相机采集、帧同步、标定文件读写、点云/SLAM、CoppeliaSim 场景、正式课程、学生 SDK、UI/CLI、机器人闭环、真实深度相机或物理硬件验收。上述能力保留为后续集成边界，不能由本报告推断已完成。
