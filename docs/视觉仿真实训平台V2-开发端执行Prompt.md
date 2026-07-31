# 视觉仿真实训平台 V2.1 开发端执行 Prompt

以下正文可直接发送给具备本地文件、Git、PowerShell、Python 和 CoppeliaSim 操作能力的开发端。

---

你是 Blinx-CoppeliaSim 项目 V2.1 的主开发工程师。请基于现有一期成果，完整实现“学生 Python 程序载入与受控执行”，并完成静态、PyQt 和真实 CoppeliaSim 验收。

这不是方案讨论任务。设计方向已经确认；你需要按设计和实施方案逐项开发、测试、提交和形成证据。若发现文档与实际代码冲突，先用可复现证据说明冲突，再做满足既定目标的最小修正，并同步更新三份 V2.1 文档。

## 一、唯一项目位置

```text
C:\Users\yqzhe\Documents\Robot Sim\Blinx-CoppeliaSim-clean
```

CoppeliaSim 默认位置：

```text
E:\CoppeliaSim
```

开发基线：

```text
origin/main
7546db7
```

开发分支：

```text
codex/student-program-runner-v2
```

当前仓库是专用的干净发布仓库。不要回到旧工作树开发，不要从旧功能分支创建本次分支。

## 二、必须先完整阅读

按顺序阅读：

1. `README.md`
2. `RETAINED_FILES.txt`
3. `docs/superpowers/specs/2026-07-30-student-program-runner-v2-design.md`
4. `docs/superpowers/plans/2026-07-30-student-program-runner-v2-plan.md`
5. `docs/视觉仿真实训平台使用说明.md`
6. `docs/视觉仿真实训平台自动验收报告.md`
7. `docs/仿真转真机验证清单.md`
8. `simulation/vision_lab/robot_assets_manifest.json`
9. `tests/test_acceptance/test_delivery_contract.py`

设计文档定义产品和安全边界，实施方案定义任务顺序、文件、测试和提交。两者优先于本 Prompt 的概述。

## 三、开始前操作

先检查：

```powershell
$project = 'C:\Users\yqzhe\Documents\Robot Sim\Blinx-CoppeliaSim-clean'
git -C $project status --short --branch
git -C $project rev-parse HEAD
git -C $project rev-parse origin/main
```

预期：

- `HEAD` 与 `origin/main` 的文档开发前基线为 `7546db7`；
- 未提交内容只应包含三份 V2.1 文档和 `RETAINED_FILES.txt`；
- 不得覆盖用户的其他改动。

如果 `codex/student-program-runner-v2` 尚不存在，按实施方案 Task 1 在当前专用仓库中创建该分支，并先提交文档基线。如果分支已经存在，检查后继续使用，禁止删除后重建。

创建项目自己的 Python 环境：

```powershell
Push-Location $project
try {
    powershell -ExecutionPolicy Bypass -File tools\vision_lab\bootstrap.ps1
    powershell -ExecutionPolicy Bypass -File tools\vision_lab\python.ps1 `
      -m pytest -q
} finally {
    Pop-Location
}
```

文档更新时确认的静态基线是：

```text
158 passed, 2 skipped
```

两个 skip 是显式启用的 CoppeliaSim 在线测试，只表示本次静态运行未执行在线门禁，不能记作在线 PASS。若开始开发前实际计数不同，先确认是不是基线已经更新，再记录真实结果，不要为了匹配旧计数删除或跳过测试。

## 四、必须交付的功能

### 1. 学生程序契约

学生程序只定义同步入口：

```python
def main(ctx):
    pick = (55, -55, 20)
    drop = (122, -66, 20)
    safe_z = 110

    ctx.log("程序开始")
    ctx.robot.home()
    ctx.robot.move_world(pick[0], pick[1], safe_z, speed=15)
    pick_approach_pose = ctx.robot.pose()
    ctx.robot.move_world(
        pick_approach_pose[0], pick_approach_pose[1], pick[2], speed=8
    )
    ctx.tool.on()
    pick_pose = ctx.robot.pose()
    ctx.robot.move_world(pick_pose[0], pick_pose[1], safe_z, speed=12)
    ctx.robot.move_world(drop[0], drop[1], safe_z, speed=15)
    drop_approach_pose = ctx.robot.pose()
    ctx.robot.move_world(
        drop_approach_pose[0], drop_approach_pose[1], drop[2], speed=8
    )
    ctx.tool.off()
    drop_pose = ctx.robot.pose()
    ctx.robot.move_world(drop_pose[0], drop_pose[1], safe_z, speed=12)
    ctx.robot.home()
    ctx.log("程序结束")
```

V2.1 只开放：

```text
ctx.log(message)
ctx.sleep(seconds)
ctx.checkpoint(label)
ctx.robot.home()
ctx.robot.move_world(x_mm, y_mm, z_mm, speed=...)
ctx.robot.pose()
ctx.tool.on()
ctx.tool.off()
```

学生程序不得导入或获得 `VisionLabApplication`、机器人后端、CoppeliaSim Remote API、`sim`、`simIK` 或真实硬件对象。

### 2. 静态检查

实现：

- `.py` 和 256 KiB 上限；
- UTF-8 严格解码；
- `ast.parse()`；
- 有且只有一个顶层同步 `main(ctx)`；
- 入口签名检查；
- 顶层动作调用检查；
- 禁用导入检查；
- 带错误码、中文消息、行号和列号的结构化结果。

至少禁止：

```text
robot_backends
vision_platform.application
coppeliasim_zmqremoteapi_client
subprocess
socket
ctypes
```

这些限制是教学防误操作，不得宣称为恶意代码安全沙箱。

### 3. 子进程和协议

- 使用 `multiprocessing.get_context("spawn")`；
- 学生代码只在 Worker 子进程中执行；
- `VisionLabApplication`、CoppeliaSim 连接、机器人和吸盘只在主进程；
- 使用版本化、可序列化、带 `command_id` 的白名单协议；
- 未知命令拒绝，不允许任意 `getattr` 分派；
- 处理子进程异常退出、通信中断、命令超时、总运行超时和强制回收。

### 4. 状态和控制

状态固定为：

```text
EMPTY
LOADED
VALIDATED
RUNNING
PAUSED
PASSED
FAILED
CANCELLED
RESETTING
```

必须实现：

- 运行；
- 暂停；
- 继续；
- 下一步；
- 停止；
- 场景复位；
- 窗口关闭时的有界清理。

暂停和单步发生在命令边界。暂停不能被描述成对已经进入底层 IK 的单条动作进行瞬时冻结；“下一步”只放行一条后续命令。

### 5. 安全网关

每条动作进入底层机器人前检查：

- XYZ 为有限数值且位于现有 `WorkspacePolicy`；
- `1 <= speed <= 30`；
- 总命令数不超过 200；
- 总运行时间不超过 60 秒；
- 单条命令等待不超过 10 秒；
- 单次 `sleep` 不超过 5 秒；
- 低于 `safe_z_mm` 时只允许保持 XY 不变的垂直运动；
- 有 XY 横移时，起点和终点都不低于 `safe_z_mm`；
- `tool.on()` 只允许在 `z <= 35 mm`；
- V2.1 无条件拒绝真实机器人后端，错误码为 `REAL_BACKEND_NOT_AUTHORIZED`。

停止或失败后的顺序：

1. 停止接受新命令；
2. 请求并回收学生子进程；
3. `tool.off()`；
4. 尝试垂直升到安全高度；
5. 尝试回零；
6. 把收尾错误写入 `cleanup_errors`，但不覆盖原始错误。

### 6. PyQt 学生编程页

在现有窗口的页签区增加独立 `StudentProgramPanel`，同时保留左侧相机画面和原有标定、识别、分类、验收功能。

页面必须包含：

- 程序路径；
- 打开、保存、另存为；
- UTF-8 Python 编辑器；
- 检查代码；
- 运行、暂停、继续、下一步、停止、复位；
- 状态、当前命令、命令数、运行时间和 TCP；
- 只读事件控制台；
- 证据目录。

按钮启用状态只能由统一状态映射驱动。运行时禁止编辑和切换文件；文件修改后必须从 `VALIDATED` 退回 `LOADED`。原 `VisionLabWindow` 的已有控件和测试必须保持兼容。

### 7. CLI、脚本和模板

增加：

```text
vision-platform student-validate
vision-platform student-run
tools/vision_lab/run_student_program.ps1
student_programs/templates/basic_motion.py
student_programs/templates/pick_and_place.py
student_programs/README.md
```

PowerShell 入口负责定位项目、启动或复用 CoppeliaSim、传入场景和端口并调用 Python CLI，不复制安全逻辑，也不得终止不是自己拥有的 CoppeliaSim 进程。

进程与 readiness 采用以下强制契约：

- 正式 helper 为 `vision_platform/coppeliasim_readiness.py`，必须由项目
  `.venv-vision` 调用，并用 fresh ZMQ clients 验证 RPC、目标场景路径、
  `/VisionLab` 和 `/BLX_base_link`；TCP 端口打开本身不算 ready。
  `timeout_s` 必须有限且不小于 0，retry interval 必须有限且大于 0，并在
  首次 client factory 前拒绝非法值。成功 client 必须先释放 socket
  (`linger=0`) 和 context，之后才打印 `READY`；否则打印 `NOT_READY`。
- `launch_coppeliasim.ps1` 在启动或取得既有 listener 后立即冻结规范化
  启动身份，只返回一个包含 `StartedByScript`、`ProcessId`、
  `ProcessPath`、`ProcessStartTimeUtcTicks` 的结构化对象。成功日志不得
  污染返回管道。
  使用前必须以 `Resolve-Path` 规范化 `CoppeliaRoot` 和 executable，确保
  `E:\CoppeliaSim\.` 等价路径不会破坏所有权比较。
- `run_acceptance.ps1`、`run_student_program.ps1` 和 `run_pyqt.ps1`
  仅在 `StartedByScript=true` 时于 `finally` 传递启动时捕获的 PID、
  规范路径和 UTC 启动 ticks；
  borrowed 用户 listener 永不终止，禁止按进程名群杀。四个入口必须共用
  `tools/vision_lab/process_ownership.ps1`，并检查 `WaitForExit` 布尔值；
  三元组不符、身份字段无法读取、超时或原三元组仍存活必须失败。首次
  `Get-Process` 后必须在停止前与 `ProcessStartTimeUtcTicks` 一并比较；
  不匹配时停止调用次数必须为零。停止动作必须是
  `Stop-Process -InputObject $VerifiedProcess`，不得在验证后再按 PID
  查杀；停止后的 survivor 也与原启动三元组比较，PID 复用后的不同对象
  不得被停止。若 `Start-Process` 已成功而首次启动身份捕获失败，必须调用
  `Stop-StartedProcessObject`，仅停止 `Start-Process` 返回的原始 Process
  对象并等待退出；禁止为这个回退执行 `Get-Process -Id` 或
  `Stop-Process -Id`。
- 在线 pytest fixture 不启动或终止 CoppeliaSim。缺 listener 或错误场景
  必须 FAIL 并给出 `launch_coppeliasim.ps1` 命令；fixture 只负责稳定
  borrowed listener 上的 stop→load→start 和 fresh-client teardown。
  临时场景载入后的 start/running readiness 失败也必须 fail-closed：
  fresh-client stop、恢复正式源场景，并保留原异常与 cleanup cause。
- 在线 pytest 提供 `--coppelia-host`/`--coppelia-port`，优先于
  `COPPELIA_HOST`/`COPPELIA_PORT` 和默认 `127.0.0.1:23000`。
  `run_acceptance.ps1` 的两组在线 pytest 必须透传本次端点。
- 直接运行在线 pytest 前必须先运行 `launch_coppeliasim.ps1`；完整验收
  则由 `run_acceptance.ps1` 持有并最终回收自己启动的进程。清理发生在
  写 summary 之前；清理失败必须令 summary 为 `FAIL` 且退出非零。

### 8. 运行证据

每次运行生成唯一目录，至少包含：

```text
source.py
source.sha256
manifest.json
commands.jsonl
events.jsonl
summary.json
```

`summary.json` 至少包含：

```text
schema_version
run_id
status
program_path
source_sha256
started_at
finished_at
elapsed_seconds
command_count
last_pose_mm
safety_violation_count
error
cleanup_errors
robot_backend
hardware_status
```

仿真运行的 `hardware_status` 仍写 `PENDING_HARDWARE`。证据写入失败必须使运行失败，不能继续声称 PASS。

## 五、实现顺序

严格按实施方案中的 12 个 Task 顺序推进：

1. 固化文档基线并建立开发分支；
2. 协议、状态和错误码；
3. AST 静态校验；
4. 学生 SDK 和 Worker；
5. 动作安全网关和配置；
6. 证据记录；
7. 可替换应用会话和场景复位；
8. `StudentProgramController`；
9. CLI、PowerShell 和模板；
10. PyQt 学生编程页；
11. CoppeliaSim 在线验收和学生实验说明；
12. 完整回归、发布白名单和交付。

每个 Task 都执行：

1. 写失败测试；
2. 运行并确认按预期失败；
3. 写最小实现；
4. 运行目标测试并确认通过；
5. 运行相关回归；
6. 提交一个范围清晰的 Git commit。

不要把多个未验证子系统一次性堆入同一个提交。

## 六、必须保护的内容

禁止修改：

```text
simulation/vision_lab/BL23_vision_lab.ttt
robot_backends/models/BLX_openr6.ttt
robot_backends/models/openr6_arm_coppeliasim.urdf
robot_backends/models/meshes_blx/*
simulation/vision_lab/assets/robot/*
```

使用以下文件和测试验证资产，不要引用不存在的清单：

```text
simulation/vision_lab/robot_assets_manifest.json
tests/test_simulation/test_robot_asset_provenance.py
```

还必须：

- 保留 replay、仿真和可选海康相机的现有接口；
- 保留原标定、识别、分类、PyQt、simUI 和验收功能；
- 保持 position-only，不开放 RX、RY、RZ；
- 不新增第三方 Python 依赖；
- 不把仿真结果写成真机验收；
- 不伪造教师教学效果验收。

## 七、发布白名单

这是干净发布仓库。所有新增正式 Python、测试、模板、脚本和文档必须逐项加入：

```text
RETAINED_FILES.txt
```

扩展现有：

```text
tests/test_acceptance/test_delivery_contract.py
```

不要新建独立的 delivery 测试目录。发布合同必须验证：

- 白名单路径全部存在；
- V2.1 入口、模板、文档和在线测试在白名单中；
- `artifacts/`、`.venv-vision/`、缓存和学生个人提交不在白名单中。

## 八、必须运行的验收

### 1. 目标单元与集成测试

按实施方案运行每个 Task 对应的测试，不能只在最后运行一次。

### 2. 完整静态回归

```powershell
powershell -ExecutionPolicy Bypass -File tools\vision_lab\python.ps1 `
  -m pytest -q
```

要求零失败。报告实际通过和跳过数量，不写预估数量。

### 3. PyQt 离屏回归

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
powershell -ExecutionPolicy Bypass -File tools\vision_lab\python.ps1 `
  -m pytest tests/test_vision_platform/test_pyqt_smoke.py `
  tests/test_vision_platform/test_student_program_panel.py -q
```

### 4. CoppeliaSim 在线学生程序测试

必须以 `-m coppeliasim` 显式运行并获得 PASS。skip 不是 PASS。

```powershell
powershell -ExecutionPolicy Bypass `
  -File tools\vision_lab\launch_coppeliasim.ps1

powershell -ExecutionPolicy Bypass -File tools\vision_lab\python.ps1 `
  -m pytest tests/test_acceptance/test_coppeliasim_student_program.py `
  -m coppeliasim -q
```

前一个命令必须先完成真实 RPC、正式场景和哨兵 readiness；pytest fixture
只管理 borrowed 场景生命周期，不负责启动或终止 CoppeliaSim。
在线学生测试必须逐条核对完整关键命令与六个 move 的坐标/速度；两个
下探动作的 X/Y 必须分别与其前一条 `robot.pose` 返回的实测安全高度 X/Y
一致，仅 Z 改为抓取或放置高度。测试还必须用
`application.sim` 查询
`/VisionLab/Pickables/object_01_red_square`，确认最终 XY 落入 red zone；
只看到 `tool.on`/`tool.off` 命令不能算抓放通过，硬件状态仍为
`PENDING_HARDWARE`。

### 5. 完整自动验收

```powershell
powershell -ExecutionPolicy Bypass `
  -File tools\vision_lab\run_acceptance.ps1 `
  -OutputDir artifacts\vision_lab\student-program-v2-final
```

要求原标定、六物体分类、原在线测试、学生程序在线测试、PyQt、simUI、完整静态测试和场景运行验证全部 PASS。

### 6. 资产和发布合同

```powershell
powershell -ExecutionPolicy Bypass -File tools\vision_lab\python.ps1 `
  -m pytest tests/test_simulation/test_robot_asset_provenance.py `
  tests/test_acceptance/test_delivery_contract.py -q
```

### 7. 人工界面检查

打开 PyQt 和 CoppeliaSim，检查：

- 中文标签可读；
- 编辑器、按钮、状态和事件区不重叠；
- 左侧相机画面和学生程序状态可同时观察；
- 运行、暂停、单步、停止、复位符合定义；
- 关闭窗口后没有残留学生进程；
- 证据路径可找到。

人工界面检查必须记录真实结果；离屏构造测试不能替代可读性检查。

## 九、完成条件

只有同时满足以下条件，才可写“V2.1 软件实现完成”：

- 实施方案全部 Task 已完成；
- 静态测试零失败；
- CoppeliaSim 学生程序在线测试为 PASS 而不是 skip；
- 暂停、单步、停止、超时和复位均有测试与证据；
- PyQt 人工可读性检查已记录；
- 原有一期验收无退化；
- 受保护资产匹配现有来源和哈希清单；
- `RETAINED_FILES.txt` 与实际交付一致；
- 开发分支工作树干净；
- 教师教学效果验收仍单独标记；
- 海康相机和真实机械臂仍为 `PENDING_HARDWARE`。

如果任何一项未完成，使用 `BLOCKED`、`FAIL` 或 `PENDING_HARDWARE` 如实报告，禁止用“基本完成”掩盖。

## 十、最终回复格式

最终交付回复必须包含：

1. **结果摘要：** 实现了什么；
2. **分支与提交：** 分支名和逐条提交；
3. **变更文件：** 按协议、运行器、安全、UI、入口、测试和文档分组；
4. **测试证据：** 每条命令、实际通过/跳过/失败数；
5. **CoppeliaSim 在线结果：** 明确 PASS、FAIL 或未执行；
6. **证据目录：** 完整路径；
7. **资产保护：** 清单与测试结果；
8. **发布白名单：** 检查结果；
9. **人工验收：** 已完成项与待教师项；
10. **硬件边界：** 逐项列出 `PENDING_HARDWARE`；
11. **已知限制：** 子进程不是恶意代码安全沙箱，V2.1 不支持姿态、关节角、自定义视觉算法和真机执行。

不要自动合并或推送到 `origin/main`。保留 `codex/student-program-runner-v2` 供教师审核。
