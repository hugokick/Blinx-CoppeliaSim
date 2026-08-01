# Blinx-CoppeliaSim

基于比邻星六轴机器人实验平台与 CoppeliaSim 开发的视觉虚拟仿真实训系统。

学生可先在虚拟相机、六轴机械臂和吸盘场景中完成标定、识别、抓取与分类代码调试，再通过同一任务层接口切换到实验室真实机械臂和海康相机。当前仓库不包含海康 MVS SDK 和厂家机械臂 SDK；真机项目必须按硬件清单重新标定并低速验收。

## 已实现

- CoppeliaSim 六轴机械臂、吸盘、顶视相机与分类工作区；
- 三点仿射标定和独立验证点；
- 颜色/形状识别、六物体抓取与分类闭环；
- PyQt 教学控制台和 CoppeliaSim simUI 状态面板；
- 学生 Python 模板、受控执行器、PyQt“学生编程”页和运行证据；
- replay 相机、CoppeliaSim 相机及可选海康相机适配层；
- 自动化测试、场景结构验证和仿真转真机验收清单。

## 快速开始

环境要求：Windows PowerShell、Python 3.10+、CoppeliaSim（默认 `E:\CoppeliaSim`）。

```powershell
powershell -ExecutionPolicy Bypass -File tools\vision_lab\bootstrap.ps1
powershell -ExecutionPolicy Bypass -File tools\vision_lab\run_pyqt.ps1
```

只运行历史图片识别：

```powershell
powershell -ExecutionPolicy Bypass -File tools\vision_lab\run_replay_demo.ps1
```

完整自动验收：

```powershell
powershell -ExecutionPolicy Bypass -File tools\vision_lab\run_acceptance.ps1
```

## 首批课程实验

正式实验目录精确包含 `R1-01`、`R1-02`、`R1-05`、`R1-06` 和
`R1-07`。可先列出目录：

```powershell
powershell -ExecutionPolicy Bypass -File tools\vision_lab\python.ps1 `
  -m vision_platform.cli experiment-list
```

也可从 PowerShell 直接运行一个实验，例如：

```powershell
powershell -ExecutionPolicy Bypass -File tools\vision_lab\run_experiment.ps1 `
  -Experiment R1-05
```

在 PyQt 中应先进入“实验目录”并先选择实验，再编辑和运行随实验加载的
模板。切换实验会终止旧会话并重新加载确定场景，未保存的编辑内容应先由
学生自行保存。

`R1-05`、`R1-06`、`R1-07` 每次集成运行的证据位于
`artifacts/vision_lab/experiment-runs/<运行目录>/`；原始图像在 `frames/`，
快照索引在 `snapshots.jsonl`，终态探针在 `scene-final.json`。
终态探针不是正式成绩，只用于复核场景是否到达预期状态；教学效果仍须
教师人工验收。
仿真结果也不覆盖真机、海康 MVS、急停、气路和物理抓取，这些项目均为
`PENDING_HARDWARE`。

## 学生自编程

模板位于 `student_programs/templates/`：

- `basic_motion.py`：基础回零和安全高度运动；
- `pick_and_place.py`：单物体吸取、搬运和红区放置。

先检查代码：

```powershell
.\.venv-vision\Scripts\python.exe -m vision_platform.cli student-validate `
  --program "student_programs\templates\basic_motion.py"
```

再运行仿真：

```powershell
powershell -ExecutionPolicy Bypass `
  -File tools\vision_lab\run_student_program.ps1 `
  -Program "student_programs\templates\basic_motion.py"
```

也可在 PyQt“学生编程”页打开、编辑、保存、检查和运行程序。暂停发生在
命令边界，不会瞬间冻结已经进入 IK 的动作；继续恢复运行；单步只放行
下一条命令；停止会取消学生子进程并执行安全清理；复位仅在运行结束后
重载正式场景。

每次运行的源码、哈希、命令、事件和汇总保存在
`artifacts/vision_lab/student-runs`。该执行器用于课堂可信代码和防误操作，
不是恶意代码安全沙箱。仿真通过不代表真机通过，海康相机、真实机械臂、
急停、气路和实体抓取仍为 `PENDING_HARDWARE`。详细说明见
`docs/学生自编程实验说明.md`。

## 主要入口

- 场景：`simulation/vision_lab/BL23_vision_lab.ttt`
- 平台代码：`vision_platform/`
- 机械臂适配：`robot_backends/`
- 学生工具：`tools/vision_lab/`
- 学生模板：`student_programs/templates/`
- 实验说明：`docs/实验三-CoppeliaSim视觉标定.md`、`docs/实验四-CoppeliaSim物体分类.md`
- 完整使用说明：`docs/视觉仿真实训平台使用说明.md`
- 真机验证：`docs/仿真转真机验证清单.md`
- 发布文件白名单：`RETAINED_FILES.txt`

## 验收边界

仓库中的自动证据只覆盖仿真和软件接口。本机未连接海康相机和真实机械臂，因此相机枚举、真实 TCP、急停、气路、低速空跑和物理抓取仍为 `PENDING_HARDWARE`。
