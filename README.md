# Blinx-CoppeliaSim

基于比邻星六轴机器人实验平台与 CoppeliaSim 开发的视觉虚拟仿真实训系统。

学生可先在虚拟相机、六轴机械臂和吸盘场景中完成标定、识别、抓取与分类代码调试，再通过同一任务层接口切换到实验室真实机械臂和海康相机。当前仓库不包含海康 MVS SDK 和厂家机械臂 SDK；真机项目必须按硬件清单重新标定并低速验收。

## 已实现

- CoppeliaSim 六轴机械臂、吸盘、顶视相机与分类工作区；
- 三点仿射标定和独立验证点；
- 颜色/形状识别、六物体抓取与分类闭环；
- PyQt 教学控制台和 CoppeliaSim simUI 状态面板；
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

## 主要入口

- 场景：`simulation/vision_lab/BL23_vision_lab.ttt`
- 平台代码：`vision_platform/`
- 机械臂适配：`robot_backends/`
- 学生工具：`tools/vision_lab/`
- 实验说明：`docs/实验三-CoppeliaSim视觉标定.md`、`docs/实验四-CoppeliaSim物体分类.md`
- 完整使用说明：`docs/视觉仿真实训平台使用说明.md`
- 真机验证：`docs/仿真转真机验证清单.md`
- 发布文件白名单：`RETAINED_FILES.txt`

## 验收边界

仓库中的自动证据只覆盖仿真和软件接口。本机未连接海康相机和真实机械臂，因此相机枚举、真实 TCP、急停、气路、低速空跑和物理抓取仍为 `PENDING_HARDWARE`。
