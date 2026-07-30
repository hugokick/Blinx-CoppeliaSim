# 学生程序

学生程序用于在 CoppeliaSim 仿真平台中练习机械臂控制。当前入口强制使用
`sim` 机器人和虚拟视觉后端，不会连接真实机械臂。真机验证状态仍为
`PENDING_HARDWARE`，不得把仿真通过写成真机验收通过。

## 程序结构

每个 UTF-8 编码的 `.py` 文件必须定义一个同步入口：

```python
def main(ctx):
    ctx.log("开始")
```

不要在模块顶层执行动作。可从
`student_programs/templates/basic_motion.py` 或
`student_programs/templates/pick_and_place.py` 复制后修改。

## 教学 SDK

- `ctx.robot.home()`：回到教学零位。
- `ctx.robot.move_world(x, y, z, speed=15)`：移动到世界坐标点。
- `ctx.robot.pose()`：读取当前 `(x, y, z)`。
- `ctx.tool.on()`、`ctx.tool.off()`：开启或关闭吸盘。
- `ctx.log(message)`：写入学生运行证据。
- `ctx.sleep(seconds)`：有界等待。
- `ctx.checkpoint("阶段名")`：设置带标签的单步教学检查点，对应
  `checkpoint(label)` 接口。

所有位置单位均为 `mm`，坐标采用机械臂世界坐标。V2.1 是
position-only 接口：不支持 RX/RY/RZ，也不接受学生程序修改末端姿态。

## 安全规则

默认工作空间为 X=`20..140 mm`、Y=`-90..90 mm`、Z=`10..140 mm`。
安全高度 `safe_z` 为 `100 mm`。低于安全高度时只能竖直升降，禁止水平
移动；应先升到 `safe_z`，再移动到目标上方，最后竖直下降。吸盘只能在
配置允许的抓取高度开启，默认不高于 `35 mm`。速度、运行时间、命令数量
和单次等待也受配置中的学生安全策略限制。

## 验证与运行

在项目根目录执行静态验证：

```powershell
.\.venv-vision\Scripts\python.exe -m vision_platform.cli student-validate `
  --program "student_programs\templates\basic_motion.py"
```

启动 CoppeliaSim 并运行学生程序：

```powershell
powershell -ExecutionPolicy Bypass -File `
  "tools\vision_lab\run_student_program.ps1" `
  -Program "student_programs\templates\pick_and_place.py"
```

也可直接调用 `vision_platform.cli student-run`，但必须保持
`--robot sim`。命令只在状态为 `PASS` 时返回退出码 0；静态检查失败返回
2，其他失败返回 1。

## 运行证据

默认证据目录为 `artifacts/vision_lab/student-runs`。每次运行会保存事件、
命令和 `summary.json`。提交实验时应保留该目录，并由教师结合 CoppeliaSim
动作画面人工验收。在线 CoppeliaSim 未连接或测试被跳过，都不能视为
`PASS`。
