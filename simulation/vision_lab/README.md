# CoppeliaSim 视觉实训场景

## 场景入口

场景文件：

`simulation/vision_lab/BL23_vision_lab.ttt`

启动：

```powershell
powershell -ExecutionPolicy Bypass -File tools\vision_lab\launch_coppeliasim.ps1
```

## 模型来源与运行决策

用户指定的厂家归档：

- 文件：`LC-YT1119-BL23-A000(机械臂总组装）-ZQSZXYx-V1.1.2-SJ.zip`
- 归档 SHA-256：`9ea431ee1b87417c19f1a0f579209dfe20d0c8d1e381b660010900718cb2b4dc`
- STEP SHA-256：`f3f493626792ef25b99e6b1e79f791da96f2526cee7161c90194443e9c34a9a2`
- STEP 实体数：141

该 STEP 继续作为厂家几何 provenance 和实体分区审计来源，但不进入运行外观。旧分段映射存在 link3–link4 硬断裂，逐段包围盒对齐会造成悬空、重叠和错误旋转。

运行本体改用用户提供的 `openr6_arm.7z` 作为正确装配参考：

- 归档 SHA-256：`524f968f1ce19dc3fc731d2fd3fbabc83631d1506fae94a938a3725c70c5d174`
- 七个网格与项目 `robot_backends/models/meshes_blx/*.STL` 逐文件哈希一致。
- 正式源资产不修改；只在独立场景副本中把错误放大 10 倍的 `link6` 校正为 `0.1`。
- 原模型末端不复用，在 `/BLX_tool_suction` 下建立吸盘适配器、连接杆和吸盘杯。

## 关节与坐标契约

- 路径固定为 `/BLX_joint1` 到 `/BLX_joint6`。
- TCP 固定为 `/BLX_tool_suction`。
- 控制模式为 `position-only`。
- X/Y/Z 使用世界坐标毫米。
- RX/RY/RZ 只为接口兼容保留，当前 IK 忽略。
- 基座不翻转。
- 建议工作半径不超过 150 mm。

## 场景对象

- `/VisionLab/Camera`：640×480 顶视虚拟相机。
- `/VisionLab/Calibration`：三个拟合点和一个独立验证点。
- `/VisionLab/Pickables`：六个彩色方形/圆形物体。
- `/VisionLab/Zones`：红、绿、蓝、黄四个分类区。
- `/VisionLab/Workspace`：教学工作平面。
- `/VisionLab/CameraRenderScope`：限制教学相机只渲染工作区、物体、分类区和标定点。

## 构建与验证

重建场景：

```powershell
.\.venv-vision\Scripts\python.exe -m simulation.vision_lab.build_scene
```

运行场景、六关节、link6 尺寸、吸盘父子关系和三视图验证：

```powershell
.\.venv-vision\Scripts\python.exe -m simulation.vision_lab.verify_scene
```

关键机器可读证据：

- `source_manifest.json`
- `robot_assets_manifest.json`
- `scene_manifest.json`
- `runtime_verification.json`

## 验收边界

自动门禁证明场景结构、六关节跟随、虚拟相机、标定和分类闭环。它不证明海康相机、真实 TCP、急停、气路或物理抓取已通过；这些项目保持 `PENDING_HARDWARE`。
