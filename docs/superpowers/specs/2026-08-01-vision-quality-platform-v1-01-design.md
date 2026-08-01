# V2.2-C0 视觉实验公共底座与 V1-01 设计

**文档状态：** 用户已确认范围，待按实施计划开发

**编制日期：** 2026-08-01

**项目名称：** Robot Sim

**权威仓库：** `C:\Users\yqzhe\Documents\Robot Sim\Blinx-CoppeliaSim-clean`

**开发分支：** `codex/v2-2-vision-quality-platform`

**开发基线：** `v2.2.0` / `main` / `37a385fd596d890716cbff9eb86e7feae31c11c9`

---

## 1. 决策摘要

本阶段实施 V2.2-C 的第一个可独立验收增量：

- 建设独立的 CoppeliaSim 视觉质检实验场景；
- 建立受控的虚拟相机与光照配置档；
- 完成 `V1-01 虚拟视觉系统认知` 正式实验；
- 建立通用视觉结果、图像层和运行证据契约；
- 在 PyQt 中增加通用视觉结果查看页；
- 复用 V2.1 学生运行器和 V2.2 实验目录、CLI、PowerShell、场景会话及证据链。

学生不获得任意 CoppeliaSim 对象写权限。学生程序只能选择教师预先发布的
视觉配置档、读取当前配置、采集图像和复位到基准配置。配置档把分辨率、
视场角、相机高度和两路光照作为一个原子变更单元；任何设置或回读失败都
触发回滚。

本阶段不实现 `V1-02` 至 `V1-05` 的二维视觉算法。并行分支
`codex/v2-2-vision2d-algorithm-kernel` 独占
`vision_platform/vision2d/**`、`tests/test_vision2d/**` 和
`docs/experiments/V1-02.md` 至 `V1-05.md`。本分支只提供未来接入算法结果所需
的通用接口，不导入未合并的算法包。

## 2. 背景与现状

`v2.2.0` 已具备：

- V2.1 学生 SDK、spawn 子进程、白名单协议、安全网关和运行控制；
- V2.2 实验目录、场景会话、能力检查、场景探针和证据绑定；
- R1 首批五个实验、机器人基础场景和物流场景；
- 通用 `camera.capture` 学生命令和 PNG 证据记录；
- PyQt 实验目录页、学生程序页和通用 CLI/PowerShell 运行入口；
- `1177 passed, 10 skipped` 的 `v2.2.0` 静态基线。

课程路线图将 V1 系列定义为九个二维机器视觉实验，并要求使用独立的视觉
质检场景。V1-01 是后续全部 V1 实验的入口：先让学生理解相机、视场、
分辨率、光照和采集结果，再进入尺寸、定位、轮廓、模板、条码、OCR 和
缺陷算法。

截至 2026-08-01，并行算法分支
`codex/v2-2-vision2d-algorithm-kernel` 已在 `f5bef91` 完成并推送。该分支的
留存证据显示 `tests/test_vision2d` 为 `34 passed`，其基于 V2.1 的完整静态
回归为 `658 passed, 3 skipped`。这些结果只证明纯二维视觉算法内核和原创
合成测试，不是本分支、CoppeliaSim、PyQt、机器人或真机验收。本设计仍不
提前合入该分支；V1-01 独立完成后再由单一集成端处理 V1-02 至 V1-05。

## 3. 目标

### 3.1 教学目标

完成 V1-01 后，学生应能够在虚拟环境中：

1. 解释图像分辨率与细节数量、处理成本之间的关系；
2. 解释视场角和相机高度对覆盖范围、目标像素尺寸的影响；
3. 观察光照强弱对颜色、对比度和可见细节的影响；
4. 使用受控 SDK 切换视觉配置档并采集标准件图像；
5. 对比不同配置档的图像尺寸、直方图摘要和清晰度摘要；
6. 将观察结论写入实验报告，而不是把自动状态误当成课程成绩。

### 3.2 工程目标

本阶段必须：

- 从受保护正式场景复制生成新的 `.ttt`，不覆盖原场景；
- 只创建原创几何标准件和程序生成的标识，不复制 H 盘旧素材；
- 用严格 JSON 契约描述视觉配置档和实验公开参数；
- 只允许选择已发布的配置档 ID，不接受任意对象路径或数值；
- 对配置变更执行快照、设置、回读、回滚和基准复位；
- 让配置切换、图像采集和复位进入学生命令证据；
- 将原图、中间图、标注图和结构化结果抽象为同一视觉结果包；
- 让 V1-01 在实验目录、学生模板、CLI、PowerShell 和 PyQt 中可访问；
- 提供静态测试、CoppeliaSim 在线测试和可复现运行证据；
- 保持 `hardware_status=PENDING_HARDWARE`。

## 4. 非目标

本阶段不包括：

- `V1-02` 至 `V1-05` 的 `vision_platform/vision2d` 算法实现；
- `V1-06` 模板匹配、`V1-07` 条码、`V1-08` OCR、`V1-09` 缺陷检测；
- 自动评分、批量成绩、正式课程成绩和教学效果自动结论；
- 任意 Python/OpenCV 算法上传和执行；
- 任意 CoppeliaSim Remote API、对象路径、脚本或参数写入；
- 镜头畸变、真实相机内参、真实照度和真实测量精度标定；
- 海康 MVS、真实机械臂、急停、气路和物理抓取验收；
- RGB-D、AI 模型训练、ROS2、MoveIt 2 和 V2.3 真机迁移；
- 修改一期 `vision_platform/recognition/color_shape.py`；
- 修改正式 `simulation/vision_lab/BL23_vision_lab.ttt`、URDF、STL 或机器人资产。

## 5. 方案比较

### 5.1 方案一：预定义配置档与原子切换

教师发布少量固定配置档。学生通过配置档 ID 切换相机和光照，运行时完成
设置、回读和回滚。

优点：

- 安全边界清楚，不暴露任意场景写接口；
- 每次实验输入可复现，便于教师比较和自动验收；
- 配置档可以同时改变分辨率、视场、相机高度和光照；
- 以后可为不同实验发布不同白名单，而不用扩大学生协议权限。

代价：

- 学生不能输入任意数值；
- 新教学组合需要教师更新配置档并重新发布。

### 5.2 方案二：开放连续参数滑块

PyQt 和学生 SDK 直接接受分辨率、视场角、高度和光照数值。

优点是探索自由；缺点是参数组合难以穷举验证，容易产生空画面、超大图像、
非法姿态和不可复现结果，还会扩大命令协议及安全网关。

### 5.3 方案三：为每个配置保存一份 `.ttt`

每个相机/光照组合单独保存场景，切换配置等同于切换场景。

优点是运行时不修改场景参数；缺点是场景文件重复、版本维护成本高、切换慢，
也不利于学生理解同一工作站中的参数变化。

### 5.4 选定方案

采用方案一。它在教学可见性、运行安全、可复现性和后续扩展之间最平衡。
方案二留到教师高级模式评估后再讨论；方案三不进入主线。

## 6. 总体架构

```text
V1-01 实验定义与公开配置档
        │
        ├── 实验目录 / CLI / PowerShell / PyQt 选择
        │
        └── Student SDK
              │
              ▼
        白名单命令协议
              │
              ▼
      StudentExperimentGateway
              │
              ▼
      VisionProfileController
       ├── 快照当前参数
       ├── 设置相机与光照
       ├── 回读并验证
       ├── 失败回滚
       └── 运行结束复位
              │
              ▼
 CoppeliaSim 视觉质检场景 ──► camera.capture
              │                       │
              │                       ▼
              └──────────────► VisionResultBundle
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                    运行证据                  PyQt 结果页
```

设计保持三条边界：

1. 场景控制层只认识配置档，不认识学生任意参数；
2. 视觉结果层只认识 JSON 原生结果和命名图像层，不依赖具体算法包；
3. UI 只读取已经校验的结果包和证据路径，不直接操作算法内部对象。

## 7. 独立视觉质检场景

### 7.1 场景路径

新增：

```text
simulation/vision_quality_lab/
├── __init__.py
├── BL23_vision_quality_lab.ttt
├── scene_spec.json
├── scene_manifest.json
└── profiles.json
```

场景从
`simulation/vision_lab/BL23_vision_lab.ttt` 的只读副本生成，输出使用新文件名。
构建前后都要按 `simulation/vision_lab/robot_assets_manifest.json` 复核受保护资产
哈希。

### 7.2 场景对象

新建根对象 `/VisionQualityLab`，至少包含：

```text
/VisionQualityLab
├── /InspectionBoard
├── /Samples
│   ├── /ReferenceRectangle
│   ├── /ReferenceCircle
│   ├── /ReferenceTriangle
│   └── /ResolutionTarget
├── /CameraRig
│   └── /Camera
└── /Lighting
    ├── /KeyLight
    └── /FillLight
```

标准件全部由 CoppeliaSim 基础几何体和程序生成纹理组成。构件保持静止、彼此
分离、完整位于基准视野中。`ResolutionTarget` 使用程序生成的黑白条带，不引入
外部版权素材。

### 7.3 场景构建规则

- `scene_spec.json` 是人工维护的声明式输入；
- `BL23_vision_quality_lab.ttt` 和 `scene_manifest.json` 由在线构建命令生成；
- 构建过程使用暂存文件、独占锁和原子替换；
- `template` 与 `output` 必须是不同文件；
- 场景清单记录模板哈希、输出哈希、文件大小、必需对象路径和配置档哈希；
- 失败不得覆盖已有可恢复版本；
- 正式 `.ttt` 只有在在线场景验证通过后进入提交。

## 8. 视觉配置档契约

### 8.1 固定配置档

`profiles.json` 使用 `schema_version=1`，固定三个配置档：

| ID | 分辨率 | 透视视场角 | 相机架世界 Z | 主光漫反射 | 补光漫反射 | 教学用途 |
|---|---:|---:|---:|---:|---:|---|
| `standard` | 512×512 | 60° | 0.70 m | 0.80 | 0.35 | 基准对照 |
| `wide_dim` | 256×256 | 75° | 0.80 m | 0.35 | 0.15 | 宽视场、低分辨率和弱光 |
| `detail_bright` | 768×768 | 40° | 0.60 m | 1.00 | 0.55 | 窄视场、高分辨率和强光 |

漫反射值应用于 RGB 三个通道。镜头近裁剪面固定为 `0.05 m`，远裁剪面固定为
`2.00 m`。相机架 X/Y、相机朝向、灯具位置和镜面分量由场景固定，不开放给
学生。

### 8.2 JSON 结构

```json
{
  "schema_version": 1,
  "baseline_profile_id": "standard",
  "sensor_path": "/VisionQualityLab/CameraRig/Camera",
  "camera_rig_path": "/VisionQualityLab/CameraRig",
  "key_light_path": "/VisionQualityLab/Lighting/KeyLight",
  "fill_light_path": "/VisionQualityLab/Lighting/FillLight",
  "profiles": [
    {
      "profile_id": "standard",
      "label": "标准视图",
      "resolution": [512, 512],
      "perspective_angle_deg": 60.0,
      "camera_rig_z_m": 0.70,
      "key_diffuse_rgb": [0.80, 0.80, 0.80],
      "fill_diffuse_rgb": [0.35, 0.35, 0.35]
    }
  ]
}
```

运行时类型位于 `vision_platform/vision_quality/models.py`。配置加载器要求字段
精确、ID 唯一、数值有限、分辨率在 `[128, 1024]`、视场角在 `[20°, 90°]`、
相机架 Z 在 `[0.50 m, 0.90 m]`、RGB 分量在 `[0, 1]`。所有对象路径必须等于
发布文件中的固定路径，不接受实验 JSON 覆盖。

## 9. 受控配置控制器

`vision_platform/vision_quality/controller.py` 提供：

```python
class VisionProfileController:
    def current(self) -> AppliedVisionProfile: ...
    def apply(self, profile_id: str) -> AppliedVisionProfile: ...
    def reset(self) -> AppliedVisionProfile: ...
```

### 9.1 初始化

控制器只接受已经验证的 `VisionProfileCatalog`、CoppeliaSim `sim` 对象和
已打开的 `CoppeliaSimCamera`。初始化时解析四个固定对象路径并缓存句柄，
拒绝 replay、海康或真实机器人后端。

### 9.2 原子切换

`apply()` 按以下顺序执行：

1. 验证 `profile_id` 位于配置档目录和当前实验的 `allowed_profile_ids`；
2. 读取并保存相机分辨率、视场角、相机架位置和两路灯光参数；
3. 设置目标配置；
4. 回读全部参数，并使用固定容差比较；
5. 丢弃配置切换后的第一帧，再采集稳定帧；
6. 如果任一步失败，按快照反向恢复并回读验证；
7. 回滚也失败时返回稳定错误码并请求运行器隔离后端。

控制器使用互斥锁，禁止两个命令同时修改配置。`current()` 只返回 JSON 原生
字段，不泄露句柄、对象路径以外的 Remote API 对象或客户端。

### 9.3 复位

`reset()` 等价于应用 `standard`，但单独记录为复位动作。学生显式复位、
程序停止、失败、超时和正常结束都必须最终尝试复位。运行器先复位视觉配置，
再执行吸盘关闭和机械臂安全清理；任何复位失败都进入 `cleanup_errors`，不能
把运行写成 PASS。

## 10. 学生命令协议与 SDK

新增三个白名单命令：

```text
camera.profile.get
camera.profile.apply
camera.profile.reset
```

SDK 保持在现有 `StudentCamera` 上：

```python
state = ctx.camera.get_profile()
state = ctx.camera.apply_profile("wide_dim")
state = ctx.camera.reset_profile()
frame = ctx.camera.capture()
```

命令参数契约：

- `get` 和 `reset` 不接受参数；
- `apply` 只接受一个非空 ASCII `profile_id`；
- 额外字段、对象路径、分辨率、角度、位置和光照数值一律拒绝；
- 只有实验定义声明 `camera.profile` 和 `lighting.profile` 能力时才允许调用；
- 只有 `public_parameters.allowed_profile_ids` 中的 ID 可以应用；
- 所有返回值必须能被 `json.dumps(..., allow_nan=False)` 序列化。

`camera.capture` 保持兼容。V1-01 中，快照元数据额外记录当前配置档 ID、
分辨率、视场角、相机架 Z 和光照摘要。其他实验没有视觉配置控制器时，原有
快照字段不变。

## 11. 通用视觉结果与证据

### 11.1 结果包

新增 `vision_platform/vision_quality/results.py`，定义与算法无关的：

```python
VisionImageLayer
VisionResultBundle
```

`VisionImageLayer` 包含稳定的 ASCII `layer_id`、中文标题、只读语义的 BGR
`uint8` 图像和元数据。`VisionResultBundle` 包含：

- `schema_version=1`；
- `bundle_id`；
- `experiment_id`；
- `source_snapshot_id`；
- `status`；
- `layers`；
- JSON 原生 `result`；
- 当前视觉配置摘要；
- `hardware_status=PENDING_HARDWARE`。

V1-01 每次采集形成只有 `raw` 图像层的结果包。未来 V1-02 至 V1-05 适配器
可以增加 `gray`、`foreground_mask`、`cleaned_mask`、`annotated` 等图像层，
无需修改 UI 或证据文件结构。

### 11.2 证据写入

`vision_platform/vision_quality/evidence.py` 只调用 `StudentRunEvidence` 已公开的
`record_snapshot()` 和 `record_json_artifact()`，不访问其私有写入函数。证据
沿用现有扁平、可校验的运行目录结构：

```text
frames/
├── frame-000001.png
├── frame-000001-foreground-mask.png
└── frame-000001-annotated.png
vision-bundle-frame-000001.json
```

`camera.capture` 已经生成的 `frame-000001.png` 直接作为 `raw` 层引用，不重复
写图。未来算法中间层通过额外的 `record_snapshot()` 写入。结果包 JSON 记录每
个图层的相对路径、尺寸和 SHA-256。V1-01 只包含 `raw` 层。写入器验证名称、
尺寸、PNG 哈希、相对路径和 JSON 原生字段；任何失败都不得留下一个声称完整
但缺少图层的结果包。证据目录不进入 Git。

## 12. PyQt 通用视觉结果页

新增 `vision_platform/ui/vision_result_panel.py`，在主窗口增加“视觉结果”页。
页面包含：

- 最近结果包状态和当前配置档；
- 图像层下拉框；
- 图像预览；
- 分辨率、视场角、相机高度和光照摘要；
- 只读格式化 JSON 结果；
- `PENDING_HARDWARE` 和“不是课程成绩”边界提示。

面板只加载当前运行证据目录内的相对路径，校验文件哈希后显示。它不执行
学生代码、不调用算法、不修改场景，也不读取任意外部路径。主窗口在学生运行
进入终态并生成结果包后刷新该页；没有结果包时稳定显示空状态。

UI 静态验收覆盖 100% 和 125% 缩放、窗口关闭、实验切换、空状态、损坏证据
以及长中文文本。截图证明界面可渲染，不等同于教学效果验收。

## 13. V1-01 正式实验

新增：

```text
config/experiments/V1-01.json
docs/experiments/V1-01.md
student_programs/templates/v1_01_virtual_vision.py
```

实验定义：

- `pack_id`: `V1`；
- `version`: `2.2.0`；
- 场景：`simulation/vision_quality_lab/BL23_vision_quality_lab.ttt`；
- 能力：`camera.rgb`、`camera.profile`、`lighting.profile`、
  `experiment.info`、`scene.probe`；
- 允许配置档：`standard`、`wide_dim`、`detail_bright`；
- 基准配置档：`standard`；
- 探针：`vision_profile_observation`；
- 硬件状态：`PENDING_HARDWARE`。

模板按固定顺序应用三个配置档、采集图像、记录尺寸和观察摘要，最后显式复位。
学生可以修改比较顺序和日志，但不能修改配置档定义。现有
`vision-platform experiments list/run` 和 `tools/vision_lab/run_experiment.ps1`
作为正式 CLI/PowerShell 入口，不新增重复启动器。

## 14. 能力与探针

能力注册新增：

- `camera.profile`：仅当相机后端为 `sim`、场景连接存在且配置档控制器可创建；
- `lighting.profile`：仅当两路固定灯具存在且配置档控制器可创建。

`vision_profile_observation` 探针读取：

- 必需场景对象路径；
- 当前分辨率；
- 当前视场角；
- 相机架 Z；
- 两路灯具状态和漫反射值；
- 当前参数是否精确匹配某个发布配置档。

初始探针必须匹配 `standard`。运行结束的安全清理在最终探针前恢复
`standard`，因此最终探针也必须匹配基准。学生实际应用过哪些配置档由命令
证据和结果包证明，不能仅用最终场景状态推断。

## 15. 错误模型

稳定错误码至少包括：

```text
VISION_PROFILE_CONTEXT_REQUIRED
VISION_PROFILE_ID_INVALID
VISION_PROFILE_NOT_ALLOWED
VISION_PROFILE_BACKEND_UNAVAILABLE
VISION_PROFILE_OBJECT_MISSING
VISION_PROFILE_APPLY_FAILED
VISION_PROFILE_READBACK_MISMATCH
VISION_PROFILE_ROLLBACK_FAILED
VISION_PROFILE_RESET_FAILED
VISION_RESULT_BUNDLE_INVALID
VISION_RESULT_EVIDENCE_INVALID
```

参数错误不会调用 CoppeliaSim。配置应用失败先回滚；回滚失败会隔离后端，
阻止后续命令和实验切换，直到安全复位。UI 显示稳定错误码和截断后的消息，
详细异常进入开发证据，不向学生暴露远端对象或本机敏感路径。

## 16. 数据流

### 16.1 正常运行

```text
选择 V1-01
  → 场景会话加载 vision_quality_lab
  → 能力检查
  → 初始探针确认 standard
  → 学生程序申请 wide_dim
  → 网关校验 allowlist
  → 控制器快照、设置、回读
  → camera.capture 记录配置元数据
  → 生成 raw 结果包
  → 重复 detail_bright 与 standard
  → 视觉复位
  → 最终探针确认 standard
  → 写 summary.json
  → PyQt 加载最近结果包
```

### 16.2 配置失败

```text
apply(profile)
  → 设置失败或回读不一致
  → 使用原始快照回滚
  ├── 回滚成功：当前命令 FAIL，后端可继续受控使用
  └── 回滚失败：当前命令 FAIL，后端隔离，运行 FAIL
```

## 17. 测试策略

### 17.1 静态单元测试

- 配置档严格字段、数值范围、唯一 ID 和递归不可变；
- 控制器设置顺序、回读、容差、回滚、并发锁和复位；
- 学生命令白名单、参数拒绝、能力拒绝和 JSON 返回；
- V1-01 模板只使用公开 SDK；
- 结果包、PNG/JSON 证据、路径约束和损坏文件拒绝；
- PyQt 空状态、图像层切换、JSON 显示和硬件边界；
- 场景 spec、manifest、profiles 和受保护资产合同；
- 实验目录、指南、CLI 和 PowerShell 通用入口。

### 17.2 CoppeliaSim 在线测试

在线测试必须显式启用，至少验证：

1. 新场景可以独立加载并开始仿真；
2. 必需对象路径和受保护资产合同通过；
3. 三个配置档逐一设置并回读成功；
4. 每个配置档采集的图像尺寸与配置一致且 PNG 非空；
5. 三个配置档生成不同且可复核的图像哈希/摘要；
6. 显式复位和运行结束清理都恢复 `standard`；
7. V1-01 模板通过受控学生运行器完成；
8. 初始与最终探针均 PASS；
9. summary 和视觉结果包保持 `PENDING_HARDWARE`。

普通静态套件中的在线测试可以 skip，但 skip 不能写成在线 PASS。只有保存了
命令、输出和证据路径的显式在线运行才能支持 CoppeliaSim 通过结论。

### 17.3 UI 验收

- Qt offscreen 自动测试；
- 100% 与 125% 缩放截图；
- 页面文字、表格和图像不遮挡；
- 损坏/缺失结果包显示安全错误，不崩溃；
- 教师人工检查项保持 `PENDING_HUMAN_ACCEPTANCE`。

### 17.4 完整回归

完成后运行完整静态套件，要求零失败。报告精确 pass/skip 数；skip 不计为
PASS。重新运行现有 CoppeliaSim 场景、V2.1 学生流程和 V2.2 R1 实验的显式
在线回归，且不得把仿真结果写成真机结果。

## 18. 文件所有权与并行开发边界

### 18.1 本分支主要拥有

```text
simulation/vision_quality_lab/**
vision_platform/vision_quality/**
vision_platform/ui/vision_result_panel.py
config/experiments/V1-01.json
docs/experiments/V1-01.md
student_programs/templates/v1_01_virtual_vision.py
tests/test_vision_quality/**
```

### 18.2 本分支允许的共享接入点

```text
config/experiments/catalog.json
simulation/training_scenes/build_scene.py
simulation/training_scenes/scene_contract.py
vision_platform/experiments/capabilities.py
vision_platform/experiments/probes.py
vision_platform/student/protocol.py
vision_platform/student/sdk.py
vision_platform/student/experiment_gateway.py
vision_platform/student/runner.py
vision_platform/ui/pyqt_app.py
tests/test_acceptance/**
tests/test_experiments/**
tests/test_student_programs/**
tests/test_vision_platform/**
RETAINED_FILES.txt
```

共享接入点只做 V1-01 所需的最小修改，不重构现有 R1 行为。

### 18.3 本分支禁止修改

```text
vision_platform/vision2d/**
tests/test_vision2d/**
docs/experiments/V1-02.md
docs/experiments/V1-03.md
docs/experiments/V1-04.md
docs/experiments/V1-05.md
vision_platform/recognition/color_shape.py
simulation/vision_lab/BL23_vision_lab.ttt
simulation/vision_lab/robot_assets/**
所有 URDF 和 STL
```

算法分支完成后，由单一集成端把算法提交合入 `v2.2.0` 后继分支，再单独规划
V1-02 至 V1-05 的目录、模板、结果适配和在线实验。两个分支不互相 cherry-pick
未完成提交。

## 19. 发布与验收边界

新增正式文件全部加入 `RETAINED_FILES.txt`，继续使用
`tests/test_acceptance/test_delivery_contract.py` 作为发布合同。正式发布报告必须
分别陈述：

- 静态测试结果；
- 显式 CoppeliaSim 在线结果；
- UI 自动/截图结果；
- 教师人工检查状态；
- 真机与海康硬件状态。

允许的完成声明是：

> V2.2-C0 视觉实验公共底座和 V1-01 在受控 CoppeliaSim 仿真中通过已执行的
> 自动测试与在线验收。

以下状态保持不变：

```text
教学效果：PENDING_HUMAN_ACCEPTANCE
海康相机：PENDING_HARDWARE
真实机械臂：PENDING_HARDWARE
急停与气路：PENDING_HARDWARE
物理抓取与真实测量精度：PENDING_HARDWARE
```

## 20. 完成条件

本阶段只有同时满足以下条件才可完成：

1. 新场景、配置档、V1-01 定义、指南和模板进入发布白名单；
2. 学生只能通过配置档 ID 控制相机和光照；
3. 配置设置、回读、回滚和复位均有单元测试；
4. 每次 V1-01 采集都记录配置摘要和视觉结果包；
5. PyQt 能安全显示原图层和结构化结果；
6. 通用结果包不依赖 `vision_platform/vision2d`；
7. 静态完整回归零失败；
8. 显式在线测试验证三档切换、采集、复位和学生模板；
9. 受保护正式场景、机器人资产、URDF 和 STL 无变化；
10. 并行算法分支拥有的路径无变化；
11. 所有证据边界和 PENDING 状态写入交付报告；
12. 工作树干净，提交按 Task 可追溯，并推送独立功能分支。
