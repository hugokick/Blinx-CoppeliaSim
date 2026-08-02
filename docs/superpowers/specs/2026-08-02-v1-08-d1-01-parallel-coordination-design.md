# V2.2 V1-08 与 D1-01 并行开发协调设计

**状态：** 已有课程路线和并行架构确认后的执行设计

**日期：** 2026-08-02

**共同基线：** `origin/main` 的 `53b440ee177fe25cbd98d227369a8f99b6b3c2e0`

**协调分支：** `codex/v2-2-v1-08-d1-01-coordination`

## 1. 阶段目标

本阶段采用一条正式课程主线和一条隔离仿真底座并行推进：

- 主线开发端交付 `V1-08 OCR 字符训练、编号识别与机器人分拣` 的完整虚拟教学闭环；
- 并行开发端交付 `D1-01 CoppeliaSim RGB-D 仿真底座`，证明同一传感器彩色图、米制深度图、内参和场景证据可被一致采集；
- 两端从同一精确 `origin/main` 建立独立工作树，按文件所有权隔离；
- D1-01 本轮不接入正式实验目录、学生 SDK、PyQt、CLI 或机器人控制，待 V1-08 合入主线后再进行低冲突正式课程集成。

这样安排的原因是 V1-08 必须修改课程目录、学生协议、安全网关、UI、CLI 和交付合同；若 D1-01 同时修改这些共享入口，两端会在高风险文件上直接冲突。D1-01 先把独立场景、传感器适配、帧合同、静态测试和真实 CoppeliaSim 在线探针做实，可在不阻塞 V1-08 的前提下形成下一轮可集成候选。

## 2. 启动硬门禁

两端启动时都必须重新执行以下门禁，不得仅依赖本设计记录的快照：

1. `git fetch origin --prune`；
2. `git rev-parse origin/main` 必须等于规划发布后的精确主线提交；
3. 本设计、两份逐 Task 实施计划、V1-07 正式课程、`vision_platform/vision2d/ocr.py` 和 `vision_platform/rgbd/**` 必须都存在于该提交；
4. 工作树必须干净；
5. 全仓静态测试零失败，skip 单独列出，不能算在线 PASS；
6. 若精确提交或文件门禁不满足，连续刷新三次仍不满足时报告 `BASELINE_GATE_NOT_READY` 并停止，不得从本地候选分支替代主线。

规划文件合入后，开发 Prompt 将给出新的精确 `origin/main` SHA；`53b440e` 只表示本设计开始编写时的已核验输入基线。

无人值守并行运行的端口所有权固定为：V1-08 构建、探针和在线验收使用 `23008`；D1-01 使用 `23009`。两端的 wrapper、测试和证据必须断言自己的端口，端口已被非预期进程占用时失败关闭，禁止自动改用对方端口或清理对方进程。

## 3. 总体数据流

### 3.1 主线 V1-08

```text
独立 OCR 分拣场景复位
  -> 主机读取哈希绑定的原创字形训练集
  -> 固定 KNN/seed/train-test split 训练并验证准确率门槛
  -> 固定 profile 采集四个编号构件
  -> OCR 识别 A1、A2、B1、B2
  -> 全量匹配主机白名单并冻结分拣计划
  -> 计划完整后才允许学生选择白名单条目顺序
  -> 现有安全运动和吸盘接口逐件分拣
  -> 场景探针验证两类仓位占用、机器人回零、吸盘关闭
  -> 保存同一次运行的训练、视觉、计划、运动和终态证据
```

### 3.2 并行 D1-01

```text
独立 RGB-D 场景和固定传感器
  -> CoppeliaSim 单次显式传感器处理
  -> RGB buffer 与 options=1 的米制 source-depth buffer
  -> 统一翻转和 RGB->BGR
  -> 读取分辨率、视场角、near/far 和传感器位姿
  -> 在线中心/离轴探针核验 source depth model
  -> 显式归一化为 optical-axis Z depth
  -> 构造 CameraIntrinsics 与不可变 RgbdFrame
  -> 保存彩色预览、可视化深度预览和 JSON 证据
  -> 在线探针验证尺寸对齐、有限深度、已知几何顺序和场景哈希
```

## 4. V1-08 正式课程架构

### 4.1 课程目标和固定任务

实验使用四个带原创印刷编号的可抓取构件：

| part_id | 固定编号 | 固定路由 | 教学含义 |
|---|---:|---|---|
| `part_a` | `A1` | `route_alpha` | A 类物料 |
| `part_b` | `A2` | `route_alpha` | A 类物料 |
| `part_c` | `B1` | `route_beta` | B 类物料 |
| `part_d` | `B2` | `route_beta` | B 类物料 |

学生学习训练集/测试集、字符分割、分类置信度、编号白名单和分拣计划先行原则。编号不是脚本、路径、坐标或机器人命令；它只作为实验配置中精确白名单的查表键。

### 4.2 原创训练与场景资产

新增独立场景族：

```text
simulation/vision_ocr_sorting_lab/
  BL23_vision_ocr_sorting_lab.ttt
  scene_manifest.json
  profiles.json
  ocr_assets_manifest.json
  training/
  labels/
```

字形只覆盖 `A`、`B`、`1`、`2`。必须由仓库内的确定性 5x7 位图生成器生成，不依赖系统字体、Tesseract、外部 OCR 模型、网络下载或第三方训练集。每类至少生成八个固定变体；训练样本和场景编号面共享同一原始字形语义，但不能让 held-out 测试样本与训练图片字节完全相同。

`ocr_assets_manifest.json` 记录：生成器版本、seed、字符集合、样本相对路径、用途、尺寸、SHA-256、训练参数和允许的场景编号。所有哈希绑定文本文件在 `.gitattributes` 中固定 `text eol=lf`。

不得修改现有正式 `.ttt`、URDF、STL、网格、机器人模型或 `simulation/vision_lab/BL23_vision_lab.ttt`；新场景可由受控构建脚本以该正式场景为模板生成，但必须输出自己的场景和清单。

### 4.3 主机拥有的 OCR 和分拣计划

复用已集成的纯算法：

- `train_glyph_classifier(samples, method="knn", test_fraction=0.25, seed=20260802)`；
- `recognize_text(image, model, expected_text=...)`；
- 结构化 `TrainingReport`、`OCRResult`、字符框、置信度和失败码。

新增正式适配层负责：

1. 只从哈希绑定清单读取训练图片；
2. 验证路径仍位于实验资产根目录、文件尺寸和 SHA-256 精确匹配；
3. 每次实验会话按固定参数训练一次模型，模型对象不进入学生协议、JSON 或持久化文件；
4. held-out accuracy 未达到配置门槛时失败关闭；
5. 从一张固定在线帧中提取四个场景清单绑定的编号 ROI；
6. 对每个 ROI 识别精确二字符编号并形成不可变计划；
7. 任何未知、重复、缺失、低置信度、错误长度、非法框、场景/清单不一致都在运动前拒绝。

学生侧提供两个有限接口：

- `ctx.vision2d.ocr_sorting()`：无参数，只触发主机训练、采集、识别和完整计划激活；
- `ctx.vision2d.sort_ocr_entry(entry_id)`：唯一参数是当前冻结计划中尚未消费的白名单 `entry_id`，由主机完成该条目的整段取放动作并返回只读回执。

识别命令不得接受任何参数；动作命令不得接受坐标、路径、编号、路由、速度或嵌套对象。两者都不得接受训练目录、文件路径、字符表、模型、算法方法、阈值、ROI、payload 或任意命令字符串。返回值只含 JSON 原生的教学字段和证据引用。

### 4.4 安全运动合同

新增 V1-08 专用 `OcrSortPlan` 和纯状态 `OcrSortGuard`，不复用或篡改 V1-07 payload 语义。只有完整计划原子激活后，`sort_ocr_entry(entry_id)` 才把严格 ID 交给主机守卫；守卫只生成和推进冻结的有限动作状态，不持有也不调用 raw robot/tool。Runner 的窄安全执行器逐 waypoint 复用现有工作空间/速度校验、停止检查、暂停/继续/单步、位姿更新、命令证据和 tool-off 清理，再逐步向守卫确认。学生程序不提交坐标，也不直接调用 V1-08 的机器人/吸盘动作。

安全规则：

- 全部四个编号和路由通过校验之前，任何 `robot.move_world` 或 `tool.suction` 必须被拒绝；
- V1-08 的公开 capability 列表不发布 `robot.move_world`、`robot.home`、`robot.pose` 或 `tool.suction`；学生自编程序在 EMPTY、ACTIVE、ENTRY_ACTIVE、完成或失败状态直接发送任何 raw robot/tool 命令都必须失败关闭且设备零调用；
- 学生只可选择未消费的 `entry_id` 顺序，不能提交任意世界坐标；
- 取料点、放料点、安全高度、速度、工作空间和仓位来自哈希绑定配置；
- 同一条目最多消费一次；全部动作完成后 runner 必须调用绑定同一 run/scene/part/route/slot 的逐条目占位探针，再以 `confirm_entry_probe` 原子标记消费，动作或探针失败都使计划失效；
- 每个 waypoint/tool step 前都必须执行 runner 的 stop 检查和 pause/single-step 门禁；stop/reset 必须清除计划；暂停期间不得执行新动作；
- 只有 `_command_ocr_sort_entry` 的私有安全执行路径可触达设备；内部路径仍必须复用现有运动校验、证据和清理逻辑，不能通过公开协议递归调用被禁命令；
- 运行失败时优先保留首个错误，尽力关闭吸盘和停止后续动作；安全条件满足时才回零；
- 成功时必须证明四个构件分别进入两个固定仓位、机器人回到 home、吸盘关闭且无学生子进程残留。

### 4.5 课程入口和证据

V1-08 作为正式课程必须完整接入：

- `config/experiments/V1-08.json` 与 `catalog.json`；
- 学生协议、SDK、实验网关、能力检查和运行器；
- 学生模板、CLI、PowerShell、PyQt 课程入口和只读结果面板；
- 中文指导书、静态材料合同、发布清单和验收脚本；
- 使用独立端口、精确进程归属和真实 CoppeliaSim 的在线验收。

每次运行的证据至少包括：实验/场景/资产清单 SHA、训练报告、原始帧、标注帧、四个字符级结果、冻结计划、命令/事件、最终仓位和清理状态。结果面板只读，不能把识别文本或坐标重新作为可执行输入。

## 5. D1-01 CoppeliaSim RGB-D 仿真底座架构

### 5.1 范围和独立目录

并行端新增独立范围：

```text
vision_platform/rgbd_sim/
tests/test_rgbd_sim/
simulation/rgbd_lab/
tools/rgbd_lab/
docs/validation/d1-01-rgbd-sim-foundation-validation.md
```

`vision_platform/rgbd_sim/**` 是仿真适配层；现有 `vision_platform/rgbd/**` 继续保持纯内存算法边界。适配层复用现有 `CoppeliaClientResolver` 连接模式、`CoppeliaSimCamera` 已验证的翻转/颜色转换规则以及 `CameraIntrinsics`、`RgbdFrame` 合同，但为读取实时传感器参数和显式 source-depth model，必须在新包内完成单一 RGB-D 采集事务，不能把现有相机返回的未标注 `depth_m` 直接送入反投影。任何 `sim` 对象、句柄、文件、网络连接或可变数组都不得泄漏给纯算法包。

本轮禁止修改：`vision_platform/student/**`、`vision_platform/experiments/**`、正式实验 catalog/课程 CLI、`tools/vision_lab/**` PowerShell、UI、V1-08 文件和任何现有正式场景。并行端只可在自己的 `tools/rgbd_lab/**` 下新增专用构建/探针脚本。

### 5.2 独立 RGB-D 场景

新场景 `simulation/rgbd_lab/BL23_rgbd_lab.ttt` 至少包含：

- `/RgbdLab/CameraRig/RgbdSensor`：固定 `256x256`、60° 最大开角、near/far、透视模式、RGB/depth 均启用，并强制 `explicit_handling=true`；
- `/RgbdLab/ReferencePlane`：已知背景深度；
- `/RgbdLab/Targets/near_block` 和 `/RgbdLab/Targets/far_block`：同一视野中深度有严格顺序；
- `/RgbdLab/Targets/step_low` 和 `/RgbdLab/Targets/step_high`：用于彩色/深度对齐和层级可见性；
- `/RgbdLab/ProbeAnchors/center`、`off_axis_left` 和 `off_axis_right`：用已知平面表面验证 source depth 是光轴 Z 还是射线距离；
- 场景、profile、对象路径、传感器参数和受保护资产哈希清单。

构建脚本必须可重复生成新场景，不得覆盖源模板。场景在线验证只能声明 CoppeliaSim 虚拟 RGB-D 通过，不能映射为真实深度相机精度或机器人抓取通过。

### 5.3 采集合同

采集分为两个不可混淆的不可变合同：

- `RgbdSourceCapture`：保留同周期 BGR 和 options=1 的原始米制 `source_depth_m`，场景声明仅记为 `expected_source_depth_model`；
- `RgbdSimCapture`：只有中心/离轴探针从原始 source depth 得出唯一 `observed_source_depth_model` 且与场景期望一致后才能创建，包含归一化 optical-Z `RgbdFrame`。

两个合同共同只公开：

- 不可变 `RgbdFrame(image_bgr, depth_m)`；
- `CameraIntrinsics`；
- `sensor_path`、`resolution`、`near_clip_m`、`far_clip_m`、`perspective_angle_rad`；
- `scene_path`、`scene_sha256`、单调 `sequence_id` 和采集时间；
- 明确的 `vertical_flip`、`RGB_to_BGR`、`source_depth_model`、`output_depth_model=optical_z`、深度单位 `metre` 和像素中心约定。

RGB 与深度必须来自同一个传感器处理周期，尺寸完全一致。适配器必须先断言 `getExplicitHandling(sensor)==1`，然后每次采集恰好调用一次 `handleVisionSensor(sensor)`；非显式传感器或重复处理都失败关闭，不能读取主循环留下的旧 buffer。CoppeliaSim `getVisionSensorDepth(..., 1)` 只证明返回值以米计；正式适配器不得未经验证就假设它等于纯内核反投影所需的光轴 Z。场景 profile 必须声明预期 `source_depth_model` 为 `optical_z` 或 `ray_range`，但分类器必须仅根据原始中心/离轴平面样本和已知场景表面几何得出观测模型，不能把声明当答案。声明与唯一观测不一致或两模型无法区分时失败关闭。若观测为 `ray_range`，按像素归一化坐标显式换算 `z = range / sqrt(1 + x_n^2 + y_n^2)`；只有验证和换算后才能构造 `RgbdFrame.depth_m`。不得执行隐藏的 near/far 归一化。

首版在线场景固定方形传感器，避免最大开角在非方形图像中的歧义：`fx=fy=width/(2*tan(angle/2))`，`cx=cy=(width-1)/2`。通用内参函数仍须通过横向、纵向和方形单元测试；实现前须用当前安装的 CoppeliaSim API 和在线探针确认 perspective mode、分辨率、RGB/depth ignore flag、最大开角、near/far 和 source-depth 语义。

### 5.4 预览和验证证据

底座提供受控 probe/CLI 工具，仅用于开发验收，不进入学生 SDK。输出：

- 原始 BGR PNG；
- 对零/无效值有显式颜色的深度预览 PNG；
- JSON 证据：分辨率、source/output depth model、深度最小/中位/最大、中心/离轴及固定 ROI 中位深度、内参、场景/清单哈希、进程和清理状态；
- 静态/在线测试结果以及明确的 `PENDING_HARDWARE`、`PENDING_HUMAN_ACCEPTANCE`。

probe 不写入 Git 跟踪目录中的运行产物；验收报告只记录命令、摘要和证据路径，不提交大图、日志或缓存。

本设计的 CoppeliaSim API 边界以官方文档为准：`sim.getVisionSensorDepth` 的 bit 0 只承诺“以米返回”，读取前应处理传感器；Vision Sensor 的 perspective angle 是检测体最大开角；透视模式、分辨率、RGB/depth ignore、near/far 和 angle 均有可读参数。实现和在线证据必须核对当前安装版本，而不能只依赖静态假设：

- <https://manual.coppeliarobotics.com/en/sim/simGetVisionSensorDepth.htm>
- <https://manual.coppeliarobotics.com/en/visionSensorPropertiesDialog.htm>
- <https://manual.coppeliarobotics.com/en/objectParameterIDs.htm>

## 6. 文件所有权

| 范围 | 主线 V1-08 | 并行 D1-01 |
|---|---|---|
| `vision_platform/student/**` | 独占 | 禁止修改 |
| `vision_platform/experiments/**`、catalog | 独占 | 禁止修改 |
| `vision_platform/ui/**`、正式课程 CLI、`tools/vision_lab/**` PowerShell | 独占 | 禁止修改 |
| `simulation/vision_ocr_sorting_lab/**` | 独占 | 禁止修改 |
| `vision_platform/vision2d/**` | 仅最小正式适配/导出 | 禁止修改 |
| `vision_platform/rgbd/**` | 禁止修改 | 只读复用，原则上禁止修改 |
| `vision_platform/rgbd_sim/**` | 禁止修改 | 独占 |
| `tests/test_rgbd_sim/**`、`simulation/rgbd_lab/**`、`tools/rgbd_lab/**`（含专用构建/探针 PowerShell） | 禁止修改 | 独占 |
| 各自专项测试、计划执行报告 | 各自独占 | 各自独占 |
| `RETAINED_FILES.txt` | 只追加本端文件 | 只追加本端文件，集成时取并集 |
| `.gitattributes` | 只追加 OCR 哈希文本规则 | 只追加 RGB-D 哈希文本规则，集成时取并集 |
| `tests/test_acceptance/test_delivery_contract.py` | 独占 | 禁止修改 |

两端都不得删除、重排或覆盖共享条目。发现必须跨界的改动时，先在交付报告提出接口建议，不能直接越界实现。

## 7. 分支、提交和集成顺序

规划发布后的开发分支固定为：

- 主线：`codex/v2-2-v1-08-ocr-sorting`；
- 并行：`codex/v2-2-d1-01-rgbd-sim-foundation`。

每个 Task 都执行 RED → 确认预期失败 → 最小 GREEN → 专项回归 → 相关全仓回归 → 独立提交。两端只推送自己的分支，不合并 `main`，不互相 cherry-pick，也不在开发期间合并对方分支。

集成顺序固定为：

1. 独立审核 V1-08，修复 P0/P1/P2，运行静态、在线仿真、UI 和发布合同；
2. 用户授权后先把 V1-08 合入 `main`；
3. 将 D1-01 候选重放/合并到新的主线集成分支，仅对 `.gitattributes` 和 `RETAINED_FILES.txt` 做并集式解决；
4. 运行 D1-01 静态与真实 CoppeliaSim 在线验收；
5. 再规划 D1-01 正式课程目录、学生 SDK 和 UI 接入。

## 8. 完成定义和结论边界

V1-08 完成仅在以下条件同时满足时成立：独立场景、原创训练资产、固定训练报告、四编号全量识别、计划前零运动、四次受控分拣、终态探针、同次运行证据、PyQt/CLI/PowerShell 和发布合同全部通过。

D1-01 仿真底座完成仅在以下条件同时满足时成立：独立场景、同周期彩色/米制 source depth 采集、source-depth 语义在线判别、显式 optical-Z 输出、内参合同、预览、严格失败测试、真实 CoppeliaSim 在线探针和运行清理通过。

两条线都必须保持：

- 教学效果：`PENDING_HUMAN_ACCEPTANCE`；
- 真实海康/深度相机、机械臂、急停、气路和物理抓取：`PENDING_HARDWARE`；
- 静态 skip：只能报告为 skip，不能写成在线 PASS；
- V2.3 真机迁移、D1-02 三维坐标正式课程、D1-03 深度引导抓取、V1-09 缺陷分流均不在本阶段实现。
