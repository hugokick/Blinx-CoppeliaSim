# V2.2 V1-09 构件表面缺陷检测与受控分流设计

**状态：** 对话设计已确认，书面规范待用户审核

**日期：** 2026-08-03

**设计基线：** `origin/main` / `1c11302811a070d62c4d440a012a0399027a1683`

**设计分支：** `codex/v2-2-v1-09-defect-sorting-design`

**设计原则：** 在安全门禁、证据可信和正式资产保护边界内，优先呈现完整、直观的实验仿真效果。

## 1. 目标

V1-09 将已经进入主线的纯二维表面缺陷算法内核，集成为一个正式的 CoppeliaSim 机器人视觉教学实验。学生应能观察并解释以下完整闭环：

```text
固定参考件 + 六个候选构件
          ↓
同一预运动帧中的固定 ROI 采集
          ↓
五类表面缺陷检测与合格判定
          ↓
主机冻结完整 DefectSortPlan
          ↓
学生只选择白名单 entry 的执行顺序
          ↓
私有安全执行器逐设备动作完成取放
          ↓
合格槽 + missing/hole/foreign/broken/dimension 五类缺陷槽
          ↓
同次运行探针、图像和动作证据
```

实验的核心教学目标包括：

1. 理解参考图与候选图比较、形态学清理、有限平移对齐和显式阈值；
2. 区分 `missing`、`hole`、`foreign`、`broken` 和 `dimension`；
3. 理解算法结果、失败状态、完整计划和机器人动作之间的安全边界；
4. 在不提交坐标、路径、阈值或原始机器人命令的情况下，运行完整分流流程；
5. 使用图像、掩膜示意、结构化结果、终态槽位和运行哈希解释一次实验。

## 2. 已有资产与本轮范围

### 2.1 已有且保持只读

主线已经包含：

- `vision_platform/vision2d/defect_detection.py`；
- `tests/test_vision2d/test_defect_detection.py`；
- `tests/test_vision2d/synthetic_v107_v109.py`；
- `DefectConfig`、`DefectFinding`、`DefectResult` 和 `detect_surface_defects()`；
- `PASS`、`PARTIAL`、`NO_TARGETS`、`REJECTED` 顶层状态；
- 五类缺陷、无缺陷、亮度、噪声、偏移、空图、尺寸不一致和多组件回归。

正式课程集成不得静默修改上述内核或专项测试。若实施中发现内核缺陷，应停止跨界修改，形成独立修复建议和单独审核分支。

### 2.2 本轮交付

本轮正式交付包括：

- 独立 V1-09 场景和确定性原创表面纹理资产；
- 哈希绑定的参考件、候选件、场景、ROI、路线和阈值合同；
- 主机拥有的完整缺陷检测计划、纯状态守卫和私有安全执行器；
- 学生 SDK、协议、网关、模板、CLI 和 PowerShell 入口；
- PyQt 只读教学结果页；
- 静态、真实 CoppeliaSim 在线、UI 和发布验收；
- 验证报告与 `RETAINED_FILES.txt` 发布登记。

### 2.3 明确不在本轮

本轮不实现：

- 学生自定义缺陷算法、自由图片路径、自由 ROI 或自由阈值；
- raw `robot.*`、`tool.*`、Remote API 或仿真句柄访问；
- 自动评分、模型训练、深度学习缺陷检测或 A1 课程；
- D1-02/D1-03、点云、三维抓取或高度自适应；
- V2.3 真机迁移；
- 真实海康相机、真实机械臂、急停、气路或物理抓取验收。

## 3. 方案选择

设计阶段比较了三种方案：

| 方案 | 内容 | 结论 |
|---|---|---|
| A | 只观察参考图、候选图和缺陷结果，不运动 | 改动最少，但不能形成机器人分流闭环 |
| B | 独立场景、主机完整计划、私有安全执行器、受控分流 | 选定；教学闭环完整且安全边界清楚 |
| C | 抽象 V1-07～V1-09 通用 InspectionPlan 框架 | 暂缓；共享重构和集成风险超出本轮需要 |

本设计采用方案 B，并按用户选择在正式场景中同时放置一个合格件和全部五类缺陷件。缺陷件分别进入五个类型槽，而不是合并为一个通用缺陷箱。

## 4. 不可变资产保护授权

用户明确授权后续只新建：

```text
simulation/vision_defect_sorting_lab/BL23_vision_defect_sorting_lab.ttt
```

所有已有正式 `.ttt` 均为只读，包括但不限于：

- `simulation/vision_lab/BL23_vision_lab.ttt`；
- V1-06、V1-07、V1-08 正式场景；
- D1-01 RGB-D 场景；
- 其他已发布训练场景。

不得修改 URDF、STL、网格或机器人资产。专用构建器可以在验证模板 SHA-256 后只读加载批准模板，但只能保存到新的 V1-09 路径。构建前后必须重新计算所有受保护场景哈希，并证明没有变化。

## 5. 独立场景与原创资产

### 5.1 场景目录

新增场景族：

```text
simulation/vision_defect_sorting_lab/
  BL23_vision_defect_sorting_lab.ttt
  scene_spec.json
  scene_manifest.json
  profiles.json
  defect_assets_manifest.json
  acceptance_ground_truth.json
  assets/
    reference.png
    candidate_a.png
    candidate_b.png
    candidate_c.png
    candidate_d.png
    candidate_e.png
    candidate_f.png
```

`acceptance_ground_truth.json` 只允许在线验收和探针读取；运行时检测服务不得导入它或根据预期标签生成路线。

### 5.2 原创纹理

专用生成器使用固定 seed `20260803` 和明确几何规则生成 256×256 PNG，不依赖系统字体、网络下载、第三方图片或外部模型。清单记录生成器版本、seed、用途、尺寸、相对路径、SHA-256 和原创来源声明 `project-original-generated`。

七张纹理的语义为：

| 资产 | 运行时中性 ID | 仅验收使用的预期结果 |
|---|---|---|
| `reference.png` | `reference` | 完好参考件 |
| `candidate_a.png` | `entry_a` / `part_a` | `qualified` |
| `candidate_b.png` | `entry_b` / `part_b` | `missing` |
| `candidate_c.png` | `entry_c` / `part_c` | `hole` |
| `candidate_d.png` | `entry_d` / `part_d` | `foreign` |
| `candidate_e.png` | `entry_e` / `part_e` | `broken` |
| `candidate_f.png` | `entry_f` / `part_f` | `dimension` |

运行时使用中性 entry/part ID，避免通过对象命名泄露正确分类。

### 5.3 场景对象

独立场景至少包含：

```text
/VisionDefectSortingLab
  /CameraRig/Camera
  /Reference/InspectionFace
  /Parts/part_a/InspectionFace
  /Parts/part_b/InspectionFace
  /Parts/part_c/InspectionFace
  /Parts/part_d/InspectionFace
  /Parts/part_e/InspectionFace
  /Parts/part_f/InspectionFace
  /Slots/slot_qualified
  /Slots/slot_missing
  /Slots/slot_hole
  /Slots/slot_foreign
  /Slots/slot_broken
  /Slots/slot_dimension
```

参考件固定且不可抓取。六个候选件使用相同三维几何、质量、吸取面和抓取高度，仅 `InspectionFace` 纹理不同。六个目标槽具有清晰中文/英文类别标识，并在 CoppeliaSim 视图中保持同时可见。

相机采用固定 1024×1024 标准 profile。参考件和六个候选件在首次运动前位于互不重叠的固定 ROI 中；一次帧采集同时冻结全部七个 ROI，避免候选之间出现不同光照或不同运行时刻。`scene_spec.json` 明确对象位置、相机、光照、抓取点、槽位、工作空间、安全高度和速度；`scene_manifest.json` 绑定最终对象路径、ROI、场景 SHA、模板 SHA 和资产清单 SHA。

## 6. 检测配置与正式决策语义

正式闭环只允许 `standard` 相机/光照 profile。若当前 profile、相机参数或场景状态与绑定配置不一致，分析在运动前失败关闭。学生可以在指导书和回放证据中比较阈值与形态学步骤的影响，但首版正式机器人分流不允许学生覆盖阈值，以免未经验证的参数直接生成运动计划。

正式配置冻结现有算法默认值：

| 字段 | 值 |
|---|---:|
| `missing_ratio` | `0.01` |
| `hole_ratio` | `0.005` |
| `foreign_ratio` | `0.002` |
| `dimension_ratio` | `0.08` |
| `min_component_ratio` | `0.002` |
| `morphology_kernel_size` | `3` |
| `max_alignment_shift_px` | `8.0` |
| `min_contrast` | `4.0` |

学生不得覆盖这些值。主机把配置复制为深度不可变正式合同，并计算 canonical JSON SHA-256。

每个候选只能按下列规则形成路线：

| 内核结果 | 正式决定 |
|---|---|
| `PASS`、`failure_code is None`、无 finding | `qualified` |
| `PARTIAL`、`failure_code == "DEFECTS_FOUND"`、finding 非空且所有 finding 只有一种批准类型 | 对应五类缺陷槽 |
| `NO_TARGETS` 或 `REJECTED` | 失败关闭，不生成计划 |
| `CANDIDATE_EMPTY`、`CANDIDATE_EMPTY_AFTER_ALIGNMENT` 或其他非 `DEFECTS_FOUND` 失败码 | 失败关闭，不把空件误当作 `missing` |
| 同一候选出现两种或以上 defect type | `DEFECT_SORT_DECISION_AMBIGUOUS`，不运动 |
| 缺失、重复或未知 entry/part | `DEFECT_SORT_PLAN_INCOMPLETE`，不运动 |

只有六个 entry 都得到唯一决定、六个 part ID 精确覆盖、一个且仅一个 `qualified`、五类缺陷各出现一次时，主机才原子激活计划。任何一个候选失败都会使整批计划失效，计划激活前设备调用数必须为零。

预期分类只存在于验收 ground truth 中。运行时路线必须完全来自实际 `DefectResult`，不能用候选 ID、文件名或验收标签替代算法判断。

## 7. 主机拥有的数据合同

新增相互隔离的正式组件：

- `defect_assets.py`：只负责路径限制、尺寸和 SHA 校验；
- `defect_sorting.py`：只负责不可变 entry、plan、receipt 和 canonical 序列化；
- `defect_service.py`：只负责一帧采集、ROI 裁剪、六次内核调用和完整计划构建；
- `defect_visualization.py`：只生成教学标注和 finding 区域示意，不参与判定；
- `defect_sort_guard.py`：只管理纯状态和条目消费，不持有设备对象；
- runner 中的 V1-09 私有执行路径：唯一允许触达机器人/工具适配器的组件。

内核 `DefectResult` 中的 mapping 必须在正式层复制、校验并转为稳定的 tuple/只读结构；不能直接把浅冻结对象作为可消费计划。

### 7.1 `DefectSortEntry`

每个 entry 至少绑定：

- `entry_id`、`part_id`；
- 实际检测的 `decision` 和唯一 `defect_type`；
- reference/candidate crop SHA；
- findings 和阈值的 canonical 摘要 SHA；
- 私有 `route_id`、pick/drop/safe waypoint、槽位和工作空间绑定；
- `run_id`、`frame_id`、scene/config/asset SHA。

坐标和 waypoint 不进入学生 DTO。

### 7.2 `DefectSortPlan`

计划必须包含精确六个不可变 entry、完整集合证明、创建时间、schema、run/scene/config/frame 绑定和 `plan_id`。`plan_id` 是 canonical JSON 的 SHA-256。计划一旦激活不得修改，也不得在同一 run 中重新分析覆盖；重新分析必须先执行受控 reset。

### 7.3 `DefectSortReceipt`

回执只在动作完成且逐条目探针确认后生成，至少包含 run/plan/entry、实际决定、目标槽、状态、证据摘要、硬件状态和 schema。失败动作不得伪造消费回执。

## 8. 学生协议与公开能力

V1-09 的 capability 列表只发布：

```text
camera.rgb
camera.profile
lighting.profile
vision2d.surface_defects
```

正式协议命令为：

```python
ctx.vision2d.surface_defects()
ctx.vision2d.defect_sort_entry(entry_id)
```

对应 wire command：

```text
vision2d.surface_defects
vision2d.defect_sort_entry
```

`surface_defects()` 不接受参数；`defect_sort_entry()` 只接受当前计划中尚未消费的严格 `entry_id` 字符串。两者均拒绝路径、图片、ROI、阈值、坐标、速度、槽位、缺陷类型、嵌套对象或任意命令字符串。

V1-09 不公开：

```text
robot.home
robot.pose
robot.move_world
tool.suction
Remote API
CoppeliaSim handle
```

协议层、SDK、gateway 和 runner 必须分别验证 exact keys、类型、长度、数值有限性、schema 和状态。V1-09 使用独立 DTO 和错误前缀，不复用 V1-08 OCR 的模型、置信度、编号或 guard 语义。

学生返回只包含 JSON-native 教学字段：run/plan/entry、缺陷类型、findings、阈值摘要、图像/证据 digest、路由名称、完成状态和 `PENDING_HARDWARE`。不得包含本地文件路径、真实坐标、原始设备对象或可执行内部动作。

## 9. 守卫状态机与安全执行

纯守卫状态固定为：

```text
EMPTY
  -> ANALYZING
  -> ACTIVE
  -> ENTRY_ACTIVE
  -> AWAITING_PROBE
  -> ACTIVE | COMPLETE

任一非终态 -> FAILED
受控 reset -> EMPTY
```

规则如下：

1. `EMPTY` 之外再次调用分析命令必须拒绝；
2. `ACTIVE` 之前任何 motion/tool 调用必须拒绝且设备零调用；
3. 学生只能选择尚未消费的白名单 entry；不能选择路线或坐标；
4. 每个 entry 最多消费一次；动作完成不等于消费，必须等待同 run 探针；
5. pre-entry 探针确认 part 仍在起始位且目标槽合同有效；
6. post-entry 探针确认该 part 进入计划槽、其他 part 未被错误移动；
7. 探针失败、run/scene/part/slot 不匹配或证据不完整会使整个计划进入 `FAILED`；
8. `stop` 立即禁止后续动作并使计划失效；`reset` 只有在清理后才能回到 `EMPTY`；
9. 暂停不消费 entry；继续或再次 single-step 才能推进；
10. 成功完成六个 entry 后执行最终 home、tool-off 和六槽终态探针。

runner 的私有执行器把每次真实 `robot.move_world` 或 tool 动作视为一个独立设备动作。每个设备动作之前都必须重新执行 stop、pause、single-step、工作空间、速度和计划状态检查。一次 single-step 许可最多触发一个实际设备动作，不能按逻辑 entry 或高层 action 批量消费许可。

失败时优先保留首个错误，停止后续动作，并尽力关闭工具；只有安全条件仍满足时才回 home。不得用公开协议递归调用被禁止的 raw 命令。

## 10. 同次运行证据

每次计划和回执必须绑定：

- `experiment_id=V1-09`、`run_id`、`frame_id`；
- scene、scene manifest、asset manifest、profile 和 config SHA；
- reference crop、六个 candidate crop、finding-region mask、annotated image SHA；
- 每个内核结果、decision、完整计划和 plan SHA；
- 每个 entry 的 pre/post probe、设备动作计数和事件摘要；
- 最终六槽占位、home、tool-off、子进程和端口清理状态。

所有 canonical 文本使用 UTF-8、LF、排序键和禁止 NaN 的 JSON。计划和 entry 证据不能跨 run 重放；run、scene、frame、plan 或 entry 任一不一致都返回 `DEFECT_SORT_RUN_MISMATCH`。

`defect_visualization.py` 生成的 finding-region mask 只用 detector 已发布的 bbox 绘制示意区域，不声称是像素级工业分割真值，也不得反馈给判定。PyQt 和证据包必须把它标注为“缺陷区域示意”。

## 11. PyQt 教学呈现

采用 CoppeliaSim 与 PyQt 双窗口协同：

- CoppeliaSim 是权威三维运动画面，展示六件逐件取放和六类目标槽；
- PyQt 是只读教学解释页，不实现第二套三维渲染器；
- 页面展示参考图、当前候选图、标注图、缺陷区域示意、findings、阈值、decision、计划进度、槽位结果和证据摘要；
- 页面显示 run/scene/plan digest 的短摘要和完整复制入口；
- 页面显示 `PENDING_HARDWARE` 和 `PENDING_HUMAN_ACCEPTANCE`；
- 继续使用受控暂停、继续、单步、停止、复位，不增加手动机器人或工具按钮；
- 结果页不能把识别结果、坐标或文件路径重新作为可执行输入。

UI 验收必须覆盖 Windows 100% 和 125% 缩放，并保存正常完成、暂停、单步、停止、复位和失败关闭状态的证据。

## 12. 端口与进程所有权

V1-09 固定使用端口 `23010`。不得自动回退到 `23000`、`23008` 或 `23009`。

CLI、PowerShell、scene builder 和在线测试必须：

1. 启动前确认 `23010` 没有非预期 listener；
2. 记录自己启动的精确 PID、可执行文件路径和启动时间；
3. readiness 后再次验证 listener 身份；
4. 只终止自己启动且身份仍匹配的进程；
5. 拒绝清理替换 listener 或按进程名批量终止；
6. 结束后证明 `23010` 无 listener 和无自有残留进程。

CLI 中现有 V1-08 `23008` 行为必须保持不变；V1-09 只增加窄范围固定端口映射和回归。

## 13. 失败码

正式层至少提供以下稳定失败码：

| 失败码 | 含义 |
|---|---|
| `DEFECT_SORT_ASSET_INVALID` | 资产路径、尺寸或 SHA 不匹配 |
| `DEFECT_SORT_SCENE_INVALID` | 场景、对象、ROI、profile 或 manifest 不匹配 |
| `DEFECT_SORT_ANALYSIS_REJECTED` | 内核返回不可用于正式决定的状态/失败码 |
| `DEFECT_SORT_DECISION_AMBIGUOUS` | 同一候选存在多类缺陷或决定不唯一 |
| `DEFECT_SORT_PLAN_INCOMPLETE` | 六项集合、合格数或五类覆盖不完整 |
| `DEFECT_SORT_PLAN_NOT_ACTIVE` | 计划尚未原子激活 |
| `DEFECT_SORT_ENTRY_INVALID` | entry 不在当前白名单或字段非法 |
| `DEFECT_SORT_ENTRY_CONSUMED` | entry 已完成消费 |
| `DEFECT_SORT_PROBE_FAILED` | pre/post/final 探针失败 |
| `DEFECT_SORT_RUN_MISMATCH` | run/scene/frame/plan/entry 证据重放或不一致 |
| `DEFECT_SORT_EXECUTION_FAILED` | 私有设备动作失败并触发清理 |

错误响应必须有界、JSON-native，不包含敏感路径、Python traceback、Remote API 对象或学生不可控的内部细节。

## 14. 文件所有权与并行边界

### 14.1 V1-09 新增独占范围

```text
simulation/vision_defect_sorting_lab/**
tools/vision_lab/generate_v1_09_defect_assets.py
tools/vision_lab/build_v1_09_scene.py
tools/vision_lab/build_v1_09_scene.ps1
tools/vision_lab/run_v1_09_surface_defects.ps1
vision_platform/experiments/defect_assets.py
vision_platform/experiments/defect_sorting.py
vision_platform/experiments/defect_service.py
vision_platform/experiments/defect_visualization.py
vision_platform/student/defect_sort_guard.py
config/experiments/V1-09.json
student_programs/templates/v1_09_surface_defects.py
docs/experiments/V1-09.md
V1-09 专项测试和验证报告
```

### 14.2 主线实施端拥有的共享增量

V1-09 正式集成需要窄范围修改：

- `config/experiments/catalog.json`；
- `vision_platform/experiments/__init__.py`、`catalog.py`、`capabilities.py`、`probes.py`；
- `vision_platform/student/protocol.py`、`sdk.py`、`experiment_gateway.py`、`runner.py`；
- `vision_platform/ui/vision_result_panel.py` 和课程目录面板；
- `vision_platform/cli.py`；
- `tests/test_acceptance/test_delivery_contract.py`；
- `RETAINED_FILES.txt` 与必要的 `.gitattributes` 追加项。

这些共享文件只能由一个主线实施端修改，不能在多个开发端并行编辑。

### 14.3 可选并行资产包

实施计划若确认文件完全独立，可把“原创 PNG 生成器 + asset manifest + 纯资产测试”分配给一个并行开发端。该端不得创建或修改 `.ttt`、scene builder、SDK、gateway、runner、UI、catalog、delivery contract、`RETAINED_FILES.txt` 或 `.gitattributes`；正式 `.ttt` 始终只由主线实施端在已集成资产之后创建。

### 14.4 受保护范围

以下保持只读：

- `vision_platform/vision2d/defect_detection.py` 及其专项测试；
- 所有已有正式 `.ttt`；
- URDF、STL、网格和机器人资产；
- `vision_platform/rgbd/**`、`vision_platform/rgbd_sim/**` 和 D1-01 范围；
- V1-07/V1-08 专属算法、资产、guard 和场景语义。

## 15. TDD 与验收门禁

### 15.1 RED 必须先证明

至少先写并运行以下失败测试：

- 资产 SHA、路径越界、尺寸、重复 ID 和 manifest tamper；
- reference/candidate ROI 越界、重叠、尺寸不一致和 frame 不匹配；
- `REJECTED`、`NO_TARGETS`、`CANDIDATE_EMPTY` 和多类 finding 不得激活计划；
- 六项未齐、合格数不为一或五类不完整时设备零调用；
- raw robot/tool 对 V1-09 始终拒绝；
- 未完成 pre/post probe、跨 run 重放和重复 entry 不得消费；
- 安全抬升后暂停时后续水平移动不得发生；
- 一次 single-step 许可最多一个实际 move/tool 调用；
- stop 使后续动作失效并执行 tool-off 清理；
- 任何已有正式场景哈希变化都使构建/发布测试失败。

每个 Task 执行 RED → 确认预期失败 → 最小 GREEN → 专项回归 → 相关回归 → 独立提交。

### 15.2 静态与发布测试

必须运行：

- V1-09 资产、计划、服务、守卫、SDK、gateway、runner、scene、CLI、UI 专项；
- 既有 `tests/test_vision2d/test_defect_detection.py` 只读回归；
- V1-07、V1-08、D1 相关不回退回归；
- `tests/test_acceptance/test_delivery_contract.py`；
- 全仓静态测试；
- `git diff --check`；
- UTF-8/LF、代码围栏、placeholder、发布清单唯一/存在性审计；
- local/remote SHA 和工作树干净检查。

普通测试中的 CoppeliaSim skip 只能写成 skipped，不能计为在线 PASS。

### 15.3 真实 CoppeliaSim 在线验收

显式使用 `-m coppeliasim`，固定端口 `23010`。在线验收必须证明：

1. 新场景、模板、manifest 和七张资产 SHA 精确匹配；
2. 一张预运动帧同时生成 reference 和六个 candidate ROI；
3. 实际算法输出形成一个合格决定和五类唯一缺陷决定；
4. 完整计划在首次设备动作之前冻结；
5. 学生选择的六个 entry 顺序全部完成且各消费一次；
6. 六个 part 最终进入各自预期槽位；
7. pause/continue/single-step/stop/reset 在设备调用粒度生效；
8. 最终 robot home、tool off、安全违规为零；
9. same-run 图像、计划、动作、探针和清理证据完整；
10. `23010` 和自有进程清理完成。

### 15.4 UI 与人工门禁

自动 UI smoke 和证据截图通过不等于教学效果通过。未执行的课堂讲解、学生理解和界面可用性人工检查保持 `PENDING_HUMAN_ACCEPTANCE`。

## 16. 集成顺序

1. 协调端提交本设计并由用户审核；
2. 协调端编写逐 Task 实施计划并再次审核文件边界；
3. D1-01 集成候选是否合入 `main` 由用户单独明确授权；
4. V1-09 实施前重新 fetch，并从当时精确 `origin/main` 建立隔离分支/工作树；
5. 若 D1-01 已先合入，V1-09 直接继承新主线；若尚未合入，后续集成只能对 `.gitattributes` 和 `RETAINED_FILES.txt` 做追加并集，不能覆盖另一端内容；
6. 主线实施端完成共享课程闭环；可选并行端只交付独立资产包；
7. 开发端完成自测后，协调端亲自进行代码审查、真实测试和 P0/P1/P2 裁决；
8. 只有 P0/P1 为零、阻断 P2 清零且证据完整时，才向用户报告可进入集成审核；
9. 未经用户明确授权，不合并或推送 `main`，不打 tag。

## 17. 声明边界

V1-09 完成后最多只能声明：

> 独立 CoppeliaSim 场景中的参考差分缺陷检测、六件完整计划、受控机器人分流、同次运行证据和软件接口通过。

不得外推为：

- 工业缺陷检测准确率通过；
- 真实海康相机或真实光学条件通过；
- 真实机械臂、急停、气路、吸盘或物理抓取通过；
- 课堂教学效果或学生能力通过。

真实硬件保持 `PENDING_HARDWARE`；教学人工验收保持 `PENDING_HUMAN_ACCEPTANCE`。

## 18. 完成定义

V1-09 只有在以下条件全部满足时才完成：

- 仅新增获授权的 V1-09 `.ttt`，已有正式场景和机器人资产零变化；
- 原创参考件和六个候选资产可重复生成并通过哈希合同；
- 一张预运动帧产生完整、唯一、实际算法驱动的六项计划；
- 学生只使用两个有限接口，raw robot/tool 始终不可达；
- 六件全部完成受控分流，暂停/单步/停止在设备粒度正确；
- 同次运行图像、判定、计划、动作、探针和清理证据完整；
- PyQt、CLI、PowerShell、课程材料和发布合同通过；
- 聚焦、全仓、显式在线和交付测试通过，skip 单独报告；
- 协调端独立复审 P0/P1 为零，P2 逐项有结论；
- `PENDING_HARDWARE` 和 `PENDING_HUMAN_ACCEPTANCE` 保持准确。
