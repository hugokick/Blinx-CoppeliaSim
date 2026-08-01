# V2.2-C1 二维视觉算法基础包设计

**文档状态：** 用户已确认  
**编制日期：** 2026-07-31  
**项目名称：** Robot Sim  
**权威仓库：** `C:\Users\yqzhe\Documents\Robot Sim\Blinx-CoppeliaSim-clean`  
**设计分支：** `codex/v2-2-vision2d-algorithm-kernel`  
**开发基线：** V2.1 最终提交 `4bc638f50fd590ca700615da46741a86c4146b5f`

---

## 1. 决策摘要

V2.2 首批多实验框架和五个 R1 实验正在
`codex/v2-2-first-batch-integration` 分支实施。为了提高整体开发效率，
本设计建立一条低冲突并行开发线，提前完成 V2.2-C 的二维视觉纯算法基础包。

首批算法包覆盖：

- `V1-02` 像素尺寸与构件尺寸测量；
- `V1-03` 物体定位与旋转角测量；
- `V1-04` 边缘长度、周长和面积测量；
- `V1-05` 颜色、形状与轮廓识别。

采用“分层共享流水线”：

```text
BGR 图像
  → 输入校验
  → 预处理
  → 多目标分割
  → 轮廓过滤
  → 几何测量
  → 颜色与形状分析
  → 稳定排序和编号
  → 结构化结果
```

新代码放在独立的 `vision_platform/vision2d/` 包中，不修改现有
`vision_platform/recognition/color_shape.py`。新结果字段与现有
`Detection` 的核心语义保持兼容，待首批 V2.2 集成完成后再由单一集成端
增加适配器和机器人闭环。

## 2. 背景

平台已经具备：

- BL23/OpenR6 六轴机械臂和 CoppeliaSim 视觉实验场景；
- BGR 相机帧、三点标定、颜色/形状识别和六物体分类；
- replay、CoppeliaSim 和可选海康相机适配层；
- V2.1 学生 SDK、spawn 子进程、安全网关、运行控制和证据；
- 正在集成的多实验目录、场景会话、实验探针和首批 R1 实验。

V1 二维视觉包共有九个候选实验。本阶段不一次实现全部 V1，也不提前接入
正在修改的 CLI、PyQt、实验目录和学生 SDK。先实现可独立测试的算法内核，
能够减少后续实验包的重复代码，并避免与当前 V2.2 集成端争用共享文件。

## 3. 目标

### 3.1 学习目标

算法包应支持学生理解和验证：

1. 图像像素与物理尺寸之间的显式换算；
2. 轮廓中心、旋转包围框和主方向角；
3. 面积、周长、长边和短边等几何特征；
4. HSV 颜色、轮廓顶点和圆度等分类依据；
5. 多目标图像中各目标的独立测量结果；
6. 噪声、贴边、过小目标和退化轮廓对测量的影响。

### 3.2 工程目标

算法包必须：

- 只依赖基础环境已经提供的 Python、NumPy 和 OpenCV；
- 接收内存中的 BGR 图像，不连接相机或 CoppeliaSim；
- 同时支持一幅图像中的多个独立构件；
- 对相同输入产生确定的目标顺序和结构化结果；
- 始终输出像素测量；
- 仅在调用者显式提供像素尺寸时输出毫米测量；
- 返回 JSON 可序列化的元数据；
- 将中间处理图保留在内存中，不自行决定证据路径；
- 对无法安全使用的目标给出明确拒绝原因；
- 不修改一期识别器和 V2.1 控制边界。

## 4. 非目标

本设计不包括：

- `V1-01` 虚拟相机、镜头、视场和光照交互；
- `V1-06` 模板匹配；
- `V1-07` 二维码和条形码；
- `V1-08` OCR；
- `V1-09` 表面缺陷检测；
- CoppeliaSim 场景建设和在线验收；
- 学生 SDK、CLI、PowerShell 或 PyQt 接入；
- 实验目录和场景会话接入；
- 自动机器人抓取、分仓或码垛；
- 自动评分和正式课程成绩；
- 海康相机、真实机械臂或真实光学精度验收；
- 对恶意代码提供操作系统级安全沙箱；
- 复制 H 盘授权不明的图片、模型或旧程序。

## 5. 方案比较

### 5.1 方案一：分层共享流水线

预处理、分割、几何和外观分析拆成独立模块，四个实验通过配置和结果视图
复用同一流水线。

优点：

- 避免四套实验重复实现轮廓提取；
- 几何和外观算法可以分别测试；
- 便于后续增加模板、OCR 和缺陷模块；
- 结果模型统一，接入学生 SDK 时只需一个适配层；
- 可以单独替换某一层而不破坏调用者。

代价：

- 首次实现需要先固定模块接口；
- 必须严格区分内存分析结果和未来证据层。

### 5.2 方案二：单一综合函数

一个函数完成分割、测量和分类。

优点是初期代码少；缺点是参数、异常、测试和实验差异会集中到一个大函数，
后续 V1-06 至 V1-09 很难复用。

### 5.3 方案三：四套实验完全独立

每个实验拥有自己的预处理和识别流程。

优点是单个实验容易理解；缺点是轮廓提取、数据生成和错误处理大量重复，
相同目标可能在不同实验中得到不一致的中心和角度。

### 5.4 选定方案

采用方案一。方案二不利于扩展，方案三违反复用目标。

## 6. 包结构

新增运行时代码：

```text
vision_platform/vision2d/
├── __init__.py
├── models.py
├── preprocessing.py
├── segmentation.py
├── geometry.py
├── appearance.py
├── pipeline.py
└── serialization.py
```

职责如下。

### 6.1 `models.py`

定义不可变输入配置和结果类型：

- `PixelScale`；
- `Vision2DConfig`；
- `TargetMeasurement`；
- `RejectedTarget`；
- `Vision2DResult`；
- `Vision2DAnalysis`。

该文件不导入 CoppeliaSim、相机、机器人、PyQt 或学生运行器。

### 6.2 `preprocessing.py`

负责：

- BGR 图像校验；
- HSV、灰度图生成；
- 可配置高斯滤波；
- 饱和度和亮度前景掩膜；
- 二值掩膜的开运算和闭运算；
- 返回命名中间图。

该层不查找轮廓，也不产生目标 ID。

### 6.3 `segmentation.py`

负责：

- 从前景掩膜提取外轮廓；
- 计算初始面积；
- 检查轮廓是否贴边；
- 拒绝面积过小、面积过大或矩退化的轮廓；
- 将可测轮廓和拒绝轮廓分别返回。

首版只处理相互分离的构件。粘连目标分割不属于本阶段完成条件。

### 6.4 `geometry.py`

负责：

- 轮廓矩中心；
- 轴对齐包围框；
- 最小旋转包围框；
- 长边、短边；
- 轮廓面积；
- 闭合轮廓周长；
- 主方向角；
- 像素到毫米的显式换算。

该层不进行颜色和形状命名。

### 6.5 `appearance.py`

负责：

- 轮廓区域内 HSV 统计；
- 颜色标签；
- 多边形近似；
- 顶点数；
- 圆度；
- 长宽比；
- 形状标签；
- 与一期颜色、形状标签语义的兼容检查。

### 6.6 `pipeline.py`

提供公共入口：

```python
analyze_image(
    image_bgr: np.ndarray,
    config: Vision2DConfig | None = None,
) -> Vision2DAnalysis
```

该入口组合各层、稳定排序目标、分配 ID 并生成图像级状态。

### 6.7 `serialization.py`

负责将 `Vision2DResult` 转换为只包含 Python 标量、列表、字典和 `None`
的结构。

该层不序列化原始 NumPy 图像。中间图由未来证据层单独保存。

## 7. 输入契约

### 7.1 图像

`image_bgr` 必须：

- 是 `numpy.ndarray`；
- 维度为 `H×W×3`；
- 数据类型为 `uint8`；
- 宽度和高度均大于 0；
- 使用 OpenCV BGR 通道顺序。

不接受灰度图、BGRA、浮点归一化图、空数组或对象数组。输入错误抛出
`TypeError` 或 `ValueError`，不返回伪造的空成功结果。

### 7.2 像素尺寸

`PixelScale` 包含：

```text
mm_per_pixel_x
mm_per_pixel_y
```

两个值必须是有限正数。算法始终输出像素结果。

未提供 `PixelScale` 时：

- `size_mm` 为 `null`；
- `area_mm2` 为 `null`；
- `perimeter_mm` 为 `null`；
- 不出现推测的毫米结果；
- 图像级状态不因此失败。

提供 `PixelScale` 时：

- X 方向长度使用 `mm_per_pixel_x`；
- Y 方向长度使用 `mm_per_pixel_y`；
- 旋转矩形边长使用对应边方向在 X/Y 上的投影换算；
- 面积使用
  `area_px2 × mm_per_pixel_x × mm_per_pixel_y`；
- 周长使用每个轮廓线段的各向异性物理长度求和，不能简单乘平均比例。

### 7.3 配置

`Vision2DConfig` 至少包含：

- 最小和最大面积比例；
- 饱和度下限；
- 亮度下限；
- 高斯滤波核尺寸；
- 形态学核尺寸；
- 贴边检查边距；
- 多边形近似比例；
- 正方形长宽比下限；
- 圆形圆度下限；
- 可选 `PixelScale`。

所有比例、阈值和核尺寸在构造时验证。核尺寸必须为正奇数。配置对象不可
变，流水线不得在运行时修改调用者配置。

## 8. 多目标结果模型

### 8.1 目标排序和 ID

可测目标按：

1. 中心 Y 坐标；
2. 中心 X 坐标；
3. 面积；

升序稳定排序，然后分配：

```text
det-001
det-002
det-003
```

ID 只保证同一输入和同一配置的重复运行稳定，不宣称跨视频帧跟踪同一物理
目标。跨帧目标跟踪属于后续能力。

### 8.2 `TargetMeasurement`

每个可测目标至少包含：

- `detection_id`；
- `center_px`；
- `axis_aligned_bbox_px`；
- `rotated_box_px`；
- `long_side_px`；
- `short_side_px`；
- `angle_deg`；
- `area_px2`；
- `perimeter_px`；
- `size_mm`；
- `area_mm2`；
- `perimeter_mm`；
- `color`；
- `shape`；
- `vertex_count`；
- `circularity`；
- `aspect_ratio`；
- `quality_flags`；
- `contour`。

`contour` 只保留在内存分析对象中。结构化 JSON 将轮廓转换为点列表，且不
包含任何 NumPy 标量。

### 8.3 角度语义

`angle_deg` 表示最小旋转包围框的长边相对图像 X 正方向的无向角，规范到：

```text
[-90°, 90°)
```

由于长轴是无向轴，角度相差 180° 视为同一方向。

对圆形：

- `angle_deg` 为 `null`；
- 增加 `ANGLE_UNDEFINED_FOR_CIRCLE` 质量标志；
- 不用 `0°` 假装存在方向。

对正方形：

- 旋转包围框可以给出角度；
- 增加 `ANGLE_AMBIGUOUS_FOR_SQUARE` 质量标志；
- 调用者不得将该角度当作唯一姿态真值。

### 8.4 颜色标签

首版颜色标签与一期识别器保持一致：

```text
red
yellow
green
blue
unknown
```

颜色由轮廓内部 HSV 像素的稳健统计确定。低饱和度、色相落入未定义区间或
有效像素不足时返回 `unknown`。

### 8.5 形状标签

首版形状标签与一期识别器兼容：

```text
triangle
square
rectangle
circle
polygon
unknown
```

形状使用顶点数、长宽比和圆度判定。`polygon` 表示轮廓可测但不属于前三类
标准多边形或圆形；`unknown` 表示特征不足以安全命名。

## 9. 图像级状态

`Vision2DResult.status` 使用：

- `PASS`：至少一个目标可测，且没有被拒绝的候选轮廓；
- `PARTIAL`：至少一个目标可测，同时存在被拒绝候选；
- `NO_TARGETS`：没有达到候选条件的前景目标；
- `REJECTED`：检测到候选轮廓，但全部因质量规则被拒绝。

这些状态是算法处理状态，不是课程成绩、机器人任务结果或验收结论。

## 10. 拒绝模型

`RejectedTarget` 至少包含：

- 临时轮廓序号；
- 可获得的中心和面积；
- 拒绝代码；
- 面向开发者的明确原因。

拒绝代码固定为：

```text
AREA_TOO_SMALL
AREA_TOO_LARGE
TOUCHES_IMAGE_BORDER
DEGENERATE_MOMENT
DEGENERATE_ROTATED_BOX
INSUFFICIENT_CONTOUR_POINTS
```

拒绝目标不进入未来机器人动作候选列表，但保留在分析结果中供学生理解和
教师复核。

## 11. 中间图

`Vision2DAnalysis` 同时包含：

- `result`：结构化结果；
- `intermediate_images`：只读语义的图像映射。

首版中间图键名固定为：

```text
gray
hsv
foreground_mask
cleaned_mask
annotated
```

流水线返回新数组，不修改调用者的 `image_bgr`。算法包不创建证据目录、
不保存文件，也不决定 PNG/JPEG 格式。未来接入端根据实验和证据策略选择
保存哪些图像。

## 12. 原创合成数据

首版测试数据全部由代码生成，不复制 H 盘旧图片。

测试生成器位于：

```text
tests/test_vision2d/synthetic_factory.py
```

生成器使用固定随机种子，能够产生：

- 红、黄、绿、蓝构件；
- 矩形、正方形、圆形和三角形；
- 单目标和多目标；
- 不同中心位置；
- 不同长短边；
- `0°`、`15°`、`30°`、`45°`、`60°` 等旋转；
- 可控背景亮度；
- 可控轻度高斯噪声；
- 可控亮度变化；
- 空图；
- 贴边目标；
- 过小目标；
- 退化轮廓。

生成器同时返回图像和精确真值，测试不依赖人工查看图片判断数值通过。

测试生成的图片默认不进入 Git。若后续课程文档需要示例图，应通过独立正式
生成命令创建，并记录生成参数、文件哈希和来源。

## 13. 测试设计

新增测试目录：

```text
tests/test_vision2d/
├── __init__.py
├── synthetic_factory.py
├── test_models.py
├── test_preprocessing.py
├── test_segmentation.py
├── test_geometry.py
├── test_appearance.py
├── test_pipeline.py
├── test_serialization.py
└── test_compatibility.py
```

### 13.1 输入和配置

验证：

- 合法 BGR `uint8` 图像通过；
- 灰度、BGRA、浮点、空数组和非数组被拒绝；
- 非有限或非正像素尺寸被拒绝；
- 偶数、零或负数核尺寸被拒绝；
- 配置对象不会被流水线修改。

### 13.2 多目标分割

验证：

- 单目标只产生一个可测轮廓；
- 多个分离目标数量正确；
- 轻度噪声不会产生大量伪目标；
- 过小、过大、贴边和退化目标进入拒绝结果；
- 空图状态为 `NO_TARGETS`。

### 13.3 几何精度

对清晰合成图：

- 中心误差不超过 `1.5 px`；
- 长边和短边误差不超过 `2 px`；
- 非圆形主方向角误差不超过 `2°`；
- 矩形面积相对误差不超过 `5%`；
- 轮廓周长相对误差不超过 `5%`。

角度误差按无向轴计算：

```text
min(|a-b|, 180-|a-b|)
```

圆形角度必须为 `null`。正方形必须带角度歧义标志。

### 13.4 毫米换算

验证：

- 无标定时所有毫米字段为 `null`；
- 各向同性像素尺寸换算正确；
- X/Y 不同像素尺寸时，面积和轮廓线段长度按各向异性公式计算；
- 不使用平均比例替代各向异性周长；
- 非有限和非正标定值在分析前失败。

### 13.5 颜色和形状

验证：

- 红、黄、绿、蓝标签正确；
- 低饱和度目标为 `unknown`；
- 三角形、正方形、矩形、圆形和一般多边形标签正确；
- 与一期 `ColorShapeRecognizer` 共用标签集合；
- 新包不修改或调用一期识别器的私有方法。

### 13.6 确定性和序列化

验证：

- 相同输入和配置重复运行，目标顺序相同；
- ID 从 `det-001` 连续编号；
- 结构化结果重复运行完全相等；
- JSON 结果不包含 NumPy 数组或 NumPy 标量；
- 中间图不进入结构化 JSON；
- 原始输入图在运行前后逐像素相同。

### 13.7 完整回归

开发完成后运行完整静态套件，要求零失败。

普通静态套件中的 CoppeliaSim 测试可以保持显式 skip，但 skip 不能被写成
本算法包在线 `PASS`。本分支不声明 CoppeliaSim、PyQt、机器人或真机通过。

## 14. 实验视图

四个实验使用同一 `Vision2DResult`，不复制流水线：

- `V1-02` 展示中心、长短边、像素尺寸和可选毫米尺寸；
- `V1-03` 展示中心、旋转包围框、主方向角和角度质量标志；
- `V1-04` 展示面积、周长、圆度、长宽比和轮廓；
- `V1-05` 展示颜色、形状、轮廓和外观判定依据。

本分支只定义结果字段和课程说明，不创建正式实验目录 JSON。正式实验目录、
学生模板和 PyQt 参数面板由后续集成计划完成。

## 15. 与现有一期识别器的关系

`vision_platform/recognition/color_shape.py` 保持不变。

兼容范围：

- `detection_id` 继续使用 `det-NNN`；
- `center_px` 继续使用 `(x, y)`；
- `color` 使用相同英文标签；
- `shape` 使用相同英文标签；
- `angle_deg` 继续以度为单位；
- `area_px2` 在适配时映射到一期 `area_px`；
- 新包可以提供比一期更多的几何和质量字段。

有意差异：

- 新包要求输入必须是 `uint8`；
- 新包区分可测目标和拒绝目标；
- 新包对圆形使用 `angle_deg=null`，而一期圆形角度为 `0.0`；
- 新包不把颜色和形状分数平均后称为机器人动作置信度；
- 新包保留图像级状态和中间处理图。

后续适配器必须显式处理这些差异，不能用修改一期识别器来掩盖差异。

## 16. 并行开发边界

本分支可以新增：

```text
vision_platform/vision2d/
tests/test_vision2d/
docs/superpowers/specs/
docs/superpowers/plans/
docs/experiments/V1-02.md
docs/experiments/V1-03.md
docs/experiments/V1-04.md
docs/experiments/V1-05.md
```

本分支不得修改：

```text
vision_platform/cli.py
vision_platform/application.py
vision_platform/session.py
vision_platform/student/
vision_platform/ui/
vision_platform/recognition/color_shape.py
tools/vision_lab/run_acceptance.ps1
config/experiments/
simulation/**/*.ttt
```

`RETAINED_FILES.txt` 只在独立提交中登记本分支新增正式文件。集成端解决路径
并集时，不得删除 V2.1 或 V2.2 首批实验的已有条目。

## 17. 后续集成

只有在 `codex/v2-2-first-batch-integration` 完成并通过完整验收后，才建立
新的 V2.2-C 集成分支。

后续集成端负责：

1. 将本算法包合入已完成的 V2.2 多实验框架；
2. 增加 V1-02 至 V1-05 正式实验目录 JSON；
3. 把 CoppeliaSim 相机快照传给算法包；
4. 将结构化结果放入学生 SDK 和运行证据；
5. 建设视觉质检场景；
6. 增加 CLI 和 PyQt 实验入口；
7. 将有效测量结果转换为受控机器人决策；
8. 运行真实 CoppeliaSim 在线测试；
9. 保留教师人工验收和 `PENDING_HARDWARE`。

本算法分支不提前实现上述集成工作。

## 18. 完成定义

只有以下条件全部满足，才能声明“V1-02 至 V1-05 二维视觉算法基础包完成”：

1. 新代码只位于确认的独立包和测试目录；
2. 现有 `color_shape.py` 和共享集成文件没有变化；
3. 输入校验、配置和像素尺寸模型测试通过；
4. 单目标和多目标分割测试通过；
5. 中心误差不超过 `1.5 px`；
6. 长短边误差不超过 `2 px`；
7. 非圆形角度误差不超过 `2°`；
8. 清晰合成图面积和周长相对误差不超过 `5%`；
9. 无标定时不产生毫米结果；
10. 各向同性和各向异性毫米换算测试通过；
11. 红、黄、绿、蓝及标准形状测试通过；
12. 空图、噪声、贴边、过小和退化目标行为明确；
13. 相同输入的 ID、顺序和 JSON 结果可重复；
14. 输入图不会被原地修改；
15. 结构化 JSON 不包含 NumPy 对象或中间图；
16. 四份实验说明与结果字段一致；
17. 所有新增正式文件进入 `RETAINED_FILES.txt`；
18. 完整静态回归零失败；
19. 文档没有把算法测试写成 CoppeliaSim、机器人、教学效果或真机通过；
20. 海康相机和真实机械臂相关结论保持 `PENDING_HARDWARE`。

完成本分支只能声明纯算法和原创合成测试通过。CoppeliaSim 场景、机器人
闭环、PyQt、教学效果和真实硬件必须由后续阶段分别验收。
