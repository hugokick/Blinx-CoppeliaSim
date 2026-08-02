# V2.2 纯二维视觉内核 V1-07～V1-09 设计

**状态：** 已确认，进入无人值守 TDD 实施

**日期：** 2026-08-02

**基线：** `origin/main` / `9af91087d29cf3186ef84f3528c5f4c495b1ff71`

**目标分支：** `codex/v2-2-v1-07-v1-09-algorithm-kernels`

**工作树：**
`C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-07-v1-09-algorithm-kernels`

## 1. 目标与边界

本工作包只增加三个可独立导入的、内存图像到结构化结果的二维算法模块：

```text
vision_platform/vision2d/code_recognition.py
vision_platform/vision2d/ocr.py
vision_platform/vision2d/defect_detection.py
```

它们服务于 V1-07 条码/二维码识别、V1-08 OCR 和 V1-09 表面缺陷检测的算法
内核验证，不进行正式课程集成。禁止修改公共 `vision2d.__init__` 导出、学生
运行器、CLI、PyQt、网关、实验目录、模板、CoppeliaSim 场景、URDF、STL、机器人
资产和 V1-06 模板匹配代码。调用方必须提供 BGR `numpy.ndarray`，模块不读写
图片路径、不下载字体/模型/数据、不访问网络。

每个模块都必须：

1. 对形状、dtype、通道和有限值做显式校验；
2. 返回冻结的数据对象，包含状态、失败原因（如有）、定位/文本/缺陷、置信度
   或阈值、处理耗时和 schema 版本；
3. 在输入无目标、目标损坏或模型/配置不适用时返回可解释状态，不能静默跳过；
4. 使用固定算法参数和固定随机种子，性能有明确上限，不启动后台任务；
5. 只由原创合成测试数据证明算法行为；合成数据不代表真机精度或教学验收。

公共状态集合为 `PASS`、`PARTIAL`、`NO_TARGETS`、`REJECTED`；具体模块可以在
结果中使用更细的 `failure_code`，但不能改变这四类顶层状态的含义。

## 2. V1-07 条码与二维码识别

### 2.1 输入与结果合同

入口为：

```python
recognize_codes(image_bgr, *, max_codes=16) -> CodeRecognitionResult
```

`CodeReading` 为不可变对象，字段为：

```text
code_type: "qr" | "rs1d"
data: str | None
polygon_px: tuple[tuple[float, float], ...]  # 原图坐标，顺时针
bbox_px: (x, y, width, height)
center_px: (x, y)
confidence: float                 # [0, 1]
decoded: bool
failure_code: str | None
```

`CodeRecognitionResult` 至少包含 `status`、`readings`、`failure_code`、
`image_size`、`detector_order`、`processing_ms` 和 `schema_version=1`。空白图返回
`NO_TARGETS`；检测到定位点但解码失败返回 `PARTIAL` 并保留位置；无效输入返回
`REJECTED` 和 `INPUT_INVALID`。

### 2.2 二维 QR 路径

使用 OpenCV 4.14 自带 `QRCodeDetector` 的单码和多码接口。测试数据用
`QRCodeEncoder_create` 在进程内生成并添加白色静区、旋转、缩放、亮度变化和高斯
噪声。调用者只能得到检测器给出的四边形原图坐标；空字符串不能伪装为成功解码。

### 2.3 原创一维 RS1D 路径

项目定义一个明确标注为教学/测试用途的 `RS1D` 码制，不声称兼容 Code128、EAN
或任何商业条码。编码器 `encode_rs1d_payload` 只用于可复现测试夹具和算法回归：

```text
quiet zone | 4-module start guard | bit bars | 4-module stop guard | quiet zone
```

每个 bit 都由黑条和一个白间隔组成，`0` 为 1 module、`1` 为 2 modules；数据帧
为 `magic=0xA7`、1-byte 长度、payload、8-bit sum checksum。解码先验证 guard、
长度、magic 和 checksum，再返回 payload，因此任意黑白纹理不会被静默当作条码。
检测器对固定角度集合（含水平、垂直和 ±30° 范围）做受限 deskew，在若干高对比度
扫描行上解析 run-length。成功后把候选四角逆变换回原图坐标；同一 payload/中心
只保留一次。模块、最大 payload 和扫描角度都有常量上限，防止异常输入导致无限
搜索。

若某一真实商业一维码不能被该自有格式表示，结果文档必须把原因写成
`RS1D_FORMAT_ONLY`，而不是声称完成商业条码兼容。

## 3. V1-08 OCR

### 3.1 训练合同

入口为：

```python
train_glyph_classifier(
    samples: Mapping[str, Sequence[np.ndarray]],
    *, method="knn", test_fraction=0.25, seed=20260802
) -> GlyphModel
```

`samples` 是调用方在内存中提供的单字符图像；训练函数将每个样本归一化为
20×20 `float32` 特征，并按标签内索引使用 `numpy.random.default_rng(seed)` 做
可复现的 train/test 分离。测试集从未用于 OpenCV ML 的 `train` 调用。`method`
支持 OpenCV `ml.KNearest_create`（k=3）和 `ml.SVM_create`（线性核）；结果保存
训练/测试计数、标签集合、seed、特征尺寸、held-out accuracy 和 schema 版本。
样本不足、标签少于两个或 test split 为空时返回 `REJECTED`/抛出稳定的
`TrainingError`，不自动用训练集冒充测试集。

### 3.2 识别合同

入口为：

```python
recognize_text(image_bgr, model, *, expected_text=None) -> OCRResult
```

输入先做灰度、轻度去噪、自适应阈值、开闭运算和受限 deskew；使用连接组件和
垂直投影排序字符框。对断裂字符使用小核闭运算；对粘连字符在投影谷处拆分；
无法安全拆分时返回 `PARTIAL`/`SEGMENTATION_STUCK`，不猜测字符。每个
`OCRCharacter` 包含识别字符、原图 bbox、[0,1] 置信度和失败原因。结果还包含
`text`、`status`、`failure_code`、`threshold_method`、`character_count`、
`processing_ms`、`image_size` 和 schema 版本。空白、纯噪声、模型不匹配、低置信
度和 expected_text 不一致都必须在状态或失败码中可见。

字形测试素材采用代码内原创 5×7 bitmap（至少数字和大写拉丁字母），由测试夹具
自己放大、旋转、断笔、粘连、亮度变化和加噪；不依赖系统字体、Tesseract、下载的
模型或外部 OCR 服务。

## 4. V1-09 表面缺陷检测

### 4.1 输入与配置

入口为：

```python
detect_surface_defects(
    reference_image, candidate_image, *, config=DefectConfig()
) -> DefectResult
```

两张图必须同尺寸、uint8、单通道或 BGR。`DefectConfig` 显式发布相对面积、
dimension ratio、形态学核尺寸、亮度归一化开关和最大平移量；阈值按参考前景面积
和图像面积计算，结果中同时回显实际像素阈值和相对阈值，避免散落 magic number。

### 4.2 算法与输出

两幅图分别通过边界极性判断、Otsu/adaptive threshold 和开闭运算得到前景 mask，
再以质心做有限平移对齐。随后计算 reference/candidate 的差分、连接组件、轮廓
层级、面积、周长和 bounding box：

- `missing`：参考前景而候选缺失的显著组件；
- `hole`：候选前景内部由轮廓层级证明的孔洞；
- `foreign`：候选新增且不与参考主体相交的组件；
- `broken`：参考单主体被候选分割为多个主要连接组件；
- `dimension`：面积或长短边相对差超过配置阈值。

`DefectFinding` 至少包含 `defect_type`、原图 `bbox_px`、`area_px2`、
`relative_area`、`metric`、`threshold` 和 `confidence`。`DefectResult` 包含
`status`（无发现为 `PASS`，有发现为 `PARTIAL`）、`defects`、
`reference_metrics`、`candidate_metrics`、`alignment_shift_px`、
`thresholds`、`failure_code`、`processing_ms` 和 schema 版本。空参考、尺寸不符、
全背景和不确定极性分别给出稳定错误码。亮度偏移、低幅噪声和小孤立点由归一化与
形态学清理吸收，但不能通过放宽阈值掩盖真实大缺陷。

测试夹具至少生成 missing、hole、foreign、broken、dimension 五类候选，并覆盖
无缺陷、亮度偏移、噪声、空白和不匹配尺寸。

## 5. 文件、测试与集成边界

新增算法专项测试直接导入三个子模块，不修改 `vision_platform/vision2d/__init__.py`：

```text
tests/test_vision2d/test_code_recognition.py
tests/test_vision2d/test_ocr.py
tests/test_vision2d/test_defect_detection.py
tests/test_vision2d/synthetic_v107_v109.py
```

每个 V1 任务按 RED→GREEN→专项回归→`tests/test_vision2d` 回归→独立提交执行。
最终必须再运行全仓静态测试、静态检查和 `git diff --check`。不添加二进制图像、
模型缓存、下载文件或 CoppeliaSim 在线验收；`RETAINED_FILES.txt` 只登记源代码、
专项测试、设计和计划。所有硬件、正式课程、人工教学验收保持未执行状态。

## 6. 性能与失败边界

在 512×512 BGR 输入和默认参数下，单次 V1-07/V1-09 处理目标为 1 秒内，OCR
训练（小于 64 个标签样本）目标为 2 秒内；测试将记录有限耗时而不对不同 CPU
设置脆弱的纳秒级断言。任何 OpenCV 异常都必须转为模块稳定错误或显式抛出输入/
训练异常；禁止返回半填充的成功对象。模块不承诺真实相机、商业条码、复杂字体、
工业缺陷精度或 CoppeliaSim/硬件闭环能力。
