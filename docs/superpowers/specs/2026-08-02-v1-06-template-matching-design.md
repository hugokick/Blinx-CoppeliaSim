# V1-06 模板匹配视觉实验设计

## 1. 范围与教学目标

V1-06 在现有“视觉实验公共底座”和 V1-01～V1-05 能力之上，增加一个只读、可复现实验：学生通过 `vision2d.template_match` 查询固定模板在相机图像中的位置，理解模板、搜索区域、相似度阈值、像素坐标和证据包之间的关系。实验只使用 CoppeliaSim 中已经存在的视觉场景，不移动机器人、不执行学生任意代码、不引入真实相机或硬件控制。

本设计严格限定在 V1-06。V1-07～V1-09、V1-02～V1-05 的算法实现、URDF/STL/正式机器人资产以及 `simulation/vision_lab/BL23_vision_lab.ttt` 均不属于本变更。

## 2. 输入、输出与确定性

### 2.1 输入

输入是公共底座从场景相机取得的一帧 BGR `uint8` 图像和经过场景清单校验的固定模板资产。模板清单声明：

- `template_id = v1_06_red_rectangle`
- `template_version = 1.0.0`
- `method = TM_CCOEFF_NORMED`
- `threshold = 0.72`
- `search_roi_px = [0, 0, 256, 256]`（`x, y, width, height`）；该窗口适配现有质量场景的最小 `256×256` profile。

模板由仓库内的确定性生成脚本生成，不能从用户输入、网络或本机任意路径加载。资产路径和 SHA-256 由 `scene_manifest.json` 指向的模板清单固定。

### 2.2 匹配规则

`vision_platform/vision2d/template_matching.py` 提供无副作用的纯函数：

1. 严格验证图像和模板为二维/三通道 `uint8`，尺寸和 ROI 合法；不隐式转换浮点或灰度输入。
2. 将图像和模板转换为灰度，使用 OpenCV `cv2.TM_CCOEFF_NORMED`，仅在固定 ROI 内搜索。
3. 取最高分；分数相差不超过 `1e-12` 时按最小 `y`、再最小 `x` 选取，保证不同 OpenCV 版本下的确定性。
4. 返回原图坐标中的 `[x, y, width, height]`、中心点 `[cx, cy]`、最高分、阈值和 `matched = score >= threshold`。
5. 结果分数必须有限，模板不得大于搜索窗口；无效输入使用稳定错误码，不抛出内部 OpenCV 错误。

公开结果模型包含 `schema_version`, `template_id`, `template_version`, `status`, `matched`, `score`, `threshold`, `bbox_px`, `center_px`, `image_size`, `search_roi_px`, `method`。不公开本机路径或实现细节。

### 2.3 图层和证据

主机为一次请求取得一帧，并生成 `VisionResultBundle` 三层：

- `raw`：相机原始帧；
- `template`：已验证的模板图；
- `annotated`：在原图上绘制 ROI、模板框、中心和分数的只读证据图。

证据包的元数据保存 `snapshot_id`, `template_id`, `template_version`, `method`, `threshold`, `search_roi_px`, `score`, `bbox_px`, `center_px`, `image_size` 和状态；不含学生任意文件或未验证路径。

## 3. 公共底座、SDK 与安全边界

新增唯一命令 `vision2d.template_match`。学生 SDK 提供：

```python
result = ctx.vision2d.template_match()
# Python 友好别名
result = ctx.vision2d.match_template()
```

命令只能由已认证的网关路由。网关读取已经通过场景清单、模板清单和 SHA-256 校验的固定资产，调用纯函数，写入证据包并返回严格响应；学生端不能传入模板路径、阈值、ROI、命令字符串或机器人动作。结果面板只渲染已验证的 bundle，不能编辑或执行结果。

`vision2d.template_matching` 能力仅在 sim backend、CoppeliaSim 在线、相机就绪并且完整视觉 profile 生效时报告 `ready`。普通静态测试中的 CoppeliaSim skip 仍是 `PENDING_ONLINE`，不能写成在线通过。

## 4. 场景与配置

正式场景只能是 `simulation/vision_quality_lab/BL23_vision_quality_lab.ttt`，本次不修改该二进制场景。现有 `scene_manifest.json` 增加可选 `template_catalog` 绑定，绑定模板清单路径、版本和哈希；网关启动时验证绑定。V1-06 实验配置复用质量场景，公开能力为 `camera.rgb`, `camera.profile`, `vision2d.template_matching`, `experiment.info`, `scene.probe`，不提供分数等级、自动评分或动作命令。

## 5. 课程材料与验收

新增 `docs/experiments/V1-06.md`、配置/模板/CLI/PowerShell 入口和独立在线测试。课程材料说明：学生编辑调用代码、启动专用 CoppeliaSim 端口、观察结果面板、保存证据和解释阈值/坐标；教学效果保持 `PENDING_HUMAN_ACCEPTANCE`。海康相机、真实机械臂、急停、气路、物理抓取和真实光学精度保持 `PENDING_HARDWARE`。

自动验收分层如下：

1. 模板匹配纯函数和错误合同；
2. 确定性模板资产、清单和发布合同；
3. 协议、SDK、网关、回滚网关和 runner；
4. V1-06 课程/配置/CLI/PowerShell；
5. 证据包和 PyQt 面板 100% 与 125% 缩放检查；
6. 显式启用 CoppeliaSim 专用端口的在线验收。

所有正式交付文件列入 `RETAINED_FILES.txt`。完成前检查受保护资产相对 `v2.2.0` 无变化、工作树干净、完整静态回归、发布合同和独立代码审查；只有真实执行的在线测试才可标记通过。

## 6. 文件边界

允许新增/修改：`vision_platform/vision2d/template_matching.py`、模板生成器和模板资产、公共协议/SDK/网关/runner 的最小扩展、V1-06 配置/课程/测试/证据及清单文档。禁止修改 `vision_platform/vision2d/**` 中 V1-02～V1-05 既有算法文件和其测试，禁止修改 `docs/experiments/V1-02.md`～`V1-05.md`、算法分支和正式机器人资产。
