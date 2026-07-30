# CoppeliaSim 视觉机器人“仿真—真机同代码”一期实施计划

> **执行要求：** 使用 `executing-plans` 逐任务实施；任何生产代码都必须遵守 `test-driven-development` 的 RED → GREEN → REFACTOR 顺序。

**目标：** 基于指定厂家 STEP 模型，完成实验三视觉标定与实验四物体分类的 CoppeliaSim 全闭环仿真，同时保留同一高层代码切换真实 BLX 机械臂和海康相机的接口。

**总体设计：** `docs/superpowers/specs/2026-07-30-coppeliasim-vision-sim-to-real-mvp-design.md`

**隔离工作树：**

```text
C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\coppeliasim-vision-mvp
```

**分支：**

```text
codex/coppeliasim-vision-mvp
```

**技术栈：** Python 3.11、pytest、numpy、OpenCV、PyQt5、CoppeliaSim ZMQ Remote API、simUI

## 执行纪律

- 不修改或覆盖 `robot_backends/models/BLX_openr6.ttt`、正式 URDF、正式 meshes。
- 不回退 ABB/IRB140。
- 保持 `move_coordinate_all` 为 position-only；RX/RY/RZ 只传 0。
- 不把旧 5 STL和旧 candidate contract当作当前几何主线。
- 厂家 ZIP/STEP只读，来源 SHA-256 必须为：

  ```text
  f3f493626792ef25b99e6b1e79f791da96f2526cee7161c90194443e9c34a9a2
  ```

- 不在旧实验四主程序内继续堆条件分支；新逻辑进入 `vision_platform/`。
- 海康 SDK只在 `VISION_BACKEND=hik` 时延迟导入。
- 不使用 `PyThreadState_SetAsyncExc`。
- 每个任务单独提交，暂存时使用精确路径，禁止 `git add .`。
- 当前 clean HEAD 基线为 `32 passed, 12 failed`；12 个失败是 IRB140 陈旧断言和 FakeSimIK 契约缺失，Task 1 负责收口。

---

## Task 1：收口 BLX 测试基线

**文件：**

- Modify: `tests/test_robot_backends/test_coppeliasim_robot.py`
- Modify: `tests/test_robot_backends/test_factory.py`
- Verify: `tests/test_blx_integration.py`

### Step 1：确认可信 BLX 集成入口的执行类型

```powershell
python -m pytest tests/test_blx_integration.py --collect-only -q
```

该文件是连接真实 CoppeliaSim 的脚本式验收入口，不包含 pytest test function，因此预期收集 0 项。不要把“0 项”写成通过；实际脚本运行推迟到 Task 17 的 CoppeliaSim 运行验收。

### Step 2：用已有失败证明旧测试已过时

```powershell
python -m pytest `
  tests/test_robot_backends/test_coppeliasim_robot.py `
  tests/test_robot_backends/test_factory.py -q
```

预期失败：

- 断言期望 `/IRB140`，实际为 `/BLX_base_link`；
- 断言期望 `/IRB140/joint1`，实际为 `/BLX_joint1`；
- `FakeSimIK` 缺 `constraint_position`。

### Step 3：最小修正测试契约

更新旧断言：

```python
assert sim_settings["base_path"] == "/BLX_base_link"
assert sim_settings["tip_path"] == "/BLX_tool_suction"
assert settings["sim"]["joint_paths"][0] == "/BLX_joint1"
```

在 `FakeSimIK` 增加真实 API 所需常量：

```python
constraint_position = 1
```

测试输入 settings 全部改为 BLX路径、`joint5_offset_deg=0`，不得改生产 backend 来迎合旧 IRB140 fake。

### Step 4：运行局部测试

```powershell
python -m pytest `
  tests/test_robot_backends/test_coppeliasim_robot.py `
  tests/test_robot_backends/test_factory.py `
  tests/test_blx_integration.py -q
```

预期：全部通过。

### Step 5：运行完整基线

```powershell
python -m pytest -q
```

预期：44 项全部通过。

### Step 6：提交

```powershell
git add -- `
  tests/test_robot_backends/test_coppeliasim_robot.py `
  tests/test_robot_backends/test_factory.py
git commit -m "test: align legacy backend tests with BLX"
```

---

## Task 2：建立可复现视觉环境

**文件：**

- Modify: `.gitignore`
- Create: `requirements-vision.txt`
- Create: `requirements-vision-cad.txt`
- Create: `tools/vision_lab/bootstrap.ps1`
- Create: `tools/vision_lab/python.ps1`
- Test: `tests/test_vision_platform/test_environment_contract.py`

### Step 1：写失败测试

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_vision_environment_contract_declares_required_packages():
    requirements = (ROOT / "requirements-vision.txt").read_text(encoding="utf-8")
    for package in (
        "numpy", "opencv-python", "PyQt5", "pyzmq", "cbor2",
        "pytest", "pytest-qt",
    ):
        assert package.lower() in requirements.lower()


def test_cad_environment_declares_mesh_and_ocp_tools():
    requirements = (ROOT / "requirements-vision-cad.txt").read_text(encoding="utf-8")
    assert "trimesh" in requirements.lower()
    assert "cadquery-ocp" in requirements.lower()


def test_bootstrap_uses_project_local_venv_and_coppeliasim_root():
    script = (ROOT / "tools/vision_lab/bootstrap.ps1").read_text(encoding="utf-8")
    assert ".venv-vision" in script
    assert "COPPELIASIM_ROOT" in script
    assert "zmqRemoteApi" in script
```

运行：

```powershell
python -m pytest tests/test_vision_platform/test_environment_contract.py -q
```

预期：因文件不存在而失败。

### Step 2：实现最小环境文件

`.gitignore` 增加：

```gitignore
.venv-vision/
artifacts/vision_lab/
```

`requirements-vision.txt` 使用兼容 Python 3.11 的固定主次版本范围，不引入实验五至七的 onnxruntime。`requirements-vision-cad.txt` 单独包含 `trimesh` 与提供 `OCP` 命名空间的 `cadquery-ocp`。

`bootstrap.ps1`：

1. 优先查找 `py -3.11`；
2. 创建 `.venv-vision`；
3. 升级 pip；
4. 安装 `requirements-vision.txt` 和 `requirements-vision-cad.txt`；
5. 验证 `E:\CoppeliaSim` 或 `COPPELIASIM_ROOT`；
6. 把 ZMQ client `src` 写入 `.venv-vision\Lib\site-packages\coppeliasim-local.pth`；
7. 打印 Python、OpenCV、PyQt和 remote client导入版本。

`python.ps1` 统一调用 `.venv-vision\Scripts\python.exe`，不存在时给出 bootstrap 指令。

### Step 3：运行环境契约测试

```powershell
python -m pytest tests/test_vision_platform/test_environment_contract.py -q
```

预期：通过。

### Step 4：实际创建环境并验证导入

```powershell
powershell -ExecutionPolicy Bypass -File tools/vision_lab/bootstrap.ps1
.\.venv-vision\Scripts\python.exe -c "import cv2,numpy,PyQt5,zmq,cbor2; from coppeliasim_zmqremoteapi_client import RemoteAPIClient; print('VISION_ENV_OK')"
```

预期输出包含 `VISION_ENV_OK`。

### Step 5：提交

```powershell
git add -- `
  .gitignore `
  requirements-vision.txt `
  requirements-vision-cad.txt `
  tools/vision_lab/bootstrap.ps1 `
  tools/vision_lab/python.ps1 `
  tests/test_vision_platform/test_environment_contract.py
git commit -m "build: add reproducible vision lab environment"
```

---

## Task 3：定义核心数据、错误和配置契约

**文件：**

- Create: `vision_platform/__init__.py`
- Create: `vision_platform/models.py`
- Create: `vision_platform/errors.py`
- Create: `vision_platform/config.py`
- Create: `vision_platform/capabilities.py`
- Create: `config/vision_lab.default.json`
- Test: `tests/test_vision_platform/test_models.py`
- Test: `tests/test_vision_platform/test_config.py`
- Test: `tests/test_vision_platform/test_capabilities.py`

### Step 1：写 Frame/Detection/TaskResult 失败测试

```python
import numpy as np
import pytest

from vision_platform.models import Frame, Detection, TaskResult


def test_frame_rejects_dimensions_that_do_not_match_image():
    with pytest.raises(ValueError, match="dimensions"):
        Frame(
            image_bgr=np.zeros((10, 20, 3), dtype=np.uint8),
            width=21,
            height=10,
            timestamp_s=1.0,
            source="replay",
            sequence_id=1,
        )


def test_detection_exposes_pixel_center_as_float_pair():
    detection = Detection(
        detection_id="d1",
        center_px=(12.5, 30.0),
        color="red",
        shape="square",
        angle_deg=0.0,
        area_px=400.0,
        confidence=1.0,
    )
    assert detection.center_px == (12.5, 30.0)


def test_task_result_serializes_without_numpy_objects():
    result = TaskResult(task_id="t1", status="PASS", events=[], metrics={"count": 1})
    assert result.to_dict()["metrics"] == {"count": 1}
```

运行后预期：模块不存在。

### Step 2：实现不可变数据类

`models.py` 使用 dataclass：

```python
@dataclass(frozen=True)
class Frame:
    image_bgr: np.ndarray
    width: int
    height: int
    timestamp_s: float
    source: str
    sequence_id: int
    intrinsics: dict | None = None
    depth_m: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.image_bgr.shape[:2] != (self.height, self.width):
            raise ValueError("frame dimensions do not match image")
```

同时实现 `Detection`、`CalibrationRecord`、`TaskEvent`、`TaskResult`，序列化时转换 tuple/numpy 标量。

### Step 3：写配置优先级失败测试

```python
def test_environment_overrides_default_json(monkeypatch, tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"camera_backend":"replay","robot_backend":"sim"}')
    monkeypatch.setenv("VISION_BACKEND", "sim")
    cfg = load_config(config_file)
    assert cfg.camera_backend == "sim"
```

### Step 4：实现配置和能力检测

- 默认 JSON包含 camera、robot、workspace、calibration、recognition、task、scene、ui；
- 环境变量覆盖 JSON；
- 配置路径全部转绝对 `Path`；
- 能力检测不导入海康 SDK，只检查候选路径；
- 返回 `Capability(name, available, reason)`。

### Step 5：测试

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision_platform/test_models.py `
  tests/test_vision_platform/test_config.py `
  tests/test_vision_platform/test_capabilities.py -q
```

预期：全部通过。

### Step 6：提交

```powershell
git add -- `
  vision_platform/__init__.py `
  vision_platform/models.py `
  vision_platform/errors.py `
  vision_platform/config.py `
  vision_platform/capabilities.py `
  config/vision_lab.default.json `
  tests/test_vision_platform/test_models.py `
  tests/test_vision_platform/test_config.py `
  tests/test_vision_platform/test_capabilities.py
git commit -m "feat: define vision platform core contracts"
```

---

## Task 4：实现 CameraBackend、ReplayCamera 和工厂

**文件：**

- Create: `vision_platform/cameras/__init__.py`
- Create: `vision_platform/cameras/base.py`
- Create: `vision_platform/cameras/replay.py`
- Create: `vision_platform/cameras/factory.py`
- Create: `config/replay_manifest.json`
- Test: `tests/test_vision_platform/test_replay_camera.py`
- Test: `tests/test_vision_platform/test_camera_factory.py`

### Step 1：写 ReplayCamera 失败测试

```python
import cv2
import numpy as np

from vision_platform.cameras.replay import ReplayCamera


def test_replay_camera_returns_bgr_frame_with_monotonic_sequence(tmp_path):
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    image_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(image_path), image)

    camera = ReplayCamera([image_path], loop=True)
    first = camera.read()
    second = camera.read()

    assert first.image_bgr[0, 0].tolist() == [10, 20, 30]
    assert first.source == "replay"
    assert second.sequence_id == first.sequence_id + 1
```

### Step 2：确认 RED

```powershell
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision_platform/test_replay_camera.py -q
```

预期：模块不存在。

### Step 3：实现最小后端

`CameraBackend`：

```python
class CameraBackend(ABC):
    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self, timeout_s: float = 1.0) -> Frame: ...

    @abstractmethod
    def close(self) -> None: ...
```

`ReplayCamera` 支持：

- 图片列表；
- manifest；
- 循环/单次；
- 明确的 `FRAME_TIMEOUT` / 文件不存在错误；
- context manager；
- 单调 sequence_id。

### Step 4：写工厂延迟导入测试

```python
def test_factory_does_not_import_hikvision_for_replay(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "MvImport.MvCameraControl_class", None)
    camera = create_camera("replay", {"paths": [tmp_path / "one.png"]})
    assert camera.__class__.__name__ == "ReplayCamera"
```

### Step 5：实现工厂

工厂只在对应分支 import：

```python
if backend == "replay":
    from .replay import ReplayCamera
elif backend == "sim":
    from .coppeliasim import CoppeliaSimCamera
elif backend == "hik":
    from .hikvision import HikvisionCamera
```

### Step 6：测试并提交

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision_platform/test_replay_camera.py `
  tests/test_vision_platform/test_camera_factory.py -q
```

```powershell
git add -- `
  vision_platform/cameras `
  config/replay_manifest.json `
  tests/test_vision_platform/test_replay_camera.py `
  tests/test_vision_platform/test_camera_factory.py
git commit -m "feat: add camera contract and replay backend"
```

---

## Task 5：实现实验三仿射标定

**文件：**

- Create: `vision_platform/calibration/__init__.py`
- Create: `vision_platform/calibration/affine.py`
- Create: `vision_platform/calibration/store.py`
- Test: `tests/test_vision_platform/test_affine_calibration.py`
- Test: `tests/test_vision_platform/test_calibration_store.py`

### Step 1：写准确变换失败测试

```python
import numpy as np

from vision_platform.calibration.affine import AffineCalibration


def test_three_point_affine_maps_pixel_to_world():
    calibration = AffineCalibration.fit(
        pixel_points=[(0, 0), (100, 0), (0, 100)],
        world_points_mm=[(10, 20), (110, 20), (10, 220)],
        image_size=(640, 480),
        plane_z_mm=30,
    )
    assert np.allclose(calibration.pixel_to_world((25, 50)), (35, 120), atol=1e-6)
```

### Step 2：写退化点失败测试

```python
def test_affine_rejects_collinear_points():
    with pytest.raises(CalibrationError, match="collinear"):
        AffineCalibration.fit(
            [(0, 0), (10, 10), (20, 20)],
            [(0, 0), (10, 0), (20, 0)],
            image_size=(100, 100),
            plane_z_mm=0,
        )
```

### Step 3：实现最小标定

- 使用 `cv2.getAffineTransform` 或 numpy最小二乘；
- 至少 3 点；
- 对三点检查三角形面积；
- 多于 3 点时使用最小二乘；
- `evaluate(validation_pairs)` 返回 RMS/max；
- 图像尺寸不匹配时报 `CALIBRATION_INVALID`。

### Step 4：实现 JSON 保存/加载

写 round-trip 测试，要求：

- schema_version；
- matrix；
- points；
- image_size；
- plane_z_mm；
- source/scene version；
- error metrics。

### Step 5：测试

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision_platform/test_affine_calibration.py `
  tests/test_vision_platform/test_calibration_store.py -q
```

### Step 6：提交

```powershell
git add -- vision_platform/calibration tests/test_vision_platform/test_affine_calibration.py tests/test_vision_platform/test_calibration_store.py
git commit -m "feat: add teachable affine camera calibration"
```

---

## Task 6：实现实验四多对象颜色/形状识别

**文件：**

- Create: `vision_platform/recognition/__init__.py`
- Create: `vision_platform/recognition/color_shape.py`
- Create: `vision_platform/cli.py`
- Test: `tests/test_vision_platform/test_color_shape_recognition.py`

### Step 1：用内存图像写失败测试

```python
import cv2
import numpy as np

from vision_platform.recognition.color_shape import ColorShapeRecognizer


def test_recognizer_finds_red_square_and_blue_circle():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (80, 100), (180, 200), (0, 0, 255), -1)
    cv2.circle(image, (420, 260), 55, (255, 0, 0), -1)

    detections = ColorShapeRecognizer().detect(image)

    assert [(d.color, d.shape) for d in detections] == [
        ("red", "square"),
        ("blue", "circle"),
    ]
```

### Step 2：写边界和空图测试

- 小于动态面积阈值的噪点不返回；
- 靠图像边缘的 ROI不抛异常；
- 无对象返回空列表；
- 多对象按 `(center_y, center_x)` 稳定排序。

### Step 3：确认 RED

```powershell
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision_platform/test_color_shape_recognition.py -q
```

### Step 4：实现最小识别器

- HSV 前景；
- 动态面积阈值 `image_area * min_area_ratio`；
- contour centroid；
- `approxPolyDP`；
- 圆形使用 circularity + 边数；
- 颜色读取 contour mask内部中位 HSV，而不是固定中心 10×10；
- 置信度基于面积、颜色间隔和形状稳定性；
- 可选 `annotate(image, detections)`。

同时创建最小 `vision_platform.cli`，本任务只提供 `recognize` 子命令；后续 Task 12 在同一 CLI 上增加 `accept` 等子命令。

### Step 5：运行现有实验四真实图片回放

```powershell
.\.venv-vision\Scripts\python.exe -m vision_platform.cli recognize `
  --image "data\replay\experiment-4-result.jpg" `
  --output artifacts/vision_lab/recognize-exp4.json
```

该步骤只记录真实图片结果，不要求用受控仿真阈值伪造真实集 100%。

### Step 6：测试并提交

```powershell
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision_platform/test_color_shape_recognition.py -q
```

```powershell
git add -- vision_platform/recognition vision_platform/cli.py tests/test_vision_platform/test_color_shape_recognition.py
git commit -m "feat: add deterministic color and shape recognition"
```

---

## Task 7：实现机器人适配、安全路径和工具契约

**文件：**

- Create: `vision_platform/robot/__init__.py`
- Create: `vision_platform/robot/adapter.py`
- Create: `vision_platform/robot/safety.py`
- Create: `vision_platform/robot/tool.py`
- Test: `tests/test_vision_platform/test_robot_adapter.py`
- Test: `tests/test_vision_platform/test_motion_safety.py`

### Step 1：写适配器失败测试

```python
def test_move_world_preserves_position_only_backend_contract():
    backend = RecordingBackend()
    robot = RobotAdapter(backend)

    robot.move_world(80, -20, 50, speed=15)

    assert backend.coordinate_calls == [(80, -20, 50, 0, 0, 0, 15)]
```

### Step 2：写门字形路径失败测试

```python
def test_gate_path_never_moves_horizontally_below_safe_z():
    policy = WorkspacePolicy(
        x_mm=(20, 140),
        y_mm=(-90, 90),
        z_mm=(10, 140),
        safe_z_mm=100,
    )
    path = plan_gate_path(
        pick=(70, -30, 20),
        drop=(110, 50, 20),
        policy=policy,
    )
    for previous, current in pairwise(path):
        horizontal = previous[:2] != current[:2]
        if horizontal:
            assert previous[2] >= 100
            assert current[2] >= 100
```

### Step 3：实现

- `RobotAdapter` 包装现有 backend；
- `ToolService` 抽象 `on/off/is_attached`；
- real模式默认 `BackendPumpTool` 调用 `pump_on/off`；
- `WorkspacePolicy` 统一检查 X/Y/Z、安全 Z、最大单步；
- `plan_gate_path` 返回明确动作列表，不直接执行；
- 错误使用稳定错误码。

### Step 4：测试

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision_platform/test_robot_adapter.py `
  tests/test_vision_platform/test_motion_safety.py -q
```

### Step 5：提交

```powershell
git add -- vision_platform/robot tests/test_vision_platform/test_robot_adapter.py tests/test_vision_platform/test_motion_safety.py
git commit -m "feat: add robot adapter and safe gate motion"
```

---

## Task 8：实现分类任务状态机和无硬件闭环

**文件：**

- Create: `vision_platform/events.py`
- Create: `vision_platform/tasks/__init__.py`
- Create: `vision_platform/tasks/state_machine.py`
- Create: `vision_platform/tasks/classify.py`
- Test: `tests/test_vision_platform/test_classification_state_machine.py`
- Test: `tests/test_acceptance/test_fake_closed_loop.py`

### Step 1：写完整状态序列失败测试

```python
EXPECTED = [
    "IDLE", "ACQUIRE", "DETECT", "TRANSFORM", "APPROACH", "DESCEND",
    "ATTACH", "LIFT", "TRANSFER", "RELEASE", "VERIFY", "COMPLETE",
]


def test_successful_classification_emits_complete_state_sequence():
    rig = FakeRig.single_object(color="red", shape="square")
    result = rig.run()
    assert [event.state for event in result.events] == EXPECTED
    assert result.status == "PASS"
```

### Step 2：写 SAFE_STOP 失败测试

```python
def test_attach_failure_turns_tool_off_and_enters_safe_stop():
    rig = FakeRig.single_object(tool_attaches=False)
    result = rig.run()
    assert result.status == "FAIL"
    assert result.error_code == "SUCTION_ATTACH_FAILED"
    assert rig.tool.off_calls >= 1
    assert result.events[-1].state == "SAFE_STOP"
```

### Step 3：实现同步可观察状态机

- 状态机本身同步、确定性；
- 每次状态变化发布 `TaskEvent`；
- UI worker负责异步运行，不把线程并发塞入核心；
- 支持 cancel token；
- 支持单步模式；
- 多对象任务每次重新取帧，避免使用搬运前过期位置；
- 选择规则明确：最高置信度且未处理对象。

### Step 4：实现分类区映射和验证端口

```python
class PlacementVerifier(Protocol):
    def verify(self, detection, expected_zone: str) -> PlacementEvidence: ...
```

默认支持按 `color` 或 `shape` 分类。

### Step 5：测试

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision_platform/test_classification_state_machine.py `
  tests/test_acceptance/test_fake_closed_loop.py -q
```

### Step 6：提交

```powershell
git add -- `
  vision_platform/events.py `
  vision_platform/tasks `
  tests/test_vision_platform/test_classification_state_machine.py `
  tests/test_acceptance/test_fake_closed_loop.py
git commit -m "feat: add observable closed-loop classification task"
```

---

## Task 9：实现 CoppeliaSim 虚拟相机后端

**文件：**

- Create: `vision_platform/coppelia.py`
- Create: `vision_platform/cameras/coppeliasim.py`
- Test: `tests/test_vision_platform/test_coppeliasim_camera.py`

### Step 1：写 RGB/方向转换失败测试

```python
def test_coppeliasim_rgb_bytes_are_flipped_and_converted_to_bgr():
    # Coppelia返回从下到上的 RGB：bottom red, top blue
    raw = bytes([
        255, 0, 0, 255, 0, 0,
        0, 0, 255, 0, 0, 255,
    ])
    sim = FakeSim(raw=raw, resolution=[2, 2])
    camera = CoppeliaSimCamera(sim=sim, sensor_path="/VisionLab/Camera")

    frame = camera.read()

    assert frame.image_bgr[0, 0].tolist() == [255, 0, 0]
    assert frame.image_bgr[1, 0].tolist() == [0, 0, 255]
```

根据本机 API实测确认垂直方向；如果实测与 fake假设相反，先修测试说明再实现，不能盲目翻转两次。

### Step 2：写超时/分辨率失败测试

- 空 bytes → `FRAME_TIMEOUT`；
- resolution变化 → Frame按新尺寸生成；
- 长度不等于 `width*height*3` → 显式错误；
- sensor路径解析一次并缓存。

### Step 3：实现

- `CoppeliaClientResolver` 从已注入 sim/client或本机 remote API创建连接；
- `CoppeliaSimCamera.open()` 解析 Vision Sensor；
- `read()` 调 `sim.getVisionSensorImg`；
- RGB→BGR、方向归一；
- 可选调用 `sim.getVisionSensorDepth(..., options=1)`；
- `close()` 不关闭共享 client，除非实例拥有它。

### Step 4：测试

```powershell
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision_platform/test_coppeliasim_camera.py -q
```

### Step 5：提交

```powershell
git add -- vision_platform/coppelia.py vision_platform/cameras/coppeliasim.py tests/test_vision_platform/test_coppeliasim_camera.py
git commit -m "feat: add CoppeliaSim vision sensor camera backend"
```

---

## Task 10：实现模拟吸盘附着、释放和分类区验证

**文件：**

- Create: `vision_platform/robot/coppeliasim_suction.py`
- Create: `vision_platform/tasks/coppeliasim_verifier.py`
- Test: `tests/test_vision_platform/test_coppeliasim_suction.py`
- Test: `tests/test_vision_platform/test_coppeliasim_placement_verifier.py`

### Step 1：写最近对象附着失败测试

```python
def test_suction_attaches_nearest_eligible_object_and_preserves_world_pose():
    sim = FakeScene.with_pickables(
        tcp=(0.08, 0.00, 0.03),
        objects={
            101: (0.081, 0.002, 0.02),
            102: (0.12, 0.00, 0.02),
        },
    )
    tool = CoppeliaSimSuction(sim, tcp_path="/BLX_tool_suction")

    evidence = tool.on()

    assert evidence.object_handle == 101
    assert sim.parent_of(101) == sim.handle("/BLX_tool_suction")
    assert sim.keep_in_place_calls[-1] is True
```

### Step 2：写失败和释放测试

- 无对象在容差内 → `SUCTION_ATTACH_FAILED`；
- 二次 `on()` 幂等；
- `off()` 恢复工作区 parent和动态属性；
- 未附着时 `off()` 安全；
- 对象不属于 pickables集合时禁止附着。

### Step 3：实现

优先使用对象路径/collection筛选，计算 TCP到对象参考点距离；保存：

- original parent；
- model/dynamic flags；
- attachment handle；
- world pose。

释放时恢复 parent和属性。

### Step 4：写分类区验证测试

```python
def test_verifier_passes_only_when_object_center_is_inside_expected_zone():
    verifier = CoppeliaSimPlacementVerifier(
        sim=FakeScene.object_in_zone("red", object_handle=101),
        zone_paths={"red": "/VisionLab/Zones/red"},
    )
    evidence = verifier.verify(101, expected_zone="red")
    assert evidence.ok is True
```

### Step 5：测试并提交

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision_platform/test_coppeliasim_suction.py `
  tests/test_vision_platform/test_coppeliasim_placement_verifier.py -q
```

```powershell
git add -- `
  vision_platform/robot/coppeliasim_suction.py `
  vision_platform/tasks/coppeliasim_verifier.py `
  tests/test_vision_platform/test_coppeliasim_suction.py `
  tests/test_vision_platform/test_coppeliasim_placement_verifier.py
git commit -m "feat: add simulated suction and placement verification"
```

---

## Task 11：建立指定 STEP 几何 provenance 和独立场景构建器

**文件：**

- Create: `simulation/vision_lab/__init__.py`
- Create: `simulation/vision_lab/source_manifest.json`
- Create: `simulation/vision_lab/import_robot_assets.py`
- Create: `simulation/vision_lab/build_scene.py`
- Create: `simulation/vision_lab/verify_scene.py`
- Create: `simulation/vision_lab/assets/robot/README.md`
- Create/generated: `simulation/vision_lab/assets/robot/*.STL`
- Create/generated: `simulation/vision_lab/BL23_vision_lab.ttt`
- Test: `tests/test_simulation/test_robot_asset_provenance.py`
- Test: `tests/test_simulation/test_scene_manifest.py`

### Step 1：写来源哈希失败测试

```python
EXPECTED_STEP_SHA256 = "f3f493626792ef25b99e6b1e79f791da96f2526cee7161c90194443e9c34a9a2"


def test_source_manifest_pins_user_selected_step():
    manifest = json.loads(
        (ROOT / "simulation/vision_lab/source_manifest.json")
        .read_text(encoding="utf-8")
    )
    assert manifest["source_step_sha256"] == EXPECTED_STEP_SHA256
    assert manifest["segments"] == [
        "base", "link1", "link2", "link3", "link4", "link5",
        "link6", "tool",
    ]
```

### Step 2：写资产门测试

每个 segment必须：

- 文件存在；
- output SHA-256匹配 manifest；
- triangle count > 0；
- bounds有限；
- 尺寸单位换算明确；
- 来源包含 141 volume tag列表；
- 不引用旧 5 STL contract作为 source-of-truth。

如果 `trimesh.is_watertight` 未通过，只允许：

- 重新 OCP实体级导出；或
- 明确标记 `visual_only` 并生成独立简化碰撞体。

不能把 `repaired_v4` 文件名本身当作质量证明。

### Step 3：实现导入/生成脚本

脚本输入：

```text
厂家 STEP
simulation/vision_lab/provenance/step_volumes_2026-06-08.json
simulation/vision_lab/provenance/segment_volume_mapping_2026-06-08.json
```

执行顺序：

1. 校验 STEP SHA-256；
2. 从仓库内固定的 provenance 快照读取实体与分段记录；
3. 对每段运行三角面、bounds、连通性和 watertight检查；
4. 不合格段调用受控 OCP重导脚本；
5. 写入 `simulation/vision_lab/assets/robot/`；
6. 生成带输入/输出 hash的 manifest；
7. 不修改厂家原始 ZIP/STEP。

### Step 4：写场景 manifest 失败测试

要求：

```python
required_paths = {
    "/BLX_base_link",
    "/BLX_joint1", "/BLX_joint2", "/BLX_joint3",
    "/BLX_joint4", "/BLX_joint5", "/BLX_joint6",
    "/BLX_tool_suction",
    "/VisionLab/Camera",
    "/VisionLab/Workspace",
    "/VisionLab/Pickables",
    "/VisionLab/Zones",
}
```

同时要求 3 个标定点、额外验证点和至少 6 个受控对象。

### Step 5：实现可重复场景构建器

`build_scene.py`：

1. 校验正式 BLX资产 pre-hash；
2. 把正式 BLX场景复制到 ASCII staging；
3. 启动/连接 CoppeliaSim；
4. 加载 staging；
5. 隐藏正式视觉 meshes但保留运动学对象；
6. 导入 STEP派生 mesh并 parent到对应 joint；
7. 创建简化碰撞体；
8. 创建 `/VisionLab` 工作台、Vision Sensor、标定点、对象、分类区；
9. 生成至少 6 个可达对象布局；
10. 保存到 ASCII staging；
11. 复制到 `simulation/vision_lab/BL23_vision_lab.ttt`；
12. 写 scene manifest与 hash；
13. 校验正式 BLX资产 post-hash等于 pre-hash。

### Step 6：运行静态来源测试

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_simulation/test_robot_asset_provenance.py `
  tests/test_simulation/test_scene_manifest.py -q
```

### Step 7：运行场景构建

```powershell
.\.venv-vision\Scripts\python.exe -m simulation.vision_lab.import_robot_assets
.\.venv-vision\Scripts\python.exe -m simulation.vision_lab.build_scene `
  --coppeliasim-root E:\CoppeliaSim
```

预期：

- 输出场景；
- 输出 manifest；
- 正式资产 hash unchanged；
- 无对象路径缺失。

### Step 8：运行场景验证

```powershell
.\.venv-vision\Scripts\python.exe -m simulation.vision_lab.verify_scene `
  --scene simulation/vision_lab/BL23_vision_lab.ttt `
  --report artifacts/vision_lab/scene-verification.json
```

验证：

- scene可加载；
- 六关节可访问；
- STEP派生 8 段对象在位；
- j1–j6各 ±10° 后子 mesh跟随；
- TCP存在；
- Vision Sensor可取图；
- 6 个对象和分类区存在；
- 仿真停止/启动正常。

### Step 9：提交

二进制和 mesh提交前再次检查文件大小、hash和来源。精确暂存：

```powershell
git add -- `
  simulation/vision_lab `
  tests/test_simulation/test_robot_asset_provenance.py `
  tests/test_simulation/test_scene_manifest.py
git commit -m "feat: build isolated BL23 vision training scene"
```

---

## Task 12：接通仿真标定和完整分类闭环

**文件：**

- Create: `vision_platform/application.py`
- Create: `vision_platform/acceptance.py`
- Modify: `vision_platform/cli.py`
- Create: `tests/test_acceptance/conftest.py`
- Test: `tests/test_acceptance/test_coppeliasim_calibration.py`
- Test: `tests/test_acceptance/test_coppeliasim_classification.py`

### Step 1：写标定验收测试

测试调用真实 CoppeliaSim时使用 marker：

```python
@pytest.mark.coppeliasim
def test_sim_calibration_meets_teaching_error_budget(running_vision_scene):
    report = run_calibration_acceptance(running_vision_scene)
    assert report.rms_error_mm <= 3.0
    assert report.max_error_mm <= 5.0
```

没有运行 sim时明确 skip，不把 skip当 PASS。

`tests/test_acceptance/conftest.py` 负责：

- 只在 `-m coppeliasim` 时启动/复用 CoppeliaSim；
- 通过独立场景副本工作；
- 等待 ZMQ端口就绪；
- 测试结束停止仿真并关闭本次 fixture拥有的进程；
- 把 skip原因写清楚，最终验收脚本对 skip作失败处理。

### Step 2：写 6 对象完整分类测试

```python
@pytest.mark.coppeliasim
def test_six_objects_are_sorted_into_expected_zones(running_vision_scene):
    result = run_classification_acceptance(
        scene=running_vision_scene,
        object_count=6,
    )
    assert result.status == "PASS"
    assert result.metrics["objects_completed"] == 6
    assert result.metrics["attach_success"] == 6
    assert result.metrics["release_success"] == 6
    assert result.metrics["correct_zone"] == 6
```

### Step 3：实现应用装配

`VisionLabApplication.from_config()` 负责：

- camera factory；
- calibration store；
- recognizer；
- robot backend factory；
- tool service；
- placement verifier；
- task状态机；
- event bus；
- 统一 close。

### Step 4：实现验收 CLI

```powershell
.\.venv-vision\Scripts\python.exe -m vision_platform.cli accept `
  --robot sim `
  --camera sim `
  --scene simulation/vision_lab/BL23_vision_lab.ttt `
  --output artifacts/vision_lab/acceptance
```

输出：

- `acceptance.json`
- `events.jsonl`
- 标定原图/标注图；
- 每个对象抓取前、释放后关键帧；
- 配置快照；
- 场景/STEP hash；
- `PENDING_HARDWARE` 项。

### Step 5：运行真实 CoppeliaSim 验收

```powershell
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_acceptance/test_coppeliasim_calibration.py `
  tests/test_acceptance/test_coppeliasim_classification.py `
  -m coppeliasim -q
```

必须看到 2 项实际 PASS，不能是 skip。

### Step 6：提交

```powershell
git add -- `
  vision_platform/application.py `
  vision_platform/acceptance.py `
  vision_platform/cli.py `
  tests/test_acceptance/conftest.py `
  tests/test_acceptance/test_coppeliasim_calibration.py `
  tests/test_acceptance/test_coppeliasim_classification.py
git commit -m "feat: complete CoppeliaSim calibration and sorting loop"
```

---

## Task 13：实现 Hikvision 延迟加载适配器

**文件：**

- Create: `vision_platform/cameras/hikvision.py`
- Test: `tests/test_vision_platform/test_hikvision_camera.py`

### Step 1：写无 SDK 失败测试

```python
def test_hikvision_backend_reports_actionable_unavailable_error(monkeypatch):
    monkeypatch.delenv("HIK_MVS_ROOT", raising=False)
    with pytest.raises(CameraUnavailableError) as exc:
        HikvisionCamera().open()
    assert exc.value.code == "CAMERA_UNAVAILABLE"
    assert "MVS" in str(exc.value)
```

### Step 2：写 fake SDK帧转换测试

fake覆盖：

- 枚举；
- 打开；
- 开始取流；
- BGR/RGB像素格式；
- timeout；
- stop/close；
- SDK错误码转中文消息。

### Step 3：实现

- 不在模块顶层加载 `.so` / `.dll`；
- 从 `HIK_MVS_ROOT`、项目旧 MvImport位置和平台默认位置发现 SDK；
- 把 SDK调用封装在小 adapter；
- 不复制旧主程序中的 UI线程和强杀线程逻辑；
- 当前环境运行结果应为清楚的 unavailable，而不是 import crash。

### Step 4：测试并提交

```powershell
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision_platform/test_hikvision_camera.py -q
```

```powershell
git add -- vision_platform/cameras/hikvision.py tests/test_vision_platform/test_hikvision_camera.py
git commit -m "feat: add optional Hikvision camera adapter"
```

---

## Task 14：实现共享 ViewModel 与 PyQt 教学界面

**文件：**

- Create: `vision_platform/ui/__init__.py`
- Create: `vision_platform/ui/view_model.py`
- Create: `vision_platform/ui/pyqt_app.py`
- Create: `tools/vision_lab/run_pyqt.ps1`
- Test: `tests/test_vision_platform/test_view_model.py`
- Test: `tests/test_vision_platform/test_pyqt_smoke.py`

### Step 1：写 ViewModel 状态投影失败测试

```python
def test_view_model_projects_task_event_to_student_status():
    vm = VisionLabViewModel()
    vm.apply(TaskEvent(state="DETECT", message="识别到红色正方形"))
    assert vm.state == "DETECT"
    assert vm.status_text == "识别到红色正方形"
```

### Step 2：写 PyQt offscreen smoke 失败测试

```python
@pytest.mark.qt
def test_pyqt_window_constructs_without_camera_or_robot_connection(qtbot):
    window = VisionLabWindow(application=DisconnectedApplication())
    qtbot.addWidget(window)
    assert window.camera_backend_combo.count() == 3
    assert window.robot_backend_combo.count() == 2
    assert "未连接" in window.status_label.text()
```

### Step 3：实现 ViewModel

- 纯 Python，无 Qt依赖；
- 保存连接、帧、标定、识别、任务、错误和验收状态；
- 事件订阅线程安全；
- UI只读 ViewModel快照。

### Step 4：实现 PyQt

界面：

- sim/replay/hik；
- sim/real；
- 能力状态；
- 图像预览；
- 三点标定表；
- 检测表；
- 像素/世界坐标；
- 启动、暂停、继续、复位、急停；
- 单步/全自动；
- 日志和最终报告。

任务运行使用 `QThread` + signals，取消使用 `threading.Event`；窗口关闭时等待 worker退出，不强杀线程。

### Step 5：测试

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv-vision\Scripts\python.exe -m pytest `
  tests/test_vision_platform/test_view_model.py `
  tests/test_vision_platform/test_pyqt_smoke.py -q
```

### Step 6：提交

```powershell
git add -- `
  vision_platform/ui/__init__.py `
  vision_platform/ui/view_model.py `
  vision_platform/ui/pyqt_app.py `
  tools/vision_lab/run_pyqt.ps1 `
  tests/test_vision_platform/test_view_model.py `
  tests/test_vision_platform/test_pyqt_smoke.py
git commit -m "feat: add shared view model and PyQt vision lab UI"
```

---

## Task 15：实现 CoppeliaSim simUI 简化面板

**文件：**

- Create: `vision_platform/ui/simui_panel.py`
- Test: `tests/test_vision_platform/test_simui_panel.py`

### Step 1：写 XML 契约失败测试

```python
def test_simui_xml_contains_required_teaching_controls():
    xml = build_simui_xml()
    for text in ("启动", "暂停", "复位", "标定", "识别结果", "世界坐标"):
        assert text in xml
```

### Step 2：写生命周期失败测试

```python
def test_panel_creates_updates_and_destroys_one_ui():
    simui = RecordingSimUI()
    panel = SimUIPanel(simui=simui, view_model=VisionLabViewModel())
    panel.open()
    panel.refresh()
    panel.close()
    assert len(simui.created) == 1
    assert simui.destroyed == [simui.created[0]]
```

### Step 3：实现

- 使用本机 simUI XML语法；
- 显示状态、相机缩略图、识别、像素/世界坐标；
- 按钮回调发送共享命令；
- 不在回调内执行长任务；
- `close()` 幂等；
- simUI不可用时 PyQt仍可独立运行。

### Step 4：连接真实 CoppeliaSim做生命周期检查

```powershell
.\.venv-vision\Scripts\python.exe -m vision_platform.cli simui-smoke `
  --duration 5 `
  --output artifacts/vision_lab/simui-smoke.json
```

要求报告 `created=true`、至少一次 update、`destroyed=true`。

### Step 5：测试并提交

```powershell
.\.venv-vision\Scripts\python.exe -m pytest tests/test_vision_platform/test_simui_panel.py -q
```

```powershell
git add -- vision_platform/ui/simui_panel.py tests/test_vision_platform/test_simui_panel.py
git commit -m "feat: add CoppeliaSim teaching control panel"
```

---

## Task 16：教学入口、说明和验收包

**文件：**

- Create: `tools/vision_lab/launch_coppeliasim.ps1`
- Create: `tools/vision_lab/run_acceptance.ps1`
- Create: `tools/vision_lab/run_replay_demo.ps1`
- Create: `docs/视觉仿真实训平台使用说明.md`
- Create: `docs/实验三-CoppeliaSim视觉标定.md`
- Create: `docs/实验四-CoppeliaSim物体分类.md`
- Create: `docs/仿真转真机验证清单.md`
- Create: `simulation/vision_lab/README.md`
- Test: `tests/test_acceptance/test_delivery_contract.py`

### Step 1：写交付契约失败测试

```python
def test_delivery_has_one_command_for_each_student_workflow():
    required = [
        ROOT / "tools/vision_lab/run_pyqt.ps1",
        ROOT / "tools/vision_lab/run_replay_demo.ps1",
        ROOT / "tools/vision_lab/run_acceptance.ps1",
    ]
    assert all(path.exists() for path in required)


def test_hardware_checklist_does_not_claim_camera_acceptance():
    text = (ROOT / "docs/仿真转真机验证清单.md").read_text(encoding="utf-8")
    assert "PENDING_HARDWARE" in text
    assert "海康相机" in text
```

### Step 2：实现启动脚本

`launch_coppeliasim.ps1`：

- 解析 CoppeliaSim路径；
- 使用 `-s` 或受支持参数打开独立场景；
- 隐藏后台 helper窗口；
- 等待端口就绪；
- 超时输出进程和端口诊断。

`run_acceptance.ps1`：

- 检查 venv；
- 启动/复用 CoppeliaSim；
- 运行场景验证；
- 运行实验三和四 E2E；
- 运行完整 pytest；
- 汇总 `acceptance-summary.json`；
- 不把 skipped硬件项计为通过。

### Step 3：编写教学文档

每份实验说明包含：

- 任务；
- 入口；
- 原理；
- 步骤；
- 完成条件；
- 常见错误；
- 仿真与真机差异；
- 结果保存；
- 人工验收点。

真机清单明确：

- Linux/Windows MVS路径；
- 相机枚举；
- 曝光/触发；
- 真实三点重标定；
- 抓取 Z和 TCP偏差；
- 急停；
- 低速空跑；
- 单物体抓取；
- 多物体分类；
- 所有项 `PENDING_HARDWARE`，待实验室签字。

### Step 4：测试并提交

```powershell
.\.venv-vision\Scripts\python.exe -m pytest tests/test_acceptance/test_delivery_contract.py -q
```

```powershell
git add -- `
  tools/vision_lab/launch_coppeliasim.ps1 `
  tools/vision_lab/run_acceptance.ps1 `
  tools/vision_lab/run_replay_demo.ps1 `
  docs/视觉仿真实训平台使用说明.md `
  docs/实验三-CoppeliaSim视觉标定.md `
  docs/实验四-CoppeliaSim物体分类.md `
  docs/仿真转真机验证清单.md `
  simulation/vision_lab/README.md `
  tests/test_acceptance/test_delivery_contract.py
git commit -m "docs: deliver vision simulation teaching workflows"
```

---

## Task 17：全量验证、资产保护审计与最终人工验收包

**文件：**

- Create/generated: `artifacts/vision_lab/final/*`（默认 gitignored）
- Create: `docs/视觉仿真实训平台自动验收报告.md`
- Modify: `HERMES_BRIEF.md`（只在隔离分支追加本阶段最终状态）

### Step 1：建立完成审计清单

逐条重新读取：

- 总体设计第 14、17 节；
- 本计划 Task 1–16；
- 用户要求；
- 最新 HERMES边界。

为每项记录“证据文件/命令/结果”，缺证据即未完成。

### Step 2：运行完整静态测试

```powershell
.\.venv-vision\Scripts\python.exe -m pytest -q
```

要求：0 failed。记录总项数。

### Step 3：运行 CoppeliaSim 场景验证

```powershell
powershell -ExecutionPolicy Bypass -File tools/vision_lab/run_acceptance.ps1 `
  -OutputDir artifacts/vision_lab/final
```

要求：

- STEP hash匹配；
- scene integrity PASS；
- calibration RMS/max达标；
- 6 对象完整分类；
- 6 次 attach/release；
- 6 次 correct zone；
- simUI lifecycle PASS；
- PyQt offscreen smoke PASS。

### Step 4：运行 replay 验收

```powershell
powershell -ExecutionPolicy Bypass -File tools/vision_lab/run_replay_demo.ps1 `
  -OutputDir artifacts/vision_lab/final/replay
```

报告真实图片结果，不以仿真 100%替代。

### Step 5：验证正式资产未变化

比较实现前记录的 SHA-256：

```powershell
Get-FileHash robot_backends/models/BLX_openr6.ttt -Algorithm SHA256
Get-FileHash robot_backends/models/openr6_arm_coppeliasim.urdf -Algorithm SHA256
Get-ChildItem robot_backends/models/meshes_blx -File |
  Get-FileHash -Algorithm SHA256
```

要求与基线完全一致。

### Step 6：视觉 QA

生成并人工检查：

- 场景全景；
- 机器人厂家 STEP外观；
- 虚拟相机原始帧；
- 标定点与误差叠加图；
- 识别叠加图；
- 抓取前；
- 吸附后抬升；
- 分类区释放后；
- PyQt 主界面；
- simUI 面板。

自动检查不能代替该视觉 QA；若只能完成结构验证，报告必须明确。

### Step 7：写自动验收报告

报告分为：

- 已自动通过；
- 自动证据路径；
- 已知限制；
- `PENDING_HARDWARE`；
- 最终教师教学效果验收步骤。

### Step 8：更新 HERMES

只追加：

- 新架构入口；
- 独立场景；
- 验证命令；
- position-only；
- STEP provenance；
- 正式资产保护；
- 硬件待验；
- 当前分支/提交。

不得删除或重写用户现有历史记录。

### Step 9：最终全量复验

提交前重新运行：

```powershell
git diff --check
.\.venv-vision\Scripts\python.exe -m pytest -q
powershell -ExecutionPolicy Bypass -File tools/vision_lab/run_acceptance.ps1 `
  -OutputDir artifacts/vision_lab/final-rerun
git status --short
```

### Step 10：提交

```powershell
git add -- `
  docs/视觉仿真实训平台自动验收报告.md `
  HERMES_BRIEF.md
git commit -m "docs: close vision simulation MVP acceptance"
```

### Step 11：分支收口

使用 `finishing-a-development-branch`：

- 再次验证分支；
- 汇总提交；
- 检查与 H 盘用户脏工作区的重叠文件；
- 只把本分支明确新增/修改的路径同步回用户工作区；
- 不覆盖用户旧诊断、考核材料或 candidate历史文件；
- 保留可回滚的分支和提交。

最终交给教师的唯一人工门：

```text
按“视觉仿真实训平台使用说明”完成一次实验三和一次实验四，
评价界面、节奏、提示、理解难度和仿真—真机迁移感受。
```

---

## 最终完成检查表

- [ ] Task 1：BLX 基线 0 failed
- [ ] Task 2：独立环境可复现
- [ ] Task 3：核心契约
- [ ] Task 4：ReplayCamera
- [ ] Task 5：实验三标定
- [ ] Task 6：颜色/形状识别
- [ ] Task 7：机器人适配与安全路径
- [ ] Task 8：任务状态机
- [ ] Task 9：CoppeliaSimCamera
- [ ] Task 10：模拟吸盘与放置验证
- [ ] Task 11：指定 STEP 独立场景
- [ ] Task 12：真实 CoppeliaSim 闭环
- [ ] Task 13：Hikvision 延迟加载
- [ ] Task 14：PyQt
- [ ] Task 15：simUI
- [ ] Task 16：教学入口与文档
- [ ] Task 17：全量审计
- [ ] 正式 BLX资产哈希不变
- [ ] 实验三 RMS ≤ 3 mm、max ≤ 5 mm
- [ ] 6 个受控对象 6/6 正确分类
- [ ] 真实相机/真机明确 `PENDING_HARDWARE`
- [ ] 只剩最终教学效果人工验收
