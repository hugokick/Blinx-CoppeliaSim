# Robot Curriculum V2.2 First Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在已完成的 V2.1 学生程序受控执行基线上，交付多实验框架、相机快照 SDK、机器人基础场景、物流与码垛场景，以及 R1-01、R1-02、R1-05、R1-06、R1-07 五个可运行实验。

**Architecture:** 新增机器可读实验目录和 `ExperimentSession`，由它在学生进程停止后选择实验、校验能力并替换 `VisionLabApplication`；V2.1 `StudentProgramController` 仍是唯一学生命令网关。相机帧由主进程采集并编码为 PNG 字节，通过白名单响应返回子进程，同时把同一图像写入本次运行证据；两个新场景从现有正式场景只读加载，在内存中删除旧 `/VisionLab` 教学根节点后构建新的独立教学根节点和清单，绝不覆盖原正式场景。

**Tech Stack:** Python 3.11、标准库 `dataclasses` / `json` / `hashlib` / `multiprocessing`、NumPy、OpenCV、PyQt5、pytest、pytest-qt、CoppeliaSim ZMQ Remote API、PowerShell

---

## 0. 实施边界和执行前提

- 总体规格：`docs/superpowers/specs/2026-07-31-robot-curriculum-coppeliasim-roadmap-design.md`
- 本计划只覆盖 V2.2-A 和 V2.2-B。
- 实施分支建议命名：`codex/robot-curriculum-v2-2-first-batch`。
- 必须从 V2.1 经审核的最终提交创建实施分支，不能直接从本规划分支或当前未完成的 V2.1 中间提交开始。
- 当前编写计划时，V2.1 开发分支停在 `23a0503`，仅完成原实施方案 Task 1 至 Task 5；因此 Task 1 把 V2.1 完成条件设为硬门禁。
- 不修改或覆盖：
  - `simulation/vision_lab/BL23_vision_lab.ttt`
  - `robot_backends/models/BLX_openr6.ttt`
  - `robot_backends/models/openr6_arm_coppeliasim.urdf`
  - `robot_backends/models/meshes_blx/*`
  - `simulation/vision_lab/assets/robot/*`
- 两个新 `.ttt` 是独立正式交付物，必须有独立规格、清单、静态检查和真实 CoppeliaSim 在线检查。
- V2.2 不开放真实机器人后端，不生成正式成绩，不把场景探针结果称为自动评分。
- `hardware_status` 在全部软件、证据和报告中固定为 `PENDING_HARDWARE`。
- 新增正式文件逐项登记到 `RETAINED_FILES.txt`；运行生成的 `artifacts/`、缓存和学生个人文件不得登记。
- 基础环境不增加新的第三方依赖；使用现有 NumPy、OpenCV、PyQt5 和 CoppeliaSim 客户端。

## 1. 文件结构

### 1.1 实验框架

| 文件 | 单一职责 |
|---|---|
| `vision_platform/experiments/models.py` | 实验、场景、能力和运行上下文的不可变数据模型 |
| `vision_platform/experiments/catalog.py` | 读取并严格校验实验 JSON，解析项目内路径 |
| `vision_platform/experiments/capabilities.py` | 根据应用和后端状态检查实验能力 |
| `vision_platform/experiments/session.py` | 在学生运行停止后切换实验、应用和场景 |
| `vision_platform/experiments/probes.py` | 读取场景状态并生成非评分式终态证据 |
| `config/experiments/catalog.json` | 正式实验索引和显示顺序 |
| `config/experiments/R1-*.json` | 五个实验的独立正式清单 |

### 1.2 学生 SDK 与证据

| 文件 | 责任 |
|---|---|
| `vision_platform/student/protocol.py` | 增加 `camera.capture` 和 `experiment.info` 白名单命令 |
| `vision_platform/student/sdk.py` | 提供 `StudentCamera.capture()`、`StudentFrame` 和只读实验信息 |
| `vision_platform/student/runner.py` | 在主进程分派两条新命令，不改变运动安全路径 |
| `vision_platform/student/evidence.py` | 保存实验元数据和相机快照 |

PNG 字节仅用于父子进程传输。学生侧返回独立的 NumPy 数组；主进程证据只记录快照路径、哈希和元数据，不把二进制内容写入 JSONL。

### 1.3 场景

| 文件 | 责任 |
|---|---|
| `simulation/training_scenes/build_scene.py` | 从只读正式场景模板构建新场景 |
| `simulation/training_scenes/scene_contract.py` | 静态规格和清单一致性检查 |
| `simulation/training_scenes/generate_labels.py` | 生成原创数字和构件类别纹理 |
| `simulation/robot_basics/scene_spec.json` | 机器人基础场景声明 |
| `simulation/robot_basics/scene_manifest.json` | 构建产物、哈希和对象路径 |
| `simulation/robot_basics/BL23_robot_basics.ttt` | R1-01、R1-02 正式场景 |
| `simulation/logistics_lab/scene_spec.json` | 物流场景、物体、仓位和堆垛槽声明 |
| `simulation/logistics_lab/scene_manifest.json` | 构建产物、哈希和对象路径 |
| `simulation/logistics_lab/BL23_logistics_lab.ttt` | R1-05、R1-06、R1-07 正式场景 |
| `simulation/logistics_lab/assets/labels/*.png` | 脚本生成的原创纹理 |
| `simulation/logistics_lab/assets/labels/manifest.json` | 纹理生成方式和哈希 |

### 1.4 教学入口

| 文件 | 责任 |
|---|---|
| `student_programs/templates/r1_*.py` | 五个可运行参考模板 |
| `docs/experiments/R1-*.md` | 五份教学说明 |
| `vision_platform/ui/experiment_catalog_panel.py` | 实验选择、说明、能力和硬件状态显示 |
| `vision_platform/cli.py` | `experiment-list`、`experiment-show`、`experiment-run` |
| `tools/vision_lab/run_experiment.ps1` | 定位仓库并调用统一 CLI |

### Task 1: 固化 V2.1 完成基线并创建实施分支

**Files:**
- Read: `docs/superpowers/plans/2026-07-30-student-program-runner-v2-plan.md`
- Read: `vision_platform/student/runner.py`
- Read: `vision_platform/student/evidence.py`
- Read: `vision_platform/session.py`
- Read: `vision_platform/ui/student_program_panel.py`
- Modify: `docs/superpowers/specs/2026-07-31-robot-curriculum-coppeliasim-roadmap-design.md`
- Modify: `RETAINED_FILES.txt`

**执行记录：** 开发端执行 Prompt 覆盖本计划中的建议分支名和工作树名；实际使用分支 `codex/v2-2-first-batch-integration`、工作树 `v2-2-first-batch-integration`，并从精确基线 `4bc638f50fd590ca700615da46741a86c4146b5f` 创建。

- [x] **Step 1: 确认 V2.1 不是中间提交**

Run:

```powershell
$project = 'C:\Users\yqzhe\Documents\Robot Sim\Blinx-CoppeliaSim-clean'
git -C $project status --short --branch
git -C $project log -1 --oneline
$required = @(
  'vision_platform/student/runner.py',
  'vision_platform/student/evidence.py',
  'vision_platform/session.py',
  'vision_platform/ui/student_program_panel.py',
  'tools/vision_lab/run_student_program.ps1',
  'tests/test_acceptance/test_coppeliasim_student_program.py'
)
$required | ForEach-Object {
  if (-not (Test-Path -LiteralPath (Join-Path $project $_) -PathType Leaf) {
    throw "V2.1 未完成，缺少 $_"
  }
}
```

Expected: 工作树干净，六个 V2.1 交付文件全部存在。任一文件缺失就停止本计划，不在 V2.1 中间接口上实施 V2.2。

- [x] **Step 2: 运行 V2.1 静态和在线完成门禁**

Run:

```powershell
$python = 'C:\Users\yqzhe\Documents\Robot Sim\Blinx-CoppeliaSim-clean\.venv-vision\Scripts\python.exe'
& $python -m pytest -q
& $python -m pytest -m coppeliasim tests/test_acceptance/test_coppeliasim_student_program.py -q
```

Expected:

- 静态套件零失败；两个显式在线测试可以在静态套件中保持 skip；
- `test_coppeliasim_student_program.py` 在已启动 CoppeliaSim 环境中实际 PASS，不能是 skip；
- 若在线门禁未 PASS，V2.1 仍不满足本计划前提。

- [x] **Step 3: 创建独立实施工作树**

Run:

```powershell
$project = 'C:\Users\yqzhe\Documents\Robot Sim\Blinx-CoppeliaSim-clean'
$worktree = 'C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-first-batch'
$branch = 'codex/robot-curriculum-v2-2-first-batch'
$baseline = git -C $project rev-parse HEAD
git -C $project worktree add $worktree -b $branch $baseline
git -C $worktree status --short --branch
git -C $worktree rev-parse HEAD
```

Expected: 新工作树在 V2.1 最终提交上，分支为 `codex/robot-curriculum-v2-2-first-batch`，状态干净。

- [x] **Step 4: 固化集成基线记录**

在总体规格的“当前开发前置”下增加实际 V2.1 最终提交：

```markdown
**V2.1 集成基线：** `<Task 1 Step 3 输出的完整提交哈希>`
```

同时确认 `RETAINED_FILES.txt` 已保留本实施方案条目：

```text
docs/superpowers/plans/2026-07-31-robot-curriculum-v2-2-first-batch-plan.md
```

- [x] **Step 5: 提交基线记录**

Run:

```powershell
git add `
  docs/superpowers/specs/2026-07-31-robot-curriculum-coppeliasim-roadmap-design.md `
  docs/superpowers/plans/2026-07-31-robot-curriculum-v2-2-first-batch-plan.md `
  RETAINED_FILES.txt
git commit -m "docs: start first robot curriculum expansion"
```

Expected: 提交只包含已确认规格、实施方案和发布白名单记录。

### Task 2: 定义实验数据模型和严格 JSON 装载

**Files:**
- Create: `vision_platform/experiments/__init__.py`
- Create: `vision_platform/experiments/models.py`
- Create: `vision_platform/experiments/catalog.py`
- Create: `tests/test_experiments/__init__.py`
- Create: `tests/test_experiments/test_catalog.py`

- [x] **Step 1: 写目录装载失败测试**

Create `tests/test_experiments/test_catalog.py`:

```python
from __future__ import annotations

import json

import pytest

from vision_platform.experiments.catalog import ExperimentCatalog


def _write_experiment(root, *, experiment_id="R1-01", scene="scene.ttt"):
    experiments = root / "config" / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    guide = root / "docs" / "experiments" / f"{experiment_id}.md"
    guide.parent.mkdir(parents=True, exist_ok=True)
    guide.write_text("# guide\n", encoding="utf-8")
    template = root / "student_programs" / "templates" / f"{experiment_id}.py"
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("def main(ctx):\n    ctx.robot.home()\n", encoding="utf-8")
    scene_path = root / scene
    scene_path.write_bytes(b"scene")
    payload = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "pack_id": "R1",
        "title": "机械臂认知和基础操作",
        "version": "2.2.0",
        "scene": scene,
        "scene_manifest": "scene_manifest.json",
        "student_template": template.relative_to(root).as_posix(),
        "guide": guide.relative_to(root).as_posix(),
        "capabilities": ["robot.home", "robot.pose"],
        "workspace": {
            "x_mm": [20, 140],
            "y_mm": [-90, 90],
            "z_mm": [10, 140],
            "safe_z_mm": 100,
        },
        "public_parameters": {"observation_pose_mm": [100, 0, 120]},
        "acceptance": {
            "probe_kind": "motion_observation",
            "automated_checks": ["robot_paths", "command_trace"],
            "human_checks": ["学生能够解释六个关节"],
        },
        "hardware_status": "PENDING_HARDWARE",
    }
    path = experiments / f"{experiment_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (root / "scene_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scene": {"path": scene, "sha256": "0" * 64},
                "required_paths": ["/BLX_base_link"],
            }
        ),
        encoding="utf-8",
    )
    (experiments / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiments": [f"{experiment_id}.json"],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_catalog_loads_and_resolves_project_paths(tmp_path):
    _write_experiment(tmp_path)

    catalog = ExperimentCatalog.load(
        tmp_path / "config" / "experiments" / "catalog.json",
        project_root=tmp_path,
    )
    experiment = catalog.require("R1-01")

    assert experiment.experiment_id == "R1-01"
    assert experiment.scene == (tmp_path / "scene.ttt").resolve()
    assert experiment.student_template.name == "R1-01.py"
    assert experiment.hardware_status == "PENDING_HARDWARE"
    assert catalog.ids == ("R1-01",)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("experiment_id", "r1-01", "experiment_id"),
        ("hardware_status", "PASS", "PENDING_HARDWARE"),
        ("capabilities", ["robot.home", "robot.home"], "duplicate"),
    ],
)
def test_catalog_rejects_invalid_contract(tmp_path, field, value, message):
    path = _write_experiment(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        ExperimentCatalog.load(
            tmp_path / "config" / "experiments" / "catalog.json",
            project_root=tmp_path,
        )


def test_catalog_rejects_paths_outside_project(tmp_path):
    path = _write_experiment(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scene"] = "../outside.ttt"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="inside project"):
        ExperimentCatalog.load(
            tmp_path / "config" / "experiments" / "catalog.json",
            project_root=tmp_path,
        )
```

- [x] **Step 2: 运行测试并确认先失败**

Run:

```powershell
python -m pytest tests/test_experiments/test_catalog.py -q
```

Expected: FAIL with `ModuleNotFoundError: vision_platform.experiments`。

- [x] **Step 3: 实现不可变模型**

Create `vision_platform/experiments/models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ExperimentAcceptance:
    probe_kind: str
    automated_checks: tuple[str, ...]
    human_checks: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentDefinition:
    experiment_id: str
    pack_id: str
    title: str
    version: str
    scene: Path
    scene_manifest: Path
    student_template: Path
    guide: Path
    capabilities: tuple[str, ...]
    workspace: Mapping[str, Any]
    public_parameters: Mapping[str, Any]
    acceptance: ExperimentAcceptance
    hardware_status: str


@dataclass(frozen=True)
class CapabilityReport:
    available: tuple[str, ...]
    missing: tuple[str, ...]
    reasons: Mapping[str, str] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return not self.missing


@dataclass(frozen=True)
class ExperimentRunContext:
    experiment_id: str
    experiment_version: str
    scene_path: Path
    scene_sha256: str
    scene_manifest_path: Path
    public_parameters: Mapping[str, Any]
    hardware_status: str = "PENDING_HARDWARE"

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_version": self.experiment_version,
            "scene_sha256": self.scene_sha256,
            "public_parameters": dict(self.public_parameters),
            "hardware_status": self.hardware_status,
        }
```

- [x] **Step 4: 实现严格目录装载**

Create `vision_platform/experiments/catalog.py`:

```python
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from vision_platform.experiments.models import (
    ExperimentAcceptance,
    ExperimentDefinition,
)


_EXPERIMENT_ID = re.compile(r"^[A-Z][A-Z0-9]*-[0-9]{2}$")
_REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "pack_id",
    "title",
    "version",
    "scene",
    "scene_manifest",
    "student_template",
    "guide",
    "capabilities",
    "workspace",
    "public_parameters",
    "acceptance",
    "hardware_status",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _inside(root: Path, raw: str, field: str, *, must_exist: bool = True) -> Path:
    path = (root / raw).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"{field} must stay inside project")
    if must_exist and not path.is_file():
        raise ValueError(f"{field} does not exist: {raw}")
    return path


def _strings(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{field} must be a non-empty string list")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} contains duplicate values")
    return result


def _definition(path: Path, root: Path) -> ExperimentDefinition:
    payload = _load_json(path)
    missing = sorted(_REQUIRED_FIELDS - payload.keys())
    extra = sorted(payload.keys() - _REQUIRED_FIELDS)
    if missing or extra:
        raise ValueError(f"experiment fields mismatch: missing={missing}, extra={extra}")
    if payload["schema_version"] != 1:
        raise ValueError("experiment schema_version must be 1")
    experiment_id = str(payload["experiment_id"])
    if not _EXPERIMENT_ID.fullmatch(experiment_id):
        raise ValueError(f"invalid experiment_id: {experiment_id}")
    if payload["hardware_status"] != "PENDING_HARDWARE":
        raise ValueError("hardware_status must remain PENDING_HARDWARE")
    capabilities = _strings(payload["capabilities"], "capabilities")
    acceptance_payload = payload["acceptance"]
    if not isinstance(acceptance_payload, dict):
        raise ValueError("acceptance must be an object")
    acceptance = ExperimentAcceptance(
        probe_kind=str(acceptance_payload["probe_kind"]),
        automated_checks=_strings(
            acceptance_payload["automated_checks"],
            "acceptance.automated_checks",
        ),
        human_checks=_strings(
            acceptance_payload["human_checks"],
            "acceptance.human_checks",
        ),
    )
    return ExperimentDefinition(
        experiment_id=experiment_id,
        pack_id=str(payload["pack_id"]),
        title=str(payload["title"]),
        version=str(payload["version"]),
        scene=_inside(root, str(payload["scene"]), "scene"),
        scene_manifest=_inside(
            root,
            str(payload["scene_manifest"]),
            "scene_manifest",
        ),
        student_template=_inside(
            root,
            str(payload["student_template"]),
            "student_template",
        ),
        guide=_inside(root, str(payload["guide"]), "guide"),
        capabilities=capabilities,
        workspace=dict(payload["workspace"]),
        public_parameters=dict(payload["public_parameters"]),
        acceptance=acceptance,
        hardware_status="PENDING_HARDWARE",
    )


class ExperimentCatalog:
    def __init__(self, definitions: tuple[ExperimentDefinition, ...]) -> None:
        by_id = {item.experiment_id: item for item in definitions}
        if len(by_id) != len(definitions):
            raise ValueError("catalog contains duplicate experiment_id")
        self._definitions = definitions
        self._by_id = by_id

    @classmethod
    def load(cls, path: str | Path, *, project_root: str | Path) -> "ExperimentCatalog":
        selected = Path(path).expanduser().resolve()
        root = Path(project_root).expanduser().resolve()
        payload = _load_json(selected)
        if payload.get("schema_version") != 1:
            raise ValueError("catalog schema_version must be 1")
        files = _strings(payload.get("experiments"), "experiments")
        definitions = tuple(
            _definition(_inside(selected.parent, name, "experiment"), root)
            for name in files
        )
        return cls(definitions)

    @property
    def definitions(self) -> tuple[ExperimentDefinition, ...]:
        return self._definitions

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(item.experiment_id for item in self._definitions)

    def require(self, experiment_id: str) -> ExperimentDefinition:
        try:
            return self._by_id[experiment_id]
        except KeyError as error:
            raise KeyError(f"Unknown experiment: {experiment_id}") from error
```

Create `vision_platform/experiments/__init__.py`:

```python
from vision_platform.experiments.catalog import ExperimentCatalog
from vision_platform.experiments.models import ExperimentDefinition

__all__ = ["ExperimentCatalog", "ExperimentDefinition"]
```

- [x] **Step 5: 运行测试并确认通过**

Run:

```powershell
python -m pytest tests/test_experiments/test_catalog.py -q
```

Expected: `5 passed`。

- [x] **Step 6: 提交实验模型**

Run:

```powershell
git add vision_platform/experiments tests/test_experiments
git commit -m "feat(experiments): add strict experiment catalog"
```

### Task 3: 建立五个正式实验清单

**Files:**
- Create: `config/experiments/catalog.json`
- Create: `config/experiments/R1-01.json`
- Create: `config/experiments/R1-02.json`
- Create: `config/experiments/R1-05.json`
- Create: `config/experiments/R1-06.json`
- Create: `config/experiments/R1-07.json`
- Create: `tests/test_experiments/test_formal_catalog.py`

- [x] **Step 1: 写正式目录合同测试**

Create `tests/test_experiments/test_formal_catalog.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config" / "experiments"


def _payload(experiment_id):
    return json.loads(
        (CONFIG_DIR / f"{experiment_id}.json").read_text(encoding="utf-8")
    )


def test_first_batch_catalog_has_exact_order_and_hardware_boundary():
    catalog = json.loads(
        (CONFIG_DIR / "catalog.json").read_text(encoding="utf-8")
    )
    names = catalog["experiments"]
    payloads = [_payload(Path(name).stem) for name in names]

    assert names == [
        "R1-01.json",
        "R1-02.json",
        "R1-05.json",
        "R1-06.json",
        "R1-07.json",
    ]
    assert all(
        item["hardware_status"] == "PENDING_HARDWARE" for item in payloads
    )


def test_first_batch_experiments_use_two_independent_scenes():
    scenes = {
        experiment_id: _payload(experiment_id)["scene"]
        for experiment_id in ("R1-01", "R1-02", "R1-05", "R1-06", "R1-07")
    }

    assert scenes["R1-01"] == "simulation/robot_basics/BL23_robot_basics.ttt"
    assert scenes["R1-02"] == scenes["R1-01"]
    assert scenes["R1-05"] == "simulation/logistics_lab/BL23_logistics_lab.ttt"
    assert scenes["R1-06"] == scenes["R1-05"]
    assert scenes["R1-07"] == scenes["R1-05"]
    assert all("BL23_vision_lab.ttt" not in value for value in scenes.values())


def test_each_experiment_declares_automated_and_human_checks():
    for experiment_id in ("R1-01", "R1-02", "R1-05", "R1-06", "R1-07"):
        experiment = _payload(experiment_id)
        assert experiment["acceptance"]["automated_checks"]
        assert experiment["acceptance"]["human_checks"]
        assert "camera.rgb" in experiment["capabilities"] or experiment_id in {
            "R1-01",
            "R1-02",
        }
```

- [x] **Step 2: 运行正式目录测试并确认先失败**

Run:

```powershell
python -m pytest tests/test_experiments/test_formal_catalog.py -q
```

Expected: FAIL，指出 `config/experiments/catalog.json` 不存在。

- [x] **Step 3: 创建正式目录索引**

Create `config/experiments/catalog.json`:

```json
{
  "schema_version": 1,
  "experiments": [
    "R1-01.json",
    "R1-02.json",
    "R1-05.json",
    "R1-06.json",
    "R1-07.json"
  ]
}
```

- [x] **Step 4: 创建 R1-01 和 R1-02 清单**

Create `config/experiments/R1-01.json`:

```json
{
  "schema_version": 1,
  "experiment_id": "R1-01",
  "pack_id": "R1",
  "title": "机械臂认知和基础操作",
  "version": "2.2.0",
  "scene": "simulation/robot_basics/BL23_robot_basics.ttt",
  "scene_manifest": "simulation/robot_basics/scene_manifest.json",
  "student_template": "student_programs/templates/r1_01_robot_basics.py",
  "guide": "docs/experiments/R1-01.md",
  "capabilities": ["robot.home", "robot.pose", "robot.move_world", "experiment.info", "scene.probe"],
  "workspace": {
    "x_mm": [20, 140],
    "y_mm": [-90, 90],
    "z_mm": [10, 140],
    "safe_z_mm": 100
  },
  "public_parameters": {
    "camera_path": "/RobotBasics/Camera",
    "observation_pose_mm": [100, 0, 120],
    "joint_paths": ["/BLX_joint1", "/BLX_joint2", "/BLX_joint3", "/BLX_joint4", "/BLX_joint5", "/BLX_joint6"]
  },
  "acceptance": {
    "probe_kind": "motion_observation",
    "automated_checks": ["required_paths", "home_and_pose_commands", "safe_cleanup"],
    "human_checks": ["学生能够指出六个关节和 TCP", "学生能够解释仿真与真机差异"]
  },
  "hardware_status": "PENDING_HARDWARE"
}
```

Create `config/experiments/R1-02.json`:

```json
{
  "schema_version": 1,
  "experiment_id": "R1-02",
  "pack_id": "R1",
  "title": "机械臂示教和运动控制",
  "version": "2.2.0",
  "scene": "simulation/robot_basics/BL23_robot_basics.ttt",
  "scene_manifest": "simulation/robot_basics/scene_manifest.json",
  "student_template": "student_programs/templates/r1_02_teach_points.py",
  "guide": "docs/experiments/R1-02.md",
  "capabilities": ["robot.home", "robot.pose", "robot.move_world", "experiment.info", "scene.probe"],
  "workspace": {
    "x_mm": [20, 140],
    "y_mm": [-90, 90],
    "z_mm": [10, 140],
    "safe_z_mm": 100
  },
  "public_parameters": {
    "camera_path": "/RobotBasics/Camera",
    "teach_points_mm": [[60, -50, 120], [110, -50, 120], [110, 50, 120], [60, 50, 120]]
  },
  "acceptance": {
    "probe_kind": "motion_observation",
    "automated_checks": ["required_paths", "four_teach_points", "speed_policy", "safe_cleanup"],
    "human_checks": ["学生能够解释世界坐标和安全高度", "学生能够使用暂停和单步观察运动"]
  },
  "hardware_status": "PENDING_HARDWARE"
}
```

- [x] **Step 5: 创建 R1-05、R1-06 和 R1-07 清单**

Create `config/experiments/R1-05.json`:

```json
{
  "schema_version": 1,
  "experiment_id": "R1-05",
  "pack_id": "R1",
  "title": "基于视觉的物体码垛",
  "version": "2.2.0",
  "scene": "simulation/logistics_lab/BL23_logistics_lab.ttt",
  "scene_manifest": "simulation/logistics_lab/scene_manifest.json",
  "student_template": "student_programs/templates/r1_05_visual_stacking.py",
  "guide": "docs/experiments/R1-05.md",
  "capabilities": ["robot.home", "robot.pose", "robot.move_world", "tool.suction", "camera.rgb", "experiment.info", "scene.probe"],
  "workspace": {
    "x_mm": [20, 140],
    "y_mm": [-90, 90],
    "z_mm": [10, 140],
    "safe_z_mm": 100
  },
  "public_parameters": {
    "camera_path": "/LogisticsLab/Camera",
    "scene_group_path": "/LogisticsLab/Tasks/Stack",
    "pickables_path": "/LogisticsLab/Tasks/Stack/Pickables",
    "calibration_matrix": [[0.203125, 0.0, 35.0], [0.0, -0.2916666667, 70.0]],
    "pick_region_world_mm": {"x_max": 108},
    "pick_z_mm": 20,
    "stack_slots_mm": [[118, -45, 20], [118, 45, 20], [118, -45, 38], [118, 45, 38], [118, -45, 56], [118, 45, 56]],
    "safe_z_mm": 100
  },
  "acceptance": {
    "probe_kind": "stack_2x3",
    "automated_checks": ["six_objects", "two_columns_three_layers", "reset_positions", "safe_cleanup"],
    "human_checks": ["学生能够解释像素到世界坐标转换", "学生能够解释分层码垛顺序"]
  },
  "hardware_status": "PENDING_HARDWARE"
}
```

Create `config/experiments/R1-06.json`:

```json
{
  "schema_version": 1,
  "experiment_id": "R1-06",
  "pack_id": "R1",
  "title": "基于视觉的数字排序",
  "version": "2.2.0",
  "scene": "simulation/logistics_lab/BL23_logistics_lab.ttt",
  "scene_manifest": "simulation/logistics_lab/scene_manifest.json",
  "student_template": "student_programs/templates/r1_06_digit_sort.py",
  "guide": "docs/experiments/R1-06.md",
  "capabilities": ["robot.home", "robot.pose", "robot.move_world", "tool.suction", "camera.rgb", "experiment.info", "scene.probe"],
  "workspace": {
    "x_mm": [20, 140],
    "y_mm": [-90, 90],
    "z_mm": [10, 140],
    "safe_z_mm": 100
  },
  "public_parameters": {
    "camera_path": "/LogisticsLab/Camera",
    "scene_group_path": "/LogisticsLab/Tasks/Digits",
    "pickables_path": "/LogisticsLab/Tasks/Digits/Pickables",
    "calibration_matrix": [[0.203125, 0.0, 35.0], [0.0, -0.2916666667, 70.0]],
    "pick_region_world_mm": {"x_max": 108},
    "digit_reference_dir": "simulation/logistics_lab/assets/labels/digits",
    "order": [1, 2, 3],
    "reverse_order": [3, 2, 1],
    "drop_slots_mm": [[118, -55, 20], [118, 0, 20], [118, 55, 20]],
    "pick_z_mm": 20,
    "safe_z_mm": 100
  },
  "acceptance": {
    "probe_kind": "ordered_slots",
    "automated_checks": ["digit_objects", "ascending_order", "descending_mode", "safe_cleanup"],
    "human_checks": ["学生能够说明模板匹配结果", "学生能够切换升序和倒序规则"]
  },
  "hardware_status": "PENDING_HARDWARE"
}
```

Create `config/experiments/R1-07.json`:

```json
{
  "schema_version": 1,
  "experiment_id": "R1-07",
  "pack_id": "R1",
  "title": "基于视觉的施工构件与物流目标分类",
  "version": "2.2.0",
  "scene": "simulation/logistics_lab/BL23_logistics_lab.ttt",
  "scene_manifest": "simulation/logistics_lab/scene_manifest.json",
  "student_template": "student_programs/templates/r1_07_component_sort.py",
  "guide": "docs/experiments/R1-07.md",
  "capabilities": ["robot.home", "robot.pose", "robot.move_world", "tool.suction", "camera.rgb", "experiment.info", "scene.probe"],
  "workspace": {
    "x_mm": [20, 140],
    "y_mm": [-90, 90],
    "z_mm": [10, 140],
    "safe_z_mm": 100
  },
  "public_parameters": {
    "camera_path": "/LogisticsLab/Camera",
    "scene_group_path": "/LogisticsLab/Tasks/Classes",
    "pickables_path": "/LogisticsLab/Tasks/Classes/Pickables",
    "calibration_matrix": [[0.203125, 0.0, 35.0], [0.0, -0.2916666667, 70.0]],
    "pick_region_world_mm": {"x_max": 108},
    "classes": ["red_block", "blue_block", "green_cylinder", "yellow_cylinder"],
    "drop_poses_mm": {
      "red_block": [118, -60, 20],
      "blue_block": [118, -20, 20],
      "green_cylinder": [118, 20, 20],
      "yellow_cylinder": [118, 60, 20]
    },
    "pick_z_mm": 20,
    "safe_z_mm": 100
  },
  "acceptance": {
    "probe_kind": "class_zones",
    "automated_checks": ["four_classes", "configured_routes", "reset_positions", "safe_cleanup"],
    "human_checks": ["学生能够解释颜色和形状特征", "学生能够说明误识别时为何不应自动抓取"]
  },
  "hardware_status": "PENDING_HARDWARE"
}
```

- [x] **Step 6: 运行正式目录测试**

Run:

```powershell
python -m pytest tests/test_experiments/test_catalog.py tests/test_experiments/test_formal_catalog.py -q
```

Expected: `8 passed`。场景、模板和说明路径将在对应后续任务创建并由最终发布合同解析。

- [x] **Step 7: 只提交目录清单和测试**

Run:

```powershell
git add config/experiments tests/test_experiments/test_formal_catalog.py
git commit -m "feat(experiments): declare first five robot labs"
```

Expected: 提交只包含正式目录 JSON 和目录合同测试；引用的场景、模板和说明在后续任务创建。

### Task 4: 实现实验能力检查

**Files:**
- Create: `vision_platform/experiments/capabilities.py`
- Create: `tests/test_experiments/test_capabilities.py`

- [x] **Step 1: 写能力检查失败测试**

Create `tests/test_experiments/test_capabilities.py`:

```python
from types import SimpleNamespace

from vision_platform.experiments.capabilities import check_capabilities


def _application(*, camera=True, backend="sim", tool=True, sim=True):
    return SimpleNamespace(
        camera=object() if camera else None,
        robot=object(),
        tool=object() if tool else None,
        sim=object() if sim else None,
        config=SimpleNamespace(robot_backend=backend),
    )


def test_sim_application_satisfies_first_batch_capabilities():
    report = check_capabilities(
        _application(),
        (
            "robot.home",
            "robot.pose",
            "robot.move_world",
            "tool.suction",
            "camera.rgb",
            "experiment.info",
            "scene.probe",
        ),
    )

    assert report.ready
    assert report.missing == ()


def test_real_backend_is_rejected_even_when_objects_exist():
    report = check_capabilities(
        _application(backend="real"),
        ("robot.home", "camera.rgb"),
    )

    assert not report.ready
    assert "robot.home" in report.missing
    assert report.reasons["robot.home"] == "V2.2 禁止真实机器人后端"


def test_missing_camera_has_stable_reason():
    report = check_capabilities(
        _application(camera=False),
        ("camera.rgb",),
    )

    assert report.missing == ("camera.rgb",)
    assert report.reasons["camera.rgb"] == "当前应用没有可用相机"


def test_unknown_capability_is_not_silently_accepted():
    report = check_capabilities(_application(), ("robot.fly",))

    assert report.missing == ("robot.fly",)
    assert "未注册能力" in report.reasons["robot.fly"]
```

- [x] **Step 2: 运行测试并确认先失败**

Run:

```powershell
python -m pytest tests/test_experiments/test_capabilities.py -q
```

Expected: FAIL with `ModuleNotFoundError: vision_platform.experiments.capabilities`。

- [x] **Step 3: 实现固定能力注册表**

Create `vision_platform/experiments/capabilities.py`:

```python
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from vision_platform.experiments.models import CapabilityReport


_ROBOT = frozenset({"robot.home", "robot.pose", "robot.move_world"})
_KNOWN = _ROBOT | frozenset(
    {
        "tool.suction",
        "camera.rgb",
        "experiment.info",
        "scene.probe",
    }
)


def check_capabilities(
    application: Any,
    required: Iterable[str],
) -> CapabilityReport:
    available: list[str] = []
    missing: list[str] = []
    reasons: dict[str, str] = {}
    backend = str(getattr(application.config, "robot_backend", ""))

    for capability in required:
        if capability not in _KNOWN:
            missing.append(capability)
            reasons[capability] = f"未注册能力：{capability}"
        elif capability in _ROBOT and backend != "sim":
            missing.append(capability)
            reasons[capability] = "V2.2 禁止真实机器人后端"
        elif capability in _ROBOT and getattr(application, "robot", None) is None:
            missing.append(capability)
            reasons[capability] = "当前应用没有可用机械臂"
        elif capability == "tool.suction" and getattr(application, "tool", None) is None:
            missing.append(capability)
            reasons[capability] = "当前应用没有可用吸盘"
        elif capability == "camera.rgb" and getattr(application, "camera", None) is None:
            missing.append(capability)
            reasons[capability] = "当前应用没有可用相机"
        elif capability == "scene.probe" and getattr(application, "sim", None) is None:
            missing.append(capability)
            reasons[capability] = "当前应用没有 CoppeliaSim 场景连接"
        else:
            available.append(capability)

    return CapabilityReport(
        available=tuple(available),
        missing=tuple(missing),
        reasons=reasons,
    )
```

- [x] **Step 4: 运行测试并提交**

Run:

```powershell
python -m pytest tests/test_experiments/test_capabilities.py -q
git add vision_platform/experiments/capabilities.py tests/test_experiments/test_capabilities.py
git commit -m "feat(experiments): check runtime capabilities"
```

Expected: `4 passed`，提交成功。

### Task 5: 实现安全的实验切换会话

**Files:**
- Modify: `vision_platform/session.py`
- Create: `vision_platform/experiments/session.py`
- Create: `tests/test_experiments/test_session.py`
- Modify: `tests/test_vision_platform/test_session.py`

- [x] **Step 1: 写应用替换和运行中拒绝测试**

Create `tests/test_experiments/test_session.py`:

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vision_platform.experiments.catalog import ExperimentCatalog
from vision_platform.experiments.session import ExperimentSession
from vision_platform.session import VisionLabSession


class FakeApplication:
    def __init__(self, config):
        self.config = config
        self.camera = object()
        self.robot = object()
        self.tool = object()
        self.sim = object()
        self.loaded = []
        self.open_calls = 0
        self.close_calls = 0

    def load_and_start_scene(self, path):
        self.loaded.append(path)

    def open(self):
        self.open_calls += 1

    def close(self):
        self.close_calls += 1


def _catalog(tmp_path):
    scene = tmp_path / "scene.ttt"
    scene.write_bytes(b"formal-scene")
    manifest = tmp_path / "scene_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scene": {
                    "path": "scene.ttt",
                    "sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
                },
                "required_paths": ["/BLX_base_link"],
            }
        ),
        encoding="utf-8",
    )
    guide = tmp_path / "guide.md"
    guide.write_text("# guide\n", encoding="utf-8")
    template = tmp_path / "template.py"
    template.write_text("def main(ctx):\n    ctx.robot.home()\n", encoding="utf-8")
    definition = {
        "schema_version": 1,
        "experiment_id": "R1-01",
        "pack_id": "R1",
        "title": "机械臂基础",
        "version": "2.2.0",
        "scene": "scene.ttt",
        "scene_manifest": "scene_manifest.json",
        "student_template": "template.py",
        "guide": "guide.md",
        "capabilities": ["robot.home", "camera.rgb", "scene.probe"],
        "workspace": {
            "x_mm": [20, 140],
            "y_mm": [-90, 90],
            "z_mm": [10, 140],
            "safe_z_mm": 100,
        },
        "public_parameters": {"safe_z_mm": 100},
        "acceptance": {
            "probe_kind": "motion_observation",
            "automated_checks": ["required_paths"],
            "human_checks": ["观察机器人"],
        },
        "hardware_status": "PENDING_HARDWARE",
    }
    experiments = tmp_path / "config" / "experiments"
    experiments.mkdir(parents=True)
    (experiments / "R1-01.json").write_text(
        json.dumps(definition, ensure_ascii=False),
        encoding="utf-8",
    )
    (experiments / "catalog.json").write_text(
        json.dumps(
            {"schema_version": 1, "experiments": ["R1-01.json"]}
        ),
        encoding="utf-8",
    )
    return ExperimentCatalog.load(
        experiments / "catalog.json",
        project_root=tmp_path,
    )


def _base_config():
    return SimpleNamespace(
        camera_backend="replay",
        robot_backend="sim",
        coppelia_scene=None,
        camera_options={"sim": {"sensor_path": "/VisionLab/Camera"}},
        task={"pickables_path": "/VisionLab/Pickables"},
        workspace=None,
    )


def test_select_replaces_application_and_returns_hashed_context(tmp_path):
    first = FakeApplication(_base_config())
    vision_session = VisionLabSession(
        application=first,
        factory=lambda: FakeApplication(_base_config()),
    )
    created = []

    def factory(config):
        app = FakeApplication(config)
        created.append(app)
        return app

    session = ExperimentSession(
        catalog=_catalog(tmp_path),
        vision_session=vision_session,
        base_config=_base_config(),
        application_factory=factory,
        student_is_idle=lambda: True,
    )

    context = session.select("R1-01")

    assert first.close_calls == 1
    assert created[0].loaded == [context.scene_path]
    assert created[0].open_calls == 1
    assert created[0].config.camera_backend == "sim"
    assert created[0].config.robot_backend == "sim"
    assert context.experiment_id == "R1-01"
    assert context.scene_sha256 == hashlib.sha256(b"formal-scene").hexdigest()
    assert context.hardware_status == "PENDING_HARDWARE"
    assert vision_session.application is created[0]


def test_select_refuses_to_switch_while_student_is_active(tmp_path):
    first = FakeApplication(_base_config())
    vision_session = VisionLabSession(
        application=first,
        factory=lambda: FakeApplication(_base_config()),
    )
    session = ExperimentSession(
        catalog=_catalog(tmp_path),
        vision_session=vision_session,
        base_config=_base_config(),
        application_factory=FakeApplication,
        student_is_idle=lambda: False,
    )

    with pytest.raises(RuntimeError, match="学生程序"):
        session.select("R1-01")

    assert first.close_calls == 0
```

- [x] **Step 2: 扩展 V2.1 会话替换测试**

Append to `tests/test_vision_platform/test_session.py`:

```python
def test_session_replace_closes_old_then_loads_and_opens_new():
    events = []

    class OrderedApplication(FakeApplication):
        def close(self):
            super().close()
            events.append(f"close-{self.number}")

        def load_and_start_scene(self, path):
            self.load_calls += 1
            events.append(f"load-{self.number}-{path}")

        def open(self):
            self.open_calls += 1
            events.append(f"open-{self.number}")

    old = OrderedApplication(1)
    session = VisionLabSession(application=old, factory=lambda: old)
    replacement = OrderedApplication(2)

    result = session.replace_application(
        lambda: replacement,
        scene_path="new-scene.ttt",
    )

    assert result is replacement
    assert session.application is replacement
    assert events == [
        "close-1",
        "load-2-new-scene.ttt",
        "open-2",
    ]
```

- [x] **Step 3: 运行测试并确认先失败**

Run:

```powershell
python -m pytest tests/test_experiments/test_session.py tests/test_vision_platform/test_session.py -q
```

Expected: FAIL，指出 `ExperimentSession` 或 `replace_application` 不存在。

- [x] **Step 4: 给 V2.1 会话增加有界替换操作**

Add this method to `VisionLabSession` in `vision_platform/session.py`:

```python
    def replace_application(self, factory, *, scene_path):
        with self._lock:
            previous = self._application
        previous.close()
        replacement = factory()
        try:
            replacement.load_and_start_scene(scene_path)
            replacement.open()
        except BaseException:
            try:
                replacement.close()
            finally:
                raise
        with self._lock:
            self._application = replacement
            handlers = tuple(self._handlers)
        for handler in handlers:
            handler(replacement)
        return replacement
```

该实现直接复用 V2.1 已定义的 `_lock`、`_application` 和 `_handlers`，顺序固定为“旧应用关闭 → 新场景加载 → 新应用打开 → 发布替换事件”。

- [x] **Step 5: 实现实验会话**

Create `vision_platform/experiments/session.py`:

```python
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from vision_platform.config import WorkspaceConfig
from vision_platform.experiments.capabilities import check_capabilities
from vision_platform.experiments.catalog import ExperimentCatalog
from vision_platform.experiments.models import ExperimentRunContext


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExperimentSession:
    def __init__(
        self,
        *,
        catalog: ExperimentCatalog,
        vision_session: Any,
        base_config: Any,
        application_factory: Callable[[Any], Any],
        student_is_idle: Callable[[], bool],
    ) -> None:
        self.catalog = catalog
        self.vision_session = vision_session
        self.base_config = base_config
        self.application_factory = application_factory
        self.student_is_idle = student_is_idle
        self.current: ExperimentRunContext | None = None

    def select(self, experiment_id: str) -> ExperimentRunContext:
        if not self.student_is_idle():
            raise RuntimeError("学生程序运行、暂停或清理期间不能切换实验")
        definition = self.catalog.require(experiment_id)
        workspace = WorkspaceConfig(
            x_mm=tuple(float(value) for value in definition.workspace["x_mm"]),
            y_mm=tuple(float(value) for value in definition.workspace["y_mm"]),
            z_mm=tuple(float(value) for value in definition.workspace["z_mm"]),
            safe_z_mm=float(definition.workspace["safe_z_mm"]),
        )
        camera_options = deepcopy(dict(self.base_config.camera_options))
        camera_options.setdefault("sim", {})["sensor_path"] = str(
            definition.public_parameters["camera_path"]
        )
        task_options = dict(self.base_config.task)
        pickables_path = definition.public_parameters.get("pickables_path")
        if pickables_path is not None:
            task_options["pickables_path"] = str(pickables_path)
        config = replace(
            self.base_config,
            camera_backend="sim",
            robot_backend="sim",
            coppelia_scene=definition.scene,
            camera_options=camera_options,
            task=task_options,
            workspace=workspace,
        )
        application = self.vision_session.replace_application(
            lambda: self.application_factory(config),
            scene_path=definition.scene,
        )
        report = check_capabilities(application, definition.capabilities)
        if not report.ready:
            application.close()
            reasons = "; ".join(
                f"{name}: {report.reasons[name]}" for name in report.missing
            )
            raise RuntimeError(f"实验能力不可用：{reasons}")
        manifest = json.loads(
            definition.scene_manifest.read_text(encoding="utf-8")
        )
        actual_hash = _sha256(definition.scene)
        expected_hash = str(manifest["scene"]["sha256"])
        if actual_hash != expected_hash:
            application.close()
            raise RuntimeError(
                f"场景哈希不匹配：expected={expected_hash}, actual={actual_hash}"
            )
        context = ExperimentRunContext(
            experiment_id=definition.experiment_id,
            experiment_version=definition.version,
            scene_path=definition.scene,
            scene_sha256=actual_hash,
            scene_manifest_path=definition.scene_manifest,
            public_parameters=definition.public_parameters,
        )
        self.current = context
        return context
```

- [x] **Step 6: 运行会话测试**

Run:

```powershell
python -m pytest tests/test_experiments/test_session.py tests/test_vision_platform/test_session.py -q
```

Expected: 全部 PASS。

- [x] **Step 7: 提交实验会话**

Run:

```powershell
git add `
  vision_platform/session.py `
  vision_platform/experiments/session.py `
  tests/test_experiments/test_session.py `
  tests/test_vision_platform/test_session.py
git commit -m "feat(experiments): switch isolated experiment scenes"
```

### Task 6: 扩展运行证据以记录实验和相机快照

**Files:**
- Modify: `vision_platform/student/evidence.py`
- Create: `tests/test_student_programs/test_experiment_evidence.py`

- [x] **Step 1: 写实验证据失败测试**

Create `tests/test_student_programs/test_experiment_evidence.py`:

```python
from __future__ import annotations

import hashlib
import json

from vision_platform.student.evidence import StudentRunEvidence


def test_evidence_records_experiment_and_snapshot(tmp_path):
    program = tmp_path / "student.py"
    program.write_text("def main(ctx):\n    pass\n", encoding="utf-8")
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=program,
        robot_backend="sim",
        run_id="run-r1-05",
        run_metadata={
            "experiment_id": "R1-05",
            "experiment_version": "2.2.0",
            "scene_sha256": "a" * 64,
            "hardware_status": "PENDING_HARDWARE",
        },
    )

    record = evidence.record_snapshot(
        snapshot_id="frame-000001",
        png_bytes=b"\x89PNG\r\n\x1a\npayload",
        metadata={
            "width": 640,
            "height": 480,
            "source": "coppeliasim",
            "sequence_id": 1,
        },
    )
    summary_path = evidence.finalize(
        status="PASS",
        command_count=1,
        last_pose_mm=(100, 0, 120),
        safety_violation_count=0,
        error=None,
        cleanup_errors=[],
    )

    assert record["path"] == "frames/frame-000001.png"
    assert record["sha256"] == hashlib.sha256(
        b"\x89PNG\r\n\x1a\npayload"
    ).hexdigest()
    assert (evidence.directory / record["path"]).read_bytes().startswith(b"\x89PNG")
    manifest = json.loads(
        (evidence.directory / "manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    snapshots = [
        json.loads(line)
        for line in (evidence.directory / "snapshots.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert manifest["experiment_id"] == "R1-05"
    assert summary["experiment_id"] == "R1-05"
    assert summary["hardware_status"] == "PENDING_HARDWARE"
    assert snapshots == [record]


def test_snapshot_id_cannot_escape_evidence_directory(tmp_path):
    program = tmp_path / "student.py"
    program.write_text("def main(ctx):\n    pass\n", encoding="utf-8")
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=program,
        robot_backend="sim",
        run_id="run-safe-path",
        run_metadata={"hardware_status": "PENDING_HARDWARE"},
    )

    try:
        evidence.record_snapshot(
            snapshot_id="../escape",
            png_bytes=b"png",
            metadata={},
        )
    except ValueError as error:
        assert "snapshot_id" in str(error)
    else:
        raise AssertionError("unsafe snapshot_id was accepted")
```

- [x] **Step 2: 运行测试并确认先失败**

Run:

```powershell
python -m pytest tests/test_student_programs/test_experiment_evidence.py -q
```

Expected: FAIL，指出 `run_metadata` 或 `record_snapshot` 尚不支持。

- [x] **Step 3: 增加元数据和快照写入**

在 `vision_platform/student/evidence.py` 中：

1. 把类型导入改为：

```python
from typing import Any, Mapping
```

2. 给 `StudentRunEvidence` 增加字段：

```python
    run_metadata: dict[str, Any]
```

3. 给 `StudentRunEvidence.create()` 增加可选参数：

```python
        run_metadata: Mapping[str, Any] | None = None,
```

4. 创建对象时把元数据复制到实例字段，并合并进 `manifest.json`：

```python
metadata = dict(run_metadata or {})
if metadata.get("hardware_status", "PENDING_HARDWARE") != "PENDING_HARDWARE":
    raise ValueError("hardware_status must remain PENDING_HARDWARE")
metadata["hardware_status"] = "PENDING_HARDWARE"
manifest.update(metadata)
```

构造返回值时加入：

```python
            run_metadata=metadata,
```

5. 在 `finalize()` 生成的摘要中加入：

```python
        payload.update(self.run_metadata)
        payload["hardware_status"] = "PENDING_HARDWARE"
```

6. 给类增加完整快照方法：

```python
    def record_snapshot(
        self,
        *,
        snapshot_id: str,
        png_bytes: bytes,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        if (
            not snapshot_id
            or not snapshot_id.replace("-", "").replace("_", "").isalnum()
        ):
            raise ValueError("snapshot_id must use letters, digits, '-' or '_'")
        if not isinstance(png_bytes, bytes) or not png_bytes:
            raise ValueError("png_bytes must not be empty")
        frames = self.directory / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        relative = Path("frames") / f"{snapshot_id}.png"
        target = self.directory / relative
        target.write_bytes(png_bytes)
        record = {
            "snapshot_id": snapshot_id,
            "path": relative.as_posix(),
            "sha256": hashlib.sha256(png_bytes).hexdigest(),
            **dict(metadata),
        }
        self._append("snapshots.jsonl", record)
        return record
```

- [x] **Step 4: 运行证据回归**

Run:

```powershell
python -m pytest `
  tests/test_student_programs/test_evidence.py `
  tests/test_student_programs/test_experiment_evidence.py `
  -q
```

Expected: 全部 PASS，原 V2.1 证据测试不退化。

- [x] **Step 5: 提交证据扩展**

Run:

```powershell
git add `
  vision_platform/student/evidence.py `
  tests/test_student_programs/test_experiment_evidence.py
git commit -m "feat(student): record experiment camera evidence"
```

### Task 7: 增加相机快照和实验信息学生 SDK

**Files:**
- Modify: `vision_platform/student/protocol.py`
- Modify: `vision_platform/student/sdk.py`
- Modify: `vision_platform/student/runner.py`
- Create: `vision_platform/student/experiment_gateway.py`
- Create: `tests/test_student_programs/test_experiment_gateway.py`
- Modify: `tests/test_student_programs/test_protocol.py`
- Modify: `tests/test_student_programs/test_sdk.py`
- Modify: `tests/test_student_programs/test_runner.py`

- [x] **Step 1: 写协议和 SDK 失败测试**

Append to `tests/test_student_programs/test_protocol.py`:

```python
def test_v2_2_read_only_commands_are_whitelisted():
    assert "camera.capture" in ALLOWED_COMMANDS
    assert "experiment.info" in ALLOWED_COMMANDS
```

Append to `tests/test_student_programs/test_sdk.py`:

```python
import cv2
import numpy as np


def test_student_camera_decodes_read_only_png():
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[:, :, 1] = 200
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    connection = ScriptedConnection(
        [
            {
                "schema_version": 1,
                "kind": "result",
                "command_id": "000001",
                "status": "PASS",
                "value": {
                    "snapshot_id": "frame-000001",
                    "png_bytes": encoded.tobytes(),
                    "width": 4,
                    "height": 3,
                    "source": "coppeliasim",
                    "sequence_id": 9,
                },
                "error": None,
            }
        ]
    )

    frame = StudentContext(connection).camera.capture()

    assert frame.snapshot_id == "frame-000001"
    assert frame.image_bgr.shape == (3, 4, 3)
    assert frame.image_bgr.flags.writeable is False
    assert connection.sent[0]["name"] == "camera.capture"


def test_student_experiment_returns_detached_public_info():
    connection = ScriptedConnection(
        [
            {
                "schema_version": 1,
                "kind": "result",
                "command_id": "000001",
                "status": "PASS",
                "value": {
                    "experiment_id": "R1-05",
                    "public_parameters": {"safe_z_mm": 100},
                    "hardware_status": "PENDING_HARDWARE",
                },
                "error": None,
            }
        ]
    )

    info = StudentContext(connection).experiment.info()

    assert info["experiment_id"] == "R1-05"
    assert info["hardware_status"] == "PENDING_HARDWARE"
    assert connection.sent[0]["name"] == "experiment.info"
```

- [x] **Step 2: 写主进程网关失败测试**

Create `tests/test_student_programs/test_experiment_gateway.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vision_platform.experiments.models import ExperimentRunContext
from vision_platform.models import Frame
from vision_platform.student.experiment_gateway import StudentExperimentGateway


class FakeCamera:
    def read(self, timeout_s):
        assert timeout_s == 2.0
        return Frame(
            image_bgr=np.full((6, 8, 3), 127, dtype=np.uint8),
            width=8,
            height=6,
            timestamp_s=12.5,
            source="coppeliasim",
            sequence_id=4,
        )


class FakeEvidence:
    def __init__(self):
        self.calls = []

    def record_snapshot(self, **payload):
        self.calls.append(payload)
        return {"path": f"frames/{payload['snapshot_id']}.png"}


def _context(tmp_path):
    return ExperimentRunContext(
        experiment_id="R1-05",
        experiment_version="2.2.0",
        scene_path=tmp_path / "scene.ttt",
        scene_sha256="a" * 64,
        scene_manifest_path=tmp_path / "scene_manifest.json",
        public_parameters={"safe_z_mm": 100},
    )


def test_gateway_captures_png_and_records_same_bytes(tmp_path):
    evidence = FakeEvidence()
    gateway = StudentExperimentGateway(
        application=SimpleNamespace(camera=FakeCamera()),
        evidence=evidence,
        context=_context(tmp_path),
        capture_timeout_s=2.0,
    )

    value = gateway.dispatch("camera.capture", {})

    assert value["snapshot_id"] == "frame-000001"
    assert value["width"] == 8
    assert value["height"] == 6
    assert value["png_bytes"].startswith(b"\x89PNG")
    assert evidence.calls[0]["png_bytes"] == value["png_bytes"]


def test_gateway_returns_public_experiment_data_only(tmp_path):
    gateway = StudentExperimentGateway(
        application=SimpleNamespace(camera=FakeCamera()),
        evidence=FakeEvidence(),
        context=_context(tmp_path),
    )

    value = gateway.dispatch("experiment.info", {})

    assert value["experiment_id"] == "R1-05"
    assert value["hardware_status"] == "PENDING_HARDWARE"
    assert "scene_path" not in value
    assert "scene_manifest_path" not in value
```

- [x] **Step 3: 运行新增测试并确认先失败**

Run:

```powershell
python -m pytest `
  tests/test_student_programs/test_protocol.py `
  tests/test_student_programs/test_sdk.py `
  tests/test_student_programs/test_experiment_gateway.py `
  -q
```

Expected: FAIL，指出新命令、SDK 属性和网关不存在。

- [x] **Step 4: 扩展白名单和学生 SDK**

在 `vision_platform/student/protocol.py` 的 `ALLOWED_COMMANDS` 中增加：

```python
        "camera.capture",
        "experiment.info",
```

在 `vision_platform/student/sdk.py` 增加导入：

```python
from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np
```

在同文件增加：

```python
@dataclass(frozen=True)
class StudentFrame:
    snapshot_id: str
    image_bgr: np.ndarray
    width: int
    height: int
    source: str
    sequence_id: int


class StudentCamera:
    def __init__(self, rpc: _Rpc) -> None:
        self._rpc = rpc

    def capture(self) -> StudentFrame:
        value = self._rpc.call("camera.capture")
        if not isinstance(value, Mapping):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: camera.capture")
        encoded = np.frombuffer(value.get("png_bytes", b""), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: camera.capture")
        height, width = image.shape[:2]
        if width != int(value.get("width", -1)) or height != int(
            value.get("height", -1)
        ):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: camera dimensions")
        image.setflags(write=False)
        return StudentFrame(
            snapshot_id=str(value["snapshot_id"]),
            image_bgr=image,
            width=width,
            height=height,
            source=str(value["source"]),
            sequence_id=int(value["sequence_id"]),
        )


class StudentExperiment:
    def __init__(self, rpc: _Rpc) -> None:
        self._rpc = rpc

    def info(self) -> dict[str, Any]:
        value = self._rpc.call("experiment.info")
        if not isinstance(value, Mapping):
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: experiment.info")
        result = dict(value)
        if result.get("hardware_status") != "PENDING_HARDWARE":
            raise RuntimeError("PROTOCOL_RESPONSE_INVALID: hardware_status")
        return result
```

并在 `StudentContext.__init__()` 中加入：

```python
        self.camera = StudentCamera(self._rpc)
        self.experiment = StudentExperiment(self._rpc)
```

- [x] **Step 5: 实现主进程只读网关**

Create `vision_platform/student/experiment_gateway.py`:

```python
from __future__ import annotations

from itertools import count
from typing import Any, Mapping

import cv2

from vision_platform.experiments.models import ExperimentRunContext


class StudentExperimentGateway:
    def __init__(
        self,
        *,
        application: Any,
        evidence: Any,
        context: ExperimentRunContext,
        capture_timeout_s: float = 2.0,
    ) -> None:
        self.application = application
        self.evidence = evidence
        self.context = context
        self.capture_timeout_s = float(capture_timeout_s)
        self._snapshot_ids = count(1)

    def dispatch(self, name: str, args: Mapping[str, Any]) -> Any:
        if args:
            raise ValueError(f"{name} does not accept arguments")
        if name == "experiment.info":
            return self.context.to_public_dict()
        if name == "camera.capture":
            return self._capture()
        raise ValueError(f"COMMAND_NOT_ALLOWED: {name}")

    def _capture(self) -> dict[str, Any]:
        frame = self.application.camera.read(timeout_s=self.capture_timeout_s)
        ok, encoded = cv2.imencode(".png", frame.image_bgr)
        if not ok:
            raise RuntimeError("CAMERA_SNAPSHOT_ENCODE_FAILED")
        snapshot_id = f"frame-{next(self._snapshot_ids):06d}"
        png_bytes = encoded.tobytes()
        metadata = {
            "width": int(frame.width),
            "height": int(frame.height),
            "timestamp_s": float(frame.timestamp_s),
            "source": str(frame.source),
            "sequence_id": int(frame.sequence_id),
        }
        record = self.evidence.record_snapshot(
            snapshot_id=snapshot_id,
            png_bytes=png_bytes,
            metadata=metadata,
        )
        return {
            "snapshot_id": snapshot_id,
            "png_bytes": png_bytes,
            **metadata,
            "evidence_path": record["path"],
        }
```

- [x] **Step 6: 接入唯一控制器命令分派**

给 `StudentProgramController` 构造函数增加必选的 V2.2 上下文参数：

```python
        experiment_context: ExperimentRunContext | None = None,
```

创建 `StudentRunEvidence` 时传入：

```python
run_metadata = (
    experiment_context.to_public_dict()
    if experiment_context is not None
    else {"hardware_status": "PENDING_HARDWARE"}
)
```

证据创建后构造：

```python
self._experiment_gateway = (
    StudentExperimentGateway(
        application=self.session.application,
        evidence=self.evidence,
        context=experiment_context,
    )
    if experiment_context is not None
    else None
)
```

在现有显式白名单命令分派中加入：

```python
if command.name in {"camera.capture", "experiment.info"}:
    if self._experiment_gateway is None:
        raise VisionPlatformError(
            "EXPERIMENT_CONTEXT_REQUIRED",
            "当前运行没有选择 V2.2 实验",
        )
    return self._experiment_gateway.dispatch(command.name, command.args)
```

不得使用 `getattr` 动态调用。`robot.*` 和 `tool.*` 仍进入 V2.1 `StudentMotionGuard`，不经过实验网关。

- [x] **Step 7: 增加控制器集成断言**

Append to `tests/test_student_programs/test_runner.py`:

```python
def test_camera_command_without_experiment_context_fails_closed(tmp_path):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    ctx.camera.capture()\n",
        experiment_context=None,
    )
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=3)

    assert result.status == "FAILED"
    assert result.error["code"] == "EXPERIMENT_CONTEXT_REQUIRED"
```

同时给既有 `make_controller()` 辅助函数增加 `experiment_context=None` 参数，并原样传给 `StudentProgramController`。

- [x] **Step 8: 运行学生程序完整回归**

Run:

```powershell
python -m pytest tests/test_student_programs -q
```

Expected: 全部 PASS，V2.1 的运动、暂停、单步、停止、超时和证据测试均不退化。

- [x] **Step 9: 提交 SDK 扩展**

Run:

```powershell
git add `
  vision_platform/student/protocol.py `
  vision_platform/student/sdk.py `
  vision_platform/student/runner.py `
  vision_platform/student/experiment_gateway.py `
  tests/test_student_programs
git commit -m "feat(student): expose controlled camera snapshots"
```

### Task 8: 按实验激活物流任务组并绑定吸盘集合

**Files:**
- Create: `vision_platform/experiments/scene_setup.py`
- Modify: `vision_platform/experiments/session.py`
- Modify: `vision_platform/application.py`
- Create: `tests/test_experiments/test_scene_setup.py`
- Modify: `tests/test_experiments/test_session.py`
- Modify: `tests/test_vision_platform/test_coppeliasim_suction.py`

- [x] **Step 1: 写任务组激活失败测试**

Create `tests/test_experiments/test_scene_setup.py`:

```python
from vision_platform.experiments.scene_setup import activate_scene_group


class FakeSim:
    handle_world = -1

    def __init__(self):
        self.handles = {
            "/LogisticsLab/Tasks/Stack": 1,
            "/LogisticsLab/Tasks/Digits": 2,
            "/LogisticsLab/Tasks/Classes": 3,
        }
        self.positions = []

    def getObject(self, path):
        return self.handles[path]

    def setObjectPosition(self, handle, position, relative_to):
        self.positions.append((handle, list(position), relative_to))

    def resetDynamicObject(self, handle):
        pass


def test_activate_scene_group_moves_only_selected_group_into_workspace():
    sim = FakeSim()

    activate_scene_group(
        sim,
        active_path="/LogisticsLab/Tasks/Digits",
    )

    assert sim.positions == [
        (1, [0.0, 0.0, -2.0], -1),
        (2, [0.0, 0.0, 0.0], -1),
        (3, [0.0, 0.0, -4.0], -1),
    ]


def test_scene_without_group_needs_no_activation():
    sim = FakeSim()

    activate_scene_group(sim, active_path=None)

    assert sim.positions == []
```

- [x] **Step 2: 写吸盘路径配置失败测试**

Append to `tests/test_vision_platform/test_coppeliasim_suction.py`:

```python
def test_suction_accepts_experiment_pickables_path():
    tool = CoppeliaSimSuction(
        sim=object(),
        pickables_path="/LogisticsLab/Tasks/Stack/Pickables",
    )

    assert tool.pickables_path == "/LogisticsLab/Tasks/Stack/Pickables"
```

在 `tests/test_experiments/test_session.py` 的成功切换测试中断言：

```python
    assert created[0].config.task["pickables_path"] == (
        "/VisionLab/Pickables"
    )
```

该测试 fixture 的 R1-01 不声明 `pickables_path`，因此保持 V2.1 默认值。

- [x] **Step 3: 运行测试并确认先失败**

Run:

```powershell
python -m pytest `
  tests/test_experiments/test_scene_setup.py `
  tests/test_experiments/test_session.py `
  tests/test_vision_platform/test_coppeliasim_suction.py `
  -q
```

Expected: FAIL，指出 `scene_setup` 不存在；既有吸盘构造测试继续通过。

- [x] **Step 4: 实现固定物流任务组激活**

Create `vision_platform/experiments/scene_setup.py`:

```python
from __future__ import annotations

from typing import Any


LOGISTICS_GROUPS = (
    "/LogisticsLab/Tasks/Stack",
    "/LogisticsLab/Tasks/Digits",
    "/LogisticsLab/Tasks/Classes",
)


def activate_scene_group(
    sim: Any,
    *,
    active_path: str | None,
) -> None:
    if active_path is None:
        return
    if active_path not in LOGISTICS_GROUPS:
        raise ValueError(f"Unknown logistics task group: {active_path}")
    for index, path in enumerate(LOGISTICS_GROUPS, start=1):
        handle = int(sim.getObject(path))
        position = (
            [0.0, 0.0, 0.0]
            if path == active_path
            else [0.0, 0.0, -float(index + 1)]
        )
        sim.setObjectPosition(handle, position, sim.handle_world)
```

为匹配测试中的确定位置，把 `LOGISTICS_GROUPS` 非活动位置映射固定为：

```python
_PARKED_Z = {
    "/LogisticsLab/Tasks/Stack": -2.0,
    "/LogisticsLab/Tasks/Digits": -3.0,
    "/LogisticsLab/Tasks/Classes": -4.0,
}
```

并把循环中的非活动表达式替换为：

```python
            else [0.0, 0.0, _PARKED_Z[path]]
```

- [x] **Step 5: 在实验切换完成后激活任务组**

在 `vision_platform/experiments/session.py` 导入：

```python
from vision_platform.experiments.scene_setup import activate_scene_group
```

在 `replace_application()` 返回后、能力检查前调用：

```python
        activate_scene_group(
            application.sim,
            active_path=definition.public_parameters.get("scene_group_path"),
        )
```

若激活失败，执行 `application.close()` 后重新抛出异常；不允许在任务组状态未知时启动学生程序。

- [x] **Step 6: 让应用按实验配置构造吸盘**

在 `vision_platform/application.py` 中把：

```python
            tool = CoppeliaSimSuction(sim=sim)
```

替换为：

```python
            tool = CoppeliaSimSuction(
                sim=sim,
                pickables_path=str(
                    selected.task.get(
                        "pickables_path",
                        "/VisionLab/Pickables",
                    )
                ),
            )
```

该修改只改变可抓取集合路径，不改变吸盘 TCP、最大附着距离、父子关系或安全策略。

- [x] **Step 7: 运行测试并提交**

Run:

```powershell
python -m pytest `
  tests/test_experiments/test_scene_setup.py `
  tests/test_experiments/test_session.py `
  tests/test_vision_platform/test_coppeliasim_suction.py `
  -q
git add `
  vision_platform/experiments/scene_setup.py `
  vision_platform/experiments/session.py `
  vision_platform/application.py `
  tests/test_experiments `
  tests/test_vision_platform/test_coppeliasim_suction.py
git commit -m "feat(experiments): activate isolated logistics tasks"
```

Expected: 全部 PASS，提交成功。

### Task 9: 生成原创数字参考图并建立场景合同

**Files:**
- Create: `simulation/training_scenes/__init__.py`
- Create: `simulation/training_scenes/generate_labels.py`
- Create: `simulation/training_scenes/scene_contract.py`
- Create: `tests/test_simulation/test_training_labels.py`
- Create: `tests/test_simulation/test_training_scene_contract.py`

> **集成候选审计（2026-07-31）：** `91f95c2` 已带入本 Task 的生成器、
> 三张标签、清单、场景合同和四项基础测试；本集成树首次新鲜运行结果为
> `4 passed`，因此没有重现下文预期的 `ModuleNotFoundError`，也不把它记作
> 本轮 RED。针对候选缺口新增严格合同测试后，真实 RED 为
> `33 failed, 14 passed`；补充空 `required_paths` 边界时，该组真实 RED 为
> `6 failed`。最小加固后，标签、合同及两个正式场景合同的新鲜回归为
> `57 passed`。质量复审补充 hardlink、PNG staging 部分写失败和发布替换
> 失败三项边界时，三条选择器分别真实得到 `1 failed, 52 deselected`、
> `1 failed, 4 deselected` 和 `1 failed, 4 deselected`；原子发布加固后的
> 更新目标回归为 `61 passed`。两个独立临时目录生成的 PNG 和
> `manifest.json` 与正式素材逐文件字节及 SHA256 一致。

- [x] **Step 1: 写原创标签生成失败测试**

Create `tests/test_simulation/test_training_labels.py`:

```python
from __future__ import annotations

import hashlib
import json

import cv2

from simulation.training_scenes.generate_labels import generate_digit_labels


def test_digit_labels_are_deterministic_and_manifested(tmp_path):
    first = generate_digit_labels(tmp_path)
    first_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((tmp_path / "digits").glob("*.png"))
    }
    second = generate_digit_labels(tmp_path)
    second_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((tmp_path / "digits").glob("*.png"))
    }

    assert first_hashes == second_hashes
    assert sorted(first_hashes) == ["1.png", "2.png", "3.png"]
    assert first["generator"] == "simulation.training_scenes.generate_labels"
    assert first == second
    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest == first
    for path in (tmp_path / "digits").glob("*.png"):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        assert image.shape == (96, 64)
        assert image.min() == 0
        assert image.max() == 255
```

- [x] **Step 2: 写场景合同失败测试**

Create `tests/test_simulation/test_training_scene_contract.py`:

```python
from __future__ import annotations

import hashlib
import json

import pytest

from simulation.training_scenes.scene_contract import validate_scene_contract


def _write_contract(tmp_path):
    template = tmp_path / "template.ttt"
    template.write_bytes(b"protected-template")
    scene = tmp_path / "new-scene.ttt"
    scene.write_bytes(b"independent-scene")
    spec = tmp_path / "scene_spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scene_id": "robot-basics",
                "template": "template.ttt",
                "output": "new-scene.ttt",
                "root_path": "/RobotBasics",
                "required_paths": ["/BLX_base_link", "/RobotBasics/Camera"],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "scene_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scene_id": "robot-basics",
                "template": {
                    "path": "template.ttt",
                    "sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
                },
                "scene": {
                    "path": "new-scene.ttt",
                    "sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
                },
                "required_paths": ["/BLX_base_link", "/RobotBasics/Camera"],
                "protected_assets_unchanged": True,
            }
        ),
        encoding="utf-8",
    )
    return spec, manifest, template


def test_scene_contract_verifies_template_scene_and_paths(tmp_path):
    spec, manifest, _ = _write_contract(tmp_path)

    report = validate_scene_contract(spec, manifest, project_root=tmp_path)

    assert report["status"] == "PASS"
    assert report["scene_id"] == "robot-basics"


def test_scene_contract_rejects_modified_template(tmp_path):
    spec, manifest, template = _write_contract(tmp_path)
    template.write_bytes(b"changed")

    with pytest.raises(ValueError, match="template sha256"):
        validate_scene_contract(spec, manifest, project_root=tmp_path)
```

- [x] **Step 3: 运行测试并确认先失败**

Run:

```powershell
python -m pytest `
  tests/test_simulation/test_training_labels.py `
  tests/test_simulation/test_training_scene_contract.py `
  -q
```

Expected: FAIL with `ModuleNotFoundError: simulation.training_scenes`。

- [x] **Step 4: 实现确定性原创数字参考图**

Create `simulation/training_scenes/generate_labels.py`:

```python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_digit_labels(output_dir: str | Path) -> dict:
    root = Path(output_dir).expanduser().resolve()
    digits = root / "digits"
    digits.mkdir(parents=True, exist_ok=True)
    patterns = {
        1: ("b", "c"),
        2: ("a", "b", "g", "e", "d"),
        3: ("a", "b", "g", "c", "d"),
    }
    segments = {
        "a": ((20, 12), (44, 18)),
        "b": ((43, 16), (49, 46)),
        "c": ((43, 50), (49, 80)),
        "d": ((20, 78), (44, 84)),
        "e": ((15, 50), (21, 80)),
        "g": ((20, 45), (44, 51)),
    }
    files = []
    for value in (1, 2, 3):
        image = np.full((96, 64), 255, dtype=np.uint8)
        for segment in patterns[value]:
            cv2.rectangle(
                image,
                segments[segment][0],
                segments[segment][1],
                0,
                thickness=-1,
            )
        path = digits / f"{value}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write {path}")
        files.append(
            {
                "digit": value,
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "size": [64, 96],
            }
        )
    manifest = {
        "schema_version": 1,
        "generator": "simulation.training_scenes.generate_labels",
        "license": "project-original-generated",
        "rendering": "project-original seven-segment geometry",
        "files": files,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = generate_digit_labels(args.output)
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 5: 实现场景规格、清单和哈希合同**

Create `simulation/training_scenes/scene_contract.py`:

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, raw: str) -> Path:
    path = (root / raw).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"path must stay inside project: {raw}")
    if not path.is_file():
        raise ValueError(f"file does not exist: {raw}")
    return path


def validate_scene_contract(
    spec_path: str | Path,
    manifest_path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    spec = _load(Path(spec_path).expanduser().resolve())
    manifest = _load(Path(manifest_path).expanduser().resolve())
    if spec.get("schema_version") != 1 or manifest.get("schema_version") != 1:
        raise ValueError("scene schema_version must be 1")
    if spec["scene_id"] != manifest["scene_id"]:
        raise ValueError("scene_id mismatch")
    template = _inside(root, manifest["template"]["path"])
    scene = _inside(root, manifest["scene"]["path"])
    if _sha256(template) != manifest["template"]["sha256"]:
        raise ValueError("template sha256 mismatch")
    if _sha256(scene) != manifest["scene"]["sha256"]:
        raise ValueError("scene sha256 mismatch")
    if manifest.get("protected_assets_unchanged") is not True:
        raise ValueError("protected assets were not verified")
    if spec["required_paths"] != manifest["required_paths"]:
        raise ValueError("required_paths mismatch")
    return {
        "status": "PASS",
        "scene_id": spec["scene_id"],
        "scene_sha256": manifest["scene"]["sha256"],
        "required_path_count": len(spec["required_paths"]),
    }
```

Create `simulation/training_scenes/__init__.py`:

```python
"""Shared builders and contracts for modular training scenes."""
```

- [x] **Step 6: 运行测试并生成正式数字素材**

Run:

```powershell
python -m pytest `
  tests/test_simulation/test_training_labels.py `
  tests/test_simulation/test_training_scene_contract.py `
  -q
python -m simulation.training_scenes.generate_labels `
  --output simulation/logistics_lab/assets/labels
```

Expected: `3 passed`；生成 `1.png`、`2.png`、`3.png` 和 `manifest.json`，清单标记 `project-original-generated`。

- [x] **Step 7: 提交素材和合同**

Run:

```powershell
git add `
  simulation/training_scenes `
  simulation/logistics_lab/assets/labels `
  tests/test_simulation/test_training_labels.py `
  tests/test_simulation/test_training_scene_contract.py
git commit -m "feat(simulation): add original labels and scene contracts"
```

### Task 10: 构建机器人基础场景和物流场景

**Files:**
- Create: `simulation/training_scenes/build_scene.py`
- Create: `simulation/robot_basics/__init__.py`
- Create: `simulation/robot_basics/scene_spec.json`
- Create: `simulation/robot_basics/scene_manifest.json`
- Create: `simulation/robot_basics/BL23_robot_basics.ttt`
- Create: `simulation/logistics_lab/__init__.py`
- Create: `simulation/logistics_lab/scene_spec.json`
- Create: `simulation/logistics_lab/scene_manifest.json`
- Create: `simulation/logistics_lab/BL23_logistics_lab.ttt`
- Create: `tests/test_simulation/test_formal_training_scenes.py`

- [x] **Step 1: 写两个正式场景的静态失败测试**

Create `tests/test_simulation/test_formal_training_scenes.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from simulation.training_scenes.scene_contract import validate_scene_contract


ROOT = Path(__file__).resolve().parents[2]
SCENES = (
    ROOT / "simulation" / "robot_basics",
    ROOT / "simulation" / "logistics_lab",
)


def test_formal_training_scene_contracts_pass():
    reports = [
        validate_scene_contract(
            directory / "scene_spec.json",
            directory / "scene_manifest.json",
            project_root=ROOT,
        )
        for directory in SCENES
    ]

    assert [item["scene_id"] for item in reports] == [
        "robot-basics",
        "logistics-lab",
    ]
    assert all(item["status"] == "PASS" for item in reports)


def test_training_scenes_have_distinct_outputs_and_roots():
    specs = [
        json.loads(
            (directory / "scene_spec.json").read_text(encoding="utf-8")
        )
        for directory in SCENES
    ]

    assert specs[0]["output"] != specs[1]["output"]
    assert specs[0]["root_path"] == "/RobotBasics"
    assert specs[1]["root_path"] == "/LogisticsLab"
    assert all(
        item["template"] == "simulation/vision_lab/BL23_vision_lab.ttt"
        for item in specs
    )
    assert all(
        item["output"] != item["template"]
        for item in specs
    )


def test_logistics_scene_declares_three_isolated_task_groups():
    spec = json.loads(
        (SCENES[1] / "scene_spec.json").read_text(encoding="utf-8")
    )

    assert tuple(spec["tasks"]) == ("Stack", "Digits", "Classes")
    assert len(spec["tasks"]["Stack"]["objects"]) == 6
    assert [item["digit"] for item in spec["tasks"]["Digits"]["objects"]] == [
        1,
        2,
        3,
    ]
    assert len(spec["tasks"]["Classes"]["objects"]) == 4
```

- [x] **Step 2: 运行测试并确认先失败**

Run:

```powershell
python -m pytest tests/test_simulation/test_formal_training_scenes.py -q
```

Expected: FAIL，指出正式场景规格或清单不存在。

- [x] **Step 3: 创建机器人基础场景规格**

Create `simulation/robot_basics/scene_spec.json`:

```json
{
  "schema_version": 1,
  "scene_id": "robot-basics",
  "template": "simulation/vision_lab/BL23_vision_lab.ttt",
  "output": "simulation/robot_basics/BL23_robot_basics.ttt",
  "remove_paths": ["/VisionLab"],
  "root_path": "/RobotBasics",
  "workspace": {
    "alias": "Workspace",
    "size_mm": [160, 200, 10],
    "center_mm": [80, 0, 5]
  },
  "camera": {
    "alias": "Camera",
    "path": "/RobotBasics/Camera",
    "resolution": [640, 480],
    "position_m": [0.08, 0.0, 0.55],
    "near_clip_m": 0.01,
    "far_clip_m": 2.0,
    "perspective_angle_deg": 52
  },
  "markers": [
    {"alias": "TeachPoint1", "position_mm": [60, -50, 12]},
    {"alias": "TeachPoint2", "position_mm": [110, -50, 12]},
    {"alias": "TeachPoint3", "position_mm": [110, 50, 12]},
    {"alias": "TeachPoint4", "position_mm": [60, 50, 12]}
  ],
  "required_paths": [
    "/BLX_base_link",
    "/BLX_joint1",
    "/BLX_joint2",
    "/BLX_joint3",
    "/BLX_joint4",
    "/BLX_joint5",
    "/BLX_joint6",
    "/BLX_tool_suction",
    "/RobotBasics",
    "/RobotBasics/Workspace",
    "/RobotBasics/Camera",
    "/RobotBasics/TeachPoints/TeachPoint1",
    "/RobotBasics/TeachPoints/TeachPoint2",
    "/RobotBasics/TeachPoints/TeachPoint3",
    "/RobotBasics/TeachPoints/TeachPoint4"
  ]
}
```

- [x] **Step 4: 创建物流场景规格**

Create `simulation/logistics_lab/scene_spec.json`:

```json
{
  "schema_version": 1,
  "scene_id": "logistics-lab",
  "template": "simulation/vision_lab/BL23_vision_lab.ttt",
  "output": "simulation/logistics_lab/BL23_logistics_lab.ttt",
  "remove_paths": ["/VisionLab"],
  "root_path": "/LogisticsLab",
  "workspace": {
    "alias": "Workspace",
    "size_mm": [160, 200, 10],
    "center_mm": [80, 0, 5]
  },
  "camera": {
    "alias": "Camera",
    "path": "/LogisticsLab/Camera",
    "resolution": [640, 480],
    "position_m": [0.08, 0.0, 0.55],
    "near_clip_m": 0.01,
    "far_clip_m": 2.0,
    "perspective_angle_deg": 52
  },
  "tasks": {
    "Stack": {
      "objects": [
        {"alias": "stack_01", "shape": "cuboid", "size_mm": [18, 18, 18], "position_mm": [45, -60, 20], "color": [0.85, 0.20, 0.18]},
        {"alias": "stack_02", "shape": "cuboid", "size_mm": [18, 18, 18], "position_mm": [70, -60, 20], "color": [0.20, 0.55, 0.90]},
        {"alias": "stack_03", "shape": "cuboid", "size_mm": [18, 18, 18], "position_mm": [95, -60, 20], "color": [0.20, 0.70, 0.32]},
        {"alias": "stack_04", "shape": "cuboid", "size_mm": [18, 18, 18], "position_mm": [45, -25, 20], "color": [0.95, 0.75, 0.15]},
        {"alias": "stack_05", "shape": "cuboid", "size_mm": [18, 18, 18], "position_mm": [70, -25, 20], "color": [0.75, 0.30, 0.82]},
        {"alias": "stack_06", "shape": "cuboid", "size_mm": [18, 18, 18], "position_mm": [95, -25, 20], "color": [0.25, 0.80, 0.78]}
      ],
      "targets": [
        {"alias": "slot_01", "position_mm": [118, -45, 10]},
        {"alias": "slot_02", "position_mm": [118, 45, 10]},
        {"alias": "slot_03", "position_mm": [118, -45, 28]},
        {"alias": "slot_04", "position_mm": [118, 45, 28]},
        {"alias": "slot_05", "position_mm": [118, -45, 46]},
        {"alias": "slot_06", "position_mm": [118, 45, 46]}
      ]
    },
    "Digits": {
      "objects": [
        {"alias": "digit_1", "digit": 1, "position_mm": [50, -55, 20]},
        {"alias": "digit_2", "digit": 2, "position_mm": [75, -55, 20]},
        {"alias": "digit_3", "digit": 3, "position_mm": [100, -55, 20]}
      ],
      "targets": [
        {"alias": "slot_1", "position_mm": [118, -55, 10]},
        {"alias": "slot_2", "position_mm": [118, 0, 10]},
        {"alias": "slot_3", "position_mm": [118, 55, 10]}
      ]
    },
    "Classes": {
      "objects": [
        {"alias": "red_block", "shape": "cuboid", "size_mm": [18, 18, 18], "position_mm": [45, -55, 20], "color": [0.90, 0.15, 0.12]},
        {"alias": "blue_block", "shape": "cuboid", "size_mm": [18, 18, 18], "position_mm": [70, -55, 20], "color": [0.10, 0.35, 0.90]},
        {"alias": "green_cylinder", "shape": "cylinder", "size_mm": [18, 18, 18], "position_mm": [95, -55, 20], "color": [0.10, 0.75, 0.25]},
        {"alias": "yellow_cylinder", "shape": "cylinder", "size_mm": [18, 18, 18], "position_mm": [105, -55, 20], "color": [0.95, 0.78, 0.08]}
      ],
      "targets": [
        {"alias": "red_block", "position_mm": [118, -60, 10]},
        {"alias": "blue_block", "position_mm": [118, -20, 10]},
        {"alias": "green_cylinder", "position_mm": [118, 20, 10]},
        {"alias": "yellow_cylinder", "position_mm": [118, 60, 10]}
      ]
    }
  },
  "required_paths": [
    "/BLX_base_link",
    "/BLX_joint1",
    "/BLX_joint2",
    "/BLX_joint3",
    "/BLX_joint4",
    "/BLX_joint5",
    "/BLX_joint6",
    "/BLX_tool_suction",
    "/LogisticsLab",
    "/LogisticsLab/Workspace",
    "/LogisticsLab/Camera",
    "/LogisticsLab/Tasks/Stack/Pickables",
    "/LogisticsLab/Tasks/Stack/Targets",
    "/LogisticsLab/Tasks/Digits/Pickables",
    "/LogisticsLab/Tasks/Digits/Targets",
    "/LogisticsLab/Tasks/Classes/Pickables",
    "/LogisticsLab/Tasks/Classes/Targets"
  ]
}
```

- [x] **Step 5: 实现通用新场景构建器**

Create `simulation/training_scenes/build_scene.py`:

```python
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from simulation.vision_lab.hashing import asset_sha256
from vision_platform.coppelia_scene import stage_scene_for_coppeliasim


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = (
    PROJECT_ROOT / "simulation" / "vision_lab" / "source_manifest.json"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_hashes() -> dict[str, str]:
    source = _load(SOURCE_MANIFEST)
    result = {}
    for item in source["protected_formal_assets"]:
        path = PROJECT_ROOT / item["path"]
        actual = asset_sha256(path, mode=item.get("hash_mode", "bytes"))
        if actual != item["sha256"]:
            raise RuntimeError(f"protected asset hash mismatch: {item['path']}")
        result[item["path"]] = actual
    return result


def _alias(sim: Any, handle: int, name: str) -> int:
    sim.setObjectAlias(handle, name)
    return int(handle)


def _dummy(sim: Any, name: str, parent: int | None = None) -> int:
    handle = _alias(sim, int(sim.createDummy(0.005)), name)
    if parent is not None:
        sim.setObjectParent(handle, parent, True)
    return handle


def _shape(
    sim: Any,
    *,
    name: str,
    shape: str,
    size_mm: list[float],
    position_mm: list[float],
    color: list[float],
    parent: int,
    respondable: bool,
) -> int:
    primitive = (
        sim.primitiveshape_cylinder
        if shape == "cylinder"
        else sim.primitiveshape_cuboid
    )
    handle = _alias(
        sim,
        int(
            sim.createPrimitiveShape(
                primitive,
                [float(value) / 1000.0 for value in size_mm],
                0,
            )
        ),
        name,
    )
    sim.setShapeColor(
        handle,
        "",
        sim.colorcomponent_ambient_diffuse,
        [float(value) for value in color],
    )
    sim.setObjectInt32Param(handle, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(
        handle,
        sim.shapeintparam_respondable,
        int(respondable),
    )
    sim.setObjectParent(handle, parent, False)
    sim.setObjectPosition(
        handle,
        [float(value) / 1000.0 for value in position_mm],
        parent,
    )
    return handle


def _camera(sim: Any, spec: dict[str, Any], parent: int) -> int:
    resolution = [int(value) for value in spec["resolution"]]
    options = 2 | 4 | 64 | 128
    handle = int(
        sim.createVisionSensor(
            options,
            [resolution[0], resolution[1], 0, 0],
            [
                float(spec["near_clip_m"]),
                float(spec["far_clip_m"]),
                math.radians(float(spec["perspective_angle_deg"])),
                0.02,
                0.0,
                0.0,
                0.08,
                0.08,
                0.10,
                0.0,
                0.0,
            ],
        )
    )
    _alias(sim, handle, spec["alias"])
    sim.setObjectPose(
        handle,
        [*[float(value) for value in spec["position_m"]], 1.0, 0.0, 0.0, 0.0],
        sim.handle_world,
    )
    sim.setObjectParent(handle, parent, True)
    return handle


_SEGMENTS = {
    1: ("b", "c"),
    2: ("a", "b", "g", "e", "d"),
    3: ("a", "b", "g", "c", "d"),
}
_SEGMENT_POSES = {
    "a": ([0, 5, 10], [10, 2, 2]),
    "b": ([5, 0, 10], [2, 10, 2]),
    "c": ([5, -10, 10], [2, 10, 2]),
    "d": ([0, -15, 10], [10, 2, 2]),
    "e": ([-5, -10, 10], [2, 10, 2]),
    "g": ([0, -5, 10], [10, 2, 2]),
}


def _digit(sim: Any, item: dict[str, Any], parent: int) -> int:
    center = [float(value) for value in item["position_mm"]]
    parts = [
        _shape(
            sim,
            name=f"{item['alias']}_plate",
            shape="cuboid",
            size_mm=[20, 30, 8],
            position_mm=center,
            color=[0.92, 0.92, 0.92],
            parent=parent,
            respondable=True,
        )
    ]
    for segment in _SEGMENTS[int(item["digit"])]:
        offset, size = _SEGMENT_POSES[segment]
        parts.append(
            _shape(
                sim,
                name=f"{item['alias']}_{segment}",
                shape="cuboid",
                size_mm=size,
                position_mm=[
                    center[0] + offset[0],
                    center[1] + offset[1],
                    center[2] + offset[2],
                ],
                color=[0.04, 0.04, 0.04],
                parent=parent,
                respondable=False,
            )
        )
    compound = int(sim.groupShapes(parts, False))
    _alias(sim, compound, item["alias"])
    sim.setObjectParent(compound, parent, True)
    return compound


def _build_workspace(sim: Any, spec: dict[str, Any], root: int) -> None:
    _shape(
        sim,
        name=spec["alias"],
        shape="cuboid",
        size_mm=spec["size_mm"],
        position_mm=spec["center_mm"],
        color=[0.30, 0.32, 0.34],
        parent=root,
        respondable=True,
    )


def _build_basics(sim: Any, spec: dict[str, Any], root: int) -> None:
    points = _dummy(sim, "TeachPoints", root)
    for marker in spec["markers"]:
        _shape(
            sim,
            name=marker["alias"],
            shape="cylinder",
            size_mm=[8, 8, 2],
            position_mm=marker["position_mm"],
            color=[0.12, 0.75, 0.85],
            parent=points,
            respondable=False,
        )


def _build_logistics(sim: Any, spec: dict[str, Any], root: int) -> None:
    tasks = _dummy(sim, "Tasks", root)
    for index, (name, task) in enumerate(spec["tasks"].items()):
        group = _dummy(sim, name, tasks)
        sim.setObjectPosition(
            group,
            [0.0, 0.0, 0.0 if index == 0 else -float(index + 2)],
            sim.handle_world,
        )
        pickables = _dummy(sim, "Pickables", group)
        targets = _dummy(sim, "Targets", group)
        for item in task["objects"]:
            if "digit" in item:
                _digit(sim, item, pickables)
            else:
                _shape(
                    sim,
                    name=item["alias"],
                    shape=item["shape"],
                    size_mm=item["size_mm"],
                    position_mm=item["position_mm"],
                    color=item["color"],
                    parent=pickables,
                    respondable=True,
                )
        for target in task["targets"]:
            handle = _shape(
                sim,
                name=target["alias"],
                shape="cuboid",
                size_mm=[24, 24, 2],
                position_mm=target["position_mm"],
                color=[0.35, 0.75, 0.95],
                parent=targets,
                respondable=False,
            )
            transparency = getattr(sim, "colorcomponent_transparency", None)
            if transparency is not None:
                sim.setShapeColor(handle, "", transparency, [0.70])


def build_scene(*, spec_path: Path, host: str, port: int) -> dict[str, Any]:
    spec_path = spec_path.expanduser().resolve()
    spec = _load(spec_path)
    if spec.get("schema_version") != 1:
        raise ValueError("scene schema_version must be 1")
    template = (PROJECT_ROOT / spec["template"]).resolve()
    output = (PROJECT_ROOT / spec["output"]).resolve()
    if output == template:
        raise ValueError("training scene output must not overwrite template")
    before = _protected_hashes()
    template_hash = _sha256(template)
    client = RemoteAPIClient(host=host, port=port)
    sim = client.require("sim")
    if int(sim.getSimulationState()) != int(sim.simulation_stopped):
        sim.stopSimulation()
        client.step()
    sim.loadScene(stage_scene_for_coppeliasim(template).as_posix())
    for path in spec["remove_paths"]:
        handle = int(sim.getObject(path))
        sim.removeObjects([handle], False)
    root_name = spec["root_path"].rsplit("/", 1)[-1]
    root = _dummy(sim, root_name)
    _build_workspace(sim, spec["workspace"], root)
    _camera(sim, spec["camera"], root)
    if spec["scene_id"] == "robot-basics":
        _build_basics(sim, spec, root)
    elif spec["scene_id"] == "logistics-lab":
        _build_logistics(sim, spec, root)
    else:
        raise ValueError(f"unsupported scene_id: {spec['scene_id']}")
    for path in spec["required_paths"]:
        sim.getObject(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sim.saveScene(output.as_posix())
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"CoppeliaSim did not write {output}")
    after = _protected_hashes()
    if after != before or _sha256(template) != template_hash:
        raise RuntimeError("protected assets changed during scene build")
    manifest = {
        "schema_version": 1,
        "scene_id": spec["scene_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "simulation.training_scenes.build_scene",
        "template": {
            "path": Path(spec["template"]).as_posix(),
            "sha256": template_hash,
        },
        "scene": {
            "path": Path(spec["output"]).as_posix(),
            "sha256": _sha256(output),
            "size_bytes": output.stat().st_size,
        },
        "required_paths": spec["required_paths"],
        "task_contracts": spec.get("tasks", {}),
        "protected_assets_unchanged": True,
    }
    _write(spec_path.parent / "scene_manifest.json", manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    args = parser.parse_args(argv)
    manifest = build_scene(
        spec_path=args.spec,
        host=args.host,
        port=args.port,
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `simulation/robot_basics/__init__.py` and `simulation/logistics_lab/__init__.py`:

```python
"""Formal modular CoppeliaSim training scene."""
```

- [x] **Step 6: 启动 CoppeliaSim 并构建两个独立场景**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File tools\vision_lab\launch_coppeliasim.ps1 `
  -Scene simulation\vision_lab\BL23_vision_lab.ttt `
  -Port 23000
python -m simulation.training_scenes.build_scene `
  --spec simulation/robot_basics/scene_spec.json `
  --port 23000
python -m simulation.training_scenes.build_scene `
  --spec simulation/logistics_lab/scene_spec.json `
  --port 23000
```

Expected:

- 两条构建命令都输出包含 `"protected_assets_unchanged": true` 的 JSON；
- 生成两个非空 `.ttt` 和两个 `scene_manifest.json`；
- 输出路径与模板路径不同。

- [x] **Step 7: 验证静态合同和受保护文件零差异**

Run:

```powershell
python -m pytest `
  tests/test_simulation/test_training_scene_contract.py `
  tests/test_simulation/test_formal_training_scenes.py `
  tests/test_simulation/test_robot_asset_provenance.py `
  -q
git diff --exit-code -- `
  simulation/vision_lab/BL23_vision_lab.ttt `
  robot_backends/models/BLX_openr6.ttt `
  robot_backends/models/openr6_arm_coppeliasim.urdf `
  robot_backends/models/meshes_blx `
  simulation/vision_lab/assets/robot
```

Expected: 全部 PASS；受保护场景和资产零差异。

- [x] **Step 8: 提交两个正式场景**

Run:

```powershell
git add `
  simulation/training_scenes/build_scene.py `
  simulation/robot_basics `
  simulation/logistics_lab `
  tests/test_simulation/test_formal_training_scenes.py
git commit -m "feat(simulation): add robot basics and logistics scenes"
```

> **集成审计（2026-07-31）：** 候选提交 `30719dc` 已带入两份规格、
> 两个场景、清单、构建器和静态测试；首次按本 Task 运行候选测试得到
> `3 passed`，因此没有虚构“正式场景缺失”的 RED。随后针对候选构建器会
> 直接删除/覆盖正式 `.ttt`、路径和 schema 校验不足、别名可逃逸、旧清单
> 可能指向新损坏场景等真实风险补充失败测试，首次结果为
> `13 failed, 5 passed`。加固后正式场景测试为 `20 passed`，Task 9/10、
> 场景合同和机器人资产来源联合测试为 `85 passed`，发布合同为
> `32 passed`，完整静态回归为 `856 passed, 5 skipped`；五项 skip 仍是
> 显式 CoppeliaSim 在线门禁，不能计为在线 PASS，本 Task 也不提前替代
> Task 15。构建器现仅接受两个固定 spec/scene/root/output 绑定，严格拒绝
> bool schema、绝对或逃逸路径、symlink/junction/hardlink alias，并在连接
> CoppeliaSim 及任何写入前核验 `SOURCE_MANIFEST` 和 34 项保护快照；它只
> 将场景保存到输出目录内唯一 staged `.ttt`，验证后使旧清单失效，原子
> 发布场景并最后发布 staged manifest，失败仅清理本轮临时文件。实际使用
> `E:\CoppeliaSim` 启动本任务 PID `23792`（端口 `23000`），构建结果为：
> `robot-basics` 11,599,773 bytes、SHA-256
> `9a2ce252e3f61f08c7498e2cdaf682bfe5255c99d8871dc242b0255555413ea4`；
> `logistics-lab` 11,664,564 bytes、SHA-256
> `bc81a8cbf9f05a079d6fd62309b284d135ea0f1dcc251da5f53363275c8ce8c3`；
> 两次 JSON 均为 `protected_assets_unchanged=true`。随后按 PID、可执行路径
> 和启动时刻精确停止，CoppeliaSim 进程及端口监听均为 0；五类受保护资产
> 相对本 Task 基线零差异。候选将三个任务组根路径加入 `required_paths`，
> 并把 `yellow_cylinder` 的 y 坐标设为 `-25 mm`；审计保留这两项合理偏差，
> 静态合同验证三组根路径存在，且 yellow 与 green 的平面包围盒不重叠。
> 本 Task 十个正式路径均已逐条核对 `RETAINED_FILES.txt`，各出现一次；整份
> 白名单为 237 项、0 重复、0 缺失、0 禁入项。

> **质量复审加固（2026-07-31）：** 针对同一输出的跨进程交错、cleanup
> 覆盖主异常、以及 scene replace 失败前旧 manifest 被删除三项审查意见，
> 先增加确定性复现测试，首次为 `6 failed, 21 passed`；另以单独 RED 证明
> staging 名称碰撞时旧实现会误删非本轮文件，并以另一条 RED 证明保护声明
> 无效的旧 manifest 不能视为可恢复发布。构建器现按规范化正式 output
> 同时取得进程内非阻塞线程锁和系统临时目录中的 OS 文件锁；Windows 使用
> `msvcrt.LK_NBLCK`，进程异常退出后由操作系统释放，锁范围从保护快照覆盖
> 到 client、stop/load/build/save、scene/manifest 发布、独立资源清理，
> 第二个构建明确返回 `build already in progress`。client close、每个
> staged/backup 删除及锁释放均独立尝试：存在主异常时通过异常注记报告
> cleanup 诊断并重抛原异常；已完成 manifest-last 提交时只返回
> `cleanup_errors` 并发出 warning，不把已提交构建误报为普通失败。旧 output
> 与 manifest 自洽时，旧 manifest 先原子移动到本轮唯一 backup；scene
> replace 失败且旧 scene 的 hash/size 仍一致时恢复，否则保持正式 manifest
> 缺失；scene 已变化而新 manifest 发布失败时同样保持无 manifest。最终
> Task 10 测试为 `30 passed`，Task 9/10、场景合同和资产来源联合为
> `95 passed`，发布合同为 `32 passed`，完整静态回归为
> `866 passed, 5 skipped`。本次只修改构建器、测试和审计记录，没有重新
> 构建正式场景；两份 `.ttt` 和两份 manifest 的字节保持不变。

### Task 11: 记录实验初态和终态场景探针

**Files:**
- Create: `vision_platform/experiments/probes.py`
- Modify: `vision_platform/student/evidence.py`
- Modify: `vision_platform/student/experiment_gateway.py`
- Modify: `vision_platform/student/runner.py`
- Create: `tests/test_experiments/test_probes.py`
- Modify: `tests/test_student_programs/test_experiment_evidence.py`
- Modify: `tests/test_student_programs/test_experiment_gateway.py`

- [x] **Step 1: 写物流终态探针失败测试**

Create `tests/test_experiments/test_probes.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

from vision_platform.experiments.probes import probe_experiment


class FakeSim:
    handle_world = -1

    def __init__(self, positions):
        self.positions = dict(positions)

    def getObject(self, path):
        if path not in self.positions:
            raise RuntimeError(path)
        return path

    def getObjectPosition(self, handle, relative_to):
        assert relative_to == self.handle_world
        return self.positions[handle]


def _definition(experiment_id, probe_kind, parameters):
    return SimpleNamespace(
        experiment_id=experiment_id,
        public_parameters=parameters,
        acceptance=SimpleNamespace(probe_kind=probe_kind),
    )


def test_stack_probe_accepts_two_columns_and_three_layers():
    slots_mm = [
        [118, -45, 20],
        [118, 45, 20],
        [118, -45, 38],
        [118, 45, 38],
        [118, -45, 56],
        [118, 45, 56],
    ]
    positions = {
        f"/LogisticsLab/Tasks/Stack/Pickables/stack_{index:02d}": [
            value / 1000 for value in slot
        ]
        for index, slot in enumerate(slots_mm, start=1)
    }
    definition = _definition(
        "R1-05",
        "stack_2x3",
        {
            "scene_group_path": "/LogisticsLab/Tasks/Stack",
            "stack_slots_mm": slots_mm,
        },
    )

    report = probe_experiment(
        FakeSim(positions),
        definition,
        phase="final",
        scene_manifest={"task_contracts": {}},
    )

    assert report["status"] == "PASS"
    assert report["matched"] == 6
    assert report["hardware_status"] == "PENDING_HARDWARE"


def test_class_probe_reports_misrouted_object_without_assigning_grade():
    positions = {
        "/LogisticsLab/Tasks/Classes/Pickables/red_block": [0.118, 0.060, 0.020],
        "/LogisticsLab/Tasks/Classes/Pickables/blue_block": [0.118, -0.020, 0.020],
        "/LogisticsLab/Tasks/Classes/Pickables/green_cylinder": [0.118, 0.020, 0.020],
        "/LogisticsLab/Tasks/Classes/Pickables/yellow_cylinder": [0.118, -0.060, 0.020],
    }
    definition = _definition(
        "R1-07",
        "class_zones",
        {
            "scene_group_path": "/LogisticsLab/Tasks/Classes",
            "drop_poses_mm": {
                "red_block": [118, -60, 20],
                "blue_block": [118, -20, 20],
                "green_cylinder": [118, 20, 20],
                "yellow_cylinder": [118, 60, 20],
            },
        },
    )

    report = probe_experiment(
        FakeSim(positions),
        definition,
        phase="final",
        scene_manifest={"task_contracts": {}},
    )

    assert report["status"] == "FAIL"
    assert report["matched"] == 2
    assert "grade" not in report
    assert "score" not in report


def test_initial_probe_uses_manifested_reset_positions():
    objects = [
        {
            "alias": f"stack_{index:02d}",
            "position_mm": [40 + index * 10, -60, 20],
        }
        for index in range(1, 7)
    ]
    positions = {
        f"/LogisticsLab/Tasks/Stack/Pickables/{item['alias']}": [
            value / 1000 for value in item["position_mm"]
        ]
        for item in objects
    }
    definition = _definition(
        "R1-05",
        "stack_2x3",
        {
            "scene_group_path": "/LogisticsLab/Tasks/Stack",
            "stack_slots_mm": [[118, -45, 20]] * 6,
        },
    )

    report = probe_experiment(
        FakeSim(positions),
        definition,
        phase="initial",
        scene_manifest={
            "task_contracts": {"Stack": {"objects": objects}}
        },
    )

    assert report["status"] == "PASS"
    assert report["matched"] == 6
```

- [x] **Step 2: 写 JSON 证据工件测试**

Append to `tests/test_student_programs/test_experiment_evidence.py`:

```python
def test_evidence_records_named_json_artifact(tmp_path):
    program = tmp_path / "student.py"
    program.write_text("def main(ctx):\n    pass\n", encoding="utf-8")
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=program,
        robot_backend="sim",
        run_id="run-probe",
        run_metadata={"hardware_status": "PENDING_HARDWARE"},
    )

    relative = evidence.record_json_artifact(
        "scene-initial.json",
        {
            "status": "PASS",
            "phase": "initial",
            "hardware_status": "PENDING_HARDWARE",
        },
    )

    assert relative == "scene-initial.json"
    payload = json.loads(
        (evidence.directory / relative).read_text(encoding="utf-8")
    )
    assert payload["phase"] == "initial"
```

- [x] **Step 3: 运行测试并确认先失败**

Run:

```powershell
python -m pytest `
  tests/test_experiments/test_probes.py `
  tests/test_student_programs/test_experiment_evidence.py `
  -q
```

Expected: FAIL，指出 `probes` 或 `record_json_artifact` 不存在。

- [x] **Step 4: 实现非评分式场景探针**

Create `vision_platform/experiments/probes.py`:

```python
from __future__ import annotations

from math import dist
from typing import Any


def _world_mm(sim: Any, path: str) -> list[float]:
    handle = sim.getObject(path)
    return [
        float(value) * 1000.0
        for value in sim.getObjectPosition(handle, sim.handle_world)
    ]


def _match_any(
    actual: list[list[float]],
    expected: list[list[float]],
    tolerance_mm: float,
) -> int:
    remaining = list(range(len(expected)))
    matched = 0
    for position in actual:
        candidates = [
            (dist(position, expected[index]), index) for index in remaining
        ]
        if not candidates:
            continue
        distance_mm, index = min(candidates)
        if distance_mm <= tolerance_mm:
            matched += 1
            remaining.remove(index)
    return matched


def probe_experiment(
    sim: Any,
    definition: Any,
    *,
    phase: str,
    scene_manifest: Mapping[str, Any],
    tolerance_mm: float = 6.0,
) -> dict[str, Any]:
    if phase not in {"initial", "final"}:
        raise ValueError("phase must be initial or final")
    kind = definition.acceptance.probe_kind
    parameters = definition.public_parameters
    group = parameters.get("scene_group_path")
    rows: list[dict[str, Any]] = []
    matched = 0
    expected_count = 0

    if phase == "initial" and group is not None:
        task_name = str(group).rsplit("/", 1)[-1]
        objects = scene_manifest["task_contracts"][task_name]["objects"]
        for item in objects:
            alias = str(item["alias"])
            actual = _world_mm(sim, f"{group}/Pickables/{alias}")
            expected = [float(value) for value in item["position_mm"]]
            distance_mm = dist(actual, expected)
            rows.append(
                {
                    "object": alias,
                    "position_mm": actual,
                    "expected_mm": expected,
                    "distance_mm": distance_mm,
                }
            )
            matched += int(distance_mm <= tolerance_mm)
        expected_count = len(objects)
    elif kind == "motion_observation":
        joint_paths = parameters.get(
            "joint_paths",
            [f"/BLX_joint{index}" for index in range(1, 7)],
        )
        for path in [*joint_paths, "/BLX_tool_suction"]:
            sim.getObject(path)
        matched = expected_count = len(joint_paths) + 1
    elif kind == "stack_2x3":
        expected = [
            [float(value) for value in slot]
            for slot in parameters["stack_slots_mm"]
        ]
        actual = [
            _world_mm(sim, f"{group}/Pickables/stack_{index:02d}")
            for index in range(1, 7)
        ]
        matched = _match_any(actual, expected, tolerance_mm)
        expected_count = len(expected)
        rows = [
            {"object": index + 1, "position_mm": position}
            for index, position in enumerate(actual)
        ]
    elif kind == "ordered_slots":
        order = [int(value) for value in parameters["order"]]
        slots = parameters["drop_slots_mm"]
        for index, digit in enumerate(order):
            actual = _world_mm(sim, f"{group}/Pickables/digit_{digit}")
            expected = [float(value) for value in slots[index]]
            distance_mm = dist(actual, expected)
            rows.append(
                {
                    "digit": digit,
                    "position_mm": actual,
                    "expected_mm": expected,
                    "distance_mm": distance_mm,
                }
            )
            matched += int(distance_mm <= tolerance_mm)
        expected_count = len(order)
    elif kind == "class_zones":
        routes = parameters["drop_poses_mm"]
        for class_id, expected_raw in routes.items():
            actual = _world_mm(sim, f"{group}/Pickables/{class_id}")
            expected = [float(value) for value in expected_raw]
            distance_mm = dist(actual, expected)
            rows.append(
                {
                    "class_id": class_id,
                    "position_mm": actual,
                    "expected_mm": expected,
                    "distance_mm": distance_mm,
                }
            )
            matched += int(distance_mm <= tolerance_mm)
        expected_count = len(routes)
    else:
        raise ValueError(f"unsupported probe_kind: {kind}")

    return {
        "schema_version": 1,
        "experiment_id": definition.experiment_id,
        "phase": phase,
        "status": "PASS" if matched == expected_count else "FAIL",
        "matched": matched,
        "expected": expected_count,
        "tolerance_mm": float(tolerance_mm),
        "rows": rows,
        "hardware_status": "PENDING_HARDWARE",
    }
```

- [x] **Step 5: 实现 JSON 工件写入**

Add to `StudentRunEvidence`:

```python
    def record_json_artifact(
        self,
        name: str,
        payload: Mapping[str, Any],
    ) -> str:
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("artifact name must be one JSON filename")
        target = self.directory / name
        target.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return name
```

- [x] **Step 6: 把探针挂入实验网关和控制器生命周期**

给 `StudentExperimentGateway` 构造函数增加 `definition` 和 `scene_manifest`，并增加：

```python
    def record_probe(self, phase: str) -> dict[str, Any]:
        report = probe_experiment(
            self.application.sim,
            self.definition,
            phase=phase,
            scene_manifest=self.scene_manifest,
        )
        self.evidence.record_json_artifact(
            f"scene-{phase}.json",
            report,
        )
        return report
```

给 `StudentProgramController.__init__()` 增加：

```python
        experiment_definition: ExperimentDefinition | None = None,
        scene_manifest: Mapping[str, Any] | None = None,
```

同时导入：

```python
from typing import Any, Mapping

from vision_platform.experiments.models import ExperimentDefinition
```

并保存为：

```python
self.experiment_definition = experiment_definition
self.scene_manifest = dict(scene_manifest or {})
```

构造 `StudentExperimentGateway` 时传入这两个值；`experiment_context` 非空但定义或场景清单为空时，以 `EXPERIMENT_CONTEXT_INVALID` 失败关闭。

在 `StudentProgramController.start()` 创建证据和网关后、spawn 子进程前调用：

```python
self._experiment_gateway.record_probe("initial")
```

在控制器完成安全收尾后、写 `summary.json` 前调用：

```python
final_probe = self._experiment_gateway.record_probe("final")
```

把 `final_probe["status"]` 作为 `scene_probe_status` 写入摘要，但不改变学生程序自身的 `PASS`、`FAILED` 或 `CANCELLED`。场景终态是教师复核证据，本阶段不转换为成绩。

- [x] **Step 7: 运行探针和学生控制器回归**

Run:

```powershell
python -m pytest `
  tests/test_experiments/test_probes.py `
  tests/test_student_programs/test_experiment_evidence.py `
  tests/test_student_programs/test_experiment_gateway.py `
  tests/test_student_programs/test_runner.py `
  -q
```

Expected: 全部 PASS；证据目录包含 `scene-initial.json` 和 `scene-final.json`。

- [x] **Step 8: 提交场景探针**

Run:

```powershell
git add `
  vision_platform/experiments/probes.py `
  vision_platform/student/evidence.py `
  vision_platform/student/experiment_gateway.py `
  vision_platform/student/runner.py `
  tests/test_experiments/test_probes.py `
  tests/test_student_programs
git commit -m "feat(experiments): record initial and final scene probes"
```

### Task 12: 提供五个可运行学生模板和正式实验说明

**Files:**
- Create: `student_programs/__init__.py`
- Create: `student_programs/templates/__init__.py`
- Create: `student_programs/templates/r1_common.py`
- Create: `student_programs/templates/r1_01_robot_basics.py`
- Create: `student_programs/templates/r1_02_teach_points.py`
- Create: `student_programs/templates/r1_05_visual_stacking.py`
- Create: `student_programs/templates/r1_06_digit_sort.py`
- Create: `student_programs/templates/r1_07_component_sort.py`
- Create: `docs/experiments/R1-01.md`
- Create: `docs/experiments/R1-02.md`
- Create: `docs/experiments/R1-05.md`
- Create: `docs/experiments/R1-06.md`
- Create: `docs/experiments/R1-07.md`
- Create: `tests/test_experiments/test_student_templates.py`
- Create: `tests/test_acceptance/test_experiment_guides.py`

- [x] **Step 1: 写模板合同失败测试**

Create `tests/test_experiments/test_student_templates.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from student_programs.templates.r1_common import (
    VisualObject,
    pick_and_place,
    pixel_to_world,
)
from vision_platform.student.validator import validate_program


ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = (
    "r1_01_robot_basics.py",
    "r1_02_teach_points.py",
    "r1_05_visual_stacking.py",
    "r1_06_digit_sort.py",
    "r1_07_component_sort.py",
)


class FakeRobot:
    def __init__(self):
        self.moves = []
        self.home_calls = 0

    def home(self):
        self.home_calls += 1

    def move_world(self, x, y, z, *, speed):
        self.moves.append((x, y, z, speed))

    def pose(self):
        return (100.0, 0.0, 120.0)


class FakeTool:
    def __init__(self):
        self.events = []

    def on(self):
        self.events.append("on")

    def off(self):
        self.events.append("off")


class FakeContext:
    def __init__(self):
        self.robot = FakeRobot()
        self.tool = FakeTool()
        self.logs = []
        self.checkpoints = []

    def log(self, message):
        self.logs.append(message)

    def checkpoint(self, label):
        self.checkpoints.append(label)


def test_all_first_batch_templates_pass_student_validator():
    for name in TEMPLATES:
        result = validate_program(
            ROOT / "student_programs" / "templates" / name
        )
        assert result.ok, (name, result.issues)


def test_pixel_to_world_applies_two_by_three_affine_matrix():
    matrix = [[0.2, 0.0, 35.0], [0.0, -0.3, 70.0]]

    assert pixel_to_world(matrix, (100.0, 50.0)) == (55.0, 55.0)


def test_pick_and_place_always_lifts_before_horizontal_motion():
    ctx = FakeContext()

    pick_and_place(
        ctx,
        pick_xy=(50.0, -40.0),
        drop_xyz=(118.0, 45.0, 38.0),
        pick_z_mm=20.0,
        safe_z_mm=100.0,
        speed=12.0,
    )

    assert ctx.robot.moves == [
        (50.0, -40.0, 100.0, 12.0),
        (50.0, -40.0, 20.0, 8.0),
        (50.0, -40.0, 100.0, 12.0),
        (118.0, 45.0, 100.0, 12.0),
        (118.0, 45.0, 38.0, 8.0),
        (118.0, 45.0, 100.0, 12.0),
    ]
    assert ctx.tool.events == ["on", "off"]


def test_visual_object_is_immutable():
    item = VisualObject(
        center_px=(10.0, 20.0),
        world_xy_mm=(40.0, 50.0),
        color="red",
        shape="block",
        confidence=0.9,
    )

    with pytest.raises(Exception):
        item.color = "blue"
```

- [x] **Step 2: 写实验说明合同失败测试**

Create `tests/test_acceptance/test_experiment_guides.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUIDES = tuple(
    ROOT / "docs" / "experiments" / f"R1-{index:02d}.md"
    for index in (1, 2, 5, 6, 7)
)
SECTIONS = (
    "## 实验任务",
    "## 实验入口",
    "## 原理",
    "## 操作步骤",
    "## 完成条件",
    "## 常见错误",
    "## 仿真与真机差异",
    "## 结果保存",
    "## 人工验收点",
)


def test_first_batch_guides_have_complete_teaching_contract():
    for path in GUIDES:
        text = path.read_text(encoding="utf-8")
        for section in SECTIONS:
            assert section in text, f"{path.name} missing {section}"
        assert "PENDING_HARDWARE" in text
        assert "仿真 PASS 不代表真机验收通过" in text
        assert "自动评分" not in text
```

- [x] **Step 3: 运行测试并确认先失败**

Run:

```powershell
python -m pytest `
  tests/test_experiments/test_student_templates.py `
  tests/test_acceptance/test_experiment_guides.py `
  -q
```

Expected: FAIL，指出学生模板或实验说明不存在。

- [x] **Step 4: 实现模板共用视觉和运动函数**

Create `student_programs/templates/r1_common.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from math import pi
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VisualObject:
    center_px: tuple[float, float]
    world_xy_mm: tuple[float, float]
    color: str
    shape: str
    confidence: float


def pixel_to_world(matrix, center_px):
    u, v = (float(center_px[0]), float(center_px[1]))
    return (
        float(matrix[0][0]) * u
        + float(matrix[0][1]) * v
        + float(matrix[0][2]),
        float(matrix[1][0]) * u
        + float(matrix[1][1]) * v
        + float(matrix[1][2]),
    )


def _color_name(hue):
    if hue < 10 or hue >= 170:
        return "red"
    if hue < 35:
        return "yellow"
    if hue < 85:
        return "green"
    if hue < 135:
        return "blue"
    return "unknown"


def detect_colored_objects(
    image_bgr,
    *,
    calibration_matrix,
    pick_x_max_mm,
    minimum_area_px=180,
):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([0, 70, 60], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    output = []
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(minimum_area_px):
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        center = (
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        )
        world_xy = pixel_to_world(calibration_matrix, center)
        if world_xy[0] > float(pick_x_max_mm):
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = (
            4.0 * pi * area / (perimeter * perimeter)
            if perimeter > 0
            else 0.0
        )
        sample = hsv[int(round(center[1])), int(round(center[0]))]
        output.append(
            VisualObject(
                center_px=center,
                world_xy_mm=world_xy,
                color=_color_name(int(sample[0])),
                shape="cylinder" if circularity >= 0.78 else "block",
                confidence=min(1.0, area / 600.0),
            )
        )
    return sorted(output, key=lambda item: item.world_xy_mm)


def detect_digit_objects(
    image_bgr,
    *,
    calibration_matrix,
    pick_x_max_mm,
    reference_dir,
):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    bright = cv2.inRange(gray, 180, 255)
    contours, _ = cv2.findContours(
        bright,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    references = {
        digit: cv2.imread(
            str(Path(reference_dir) / f"{digit}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        for digit in (1, 2, 3)
    }
    if any(image is None for image in references.values()):
        raise RuntimeError("数字参考图读取失败")
    output = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width * height < 300:
            continue
        center = (x + width / 2.0, y + height / 2.0)
        world_xy = pixel_to_world(calibration_matrix, center)
        if world_xy[0] > float(pick_x_max_mm):
            continue
        crop = gray[y : y + height, x : x + width]
        normalized = cv2.resize(crop, (64, 96))
        scores = {
            digit: float(
                cv2.matchTemplate(
                    normalized,
                    reference,
                    cv2.TM_CCOEFF_NORMED,
                )[0, 0]
            )
            for digit, reference in references.items()
        }
        digit, confidence = max(scores.items(), key=lambda item: item[1])
        output.append((digit, world_xy, confidence))
    return sorted(output, key=lambda item: item[0])


def pick_and_place(
    ctx,
    *,
    pick_xy,
    drop_xyz,
    pick_z_mm,
    safe_z_mm,
    speed,
):
    x_mm, y_mm = (float(pick_xy[0]), float(pick_xy[1]))
    drop_x, drop_y, drop_z = (
        float(drop_xyz[0]),
        float(drop_xyz[1]),
        float(drop_xyz[2]),
    )
    ctx.robot.move_world(x_mm, y_mm, safe_z_mm, speed=speed)
    ctx.robot.move_world(x_mm, y_mm, pick_z_mm, speed=8)
    ctx.tool.on()
    ctx.robot.move_world(x_mm, y_mm, safe_z_mm, speed=speed)
    ctx.robot.move_world(drop_x, drop_y, safe_z_mm, speed=speed)
    ctx.robot.move_world(drop_x, drop_y, drop_z, speed=8)
    ctx.tool.off()
    ctx.robot.move_world(drop_x, drop_y, safe_z_mm, speed=speed)
```

Create both package initializers:

```python
"""Student-facing example programs for the robot curriculum."""
```

- [x] **Step 5: 实现 R1-01 和 R1-02 模板**

Create `student_programs/templates/r1_01_robot_basics.py`:

```python
def main(ctx):
    info = ctx.experiment.info()
    ctx.log(f"实验 {info['experiment_id']} 开始")
    ctx.robot.home()
    home_pose = ctx.robot.pose()
    ctx.log(f"回零后 TCP={home_pose}")
    observation = info["public_parameters"]["observation_pose_mm"]
    ctx.robot.move_world(*observation, speed=12)
    ctx.checkpoint("观察六个关节和 TCP")
    ctx.robot.home()
    ctx.log("实验结束；硬件状态保持 PENDING_HARDWARE")
```

Create `student_programs/templates/r1_02_teach_points.py`:

```python
def main(ctx):
    parameters = ctx.experiment.info()["public_parameters"]
    ctx.robot.home()
    for index, point in enumerate(parameters["teach_points_mm"], start=1):
        ctx.robot.move_world(*point, speed=12)
        ctx.log(f"到达示教点 {index}: {ctx.robot.pose()}")
        ctx.checkpoint(f"示教点 {index}")
    ctx.robot.home()
```

- [x] **Step 6: 实现三个视觉闭环模板**

Create `student_programs/templates/r1_05_visual_stacking.py`:

```python
from student_programs.templates.r1_common import (
    detect_colored_objects,
    pick_and_place,
)


def main(ctx):
    parameters = ctx.experiment.info()["public_parameters"]
    frame = ctx.camera.capture()
    objects = detect_colored_objects(
        frame.image_bgr,
        calibration_matrix=parameters["calibration_matrix"],
        pick_x_max_mm=parameters["pick_region_world_mm"]["x_max"],
    )
    if len(objects) != 6:
        raise RuntimeError(f"期望识别 6 个码垛物体，实际 {len(objects)} 个")
    ctx.robot.home()
    for index, (item, slot) in enumerate(
        zip(objects, parameters["stack_slots_mm"]),
        start=1,
    ):
        pick_and_place(
            ctx,
            pick_xy=item.world_xy_mm,
            drop_xyz=slot,
            pick_z_mm=parameters["pick_z_mm"],
            safe_z_mm=parameters["safe_z_mm"],
            speed=12,
        )
        ctx.checkpoint(f"完成第 {index} 块码垛")
    ctx.robot.home()
```

Create `student_programs/templates/r1_06_digit_sort.py`:

```python
from student_programs.templates.r1_common import (
    detect_digit_objects,
    pick_and_place,
)


def main(ctx):
    parameters = ctx.experiment.info()["public_parameters"]
    frame = ctx.camera.capture()
    detected = detect_digit_objects(
        frame.image_bgr,
        calibration_matrix=parameters["calibration_matrix"],
        pick_x_max_mm=parameters["pick_region_world_mm"]["x_max"],
        reference_dir=parameters["digit_reference_dir"],
    )
    by_digit = {digit: (xy, confidence) for digit, xy, confidence in detected}
    order = parameters["order"]
    if set(by_digit) != set(order):
        raise RuntimeError(f"数字识别不完整：{sorted(by_digit)}")
    ctx.robot.home()
    for index, digit in enumerate(order):
        xy, confidence = by_digit[digit]
        ctx.log(f"数字 {digit}，匹配置信度 {confidence:.3f}")
        pick_and_place(
            ctx,
            pick_xy=xy,
            drop_xyz=parameters["drop_slots_mm"][index],
            pick_z_mm=parameters["pick_z_mm"],
            safe_z_mm=parameters["safe_z_mm"],
            speed=12,
        )
    ctx.robot.home()
```

Create `student_programs/templates/r1_07_component_sort.py`:

```python
from student_programs.templates.r1_common import (
    detect_colored_objects,
    pick_and_place,
)


def _class_id(item):
    return f"{item.color}_{item.shape}"


def main(ctx):
    parameters = ctx.experiment.info()["public_parameters"]
    frame = ctx.camera.capture()
    objects = detect_colored_objects(
        frame.image_bgr,
        calibration_matrix=parameters["calibration_matrix"],
        pick_x_max_mm=parameters["pick_region_world_mm"]["x_max"],
    )
    routes = parameters["drop_poses_mm"]
    detected_ids = {_class_id(item) for item in objects}
    if detected_ids != set(parameters["classes"]):
        raise RuntimeError(f"构件类别不完整：{sorted(detected_ids)}")
    ctx.robot.home()
    for item in objects:
        class_id = _class_id(item)
        pick_and_place(
            ctx,
            pick_xy=item.world_xy_mm,
            drop_xyz=routes[class_id],
            pick_z_mm=parameters["pick_z_mm"],
            safe_z_mm=parameters["safe_z_mm"],
            speed=12,
        )
        ctx.checkpoint(f"完成 {class_id} 分仓")
    ctx.robot.home()
```

- [x] **Step 7: 创建五份完整实验说明**

每份说明必须使用 Step 2 的九个固定二级标题，并写入下列实际内容：

| 文件 | 实验任务 | 原理 | 完成条件 | 人工验收点 |
|---|---|---|---|---|
| `R1-01.md` | 识别六个关节、TCP、工作空间并执行回零和观察位移动 | 关节链、TCP、世界坐标、position-only | 程序 PASS，初末探针存在，回到 home | 学生现场指出关节与 TCP，解释 `PENDING_HARDWARE` |
| `R1-02.md` | 顺序运行四个示教点并使用暂停、继续和单步 | 示教点、安全高度、命令边界 | 四个 checkpoint、无越界拒绝、安全收尾 | 学生解释为何低空不能横移 |
| `R1-05.md` | 从快照识别六块物体并形成两列三层 | HSV 分割、像素到世界坐标、分层放置 | 程序 PASS，终态探针 `matched=6` | 学生解释检测参数与码垛顺序 |
| `R1-06.md` | 识别数字 1、2、3 并按升序放入三个槽位 | 归一化、模板匹配、业务排序 | 三个数字均识别，终态探针 `matched=3` | 学生切换为倒序并说明规则变化 |
| `R1-07.md` | 识别红/蓝方块和绿/黄圆柱并分类 | HSV 颜色、轮廓圆度、类别到仓位映射 | 四类齐全，终态探针 `matched=4` | 学生说明空结果或低置信度为何必须停止 |

“实验入口”统一写明 PyQt 实验目录和命令行 `experiment-run`；“结果保存”统一说明 `source.py`、`manifest.json`、`snapshots.jsonl`、`scene-initial.json`、`scene-final.json`、`summary.json`；“仿真与真机差异”必须逐份包含以下原句：

```text
本实验当前只完成软件与 CoppeliaSim 仿真验证，硬件状态为 PENDING_HARDWARE。仿真 PASS 不代表真机验收通过；真实相机噪声、机械臂误差、急停、气路和物理抓取需在 V2.3 单独验收。
```

- [x] **Step 8: 运行模板、说明和校验器回归**

Run:

```powershell
python -m pytest `
  tests/test_experiments/test_student_templates.py `
  tests/test_acceptance/test_experiment_guides.py `
  tests/test_student_programs/test_validator.py `
  -q
```

Expected: 全部 PASS，五个模板均满足 V2.1 学生程序契约。

- [x] **Step 9: 提交学生材料**

Run:

```powershell
git add `
  student_programs `
  docs/experiments `
  tests/test_experiments/test_student_templates.py `
  tests/test_acceptance/test_experiment_guides.py
git commit -m "feat(curriculum): add five R1 student labs"
```

### Task 13: 增加实验目录 CLI 和 PowerShell 入口

**Files:**
- Modify: `vision_platform/cli.py`
- Create: `tools/vision_lab/run_experiment.ps1`
- Create: `tests/test_experiments/test_cli.py`
- Modify: `tests/test_acceptance/test_delivery_contract.py`

- [ ] **Step 1: 写 CLI 解析和只读输出失败测试**

Create `tests/test_experiments/test_cli.py`:

```python
from __future__ import annotations

import json

from vision_platform.cli import build_parser, main


def test_experiment_run_parser_uses_catalog_id_not_arbitrary_scene():
    args = build_parser().parse_args(
        [
            "experiment-run",
            "--experiment",
            "R1-05",
            "--program",
            "student_programs/my_stack.py",
        ]
    )

    assert args.experiment == "R1-05"
    assert args.program == "student_programs/my_stack.py"
    assert not hasattr(args, "robot")
    assert not hasattr(args, "scene")


def test_experiment_list_prints_five_formal_items(capsys):
    code = main(["experiment-list"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["status"] == "PASS"
    assert [item["experiment_id"] for item in payload["experiments"]] == [
        "R1-01",
        "R1-02",
        "R1-05",
        "R1-06",
        "R1-07",
    ]
    assert all(
        item["hardware_status"] == "PENDING_HARDWARE"
        for item in payload["experiments"]
    )


def test_experiment_show_does_not_claim_hardware_pass(capsys):
    code = main(["experiment-show", "--experiment", "R1-07"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["experiment_id"] == "R1-07"
    assert payload["hardware_status"] == "PENDING_HARDWARE"
```

- [ ] **Step 2: 运行测试并确认先失败**

Run:

```powershell
python -m pytest tests/test_experiments/test_cli.py -q
```

Expected: FAIL，指出三个实验子命令不存在。

- [ ] **Step 3: 增加目录装载和只读处理函数**

Add to `vision_platform/cli.py`:

```python
def _experiment_catalog():
    from vision_platform.experiments.catalog import ExperimentCatalog

    return ExperimentCatalog.load(
        PROJECT_ROOT / "config" / "experiments" / "catalog.json",
        project_root=PROJECT_ROOT,
    )


def _experiment_list(args: argparse.Namespace) -> int:
    catalog = _experiment_catalog()
    payload = {
        "status": "PASS",
        "experiments": [
            {
                "experiment_id": item.experiment_id,
                "title": item.title,
                "version": item.version,
                "capabilities": list(item.capabilities),
                "hardware_status": item.hardware_status,
            }
            for item in catalog.definitions
        ],
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _experiment_show(args: argparse.Namespace) -> int:
    item = _experiment_catalog().require(args.experiment)
    payload = {
        "status": "PASS",
        "experiment_id": item.experiment_id,
        "title": item.title,
        "version": item.version,
        "scene": item.scene.relative_to(PROJECT_ROOT).as_posix(),
        "student_template": item.student_template.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "guide": item.guide.relative_to(PROJECT_ROOT).as_posix(),
        "capabilities": list(item.capabilities),
        "automated_checks": list(item.acceptance.automated_checks),
        "human_checks": list(item.acceptance.human_checks),
        "hardware_status": item.hardware_status,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0
```

- [ ] **Step 4: 实现实验运行处理函数**

Add to `vision_platform/cli.py`:

```python
def _experiment_run(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from vision_platform.application import VisionLabApplication
    from vision_platform.config import load_config
    from vision_platform.experiments.session import ExperimentSession
    from vision_platform.session import VisionLabSession
    from vision_platform.student.runner import StudentProgramController
    from vision_platform.student.safety import StudentExecutionPolicy

    catalog = _experiment_catalog()
    definition = catalog.require(args.experiment)
    environ = dict(os.environ)
    environ["ROBOT_BACKEND"] = "sim"
    environ["VISION_BACKEND"] = "sim"
    environ["COPPELIA_HOST"] = args.host
    environ["COPPELIA_PORT"] = str(args.port)
    base_config = load_config(
        args.config,
        project_root=PROJECT_ROOT,
        environ=environ,
    )
    base_application = VisionLabApplication.from_config(base_config)
    vision_session = VisionLabSession(
        application=base_application,
        factory=lambda: VisionLabApplication.from_config(base_config),
    )
    controller = None
    experiment_session = ExperimentSession(
        catalog=catalog,
        vision_session=vision_session,
        base_config=base_config,
        application_factory=VisionLabApplication.from_config,
        student_is_idle=lambda: (
            controller is None or not controller.process_is_alive
        ),
    )
    try:
        context = experiment_session.select(args.experiment)
        selected_config = vision_session.application.config
        student = selected_config.student
        speed_range = student["speed_range"]
        policy = StudentExecutionPolicy(
            min_speed=float(speed_range[0]),
            max_speed=float(speed_range[1]),
            max_runtime_s=float(student["max_runtime_s"]),
            max_commands=int(student["max_commands"]),
            command_timeout_s=float(student["command_timeout_s"]),
            max_sleep_s=float(student["max_sleep_s"]),
            tool_on_max_z_mm=float(student["tool_on_max_z_mm"]),
        )
        scene_manifest = json.loads(
            definition.scene_manifest.read_text(encoding="utf-8")
        )
        controller = StudentProgramController(
            session=vision_session,
            execution_policy=policy,
            output_root=args.output,
            experiment_context=context,
            experiment_definition=definition,
            scene_manifest=scene_manifest,
        )
        program = (
            Path(args.program).expanduser()
            if args.program
            else definition.student_template
        )
        controller.load(program)
        validation = controller.validate()
        if not validation.ok:
            print(
                json.dumps(
                    {
                        "status": "FAIL",
                        "issues": [
                            asdict(issue) for issue in validation.issues
                        ],
                        "hardware_status": "PENDING_HARDWARE",
                    },
                    ensure_ascii=False,
                )
            )
            return 2
        controller.start()
        result = controller.wait(
            timeout_s=policy.max_runtime_s
            + policy.command_timeout_s
            + 5
        )
        summary = json.loads(
            result.summary_path.read_text(encoding="utf-8")
        )
        print(
            json.dumps(
                {
                    "status": result.status,
                    "experiment_id": definition.experiment_id,
                    "scene_probe_status": summary.get(
                        "scene_probe_status"
                    ),
                    "summary": str(result.summary_path),
                    "evidence": str(result.evidence_dir),
                    "error": result.error,
                    "hardware_status": "PENDING_HARDWARE",
                },
                ensure_ascii=False,
            )
        )
        return 0 if result.status == "PASS" else 1
    finally:
        if controller is not None and controller.process_is_alive:
            controller.cancel()
        vision_session.close()
```

- [ ] **Step 5: 注册三个 CLI 子命令**

Inside `build_parser()` add:

```python
    experiment_list = subparsers.add_parser(
        "experiment-list",
        help="List formal CoppeliaSim curriculum experiments",
    )
    experiment_list.set_defaults(handler=_experiment_list)

    experiment_show = subparsers.add_parser(
        "experiment-show",
        help="Show one formal curriculum experiment",
    )
    experiment_show.add_argument("--experiment", required=True)
    experiment_show.set_defaults(handler=_experiment_show)

    experiment_run = subparsers.add_parser(
        "experiment-run",
        help="Run one formal experiment with the guarded student runner",
    )
    experiment_run.add_argument("--experiment", required=True)
    experiment_run.add_argument("--program")
    experiment_run.add_argument("--config")
    experiment_run.add_argument("--host", default="127.0.0.1")
    experiment_run.add_argument("--port", type=int, default=23000)
    experiment_run.add_argument(
        "--output",
        default="artifacts/vision_lab/experiment-runs",
    )
    experiment_run.set_defaults(handler=_experiment_run)
```

- [ ] **Step 6: 创建 PowerShell 入口**

Create `tools/vision_lab/run_experiment.ps1`:

```powershell
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('R1-01', 'R1-02', 'R1-05', 'R1-06', 'R1-07')]
    [string]$Experiment,
    [string]$Program,
    [string]$HostName = '127.0.0.1',
    [int]$Port = 23000,
    [string]$Output = 'artifacts/vision_lab/experiment-runs'
)

$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$PythonWrapper = Join-Path $Project 'tools\vision_lab\python.ps1'
$Arguments = @(
    '-m', 'vision_platform.cli',
    'experiment-run',
    '--experiment', $Experiment,
    '--host', $HostName,
    '--port', $Port,
    '--output', $Output
)
if ($Program) {
    $Arguments += @('--program', $Program)
}
& powershell -ExecutionPolicy Bypass -File $PythonWrapper @Arguments
exit $LASTEXITCODE
```

- [ ] **Step 7: 扩展发布入口合同**

Append to `tests/test_acceptance/test_delivery_contract.py`:

```python
def test_delivery_has_formal_experiment_cli_and_powershell_entry():
    script = ROOT / "tools" / "vision_lab" / "run_experiment.ps1"
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "experiment-run" in text
    assert "R1-01" in text
    assert "R1-07" in text
    assert "--robot" not in text
```

- [ ] **Step 8: 运行 CLI 和发布合同测试**

Run:

```powershell
python -m pytest `
  tests/test_experiments/test_cli.py `
  tests/test_acceptance/test_delivery_contract.py `
  -q
```

Expected: 全部 PASS。

- [ ] **Step 9: 提交 CLI 入口**

Run:

```powershell
git add `
  vision_platform/cli.py `
  tools/vision_lab/run_experiment.ps1 `
  tests/test_experiments/test_cli.py `
  tests/test_acceptance/test_delivery_contract.py
git commit -m "feat(cli): run formal robot curriculum experiments"
```

### Task 14: 在 PyQt 中增加实验目录页

**Files:**
- Create: `vision_platform/ui/experiment_catalog_panel.py`
- Modify: `vision_platform/ui/student_program_panel.py`
- Modify: `vision_platform/ui/pyqt_app.py`
- Modify: `vision_platform/student/runner.py`
- Create: `tests/test_vision_platform/test_experiment_catalog_panel.py`
- Modify: `tests/test_vision_platform/test_student_program_panel.py`
- Modify: `tests/test_vision_platform/test_pyqt_smoke.py`

- [ ] **Step 1: 写目录面板失败测试**

Create `tests/test_vision_platform/test_experiment_catalog_panel.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

from vision_platform.ui.experiment_catalog_panel import ExperimentCatalogPanel


class FakeCatalog:
    definitions = (
        SimpleNamespace(
            experiment_id="R1-01",
            title="机械臂认知和基础操作",
            capabilities=("robot.home", "robot.pose"),
            hardware_status="PENDING_HARDWARE",
            acceptance=SimpleNamespace(
                human_checks=("学生指出六个关节",),
            ),
        ),
        SimpleNamespace(
            experiment_id="R1-05",
            title="基于视觉的物体码垛",
            capabilities=("camera.rgb", "tool.suction"),
            hardware_status="PENDING_HARDWARE",
            acceptance=SimpleNamespace(
                human_checks=("学生解释码垛顺序",),
            ),
        ),
    )

    def require(self, experiment_id):
        return next(
            item
            for item in self.definitions
            if item.experiment_id == experiment_id
        )


def test_catalog_panel_shows_hardware_boundary_and_selects(qtbot):
    selected = []
    panel = ExperimentCatalogPanel(
        catalog=FakeCatalog(),
        on_select=selected.append,
    )
    qtbot.addWidget(panel)

    assert panel.experiment_combo.count() == 2
    assert "PENDING_HARDWARE" in panel.hardware_label.text()
    panel.experiment_combo.setCurrentIndex(1)
    panel.select_button.click()

    assert selected == ["R1-05"]
    assert "相机" in panel.capabilities_label.text()
```

- [ ] **Step 2: 写控制器重新绑定失败测试**

Append to `tests/test_student_programs/test_runner.py`:

```python
def test_bind_experiment_clears_previous_program_when_idle(tmp_path):
    controller, _, program = make_controller(
        tmp_path,
        "def main(ctx):\n    ctx.robot.home()\n",
    )

    controller.bind_experiment(
        context=SimpleNamespace(experiment_id="R1-01"),
        definition=SimpleNamespace(experiment_id="R1-01"),
        scene_manifest={"scene_id": "robot-basics"},
    )

    assert controller.state is RunState.EMPTY
    assert controller.program_path is None


def test_bind_experiment_is_rejected_while_running(tmp_path):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    while True:\n        pass\n",
        max_runtime_s=1,
    )
    assert controller.validate().ok is True
    controller.start()
    wait_until(lambda: controller.process_is_alive)

    with pytest.raises(RuntimeError, match="运行期间"):
        controller.bind_experiment(
            context=SimpleNamespace(experiment_id="R1-01"),
            definition=SimpleNamespace(experiment_id="R1-01"),
            scene_manifest={"scene_id": "robot-basics"},
        )
    controller.cancel()
    controller.wait(timeout_s=3)
```

- [ ] **Step 3: 运行新增 UI 测试并确认先失败**

Run:

```powershell
python -m pytest `
  tests/test_vision_platform/test_experiment_catalog_panel.py `
  tests/test_student_programs/test_runner.py `
  -q
```

Expected: FAIL，指出面板或 `bind_experiment` 不存在。

- [ ] **Step 4: 实现实验目录面板**

Create `vision_platform/ui/experiment_catalog_panel.py`:

```python
from __future__ import annotations

from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


_CAPABILITY_LABELS = {
    "robot.home": "机械臂回零",
    "robot.pose": "TCP 查询",
    "robot.move_world": "世界坐标运动",
    "tool.suction": "吸盘",
    "camera.rgb": "RGB 相机",
    "experiment.info": "实验参数",
    "scene.probe": "场景探针",
}


class ExperimentCatalogPanel(QWidget):
    def __init__(self, *, catalog, on_select, parent=None):
        super().__init__(parent)
        self.catalog = catalog
        self.on_select = on_select
        self.experiment_combo = QComboBox()
        for item in catalog.definitions:
            self.experiment_combo.addItem(
                f"{item.experiment_id}  {item.title}",
                item.experiment_id,
            )
        self.title_label = QLabel()
        self.hardware_label = QLabel()
        self.capabilities_label = QLabel()
        self.human_checks = QTextBrowser()
        self.human_checks.setMaximumHeight(110)
        self.select_button = QPushButton("载入实验")
        self.select_button.setMinimumHeight(44)
        form = QFormLayout()
        form.addRow("实验", self.experiment_combo)
        form.addRow("名称", self.title_label)
        form.addRow("所需能力", self.capabilities_label)
        form.addRow("硬件状态", self.hardware_label)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("教师人工复核点"))
        layout.addWidget(self.human_checks)
        layout.addWidget(self.select_button)
        self.experiment_combo.currentIndexChanged.connect(self._render)
        self.select_button.clicked.connect(self._select)
        self._render()

    def _definition(self):
        return self.catalog.require(
            str(self.experiment_combo.currentData())
        )

    def _render(self):
        item = self._definition()
        self.title_label.setText(item.title)
        self.hardware_label.setText(item.hardware_status)
        self.capabilities_label.setText(
            "、".join(
                _CAPABILITY_LABELS.get(name, name)
                for name in item.capabilities
            )
        )
        self.human_checks.setPlainText(
            "\n".join(f"• {text}" for text in item.acceptance.human_checks)
        )

    def _select(self):
        self.on_select(self._definition().experiment_id)
```

- [ ] **Step 5: 实现控制器安全重新绑定**

Add this public method to `StudentProgramController`:

```python
    def bind_experiment(
        self,
        *,
        context,
        definition,
        scene_manifest,
    ) -> None:
        if self.process_is_alive or self.state in {
            RunState.RUNNING,
            RunState.PAUSED,
            RunState.RESETTING,
        }:
            raise RuntimeError("学生程序运行期间不能重新绑定实验")
        self.experiment_context = context
        self.experiment_definition = definition
        self.scene_manifest = dict(scene_manifest)
        self._program_path = None
        self._validation = None
        self._set_state(RunState.EMPTY)
```

给控制器增加只读属性：

```python
    @property
    def program_path(self):
        return self._program_path
```

- [ ] **Step 6: 让学生编程面板装载所选模板**

Add to `StudentProgramPanel`:

```python
    def load_template(self, path) -> None:
        if self.controller.process_is_alive:
            raise RuntimeError("学生程序运行期间不能切换模板")
        selected = Path(path).expanduser().resolve()
        text = selected.read_text(encoding="utf-8")
        self.editor.setPlainText(text)
        self.current_path = selected
        self.controller.load(selected)
        self._set_dirty(False)
```

该方法使用 V2.1 面板现有编辑器、路径字段和 dirty 状态方法，不弹出文件选择框。

- [ ] **Step 7: 集成到主窗口**

给 `VisionLabWindow.__init__()` 增加可选参数：

```python
        experiment_catalog=None,
        experiment_session=None,
```

在 V2.1 `student_controller` 和 `student_program_panel` 创建完成后：

```python
if experiment_catalog is not None and experiment_session is not None:
    self.experiment_catalog = experiment_catalog
    self.experiment_session = experiment_session
    self.experiment_catalog_panel = ExperimentCatalogPanel(
        catalog=experiment_catalog,
        on_select=self._select_experiment,
    )
    self.tabs.insertTab(0, self.experiment_catalog_panel, "实验目录")
```

增加槽函数：

```python
    def _select_experiment(self, experiment_id: str) -> None:
        definition = self.experiment_catalog.require(experiment_id)
        context = self.experiment_session.select(experiment_id)
        manifest = json.loads(
            definition.scene_manifest.read_text(encoding="utf-8")
        )
        self.student_controller.bind_experiment(
            context=context,
            definition=definition,
            scene_manifest=manifest,
        )
        self.student_program_panel.load_template(
            definition.student_template
        )
        self.status_label.setText(
            f"已载入 {definition.experiment_id}：{definition.title}"
        )
```

补充 `json` 和 `ExperimentCatalogPanel` 导入。V2.1 原有页面、对象名和按钮行为不变。

- [ ] **Step 8: 增加主窗口离屏回归断言**

Append to `tests/test_vision_platform/test_pyqt_smoke.py`:

```python
def test_pyqt_window_without_experiment_services_remains_compatible(qtbot):
    window = VisionLabWindow(application=DisconnectedApplication())
    qtbot.addWidget(window)

    assert not hasattr(window, "experiment_catalog_panel")
    assert window.camera_backend_combo.count() == 3
    assert window.robot_backend_combo.count() == 2
```

并在 `tests/test_vision_platform/test_student_program_panel.py` 增加 `load_template` 测试，断言 UTF-8 内容进入编辑器、控制器收到路径、dirty 为 false。

- [ ] **Step 9: 运行全部 PyQt 离屏测试**

Run:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
python -m pytest tests/test_vision_platform/test_*panel.py tests/test_vision_platform/test_pyqt_smoke.py -q
```

Expected: 全部 PASS，原视觉页和 V2.1 学生编程页测试不退化。

- [ ] **Step 10: 提交 PyQt 实验目录**

Run:

```powershell
git add `
  vision_platform/ui/experiment_catalog_panel.py `
  vision_platform/ui/student_program_panel.py `
  vision_platform/ui/pyqt_app.py `
  vision_platform/student/runner.py `
  tests/test_vision_platform `
  tests/test_student_programs/test_runner.py
git commit -m "feat(ui): add robot experiment catalog"
```

### Task 15: 增加真实 CoppeliaSim 场景与五实验在线门禁

**Files:**
- Create: `simulation/training_scenes/verify_scene.py`
- Create: `tests/test_acceptance/test_coppeliasim_training_scenes.py`
- Create: `tests/test_acceptance/test_coppeliasim_r1_experiments.py`
- Modify: `tools/vision_lab/run_acceptance.ps1`

- [ ] **Step 1: 写在线场景验证器**

Create `simulation/training_scenes/verify_scene.py`:

```python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from simulation.training_scenes.scene_contract import validate_scene_contract
from vision_platform.cameras.coppeliasim import CoppeliaSimCamera
from vision_platform.coppelia_scene import stage_scene_for_coppeliasim
from vision_platform.experiments.scene_setup import (
    LOGISTICS_GROUPS,
    activate_scene_group,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def verify_training_scene(
    scene_directory: str | Path,
    *,
    host: str,
    port: int,
    output: str | Path,
) -> dict:
    directory = Path(scene_directory).expanduser().resolve()
    spec = json.loads(
        (directory / "scene_spec.json").read_text(encoding="utf-8")
    )
    contract = validate_scene_contract(
        directory / "scene_spec.json",
        directory / "scene_manifest.json",
        project_root=PROJECT_ROOT,
    )
    report_path = Path(output).expanduser().resolve()
    evidence_dir = report_path.parent / f"{spec['scene_id']}-frames"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    client = RemoteAPIClient(host=host, port=port)
    sim = client.require("sim")
    scene = stage_scene_for_coppeliasim(
        PROJECT_ROOT / spec["output"]
    )
    if int(sim.getSimulationState()) != int(sim.simulation_stopped):
        sim.stopSimulation()
        deadline = time.monotonic() + 5.0
        while (
            int(sim.getSimulationState()) != int(sim.simulation_stopped)
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
    sim.loadScene(scene.as_posix())
    handles = {
        path: int(sim.getObject(path)) for path in spec["required_paths"]
    }
    sim.startSimulation()
    camera = CoppeliaSimCamera(
        sim=sim,
        client=client,
        sensor_path=spec["camera"]["path"],
    )
    camera.open()
    captures = []
    groups = (
        LOGISTICS_GROUPS
        if spec["scene_id"] == "logistics-lab"
        else (None,)
    )
    try:
        for index, group in enumerate(groups, start=1):
            activate_scene_group(sim, active_path=group)
            frame = camera.read(timeout_s=5.0)
            name = (
                f"group-{index}.png"
                if group is not None
                else "overview.png"
            )
            path = evidence_dir / name
            if not cv2.imwrite(str(path), frame.image_bgr):
                raise RuntimeError(f"Could not write {path}")
            captures.append(
                {
                    "group": group,
                    "path": path.relative_to(report_path.parent).as_posix(),
                    "size": [frame.width, frame.height],
                }
            )
    finally:
        camera.close()
        sim.stopSimulation()
    report = {
        "schema_version": 1,
        "status": "PASS",
        "scene_id": spec["scene_id"],
        "contract": contract,
        "required_paths": handles,
        "captures": captures,
        "hardware_status": "PENDING_HARDWARE",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-directory", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = verify_training_scene(
        args.scene_directory,
        host=args.host,
        port=args.port,
        output=args.output,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 写两个场景在线测试**

Create `tests/test_acceptance/test_coppeliasim_training_scenes.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from simulation.training_scenes.verify_scene import verify_training_scene


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.coppeliasim
@pytest.mark.parametrize(
    ("directory", "capture_count"),
    [
        ("simulation/robot_basics", 1),
        ("simulation/logistics_lab", 3),
    ],
)
def test_training_scene_loads_paths_and_captures_camera(
    directory,
    capture_count,
    tmp_path,
):
    report = verify_training_scene(
        ROOT / directory,
        host="127.0.0.1",
        port=23000,
        output=tmp_path / f"{Path(directory).name}.json",
    )

    assert report["status"] == "PASS"
    assert len(report["captures"]) == capture_count
    assert report["hardware_status"] == "PENDING_HARDWARE"
```

- [ ] **Step 3: 写五个实验端到端在线测试**

Create `tests/test_acceptance/test_coppeliasim_r1_experiments.py`:

```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.coppeliasim
@pytest.mark.parametrize(
    "experiment_id",
    ("R1-01", "R1-02", "R1-05", "R1-06", "R1-07"),
)
def test_r1_experiment_runs_through_guarded_student_process(
    experiment_id,
    tmp_path,
):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vision_platform.cli",
            "experiment-run",
            "--experiment",
            experiment_id,
            "--host",
            "127.0.0.1",
            "--port",
            "23000",
            "--output",
            str(tmp_path / "runs"),
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    summary = json.loads(
        Path(payload["summary"]).read_text(encoding="utf-8")
    )
    evidence = Path(payload["evidence"])

    assert payload["status"] == "PASS"
    assert payload["experiment_id"] == experiment_id
    assert payload["scene_probe_status"] == "PASS"
    assert payload["hardware_status"] == "PENDING_HARDWARE"
    assert summary["hardware_status"] == "PENDING_HARDWARE"
    assert (evidence / "scene-initial.json").is_file()
    assert (evidence / "scene-final.json").is_file()
    if experiment_id in {"R1-05", "R1-06", "R1-07"}:
        assert (evidence / "snapshots.jsonl").is_file()
```

- [ ] **Step 4: 运行在线测试并按真实图像调校**

Run:

```powershell
python -m pytest -m coppeliasim `
  tests/test_acceptance/test_coppeliasim_training_scenes.py `
  tests/test_acceptance/test_coppeliasim_r1_experiments.py `
  -q
```

Expected: `7 passed`，零 skip。若视觉模板识别数量、吸盘距离或最终位置失败，只允许调校以下项目并同步测试：

- 新场景规格中的原创教学物体位置、颜色、尺寸和相机参数；
- `r1_common.py` 的颜色阈值、面积阈值、圆度阈值和数字归一化；
- 五个实验 JSON 中的仿射矩阵、抓取高度和目标位置；
- 探针容差，但不得超过 `6.0 mm`。

不得修改正式 `BL23_vision_lab.ttt`、机器人资产、V2.1 运动安全范围或吸盘最大附着距离来换取 PASS。每次改动场景规格后重新运行构建器、更新场景清单并重复本步骤。

- [ ] **Step 5: 把在线门禁加入验收脚本**

在 `tools/vision_lab/run_acceptance.ps1` 原有 V2.1 在线测试之后增加：

```powershell
& $Python -m pytest -m coppeliasim `
  tests/test_acceptance/test_coppeliasim_training_scenes.py `
  tests/test_acceptance/test_coppeliasim_r1_experiments.py `
  -q
if ($LASTEXITCODE -ne 0) {
    throw 'V2.2 first-batch CoppeliaSim acceptance failed'
}
```

验收汇总 JSON 增加：

```json
{
  "v2_2_first_batch": "PASS",
  "r1_experiments": ["R1-01", "R1-02", "R1-05", "R1-06", "R1-07"],
  "hardware_status": "PENDING_HARDWARE"
}
```

- [ ] **Step 6: 提交在线门禁**

Run:

```powershell
git add `
  simulation/training_scenes/verify_scene.py `
  tests/test_acceptance/test_coppeliasim_training_scenes.py `
  tests/test_acceptance/test_coppeliasim_r1_experiments.py `
  tools/vision_lab/run_acceptance.ps1
git commit -m "test(acceptance): verify five R1 experiments live"
```

### Task 16: 发布白名单、文档、完整回归和交付

**Files:**
- Modify: `README.md`
- Modify: `RETAINED_FILES.txt`
- Modify: `docs/视觉仿真实训平台使用说明.md`
- Modify: `docs/视觉仿真实训平台自动验收报告.md`
- Modify: `tests/test_acceptance/test_delivery_contract.py`
- Evidence: `artifacts/vision_lab/v2-2-first-batch-final/`

- [ ] **Step 1: 增加正式目录可解析发布测试**

Append to `tests/test_acceptance/test_delivery_contract.py`:

```python
def test_formal_experiment_catalog_resolves_every_delivery_path():
    from vision_platform.experiments.catalog import ExperimentCatalog

    catalog = ExperimentCatalog.load(
        ROOT / "config" / "experiments" / "catalog.json",
        project_root=ROOT,
    )

    assert catalog.ids == ("R1-01", "R1-02", "R1-05", "R1-06", "R1-07")
    for item in catalog.definitions:
        assert item.scene.is_file()
        assert item.scene_manifest.is_file()
        assert item.student_template.is_file()
        assert item.guide.is_file()
        assert item.hardware_status == "PENDING_HARDWARE"
```

- [ ] **Step 2: 登记全部新增正式文件**

把本计划创建的下列路径逐项加入 `RETAINED_FILES.txt`：

```text
config/experiments/R1-01.json
config/experiments/R1-02.json
config/experiments/R1-05.json
config/experiments/R1-06.json
config/experiments/R1-07.json
config/experiments/catalog.json
docs/experiments/R1-01.md
docs/experiments/R1-02.md
docs/experiments/R1-05.md
docs/experiments/R1-06.md
docs/experiments/R1-07.md
docs/superpowers/plans/2026-07-31-robot-curriculum-v2-2-first-batch-plan.md
docs/superpowers/specs/2026-07-31-robot-curriculum-coppeliasim-roadmap-design.md
simulation/logistics_lab/__init__.py
simulation/logistics_lab/BL23_logistics_lab.ttt
simulation/logistics_lab/assets/labels/1.png
simulation/logistics_lab/assets/labels/2.png
simulation/logistics_lab/assets/labels/3.png
simulation/logistics_lab/assets/labels/manifest.json
simulation/logistics_lab/scene_manifest.json
simulation/logistics_lab/scene_spec.json
simulation/robot_basics/__init__.py
simulation/robot_basics/BL23_robot_basics.ttt
simulation/robot_basics/scene_manifest.json
simulation/robot_basics/scene_spec.json
simulation/training_scenes/__init__.py
simulation/training_scenes/build_scene.py
simulation/training_scenes/generate_labels.py
simulation/training_scenes/scene_contract.py
simulation/training_scenes/verify_scene.py
student_programs/__init__.py
student_programs/templates/__init__.py
student_programs/templates/r1_01_robot_basics.py
student_programs/templates/r1_02_teach_points.py
student_programs/templates/r1_05_visual_stacking.py
student_programs/templates/r1_06_digit_sort.py
student_programs/templates/r1_07_component_sort.py
student_programs/templates/r1_common.py
tests/test_acceptance/test_coppeliasim_r1_experiments.py
tests/test_acceptance/test_coppeliasim_training_scenes.py
tests/test_acceptance/test_experiment_guides.py
tests/test_experiments/__init__.py
tests/test_experiments/test_capabilities.py
tests/test_experiments/test_catalog.py
tests/test_experiments/test_cli.py
tests/test_experiments/test_formal_catalog.py
tests/test_experiments/test_probes.py
tests/test_experiments/test_scene_setup.py
tests/test_experiments/test_session.py
tests/test_experiments/test_student_templates.py
tests/test_simulation/test_formal_training_scenes.py
tests/test_simulation/test_training_labels.py
tests/test_simulation/test_training_scene_contract.py
tests/test_student_programs/test_experiment_evidence.py
tests/test_student_programs/test_experiment_gateway.py
tests/test_vision_platform/test_experiment_catalog_panel.py
tools/vision_lab/run_experiment.ps1
vision_platform/experiments/__init__.py
vision_platform/experiments/capabilities.py
vision_platform/experiments/catalog.py
vision_platform/experiments/models.py
vision_platform/experiments/probes.py
vision_platform/experiments/scene_setup.py
vision_platform/experiments/session.py
vision_platform/student/experiment_gateway.py
vision_platform/ui/experiment_catalog_panel.py
```

不登记 `artifacts/`、`.venv-vision/`、`__pycache__/`、`.pytest_cache/` 或学生个人程序。

- [ ] **Step 3: 更新使用说明和 README**

在 `README.md` 和 `docs/视觉仿真实训平台使用说明.md` 增加：

```powershell
powershell -ExecutionPolicy Bypass -File tools\vision_lab\python.ps1 `
  -m vision_platform.cli experiment-list

powershell -ExecutionPolicy Bypass -File tools\vision_lab\run_experiment.ps1 `
  -Experiment R1-05
```

文档明确说明：

- 先在“实验目录”选择实验，再编辑和运行模板；
- 切换实验会终止旧会话并重新加载确定场景；
- R1-05、R1-06、R1-07 的快照和终态探针位置；
- 场景终态探针不是正式成绩；
- 人工教学效果验收仍需教师完成；
- 真机、海康 MVS、急停、气路和物理抓取均为 `PENDING_HARDWARE`。

- [ ] **Step 4: 扩展发布白名单合同**

在 `test_delivery_contract.py` 增加：

```python
def test_retained_files_contains_every_first_batch_delivery():
    retained = {
        line.strip()
        for line in (ROOT / "RETAINED_FILES.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    required = {
        "config/experiments/catalog.json",
        "simulation/robot_basics/BL23_robot_basics.ttt",
        "simulation/logistics_lab/BL23_logistics_lab.ttt",
        "tools/vision_lab/run_experiment.ps1",
        "vision_platform/experiments/catalog.py",
        "vision_platform/student/experiment_gateway.py",
        "vision_platform/ui/experiment_catalog_panel.py",
    }

    assert required <= retained
    assert all((ROOT / path).is_file() for path in retained)
    assert not any(path.startswith("artifacts/") for path in retained)
```

- [ ] **Step 5: 运行完整静态回归**

Run:

```powershell
$evidence = 'artifacts/vision_lab/v2-2-first-batch-final'
New-Item -ItemType Directory -Path $evidence -Force | Out-Null
python -m pytest -q 2>&1 |
  Tee-Object -FilePath (Join-Path $evidence 'pytest-static.txt')
if ($LASTEXITCODE -ne 0) {
    throw 'Static regression failed'
}
```

Expected: 零失败；显式 CoppeliaSim 测试在静态运行中可显示 skip，但不能作为在线 PASS。

- [ ] **Step 6: 运行完整在线验收**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File tools\vision_lab\run_acceptance.ps1 `
  -Output artifacts\vision_lab\v2-2-first-batch-final
```

Expected:

- 一期标定、分类、simUI、PyQt 和场景门禁继续 PASS；
- V2.1 学生程序在线门禁 PASS；
- 两个 V2.2 场景和五个 R1 实验在线门禁实际 PASS，零 skip；
- 汇总保留 `hardware_status=PENDING_HARDWARE`。

- [ ] **Step 7: 执行 PyQt 人工视觉门禁**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File tools\vision_lab\run_pyqt.ps1
```

人工检查并把截图保存到 `artifacts/vision_lab/v2-2-first-batch-final/ui/`：

1. “实验目录”页在 100% 和 125% Windows 缩放下文字不截断；
2. 五个实验均可选择，名称、能力、人工复核点和 `PENDING_HARDWARE` 可读；
3. 载入实验后学生编程页显示正确模板；
4. 运行、暂停、继续、单步、停止和复位按钮状态正确；
5. 原标定、识别、分类和一期学生编程页没有布局退化。

人工视觉门禁只证明界面可用性，不证明教学效果或真机通过。

- [ ] **Step 8: 检查文档、编码、占位符和受保护资产**

Run:

```powershell
$documents = @(
  'docs/superpowers/specs/2026-07-31-robot-curriculum-coppeliasim-roadmap-design.md',
  'docs/superpowers/plans/2026-07-31-robot-curriculum-v2-2-first-batch-plan.md',
  'docs/视觉仿真实训平台使用说明.md',
  'docs/视觉仿真实训平台自动验收报告.md'
)
$patterns = @(
  ('T' + 'BD'),
  ('TO' + 'DO'),
  ('FIX' + 'ME'),
  ('implement' + ' later'),
  ('fill' + ' in'),
  ('待' + '实现'),
  ('待' + '定')
)
Get-Content $documents -Encoding UTF8 | Select-String -Pattern $patterns
foreach ($document in $documents) {
  $bytes = [System.IO.File]::ReadAllBytes((Resolve-Path $document))
  if (
    $bytes.Length -ge 3 -and
    $bytes[0] -eq 0xEF -and
    $bytes[1] -eq 0xBB -and
    $bytes[2] -eq 0xBF
  ) {
    throw "UTF-8 BOM is not allowed: $document"
  }
}
git diff --check
git diff --exit-code -- `
  simulation/vision_lab/BL23_vision_lab.ttt `
  robot_backends/models/BLX_openr6.ttt `
  robot_backends/models/openr6_arm_coppeliasim.urdf `
  robot_backends/models/meshes_blx `
  simulation/vision_lab/assets/robot
```

Expected: 无未解决占位符、无 UTF-8 BOM、无 diff 错误、受保护资产零差异。验收文档可以出现“真机验收未通过/待硬件”，但不得出现宣称通过的句子；若组合词搜索误报，人工核对语义并保留准确边界。

- [ ] **Step 9: 更新自动验收报告**

在 `docs/视觉仿真实训平台自动验收报告.md` 记录：

- V2.1 最终集成提交和本分支提交；
- 完整静态测试精确通过数和 skip 数；
- 两个场景、五个实验的在线精确通过数；
- 每个实验的证据目录和场景探针状态；
- 受保护资产验证结果；
- PyQt 人工视觉检查人、日期和截图路径；
- 教师教学效果验收仍待完成；
- 真机、海康 MVS、急停、气路和物理抓取继续 `PENDING_HARDWARE`。

- [ ] **Step 10: 提交最终交付**

Run:

```powershell
git add `
  README.md `
  RETAINED_FILES.txt `
  docs/视觉仿真实训平台使用说明.md `
  docs/视觉仿真实训平台自动验收报告.md `
  tests/test_acceptance/test_delivery_contract.py
git commit -m "docs: complete first V2.2 curriculum delivery"
git status --short --branch
```

Expected: 提交成功，实施分支工作树干净。不要自动合并或推送 `origin/main`，由教师审核分支、在线证据和人工 UI 门禁后决定集成。

## 完成定义

只有以下条件全部满足，开发端才可声明“首批 V2.2 软件与 CoppeliaSim 仿真实现完成”：

1. V2.1 最终静态和在线完成门禁已作为真实集成基线；
2. 正式目录精确包含 R1-01、R1-02、R1-05、R1-06、R1-07；
3. 两个新增 `.ttt` 能独立加载、复位和在线验证；
4. 原 `BL23_vision_lab.ttt`、URDF、STL 和受保护机器人资产未变化；
5. `camera.capture` 和 `experiment.info` 经过显式白名单，不绕过 V2.1 安全网关；
6. 五个模板均能通过 spawn 子进程运行并产生完整证据；
7. 三个视觉实验保存原始快照、识别运行记录、初态和终态场景探针；
8. R1-05 形成两列三层，R1-06 按数字顺序放置，R1-07 按四类路由；
9. 停止、失败和复位后吸盘关闭、学生子进程退出、场景回到确定初态；
10. CLI、PowerShell 和 PyQt 共用相同实验目录和控制器；
11. 完整静态回归零失败，在线 V2.2 七项门禁真实 PASS 而非 skip；
12. PyQt 人工视觉门禁已留截图证据；
13. 所有新增正式文件进入 `RETAINED_FILES.txt`，发布合同通过；
14. 场景探针没有被表述为自动评分或正式课程成绩；
15. 教学效果仍由教师人工验收；
16. 真机、海康 MVS、急停、气路和物理抓取始终为 `PENDING_HARDWARE`；
17. 仿真 PASS 没有被表述为真机验收通过。
