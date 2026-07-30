# Student Program Runner V2.1 Implementation Plan

**Revision:** 2026-07-31，已按 `Blinx-CoppeliaSim-clean` 干净发布基线更新。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为现有视觉仿真实训平台增加学生 Python 程序的载入、校验、受控执行、暂停、单步、停止、场景复位和证据留档能力。

**Architecture:** 学生代码在 `multiprocessing` 的 spawn 子进程中运行，只持有代理 SDK；主进程持有 `VisionLabApplication`，通过白名单协议接收命令，并在调用 `RobotAdapter` 和吸盘前执行安全检查。PyQt 和 CLI 共用同一 `StudentProgramController`，场景复位通过可替换 `VisionLabApplication` 的会话管理器完成。

**Tech Stack:** Python 3.11、标准库 `ast` / `multiprocessing` / `threading`、PyQt5、pytest、pytest-qt、CoppeliaSim ZMQ Remote API、PowerShell

---

## 0. 开发边界

- 唯一项目源：`C:\Users\yqzhe\Documents\Robot Sim\Blinx-CoppeliaSim-clean`
- 基线：`origin/main`，当前确认提交 `7546db7`
- 开发分支：`codex/student-program-runner-v2`
- 设计规格：`docs/superpowers/specs/2026-07-30-student-program-runner-v2-design.md`
- 不修改以下正式资产：
  - `simulation/vision_lab/BL23_vision_lab.ttt`
  - `robot_backends/models/BLX_openr6.ttt`
  - `robot_backends/models/openr6_arm_coppeliasim.urdf`
  - `robot_backends/models/meshes_blx/*`
  - `simulation/vision_lab/assets/robot/*`
- 正式资产来源与哈希以 `simulation/vision_lab/robot_assets_manifest.json` 和现有来源测试为准。
- 不把 `PENDING_HARDWARE` 写成 PASS。
- 不允许学生程序直接选择或创建真实机械臂后端。
- 不引入新的第三方 Python 依赖。
- 所有新增正式文件必须加入 `RETAINED_FILES.txt`。
- 不创建或引用仓库中不存在的额外项目简报文件。
- 不回到旧工作树继续开发。

### Task 1: 固化文档基线并建立开发分支

**Files:**
- Read: `README.md`
- Read: `RETAINED_FILES.txt`
- Read: `docs/superpowers/specs/2026-07-30-student-program-runner-v2-design.md`
- Read: `docs/视觉仿真实训平台自动验收报告.md`
- Read: `simulation/vision_lab/robot_assets_manifest.json`

- [ ] **Step 1: 检查干净发布基线和文档交接改动**

Run:

```powershell
$project = 'C:\Users\yqzhe\Documents\Robot Sim\Blinx-CoppeliaSim-clean'
git -C $project status --short --branch
git -C $project rev-parse HEAD
git -C $project rev-parse origin/main
git -C $project diff --name-only
git -C $project ls-files --others --exclude-standard
```

Expected:

- `HEAD` 和 `origin/main` 均为当前记录的 `7546db7`；
- 未提交内容只允许是本设计、实施方案、开发 Prompt 和 `RETAINED_FILES.txt`；
- `origin/main` 可解析，当前记录的基线为 `7546db7`；
- 若远端后来已有经教师确认的新提交，先更新文档中的基线记录再继续；
- 不执行 `reset --hard`、`checkout --` 或 `clean`。

- [ ] **Step 2: 在当前专用干净仓库创建开发分支并提交文档基线**

Run:

```powershell
$project = 'C:\Users\yqzhe\Documents\Robot Sim\Blinx-CoppeliaSim-clean'
if (git -C $project branch --list codex/student-program-runner-v2) {
    throw "Branch already exists; inspect it instead of recreating it."
}
git -C $project switch -c codex/student-program-runner-v2
git -C $project add `
  RETAINED_FILES.txt `
  docs/superpowers/specs/2026-07-30-student-program-runner-v2-design.md `
  docs/superpowers/plans/2026-07-30-student-program-runner-v2-plan.md `
  docs/视觉仿真实训平台V2-开发端执行Prompt.md
git -C $project commit -m "docs: define student program runner v2"
git -C $project status --short --branch
```

Expected: 当前专用仓库位于 `codex/student-program-runner-v2`，文档基线已提交，工作树干净。

- [ ] **Step 3: 创建项目本地环境并运行基线静态测试**

Run:

```powershell
$project = 'C:\Users\yqzhe\Documents\Robot Sim\Blinx-CoppeliaSim-clean'
Push-Location $project
try {
    powershell -ExecutionPolicy Bypass -File tools\vision_lab\bootstrap.ps1
    powershell -ExecutionPolicy Bypass -File tools\vision_lab\python.ps1 `
      -m pytest -q
} finally {
    Pop-Location
}
```

Expected: `158 passed, 2 skipped`。两个 skip 是显式隔离的 CoppeliaSim 在线测试，不能记作在线 PASS。

- [ ] **Step 4: 验证正式资产基线**

Run:

```powershell
$project = 'C:\Users\yqzhe\Documents\Robot Sim\Blinx-CoppeliaSim-clean'
Push-Location $project
try {
    powershell -ExecutionPolicy Bypass -File tools\vision_lab\python.ps1 `
      -m pytest tests/test_simulation/test_robot_asset_provenance.py -q
} finally {
    Pop-Location
}
```

Expected: 该测试文件全部 PASS，八个分段网格、对齐锚点和正式资产均匹配现有清单。

### Task 2: 定义学生协议、运行状态和错误码

**Files:**
- Create: `vision_platform/student/__init__.py`
- Create: `vision_platform/student/protocol.py`
- Test: `tests/test_student_programs/test_protocol.py`

- [ ] **Step 1: 写协议失败测试**

Create `tests/test_student_programs/test_protocol.py`:

```python
from __future__ import annotations

import pytest

from vision_platform.student.protocol import (
    ALLOWED_COMMANDS,
    CommandMessage,
    ResponseMessage,
    RunState,
)


def test_command_round_trip():
    command = CommandMessage(
        command_id="000001",
        name="robot.move_world",
        args={"x_mm": 100.0, "y_mm": 20.0, "z_mm": 120.0, "speed": 15.0},
    )

    restored = CommandMessage.from_dict(command.to_dict())

    assert restored == command
    assert restored.to_dict()["schema_version"] == 1


def test_unknown_command_is_rejected():
    with pytest.raises(ValueError, match="COMMAND_NOT_ALLOWED"):
        CommandMessage(
            command_id="000001",
            name="sim.setObjectPosition",
            args={},
        )


def test_response_round_trip_preserves_structured_error():
    response = ResponseMessage(
        command_id="000002",
        status="FAIL",
        value=None,
        error={"code": "TARGET_OUT_OF_WORKSPACE", "message": "越界"},
    )

    assert ResponseMessage.from_dict(response.to_dict()) == response


def test_state_names_are_stable():
    assert [item.value for item in RunState] == [
        "EMPTY",
        "LOADED",
        "VALIDATED",
        "RUNNING",
        "PAUSED",
        "PASSED",
        "FAILED",
        "CANCELLED",
        "RESETTING",
    ]


def test_protocol_exposes_only_student_sdk_commands():
    assert ALLOWED_COMMANDS == frozenset(
        {
            "context.log",
            "context.sleep",
            "context.checkpoint",
            "robot.home",
            "robot.move_world",
            "robot.pose",
            "tool.on",
            "tool.off",
        }
    )
```

- [ ] **Step 2: 验证测试先失败**

Run:

```powershell
python -m pytest tests/test_student_programs/test_protocol.py -q
```

Expected: FAIL，提示 `vision_platform.student` 不存在。

- [ ] **Step 3: 实现协议**

Create `vision_platform/student/protocol.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


SCHEMA_VERSION = 1
ALLOWED_COMMANDS = frozenset(
    {
        "context.log",
        "context.sleep",
        "context.checkpoint",
        "robot.home",
        "robot.move_world",
        "robot.pose",
        "tool.on",
        "tool.off",
    }
)


class RunState(str, Enum):
    EMPTY = "EMPTY"
    LOADED = "LOADED"
    VALIDATED = "VALIDATED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    PASSED = "PASSED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RESETTING = "RESETTING"


def _require_text(value: Any, field: str) -> str:
    text = str(value)
    if not text:
        raise ValueError(f"{field} must not be empty")
    return text


@dataclass(frozen=True)
class CommandMessage:
    command_id: str
    name: str
    args: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.name not in ALLOWED_COMMANDS:
            raise ValueError(f"COMMAND_NOT_ALLOWED: {self.name}")
        _require_text(self.command_id, "command_id")
        if not isinstance(self.args, Mapping):
            raise ValueError("args must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "command",
            "command_id": self.command_id,
            "name": self.name,
            "args": dict(self.args),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CommandMessage":
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("PROTOCOL_VERSION_UNSUPPORTED")
        if payload.get("kind") != "command":
            raise ValueError("PROTOCOL_KIND_INVALID")
        return cls(
            command_id=_require_text(payload.get("command_id"), "command_id"),
            name=_require_text(payload.get("name"), "name"),
            args=payload.get("args", {}),
        )


@dataclass(frozen=True)
class ResponseMessage:
    command_id: str
    status: str
    value: Any
    error: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "FAIL", "CANCELLED"}:
            raise ValueError(f"Invalid response status: {self.status}")
        if self.status == "PASS" and self.error is not None:
            raise ValueError("PASS response must not contain error")
        if self.status != "PASS" and self.error is None:
            raise ValueError("Non-PASS response must contain error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "result",
            "command_id": self.command_id,
            "status": self.status,
            "value": self.value,
            "error": dict(self.error) if self.error is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResponseMessage":
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("PROTOCOL_VERSION_UNSUPPORTED")
        if payload.get("kind") != "result":
            raise ValueError("PROTOCOL_KIND_INVALID")
        error = payload.get("error")
        return cls(
            command_id=_require_text(payload.get("command_id"), "command_id"),
            status=_require_text(payload.get("status"), "status"),
            value=payload.get("value"),
            error=dict(error) if error is not None else None,
        )
```

Create `vision_platform/student/__init__.py`:

```python
from vision_platform.student.protocol import (
    ALLOWED_COMMANDS,
    CommandMessage,
    ResponseMessage,
    RunState,
)

__all__ = [
    "ALLOWED_COMMANDS",
    "CommandMessage",
    "ResponseMessage",
    "RunState",
]
```

- [ ] **Step 4: 运行协议测试**

Run:

```powershell
python -m pytest tests/test_student_programs/test_protocol.py -q
```

Expected: `5 passed`。

- [ ] **Step 5: 提交协议**

```powershell
git add vision_platform/student tests/test_student_programs/test_protocol.py
git commit -m "feat(student): define runner protocol and states"
```

### Task 3: 实现学生程序静态校验

**Files:**
- Create: `vision_platform/student/validator.py`
- Test: `tests/test_student_programs/test_validator.py`

- [ ] **Step 1: 写校验器失败测试**

Create `tests/test_student_programs/test_validator.py`:

```python
from __future__ import annotations

from pathlib import Path

from vision_platform.student.validator import validate_program


def _write(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path


def test_valid_main_contract_passes(tmp_path):
    result = validate_program(
        _write(
            tmp_path / "ok.py",
            "def main(ctx):\n"
            "    ctx.robot.home()\n"
            "    ctx.log('完成')\n",
        )
    )

    assert result.ok is True
    assert result.issues == ()


def test_syntax_error_reports_line(tmp_path):
    result = validate_program(
        _write(tmp_path / "bad.py", "def main(ctx)\n    pass\n")
    )

    assert result.ok is False
    assert result.issues[0].code == "SYNTAX_ERROR"
    assert result.issues[0].line == 1


def test_missing_main_is_rejected(tmp_path):
    result = validate_program(_write(tmp_path / "missing.py", "VALUE = 1\n"))

    assert [item.code for item in result.issues] == ["MAIN_MISSING"]


def test_multiple_main_functions_are_rejected(tmp_path):
    result = validate_program(
        _write(
            tmp_path / "duplicate.py",
            "def main(ctx):\n    pass\n"
            "def main(ctx):\n    pass\n",
        )
    )

    assert "MAIN_DUPLICATED" in [item.code for item in result.issues]


def test_main_signature_must_have_one_required_argument(tmp_path):
    result = validate_program(
        _write(tmp_path / "signature.py", "def main(ctx, extra):\n    pass\n")
    )

    assert "MAIN_SIGNATURE_INVALID" in [item.code for item in result.issues]


def test_direct_backend_and_process_imports_are_rejected(tmp_path):
    for module in (
        "robot_backends",
        "vision_platform.application",
        "coppeliasim_zmqremoteapi_client",
        "subprocess",
        "socket",
        "ctypes",
    ):
        result = validate_program(
            _write(tmp_path / "blocked.py", f"import {module}\ndef main(ctx):\n    pass\n")
        )
        assert "IMPORT_NOT_ALLOWED" in [item.code for item in result.issues]


def test_top_level_calls_are_rejected_but_docstrings_are_allowed(tmp_path):
    result = validate_program(
        _write(
            tmp_path / "top_level.py",
            '"""学生程序。"""\n'
            "print('不能在载入时执行')\n"
            "def main(ctx):\n    pass\n",
        )
    )

    assert "TOP_LEVEL_CALL_NOT_ALLOWED" in [
        item.code for item in result.issues
    ]


def test_non_py_and_large_files_are_rejected(tmp_path):
    text = _write(tmp_path / "program.txt", "def main(ctx):\n    pass\n")
    assert validate_program(text).issues[0].code == "FILE_EXTENSION_INVALID"

    large = tmp_path / "large.py"
    large.write_bytes(b"#" * (256 * 1024 + 1))
    assert validate_program(large).issues[0].code == "FILE_TOO_LARGE"
```

- [ ] **Step 2: 验证测试先失败**

Run:

```powershell
python -m pytest tests/test_student_programs/test_validator.py -q
```

Expected: FAIL，提示 `validator` 模块不存在。

- [ ] **Step 3: 实现校验器**

Create `vision_platform/student/validator.py`:

```python
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


MAX_PROGRAM_BYTES = 256 * 1024
BLOCKED_IMPORTS = (
    "robot_backends",
    "vision_platform.application",
    "coppeliasim_zmqremoteapi_client",
    "subprocess",
    "socket",
    "ctypes",
)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    line: int | None = None
    column: int | None = None


@dataclass(frozen=True)
class ValidationResult:
    path: Path
    ok: bool
    issues: tuple[ValidationIssue, ...]


def _blocked(module: str) -> bool:
    return any(
        module == prefix or module.startswith(prefix + ".")
        for prefix in BLOCKED_IMPORTS
    )


def validate_program(path: str | Path) -> ValidationResult:
    selected = Path(path).expanduser().resolve()
    issues: list[ValidationIssue] = []
    if selected.suffix.lower() != ".py":
        issues.append(
            ValidationIssue(
                "FILE_EXTENSION_INVALID",
                "学生程序必须使用 .py 扩展名",
            )
        )
        return ValidationResult(selected, False, tuple(issues))
    if not selected.is_file():
        issues.append(
            ValidationIssue("FILE_NOT_FOUND", f"文件不存在：{selected}")
        )
        return ValidationResult(selected, False, tuple(issues))
    if selected.stat().st_size > MAX_PROGRAM_BYTES:
        issues.append(
            ValidationIssue(
                "FILE_TOO_LARGE",
                "学生程序不得超过 256 KiB",
            )
        )
        return ValidationResult(selected, False, tuple(issues))
    try:
        source = selected.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        issues.append(
            ValidationIssue(
                "FILE_ENCODING_INVALID",
                "学生程序必须是 UTF-8 编码",
            )
        )
        return ValidationResult(selected, False, tuple(issues))
    try:
        tree = ast.parse(source, filename=str(selected))
    except SyntaxError as error:
        issues.append(
            ValidationIssue(
                "SYNTAX_ERROR",
                error.msg,
                line=error.lineno,
                column=error.offset,
            )
        )
        return ValidationResult(selected, False, tuple(issues))

    main_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    ]
    if not main_nodes:
        issues.append(
            ValidationIssue("MAIN_MISSING", "必须定义 main(ctx)")
        )
    elif len(main_nodes) > 1:
        issues.append(
            ValidationIssue(
                "MAIN_DUPLICATED",
                "只能定义一个顶层 main(ctx)",
                line=main_nodes[1].lineno,
            )
        )
    else:
        main = main_nodes[0]
        positional = [*main.args.posonlyargs, *main.args.args]
        required_count = len(positional) - len(main.args.defaults)
        valid = (
            not isinstance(main, ast.AsyncFunctionDef)
            and len(positional) == 1
            and required_count == 1
            and main.args.vararg is None
            and main.args.kwarg is None
            and not main.args.kwonlyargs
        )
        if not valid:
            issues.append(
                ValidationIssue(
                    "MAIN_SIGNATURE_INVALID",
                    "main 必须是同步函数，并且只接收一个必选参数 ctx",
                    line=main.lineno,
                )
            )

    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        for module in modules:
            if _blocked(module):
                issues.append(
                    ValidationIssue(
                        "IMPORT_NOT_ALLOWED",
                        f"禁止导入：{module}",
                        line=getattr(node, "lineno", None),
                    )
                )

    for node in tree.body:
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, str
            ):
                continue
            if isinstance(node.value, (ast.Call, ast.Await)):
                issues.append(
                    ValidationIssue(
                        "TOP_LEVEL_CALL_NOT_ALLOWED",
                        "载入文件时不得直接执行顶层调用",
                        line=node.lineno,
                    )
                )

    return ValidationResult(selected, not issues, tuple(issues))
```

- [ ] **Step 4: 运行校验器测试**

Run:

```powershell
python -m pytest tests/test_student_programs/test_validator.py -q
```

Expected: `8 passed`。

- [ ] **Step 5: 提交校验器**

```powershell
git add vision_platform/student/validator.py tests/test_student_programs/test_validator.py
git commit -m "feat(student): validate student program contract"
```

### Task 4: 实现学生 SDK 和子进程 Worker

**Files:**
- Create: `vision_platform/student/sdk.py`
- Create: `vision_platform/student/worker.py`
- Test: `tests/test_student_programs/test_sdk.py`
- Test: `tests/test_student_programs/test_worker.py`

- [ ] **Step 1: 写 SDK 失败测试**

Create `tests/test_student_programs/test_sdk.py`:

```python
from __future__ import annotations

from vision_platform.student.protocol import ResponseMessage
from vision_platform.student.sdk import StudentContext


class FakeConnection:
    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)

    def recv(self):
        command_id = self.sent[-1]["command_id"]
        name = self.sent[-1]["name"]
        value = [100.0, 20.0, 120.0] if name == "robot.pose" else None
        return ResponseMessage(
            command_id=command_id,
            status="PASS",
            value=value,
            error=None,
        ).to_dict()


def test_student_context_emits_whitelisted_commands():
    connection = FakeConnection()
    ctx = StudentContext(connection)

    ctx.log("开始")
    ctx.robot.home()
    ctx.robot.move_world(100, 20, 120, speed=15)
    pose = ctx.robot.pose()
    ctx.tool.on()
    ctx.tool.off()

    assert [item["name"] for item in connection.sent] == [
        "context.log",
        "robot.home",
        "robot.move_world",
        "robot.pose",
        "tool.on",
        "tool.off",
    ]
    assert pose == (100.0, 20.0, 120.0)


def test_failed_response_becomes_student_runtime_error():
    connection = FakeConnection()

    def failed_recv():
        command_id = connection.sent[-1]["command_id"]
        return ResponseMessage(
            command_id=command_id,
            status="FAIL",
            value=None,
            error={"code": "TARGET_OUT_OF_WORKSPACE", "message": "目标越界"},
        ).to_dict()

    connection.recv = failed_recv
    ctx = StudentContext(connection)

    try:
        ctx.robot.move_world(500, 0, 20, speed=15)
    except RuntimeError as error:
        assert "TARGET_OUT_OF_WORKSPACE" in str(error)
    else:
        raise AssertionError("Expected student SDK to raise")
```

- [ ] **Step 2: 写 Worker 失败测试**

Create `tests/test_student_programs/test_worker.py`:

```python
from __future__ import annotations

from multiprocessing import Pipe
from pathlib import Path
from threading import Thread

from vision_platform.student.protocol import ResponseMessage
from vision_platform.student.worker import run_student_worker


def test_worker_imports_program_and_reports_completion(tmp_path):
    program = tmp_path / "program.py"
    program.write_text(
        "def main(ctx):\n"
        "    ctx.log('worker-ok')\n"
        "    ctx.robot.home()\n",
        encoding="utf-8",
    )
    parent, child = Pipe(duplex=True)
    result = {}

    def worker():
        result.update(run_student_worker(program, child))

    thread = Thread(target=worker)
    thread.start()
    names = []
    while thread.is_alive():
        if parent.poll(0.05):
            command = parent.recv()
            names.append(command["name"])
            parent.send(
                ResponseMessage(
                    command_id=command["command_id"],
                    status="PASS",
                    value=None,
                    error=None,
                ).to_dict()
            )
    thread.join(timeout=1)

    assert result["status"] == "PASS"
    assert names == ["context.log", "robot.home"]


def test_worker_reports_uncaught_exception(tmp_path):
    program = tmp_path / "broken.py"
    program.write_text(
        "def main(ctx):\n    raise ValueError('student-error')\n",
        encoding="utf-8",
    )
    parent, child = Pipe(duplex=True)

    result = run_student_worker(program, child)

    assert result["status"] == "FAIL"
    assert result["error"]["code"] == "STUDENT_PROGRAM_FAILED"
    assert "student-error" in result["error"]["message"]
```

- [ ] **Step 3: 验证 SDK 和 Worker 测试先失败**

Run:

```powershell
python -m pytest `
  tests/test_student_programs/test_sdk.py `
  tests/test_student_programs/test_worker.py `
  -q
```

Expected: FAIL，提示模块不存在。

- [ ] **Step 4: 实现 SDK**

Create `vision_platform/student/sdk.py`:

```python
from __future__ import annotations

from itertools import count
from typing import Any

from vision_platform.student.protocol import CommandMessage, ResponseMessage


class _Rpc:
    def __init__(self, connection) -> None:
        self.connection = connection
        self._ids = count(1)

    def call(self, name: str, **args: Any) -> Any:
        command_id = f"{next(self._ids):06d}"
        command = CommandMessage(command_id, name, args)
        self.connection.send(command.to_dict())
        response = ResponseMessage.from_dict(self.connection.recv())
        if response.command_id != command_id:
            raise RuntimeError("PROTOCOL_COMMAND_ID_MISMATCH")
        if response.status != "PASS":
            assert response.error is not None
            raise RuntimeError(
                f"{response.error.get('code', 'COMMAND_FAILED')}: "
                f"{response.error.get('message', '命令失败')}"
            )
        return response.value


class StudentRobot:
    def __init__(self, rpc: _Rpc) -> None:
        self._rpc = rpc

    def home(self) -> None:
        self._rpc.call("robot.home")

    def move_world(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        *,
        speed: float,
    ) -> None:
        self._rpc.call(
            "robot.move_world",
            x_mm=float(x_mm),
            y_mm=float(y_mm),
            z_mm=float(z_mm),
            speed=float(speed),
        )

    def pose(self) -> tuple[float, float, float]:
        value = self._rpc.call("robot.pose")
        return (float(value[0]), float(value[1]), float(value[2]))


class StudentTool:
    def __init__(self, rpc: _Rpc) -> None:
        self._rpc = rpc

    def on(self) -> None:
        self._rpc.call("tool.on")

    def off(self) -> None:
        self._rpc.call("tool.off")


class StudentContext:
    def __init__(self, connection) -> None:
        self._rpc = _Rpc(connection)
        self.robot = StudentRobot(self._rpc)
        self.tool = StudentTool(self._rpc)

    def log(self, message: str) -> None:
        self._rpc.call("context.log", message=str(message))

    def sleep(self, seconds: float) -> None:
        self._rpc.call("context.sleep", seconds=float(seconds))

    def checkpoint(self, label: str) -> None:
        self._rpc.call("context.checkpoint", label=str(label))
```

- [ ] **Step 5: 实现 Worker**

Create `vision_platform/student/worker.py`:

```python
from __future__ import annotations

import importlib.util
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

from vision_platform.student.sdk import StudentContext


def _load_module(path: Path) -> ModuleType:
    name = f"student_program_{path.stem}_{abs(hash(path))}"
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"无法载入学生程序：{path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def run_student_worker(path: str | Path, connection) -> dict[str, Any]:
    selected = Path(path).expanduser().resolve()
    try:
        module = _load_module(selected)
        entry = getattr(module, "main")
        entry(StudentContext(connection))
        return {"status": "PASS", "error": None}
    except (EOFError, BrokenPipeError):
        return {
            "status": "CANCELLED",
            "error": {
                "code": "STUDENT_PROGRAM_CANCELLED",
                "message": "学生程序通信已取消",
            },
        }
    except BaseException as error:
        return {
            "status": "FAIL",
            "error": {
                "code": "STUDENT_PROGRAM_FAILED",
                "message": str(error),
                "type": type(error).__name__,
                "traceback": traceback.format_exc(),
            },
        }
    finally:
        try:
            connection.close()
        except Exception:
            pass
```

- [ ] **Step 6: 运行 SDK 和 Worker 测试**

Run:

```powershell
python -m pytest `
  tests/test_student_programs/test_sdk.py `
  tests/test_student_programs/test_worker.py `
  -q
```

Expected: `4 passed`。

- [ ] **Step 7: 提交 SDK 和 Worker**

```powershell
git add `
  vision_platform/student/sdk.py `
  vision_platform/student/worker.py `
  tests/test_student_programs/test_sdk.py `
  tests/test_student_programs/test_worker.py
git commit -m "feat(student): add proxy sdk and worker"
```

### Task 5: 实现学生动作安全网关

**Files:**
- Create: `vision_platform/student/safety.py`
- Modify: `vision_platform/config.py`
- Modify: `config/vision_lab.default.json`
- Test: `tests/test_student_programs/test_student_safety.py`
- Test: `tests/test_vision_platform/test_config.py`

- [ ] **Step 1: 写安全网关失败测试**

Create `tests/test_student_programs/test_student_safety.py`:

```python
from __future__ import annotations

import pytest

from vision_platform.errors import MotionSafetyError, VisionPlatformError
from vision_platform.robot.safety import WorkspacePolicy
from vision_platform.student.safety import (
    StudentExecutionPolicy,
    StudentMotionGuard,
)


def _guard():
    workspace = WorkspacePolicy(
        x_mm=(20, 140),
        y_mm=(-90, 90),
        z_mm=(10, 140),
        safe_z_mm=100,
    )
    policy = StudentExecutionPolicy(
        min_speed=1,
        max_speed=30,
        max_runtime_s=60,
        max_commands=200,
        command_timeout_s=10,
        max_sleep_s=5,
        tool_on_max_z_mm=35,
    )
    return StudentMotionGuard(workspace=workspace, policy=policy)


def test_safe_vertical_and_high_horizontal_moves_pass():
    guard = _guard()
    guard.validate_move((100, 20, 120), (100, 20, 25), speed=8)
    guard.validate_move((100, 20, 120), (120, -20, 120), speed=15)


def test_low_horizontal_move_is_rejected():
    guard = _guard()

    with pytest.raises(MotionSafetyError, match="低于安全高度"):
        guard.validate_move((100, 20, 25), (120, 20, 25), speed=8)


def test_workspace_and_speed_are_checked():
    guard = _guard()

    with pytest.raises(MotionSafetyError, match="workspace"):
        guard.validate_move((100, 20, 120), (500, 20, 120), speed=8)
    with pytest.raises(VisionPlatformError, match="速度"):
        guard.validate_move((100, 20, 120), (100, 20, 100), speed=50)


def test_tool_on_requires_low_pick_height():
    guard = _guard()

    guard.validate_tool_on((100, 20, 25))
    with pytest.raises(VisionPlatformError, match="吸盘"):
        guard.validate_tool_on((100, 20, 100))


def test_sleep_and_log_limits_are_checked():
    guard = _guard()

    assert guard.validate_sleep(0.2) == 0.2
    assert guard.validate_log("学生日志") == "学生日志"
    with pytest.raises(VisionPlatformError, match="5"):
        guard.validate_sleep(6)
    with pytest.raises(VisionPlatformError, match="500"):
        guard.validate_log("x" * 501)
```

- [ ] **Step 2: 扩展配置测试**

Append to `tests/test_vision_platform/test_config.py`:

```python
def test_default_student_execution_policy_is_safe():
    cfg = load_config()

    assert cfg.student["allow_real_backend"] is False
    assert cfg.student["max_runtime_s"] == 60
    assert cfg.student["max_commands"] == 200
    assert cfg.student["speed_range"] == [1, 30]
    assert cfg.student["tool_on_max_z_mm"] == 35
```

- [ ] **Step 3: 验证安全测试先失败**

Run:

```powershell
python -m pytest `
  tests/test_student_programs/test_student_safety.py `
  tests/test_vision_platform/test_config.py::test_default_student_execution_policy_is_safe `
  -q
```

Expected: FAIL，提示 `student.safety` 或 `VisionLabConfig.student` 不存在。

- [ ] **Step 4: 实现安全网关**

Create `vision_platform/student/safety.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from vision_platform.errors import MotionSafetyError, VisionPlatformError
from vision_platform.robot.safety import Point3, WorkspacePolicy


@dataclass(frozen=True)
class StudentExecutionPolicy:
    min_speed: float
    max_speed: float
    max_runtime_s: float
    max_commands: int
    command_timeout_s: float
    max_sleep_s: float
    tool_on_max_z_mm: float

    def __post_init__(self) -> None:
        if self.min_speed <= 0 or self.max_speed < self.min_speed:
            raise ValueError("student speed range is invalid")
        if self.max_runtime_s <= 0 or self.command_timeout_s <= 0:
            raise ValueError("student time limits must be positive")
        if self.max_commands <= 0:
            raise ValueError("student max_commands must be positive")
        if self.max_sleep_s <= 0:
            raise ValueError("student max_sleep_s must be positive")


class StudentMotionGuard:
    def __init__(
        self,
        *,
        workspace: WorkspacePolicy,
        policy: StudentExecutionPolicy,
    ) -> None:
        self.workspace = workspace
        self.policy = policy

    def validate_move(
        self,
        current: Point3,
        target: Point3,
        *,
        speed: float,
    ) -> Point3:
        start = self.workspace.validate(current)
        end = self.workspace.validate(target)
        value = float(speed)
        if (
            not isfinite(value)
            or value < self.policy.min_speed
            or value > self.policy.max_speed
        ):
            raise VisionPlatformError(
                "STUDENT_SPEED_INVALID",
                (
                    f"速度必须在 {self.policy.min_speed:g} 到 "
                    f"{self.policy.max_speed:g} 之间"
                ),
                details={"speed": value},
            )
        horizontal = abs(start[0] - end[0]) > 1e-6 or abs(
            start[1] - end[1]
        ) > 1e-6
        if horizontal and (
            start[2] < self.workspace.safe_z_mm
            or end[2] < self.workspace.safe_z_mm
        ):
            raise MotionSafetyError(
                "低于安全高度时禁止水平移动",
                start=list(start),
                target=list(end),
                safe_z_mm=self.workspace.safe_z_mm,
            )
        return end

    def validate_tool_on(self, pose: Point3) -> None:
        current = self.workspace.validate(pose)
        if current[2] > self.policy.tool_on_max_z_mm:
            raise VisionPlatformError(
                "STUDENT_TOOL_HEIGHT_INVALID",
                (
                    "吸盘只能在抓取高度开启，"
                    f"当前 Z={current[2]:.1f} mm"
                ),
                details={
                    "pose_mm": list(current),
                    "tool_on_max_z_mm": self.policy.tool_on_max_z_mm,
                },
            )

    def validate_sleep(self, seconds: float) -> float:
        value = float(seconds)
        if (
            not isfinite(value)
            or value < 0
            or value > self.policy.max_sleep_s
        ):
            raise VisionPlatformError(
                "STUDENT_SLEEP_INVALID",
                f"单次等待必须在 0 到 {self.policy.max_sleep_s:g} 秒之间",
                details={"seconds": value},
            )
        return value

    @staticmethod
    def validate_log(message: str) -> str:
        text = str(message)
        if len(text) > 500:
            raise VisionPlatformError(
                "STUDENT_LOG_TOO_LONG",
                "学生日志单条不得超过 500 个字符",
                details={"length": len(text)},
            )
        return text
```

- [ ] **Step 5: 扩展配置模型**

In `vision_platform/config.py`, add `student: Mapping[str, Any]` to `VisionLabConfig`, then pass:

```python
student=dict(payload.get("student", {})),
```

Add to `config/vision_lab.default.json`:

```json
"student": {
  "allow_real_backend": false,
  "max_runtime_s": 60,
  "max_commands": 200,
  "command_timeout_s": 10,
  "max_sleep_s": 5,
  "speed_range": [1, 30],
  "tool_on_max_z_mm": 35,
  "output": "artifacts/vision_lab/student-runs"
}
```

Place the new `student` object after `task` and before `ui`，并保持 JSON 有效。

- [ ] **Step 6: 运行安全和配置测试**

Run:

```powershell
python -m pytest `
  tests/test_student_programs/test_student_safety.py `
  tests/test_vision_platform/test_config.py `
  -q
```

Expected: 全部 PASS。

- [ ] **Step 7: 提交安全网关**

```powershell
git add `
  vision_platform/student/safety.py `
  vision_platform/config.py `
  config/vision_lab.default.json `
  tests/test_student_programs/test_student_safety.py `
  tests/test_vision_platform/test_config.py
git commit -m "feat(student): enforce student motion policy"
```

### Task 6: 实现证据记录器

**Files:**
- Create: `vision_platform/student/evidence.py`
- Test: `tests/test_student_programs/test_evidence.py`

- [ ] **Step 1: 写证据失败测试**

Create `tests/test_student_programs/test_evidence.py`:

```python
from __future__ import annotations

import hashlib
import json

from vision_platform.student.evidence import StudentRunEvidence


def test_evidence_copies_source_and_writes_summary(tmp_path):
    program = tmp_path / "my_task.py"
    program.write_text("def main(ctx):\n    pass\n", encoding="utf-8")
    evidence = StudentRunEvidence.create(
        output_root=tmp_path / "runs",
        program_path=program,
        robot_backend="sim",
        run_id="run-fixed",
    )

    evidence.record_command(
        {"command_id": "000001", "name": "robot.home", "args": {}}
    )
    evidence.record_event("RUNNING", "程序开始")
    summary_path = evidence.finalize(
        status="PASS",
        command_count=1,
        last_pose_mm=(100, 20, 120),
        safety_violation_count=0,
        error=None,
        cleanup_errors=[],
    )

    source_bytes = program.read_bytes()
    assert (evidence.directory / "source.py").read_bytes() == source_bytes
    assert (evidence.directory / "source.sha256").read_text().strip() == (
        hashlib.sha256(source_bytes).hexdigest()
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["hardware_status"] == "PENDING_HARDWARE"
    assert summary["command_count"] == 1
    assert (evidence.directory / "commands.jsonl").is_file()
    assert (evidence.directory / "events.jsonl").is_file()
```

- [ ] **Step 2: 验证测试先失败**

Run:

```powershell
python -m pytest tests/test_student_programs/test_evidence.py -q
```

Expected: FAIL，提示 `evidence` 模块不存在。

- [ ] **Step 3: 实现证据记录器**

Create `vision_platform/student/evidence.py`:

```python
from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StudentRunEvidence:
    directory: Path
    program_path: Path
    source_sha256: str
    robot_backend: str
    run_id: str
    started_at: str
    started_monotonic: float

    @classmethod
    def create(
        cls,
        *,
        output_root: str | Path,
        program_path: str | Path,
        robot_backend: str,
        run_id: str | None = None,
    ) -> "StudentRunEvidence":
        selected = Path(program_path).expanduser().resolve()
        source = selected.read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        selected_run_id = run_id or (
            datetime.now().strftime("%Y%m%d-%H%M%S")
            + "-"
            + selected.stem
            + "-"
            + uuid.uuid4().hex[:8]
        )
        directory = Path(output_root).expanduser().resolve() / selected_run_id
        directory.mkdir(parents=True, exist_ok=False)
        shutil.copy2(selected, directory / "source.py")
        (directory / "source.sha256").write_text(
            digest + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "run_id": selected_run_id,
            "program_path": str(selected),
            "source_sha256": digest,
            "robot_backend": robot_backend,
            "hardware_status": "PENDING_HARDWARE",
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return cls(
            directory=directory,
            program_path=selected,
            source_sha256=digest,
            robot_backend=robot_backend,
            run_id=selected_run_id,
            started_at=_now(),
            started_monotonic=time.monotonic(),
        )

    def _append(self, name: str, payload: dict[str, Any]) -> None:
        with (self.directory / name).open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
            )

    def record_command(self, payload: dict[str, Any]) -> None:
        self._append("commands.jsonl", {"timestamp": _now(), **payload})

    def record_event(
        self,
        state: str,
        message: str,
        **details: Any,
    ) -> None:
        self._append(
            "events.jsonl",
            {
                "timestamp": _now(),
                "state": state,
                "message": message,
                "details": details,
            },
        )

    def finalize(
        self,
        *,
        status: str,
        command_count: int,
        last_pose_mm,
        safety_violation_count: int,
        error,
        cleanup_errors,
    ) -> Path:
        path = self.directory / "summary.json"
        payload = {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": status,
            "program_path": str(self.program_path),
            "source_sha256": self.source_sha256,
            "started_at": self.started_at,
            "finished_at": _now(),
            "elapsed_seconds": round(
                time.monotonic() - self.started_monotonic,
                3,
            ),
            "command_count": int(command_count),
            "last_pose_mm": (
                list(last_pose_mm) if last_pose_mm is not None else None
            ),
            "safety_violation_count": int(safety_violation_count),
            "error": error,
            "cleanup_errors": list(cleanup_errors),
            "robot_backend": self.robot_backend,
            "hardware_status": "PENDING_HARDWARE",
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path
```

- [ ] **Step 4: 运行证据测试**

Run:

```powershell
python -m pytest tests/test_student_programs/test_evidence.py -q
```

Expected: `1 passed`。

- [ ] **Step 5: 提交证据记录器**

```powershell
git add vision_platform/student/evidence.py tests/test_student_programs/test_evidence.py
git commit -m "feat(student): record reproducible run evidence"
```

### Task 7: 实现可替换应用会话和场景复位

**Files:**
- Create: `vision_platform/session.py`
- Modify: `vision_platform/ui/pyqt_app.py`
- Test: `tests/test_vision_platform/test_session.py`
- Modify: `tests/test_vision_platform/test_pyqt_smoke.py`

- [ ] **Step 1: 写会话失败测试**

Create `tests/test_vision_platform/test_session.py`:

```python
from __future__ import annotations

from vision_platform.session import VisionLabSession


class FakeApplication:
    def __init__(self, number):
        self.number = number
        self.close_calls = 0
        self.load_calls = 0
        self.open_calls = 0

    def close(self):
        self.close_calls += 1

    def load_and_start_scene(self):
        self.load_calls += 1

    def open(self):
        self.open_calls += 1


def test_session_reset_closes_old_and_opens_fresh_application():
    created = []

    def factory():
        app = FakeApplication(len(created) + 1)
        created.append(app)
        return app

    first = factory()
    session = VisionLabSession(application=first, factory=factory)

    replacement = session.reset_simulation()

    assert first.close_calls == 1
    assert replacement is created[1]
    assert replacement.load_calls == 1
    assert replacement.open_calls == 1
    assert session.application is replacement


def test_session_notifies_application_replacement():
    created = []

    def factory():
        app = FakeApplication(len(created) + 1)
        created.append(app)
        return app

    session = VisionLabSession(application=factory(), factory=factory)
    received = []
    unsubscribe = session.subscribe(received.append)

    replacement = session.reset_simulation()
    unsubscribe()

    assert received == [replacement]
```

- [ ] **Step 2: 验证会话测试先失败**

Run:

```powershell
python -m pytest tests/test_vision_platform/test_session.py -q
```

Expected: FAIL，提示 `vision_platform.session` 不存在。

- [ ] **Step 3: 实现会话管理器**

Create `vision_platform/session.py`:

```python
from __future__ import annotations

from threading import RLock
from typing import Any, Callable


class VisionLabSession:
    def __init__(
        self,
        *,
        application: Any,
        factory: Callable[[], Any],
    ) -> None:
        self._application = application
        self._factory = factory
        self._handlers: list[Callable[[Any], None]] = []
        self._lock = RLock()

    @property
    def application(self):
        with self._lock:
            return self._application

    def subscribe(self, handler: Callable[[Any], None]):
        with self._lock:
            self._handlers.append(handler)

        def unsubscribe():
            with self._lock:
                if handler in self._handlers:
                    self._handlers.remove(handler)

        return unsubscribe

    def reset_simulation(self):
        with self._lock:
            old = self._application
        old.close()
        replacement = self._factory()
        replacement.load_and_start_scene()
        replacement.open()
        with self._lock:
            self._application = replacement
            handlers = tuple(self._handlers)
        for handler in handlers:
            handler(replacement)
        return replacement

    def close(self) -> None:
        self.application.close()
```

- [ ] **Step 4: 运行会话测试**

Run:

```powershell
python -m pytest tests/test_vision_platform/test_session.py -q
```

Expected: `2 passed`。

- [ ] **Step 5: 让 PyQt 支持会话但保持旧构造兼容**

Modify `VisionLabWindow.__init__` so it accepts `session=None`。当只传 `application` 时，保留现有行为；当传 `session` 时，所有新建 worker 都读取 `session.application`。

When a session is supplied, subscribe with:

```python
self.session = session
self._session_unsubscribe = session.subscribe(
    self._replace_application
)
```

Add a private method:

```python
def _replace_application(self, application) -> None:
    if self._event_unsubscribe is not None:
        self._event_unsubscribe()
        self._event_unsubscribe = None
    self.application = application
    event_bus = getattr(application, "event_bus", None)
    if event_bus is not None:
        self._event_unsubscribe = event_bus.subscribe(self.view_model.apply)
```

In `main()` construct:

```python
def factory():
    return VisionLabApplication.from_config(config)

application = factory()
session = VisionLabSession(application=application, factory=factory)
window = VisionLabWindow(
    application=application,
    session=session,
    output_dir=args.output,
)
```

On window close call `session.close()` exactly once when a session exists。
Also call `_session_unsubscribe()` before closing the session。

- [ ] **Step 6: 扩展 PyQt 兼容测试**

Append to `tests/test_vision_platform/test_pyqt_smoke.py`:

```python
class FakeSession:
    def __init__(self, application):
        self.application = application
        self.handlers = []
        self.close_calls = 0

    def subscribe(self, handler):
        self.handlers.append(handler)

        def unsubscribe():
            if handler in self.handlers:
                self.handlers.remove(handler)

        return unsubscribe

    def replace(self, application):
        self.application = application
        for handler in tuple(self.handlers):
            handler(application)

    def close(self):
        self.close_calls += 1
        self.application.close()


def test_pyqt_window_rebinds_to_replaced_session_application(qtbot):
    first = DisconnectedApplication()
    second = DisconnectedApplication()
    session = FakeSession(first)
    window = VisionLabWindow(application=first, session=session)
    qtbot.addWidget(window)

    session.replace(second)

    assert window.application is second
    assert second.close_calls == 0
    window.close()
    assert session.close_calls == 1
    assert second.close_calls == 1
```

Run:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
python -m pytest `
  tests/test_vision_platform/test_session.py `
  tests/test_vision_platform/test_pyqt_smoke.py `
  -q
```

Expected: 全部 PASS。

- [ ] **Step 7: 提交会话管理**

```powershell
git add `
  vision_platform/session.py `
  vision_platform/ui/pyqt_app.py `
  tests/test_vision_platform/test_session.py `
  tests/test_vision_platform/test_pyqt_smoke.py
git commit -m "feat(ui): add replaceable simulation session"
```

### Task 8: 实现 StudentProgramController

**Files:**
- Create: `vision_platform/student/runner.py`
- Modify: `vision_platform/student/__init__.py`
- Test: `tests/test_student_programs/test_runner.py`
- Test: `tests/fixtures/student_programs/infinite_loop.py`

- [ ] **Step 1: 写控制器状态与动作测试**

Create `tests/test_student_programs/test_runner.py`:

```python
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vision_platform.robot.safety import WorkspacePolicy
from vision_platform.student.protocol import RunState
from vision_platform.student.runner import StudentProgramController
from vision_platform.student.safety import StudentExecutionPolicy


class FakeRobot:
    def __init__(self):
        self.current = (100.0, 20.0, 120.0)
        self.moves = []
        self.home_calls = 0

    def current_world_pose(self):
        return self.current

    def move_world(self, x, y, z, *, speed):
        self.current = (float(x), float(y), float(z))
        self.moves.append((*self.current, float(speed)))

    def move_home(self):
        self.home_calls += 1
        self.current = (100.0, 20.0, 120.0)


class FakeTool:
    def __init__(self):
        self.on_calls = 0
        self.off_calls = 0

    def on(self):
        self.on_calls += 1

    def off(self):
        self.off_calls += 1


class FakeSession:
    def __init__(self, *, backend="sim"):
        self.application = SimpleNamespace(
            robot=FakeRobot(),
            tool=FakeTool(),
            workspace=WorkspacePolicy(
                x_mm=(20, 140),
                y_mm=(-90, 90),
                z_mm=(10, 140),
                safe_z_mm=100,
            ),
            config=SimpleNamespace(robot_backend=backend),
        )
        self.reset_calls = 0

    def reset_simulation(self):
        self.reset_calls += 1
        return self.application


def wait_until(predicate, timeout_s=2):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true before timeout")


def write_program(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "student.py"
    path.write_text(source, encoding="utf-8")
    return path


def make_controller(
    tmp_path: Path,
    source: str,
    *,
    backend: str = "sim",
    max_runtime_s: float = 2,
):
    session = FakeSession(backend=backend)
    policy = StudentExecutionPolicy(
        min_speed=1,
        max_speed=30,
        max_runtime_s=max_runtime_s,
        max_commands=200,
        command_timeout_s=1,
        max_sleep_s=0.2,
        tool_on_max_z_mm=35,
    )
    controller = StudentProgramController(
        session=session,
        execution_policy=policy,
        output_root=tmp_path / "runs",
    )
    program = write_program(tmp_path, source)
    controller.load(program)
    return controller, session, program


def test_valid_program_runs_commands_and_finishes_pass(tmp_path):
    controller, session, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    ctx.robot.move_world(100, 20, 100, speed=15)\n",
    )
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=5)

    assert result.status == "PASS"
    assert session.application.robot.moves == [
        (100.0, 20.0, 100.0, 15.0)
    ]


def test_invalid_program_never_starts_process(tmp_path):
    controller, _, program = make_controller(
        tmp_path,
        "def main(ctx)\n    pass\n",
    )
    assert controller.validate(program).ok is False
    with pytest.raises(RuntimeError, match="VALIDATED"):
        controller.start()


def test_pause_blocks_next_command_and_step_releases_one(tmp_path):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    ctx.log('one')\n"
        "    ctx.log('two')\n",
    )
    assert controller.validate().ok is True

    controller.start(paused=True)
    assert controller.state is RunState.PAUSED
    controller.step()
    wait_until(lambda: controller.command_count == 1)
    assert controller.state is RunState.PAUSED
    assert controller.command_count == 1
    controller.cancel()
    assert controller.wait(timeout_s=5).status == "CANCELLED"


def test_low_horizontal_move_fails_before_backend_call(tmp_path):
    controller, session, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    ctx.robot.move_world(100, 20, 25, speed=8)\n"
        "    ctx.robot.move_world(120, 20, 25, speed=8)\n",
    )
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=5)

    assert result.status == "FAILED"
    assert result.error["code"] == "TARGET_OUT_OF_WORKSPACE"
    assert session.application.robot.moves[0] == (
        100.0,
        20.0,
        25.0,
        8.0,
    )
    assert (120.0, 20.0, 25.0, 8.0) not in (
        session.application.robot.moves
    )


def test_cancel_closes_tool_and_terminates_worker(tmp_path):
    controller, session, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    while True:\n"
        "        pass\n",
    )
    assert controller.validate().ok is True
    controller.start()
    wait_until(lambda: controller.process_is_alive)

    controller.cancel()
    result = controller.wait(timeout_s=5)

    assert result.status == "CANCELLED"
    assert session.application.tool.off_calls >= 1
    assert controller.process_is_alive is False


def test_runtime_limit_terminates_infinite_program(tmp_path):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n"
        "    while True:\n"
        "        pass\n",
        max_runtime_s=0.2,
    )
    assert controller.validate().ok is True
    controller.start()

    result = controller.wait(timeout_s=3)

    assert result.status == "FAILED"
    assert result.error["code"] == "STUDENT_RUNTIME_TIMEOUT"


def test_real_backend_is_always_rejected_in_v2_1(tmp_path):
    controller, _, _ = make_controller(
        tmp_path,
        "def main(ctx):\n    ctx.robot.home()\n",
        backend="real",
    )
    assert controller.validate().ok is True

    with pytest.raises(RuntimeError, match="REAL_BACKEND_NOT_AUTHORIZED"):
        controller.start()
```

Use a test-local helper `wait_until(predicate, timeout_s=2)` that polls every 10 ms and raises `AssertionError` at timeout。Do not use fixed sleeps longer than 50 ms。

- [ ] **Step 2: 验证控制器测试先失败**

Run:

```powershell
python -m pytest tests/test_student_programs/test_runner.py -q
```

Expected: FAIL，提示 `runner` 不存在。

- [ ] **Step 3: 实现控制器公开契约**

`vision_platform/student/runner.py` must expose an immutable
`StudentRunResult(status, summary_path, evidence_dir, error)` and a
`StudentProgramController` with this exact public surface:

| Member | Return | Meaning |
|---|---|---|
| `state` | `RunState` | Current state |
| `command_count` | `int` | Commands accepted by the parent |
| `process_is_alive` | `bool` | Child process liveness |
| `subscribe(handler)` | unsubscribe callable | Subscribe to snapshots |
| `load(program_path)` | `Path` | Select source and enter LOADED |
| `validate(program_path=None)` | `ValidationResult` | Validate source |
| `start(paused=False)` | `None` | Spawn and begin |
| `pause()` | `None` | Pause at next command boundary |
| `resume()` | `None` | Resume continuous execution |
| `step()` | `None` | Release exactly one command |
| `cancel()` | `None` | Cancel and clean up |
| `wait(timeout_s=None)` | `StudentRunResult` | Wait for terminal result |
| `reset()` | application object | Reset session and return fresh app |

Implementation requirements:

1. Use `multiprocessing.get_context("spawn")`。
2. Parent owns `application = session.application`。
3. Reject `application.config.robot_backend != "sim"` with `REAL_BACKEND_NOT_AUTHORIZED`。
4. Spawn a child target that calls `run_student_worker()` and puts its final result on a dedicated result queue。
5. Run the parent command loop in a daemon `threading.Thread`。
6. Maintain a `threading.Condition` for pause, resume and one-command step permits。
7. Before every command, enforce total runtime and command count。
8. Dispatch commands with an explicit dictionary of callables; do not use arbitrary `getattr`。
9. For `robot.move_world`, read current pose, call `StudentMotionGuard.validate_move()`, then call `application.robot.move_world()`。
10. For `tool.on`, validate current Z before `application.tool.on()`。
11. Implement `context.sleep` as cancellable waits no longer than 50 ms per poll。
12. Convert `VisionPlatformError` to its existing `code` and `details`。
13. On failure or cancellation stop accepting commands, signal cancellation,
    terminate and join the child when necessary, then run cleanup in this
    order:
    - `application.tool.off()`；
    - if current Z is below `safe_z_mm`, move vertically to safe Z；
    - `application.robot.move_home()`。
14. Preserve cleanup errors separately。
15. Never run robot cleanup while the child can still submit commands。
16. Finalize evidence exactly once。
17. Emit immutable snapshots containing state, current command, command count, elapsed time, error and evidence directory。
18. Run a watchdog thread so a child that never sends an SDK command still
    reaches `STUDENT_RUNTIME_TIMEOUT`。

- [ ] **Step 4: 添加无限循环夹具**

Create `tests/fixtures/student_programs/infinite_loop.py`:

```python
def main(ctx):
    while True:
        pass
```

- [ ] **Step 5: 运行控制器测试**

Run:

```powershell
python -m pytest tests/test_student_programs/test_runner.py -q
```

Expected: 全部 PASS，且测试结束后不存在仍存活的学生子进程。

- [ ] **Step 6: 导出控制器**

Modify `vision_platform/student/__init__.py`:

```python
from vision_platform.student.runner import (
    StudentProgramController,
    StudentRunResult,
)
```

Add both names to `__all__`。

- [ ] **Step 7: 提交控制器**

```powershell
git add `
  vision_platform/student/runner.py `
  vision_platform/student/__init__.py `
  tests/test_student_programs/test_runner.py `
  tests/fixtures/student_programs/infinite_loop.py
git commit -m "feat(student): execute programs through guarded controller"
```

### Task 9: 增加 CLI、PowerShell 入口和学生模板

**Files:**
- Modify: `vision_platform/cli.py`
- Create: `tools/vision_lab/run_student_program.ps1`
- Create: `student_programs/README.md`
- Create: `student_programs/templates/basic_motion.py`
- Create: `student_programs/templates/pick_and_place.py`
- Test: `tests/test_student_programs/test_cli.py`
- Test: `tests/test_acceptance/test_delivery_contract.py`

- [ ] **Step 1: 写 CLI 失败测试**

Create `tests/test_student_programs/test_cli.py`:

```python
from vision_platform.cli import build_parser


def test_student_validate_parser():
    args = build_parser().parse_args(
        ["student-validate", "--program", "student_programs/my_task.py"]
    )
    assert args.command == "student-validate"


def test_student_run_requires_sim_backend():
    parser = build_parser()
    args = parser.parse_args(
        [
            "student-run",
            "--program",
            "student_programs/my_task.py",
            "--robot",
            "sim",
            "--output",
            "artifacts/vision_lab/student-runs",
        ]
    )
    assert args.robot == "sim"
```

- [ ] **Step 2: 验证 CLI 测试先失败**

Run:

```powershell
python -m pytest tests/test_student_programs/test_cli.py -q
```

Expected: FAIL，提示无对应子命令。

- [ ] **Step 3: 实现 CLI 子命令**

Add imports:

```python
from dataclasses import asdict
```

Add handlers:

```python
def _student_validate(args: argparse.Namespace) -> int:
    from vision_platform.student.validator import validate_program

    result = validate_program(args.program)
    payload = {
        "status": "PASS" if result.ok else "FAIL",
        "program": str(result.path),
        "issues": [asdict(issue) for issue in result.issues],
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if result.ok else 2


def _student_run(args: argparse.Namespace) -> int:
    from vision_platform.application import VisionLabApplication
    from vision_platform.config import load_config
    from vision_platform.session import VisionLabSession
    from vision_platform.student.runner import StudentProgramController
    from vision_platform.student.safety import StudentExecutionPolicy

    environ = dict(os.environ)
    environ["ROBOT_BACKEND"] = "sim"
    environ["VISION_BACKEND"] = "sim"
    environ["COPPELIA_HOST"] = args.host
    environ["COPPELIA_PORT"] = str(args.port)
    if args.scene:
        scene = Path(args.scene).expanduser()
        if not scene.is_absolute():
            scene = PROJECT_ROOT / scene
        environ["COPPELIA_SCENE"] = str(scene.resolve())
    config = load_config(
        args.config,
        project_root=PROJECT_ROOT,
        environ=environ,
    )

    def factory():
        return VisionLabApplication.from_config(config)

    application = factory()
    session = VisionLabSession(
        application=application,
        factory=factory,
    )
    student = config.student
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
    controller = StudentProgramController(
        session=session,
        execution_policy=policy,
        output_root=args.output,
    )
    try:
        application.load_and_start_scene(config.coppelia_scene)
        application.open()
        controller.load(args.program)
        validation = controller.validate()
        if not validation.ok:
            print(
                json.dumps(
                    {
                        "status": "FAIL",
                        "issues": [
                            asdict(issue) for issue in validation.issues
                        ],
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
        print(
            json.dumps(
                {
                    "status": result.status,
                    "summary": str(result.summary_path),
                    "evidence": str(result.evidence_dir),
                    "error": result.error,
                },
                ensure_ascii=False,
            )
        )
        return 0 if result.status == "PASS" else 1
    finally:
        if controller.process_is_alive:
            controller.cancel()
        session.close()
```

Add parsers inside `build_parser()`:

```python
    student_validate = subparsers.add_parser(
        "student-validate",
        help="Validate one student Python program",
    )
    student_validate.add_argument("--program", required=True)
    student_validate.set_defaults(handler=_student_validate)

    student_run = subparsers.add_parser(
        "student-run",
        help="Run one student program through the guarded simulator",
    )
    student_run.add_argument("--program", required=True)
    student_run.add_argument("--config")
    student_run.add_argument("--robot", choices=("sim",), default="sim")
    student_run.add_argument("--scene")
    student_run.add_argument("--host", default="127.0.0.1")
    student_run.add_argument("--port", type=int, default=23000)
    student_run.add_argument(
        "--output",
        default="artifacts/vision_lab/student-runs",
    )
    student_run.set_defaults(handler=_student_run)
```

`student-validate` prints one JSON object and exits：

- valid: exit 0, `{"status":"PASS","issues":[]}`
- invalid: exit 2, for example
  `{"status":"FAIL","issues":[{"code":"SYNTAX_ERROR","message":"expected ':'","line":1,"column":14}]}`

`student-run`:

1. Forces `ROBOT_BACKEND=sim`；
2. Creates `VisionLabApplication` and `VisionLabSession`；
3. Loads and starts the configured scene；
4. Creates `StudentProgramController`；
5. Validates and runs；
6. Prints summary path；
7. Returns 0 only for PASS；
8. Always closes session。

- [ ] **Step 4: 创建学生模板**

Create `student_programs/templates/basic_motion.py`:

```python
def main(ctx):
    ctx.log("基础点位运动开始")
    ctx.robot.home()
    ctx.robot.move_world(100, 60, 120, speed=15)
    ctx.robot.move_world(120, 20, 120, speed=15)
    ctx.robot.move_world(100, -20, 120, speed=15)
    ctx.robot.home()
    ctx.log("基础点位运动完成")
```

Create `student_programs/templates/pick_and_place.py`:

```python
def main(ctx):
    pick = (55, -55, 20)
    drop = (122, -66, 20)
    safe_z = 100

    ctx.log("单物体吸取与放置开始")
    ctx.robot.home()
    ctx.robot.move_world(pick[0], pick[1], safe_z, speed=15)
    ctx.robot.move_world(*pick, speed=8)
    ctx.tool.on()
    ctx.robot.move_world(pick[0], pick[1], safe_z, speed=12)
    ctx.robot.move_world(drop[0], drop[1], safe_z, speed=15)
    ctx.robot.move_world(*drop, speed=8)
    ctx.tool.off()
    ctx.robot.move_world(drop[0], drop[1], safe_z, speed=12)
    ctx.robot.home()
    ctx.log("单物体吸取与放置完成")
```

Create `student_programs/README.md` with the SDK contract, coordinate units, safe-height rule, launch commands and the warning that RX/RY/RZ are not supported。

- [ ] **Step 5: 创建 PowerShell 入口**

`tools/vision_lab/run_student_program.ps1` parameters:

```powershell
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Program,
    [string]$OutputDir = "artifacts\vision_lab\student-runs",
    [string]$CoppeliaRoot = $(if ($env:COPPELIASIM_ROOT) {
        $env:COPPELIASIM_ROOT
    } else {
        "E:\CoppeliaSim"
    }),
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 23000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv-vision\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment is missing: $Python"
}

if (-not [System.IO.Path]::IsPathRooted($Program)) {
    $Program = Join-Path $ProjectRoot $Program
}
$Program = (Resolve-Path -LiteralPath $Program).Path
if ([System.IO.Path]::GetExtension($Program) -ne ".py") {
    throw "Student program must be a .py file: $Program"
}

if (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $ProjectRoot $OutputDir
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

& (Join-Path $PSScriptRoot "launch_coppeliasim.ps1") `
    -CoppeliaRoot $CoppeliaRoot `
    -HostAddress $HostAddress `
    -Port $Port |
    Format-Table -AutoSize |
    Out-Host

$env:COPPELIA_HOST = $HostAddress
$env:COPPELIA_PORT = [string]$Port
$Scene = Join-Path $ProjectRoot "simulation\vision_lab\BL23_vision_lab.ttt"
$StudentExitCode = 1

Push-Location $ProjectRoot
try {
    & $Python -m vision_platform.cli student-run `
        --program $Program `
        --robot sim `
        --scene $Scene `
        --host $HostAddress `
        --port $Port `
        --output $OutputDir
    $StudentExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $StudentExitCode
```

This script intentionally does not stop any CoppeliaSim process. Process
ownership remains with `launch_coppeliasim.ps1` and the teacher。

- [ ] **Step 6: 扩展现有发布合同测试**

Append to `tests/test_acceptance/test_delivery_contract.py`，保留文件中已有测试：

```python
def test_student_launcher_has_required_contract():
    launcher = ROOT / "tools" / "vision_lab" / "run_student_program.ps1"
    source = launcher.read_text(encoding="utf-8")

    assert "[Parameter(Mandatory = $true)]" in source
    assert "[string]$Program" in source
    assert "launch_coppeliasim.ps1" in source
    assert "vision_platform.cli student-run" in source
    assert "--robot sim" in source
    assert "--scene $Scene" in source
    assert "exit $StudentExitCode" in source


def test_student_launcher_does_not_kill_unowned_simulator():
    launcher = ROOT / "tools" / "vision_lab" / "run_student_program.ps1"
    source = launcher.read_text(encoding="utf-8")

    assert "Stop-Process -Name" not in source
    assert "taskkill" not in source.lower()
```

Run:

```powershell
python -m pytest `
  tests/test_student_programs/test_cli.py `
  tests/test_acceptance/test_delivery_contract.py `
  -q
```

Expected: 全部 PASS，原发布合同测试继续保留。

- [ ] **Step 7: 提交 CLI 和模板**

```powershell
git add `
  vision_platform/cli.py `
  tools/vision_lab/run_student_program.ps1 `
  student_programs `
  tests/test_student_programs/test_cli.py `
  tests/test_acceptance/test_delivery_contract.py
git commit -m "feat(student): add cli launcher and program templates"
```

### Task 10: 在 PyQt 中增加学生编程页

**Files:**
- Modify: `vision_platform/ui/pyqt_app.py`
- Create: `vision_platform/ui/student_program_panel.py`
- Test: `tests/test_vision_platform/test_student_program_panel.py`
- Modify: `tests/test_vision_platform/test_pyqt_smoke.py`

- [ ] **Step 1: 写面板失败测试**

Create `tests/test_vision_platform/test_student_program_panel.py`:

```python
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog

from vision_platform.student.protocol import RunState
from vision_platform.student.validator import (
    ValidationIssue,
    ValidationResult,
)
from vision_platform.ui.student_program_panel import StudentProgramPanel


class FakeController:
    def __init__(self):
        self.state = RunState.EMPTY
        self.handlers = []
        self.cancel_calls = 0
        self.loaded_path = None
        self.validation_result = None

    def subscribe(self, handler):
        self.handlers.append(handler)

        def unsubscribe():
            if handler in self.handlers:
                self.handlers.remove(handler)

        return unsubscribe

    def emit_state(self, state):
        self.state = state
        snapshot = SimpleNamespace(
            state=state,
            current_command=None,
            command_count=0,
            elapsed_seconds=0.0,
            error=None,
            evidence_dir=None,
        )
        for handler in tuple(self.handlers):
            handler(snapshot)

    def load(self, path):
        self.loaded_path = Path(path)
        self.emit_state(RunState.LOADED)
        return self.loaded_path

    def validate(self, program_path=None):
        if self.validation_result is None:
            selected = Path(program_path or self.loaded_path)
            self.validation_result = ValidationResult(
                path=selected,
                ok=True,
                issues=(),
            )
        if self.validation_result.ok:
            self.emit_state(RunState.VALIDATED)
        return self.validation_result

    def start(self, *, paused=False):
        self.emit_state(RunState.PAUSED if paused else RunState.RUNNING)

    def pause(self):
        self.emit_state(RunState.PAUSED)

    def resume(self):
        self.emit_state(RunState.RUNNING)

    def step(self):
        self.emit_state(RunState.PAUSED)

    def cancel(self):
        self.cancel_calls += 1
        self.emit_state(RunState.CANCELLED)

    def reset(self):
        self.emit_state(RunState.VALIDATED)


def test_panel_exposes_file_and_run_controls(qtbot):
    panel = StudentProgramPanel(controller=FakeController())
    qtbot.addWidget(panel)
    assert panel.open_button.text() == "打开程序"
    assert panel.validate_button.text() == "检查代码"
    assert panel.run_button.text() == "运行"
    assert panel.pause_button.text() == "暂停"
    assert panel.step_button.text() == "下一步"
    assert panel.stop_button.text() == "停止"
    assert panel.reset_button.text() == "复位场景"


def test_panel_button_states_follow_runner_state(qtbot):
    controller = FakeController()
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)

    controller.emit_state(RunState.PAUSED)

    assert panel.step_button.isEnabled() is True
    assert panel.run_button.isEnabled() is False
    assert panel.stop_button.isEnabled() is True


def test_open_file_loads_utf8_source(qtbot, tmp_path, monkeypatch):
    program = tmp_path / "student.py"
    program.write_text(
        "def main(ctx):\n    ctx.robot.home()\n",
        encoding="utf-8",
    )
    controller = FakeController()
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(program), "Python (*.py)"),
    )

    qtbot.mouseClick(panel.open_button, Qt.LeftButton)

    assert "def main(ctx):" in panel.editor.toPlainText()
    assert controller.loaded_path == program


def test_validation_issues_render_line_and_code(qtbot, tmp_path):
    program = tmp_path / "student.py"
    program.write_text(
        "def main(ctx)\n    pass\n",
        encoding="utf-8",
    )
    controller = FakeController()
    controller.loaded_path = program
    controller.validation_result = ValidationResult(
        path=program,
        ok=False,
        issues=(
            ValidationIssue(
                code="SYNTAX_ERROR",
                message="expected ':'",
                line=3,
                column=10,
            ),
        ),
    )
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)
    panel.editor.setPlainText(program.read_text(encoding="utf-8"))
    panel.program_path = program
    controller.emit_state(RunState.LOADED)

    qtbot.mouseClick(panel.validate_button, Qt.LeftButton)

    assert "SYNTAX_ERROR" in panel.console.toPlainText()
    assert "第 3 行" in panel.console.toPlainText()


def test_close_cancels_running_controller(qtbot):
    controller = FakeController()
    panel = StudentProgramPanel(controller=controller)
    qtbot.addWidget(panel)
    controller.emit_state(RunState.RUNNING)

    panel.close()

    assert controller.cancel_calls == 1
```

- [ ] **Step 2: 验证面板测试先失败**

Run:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
python -m pytest tests/test_vision_platform/test_student_program_panel.py -q
```

Expected: FAIL，提示面板模块不存在。

- [ ] **Step 3: 实现独立面板**

Create `vision_platform/ui/student_program_panel.py`:

```python
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vision_platform.student.protocol import RunState


class _SnapshotBridge(QObject):
    updated = pyqtSignal(object)


class StudentProgramPanel(QWidget):
    def __init__(self, *, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.program_path: Path | None = None
        self._unsubscribe = None
        self._build_ui()

        self._bridge = _SnapshotBridge(self)
        self._bridge.updated.connect(self._render_snapshot)
        self._unsubscribe = controller.subscribe(
            self._bridge.updated.emit
        )
        self._render_snapshot(
            SimpleNamespace(
                state=controller.state,
                current_command=None,
                command_count=0,
                elapsed_seconds=0.0,
                error=None,
                evidence_dir=None,
            )
        )

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.open_button = QPushButton("打开程序")
        self.save_button = QPushButton("保存")
        self.save_as_button = QPushButton("另存为")
        self.validate_button = QPushButton("检查代码")
        self.run_button = QPushButton("运行")
        self.pause_button = QPushButton("暂停")
        self.step_button = QPushButton("下一步")
        self.stop_button = QPushButton("停止")
        self.reset_button = QPushButton("复位场景")
        for button in (
            self.open_button,
            self.save_button,
            self.save_as_button,
            self.validate_button,
            self.run_button,
            self.pause_button,
            self.step_button,
            self.stop_button,
            self.reset_button,
        ):
            button.setMinimumHeight(38)
            toolbar.addWidget(button)
        layout.addLayout(toolbar)

        self.path_label = QLabel("尚未打开学生程序")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.path_label)

        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText(
            "打开或粘贴包含 def main(ctx): 的 UTF-8 Python 程序"
        )
        self.editor.setTabStopDistance(32)
        layout.addWidget(self.editor, 3)

        status = QGridLayout()
        self.state_label = QLabel("EMPTY")
        self.command_label = QLabel("当前命令：—")
        self.metrics_label = QLabel("命令数：0　运行时间：0.0 s")
        self.evidence_label = QLabel("证据目录：—")
        status.addWidget(QLabel("运行状态："), 0, 0)
        status.addWidget(self.state_label, 0, 1)
        status.addWidget(self.command_label, 1, 0, 1, 2)
        status.addWidget(self.metrics_label, 2, 0, 1, 2)
        status.addWidget(self.evidence_label, 3, 0, 1, 2)
        layout.addLayout(status)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(130)
        layout.addWidget(self.console, 1)

        self.open_button.clicked.connect(self._open_program)
        self.save_button.clicked.connect(self._save)
        self.save_as_button.clicked.connect(self._save_as)
        self.validate_button.clicked.connect(self._validate)
        self.run_button.clicked.connect(self._run)
        self.pause_button.clicked.connect(self.controller.pause)
        self.step_button.clicked.connect(self.controller.step)
        self.stop_button.clicked.connect(self.controller.cancel)
        self.reset_button.clicked.connect(self.controller.reset)

    def _open_program(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开学生程序",
            "",
            "Python (*.py)",
        )
        if not path:
            return
        selected = Path(path).expanduser().resolve()
        source = selected.read_text(encoding="utf-8")
        self.editor.setPlainText(source)
        self.program_path = selected
        self.path_label.setText(str(selected))
        self.controller.load(selected)
        self.console.append(f"已载入：{selected}")

    def _save(self) -> bool:
        if self.program_path is None:
            return self._save_as()
        self.program_path.write_text(
            self.editor.toPlainText(),
            encoding="utf-8",
            newline="\n",
        )
        self.controller.load(self.program_path)
        self.console.append(f"已保存：{self.program_path}")
        return True

    def _save_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "另存学生程序",
            "student_program.py",
            "Python (*.py)",
        )
        if not path:
            return False
        selected = Path(path).expanduser()
        if selected.suffix.lower() != ".py":
            selected = selected.with_suffix(".py")
        self.program_path = selected.resolve()
        self.path_label.setText(str(self.program_path))
        return self._save()

    def _validate(self) -> None:
        if not self._save():
            return
        result = self.controller.validate(self.program_path)
        if result.ok:
            self.console.append("代码检查：PASS")
            return
        self.console.append("代码检查：FAIL")
        for issue in result.issues:
            location = (
                f"第 {issue.line} 行"
                if issue.line is not None
                else "文件"
            )
            self.console.append(
                f"[{issue.code}] {location}：{issue.message}"
            )

    def _run(self) -> None:
        if not self._save():
            return
        result = self.controller.validate(self.program_path)
        if not result.ok:
            self._validate()
            return
        self.controller.start()

    def _render_snapshot(self, snapshot) -> None:
        state = snapshot.state
        if not isinstance(state, RunState):
            state = RunState(str(state))
        self.state_label.setText(state.value)
        command = snapshot.current_command or "—"
        self.command_label.setText(f"当前命令：{command}")
        self.metrics_label.setText(
            f"命令数：{int(snapshot.command_count)}　"
            f"运行时间：{float(snapshot.elapsed_seconds):.1f} s"
        )
        evidence = snapshot.evidence_dir or "—"
        self.evidence_label.setText(f"证据目录：{evidence}")
        if snapshot.error:
            self.console.append(
                f"[{snapshot.error.get('code', 'FAILED')}] "
                f"{snapshot.error.get('message', '运行失败')}"
            )
        self._apply_state(state)

    def _apply_state(self, state: RunState) -> None:
        editable = state in {
            RunState.EMPTY,
            RunState.LOADED,
            RunState.VALIDATED,
            RunState.PASSED,
            RunState.FAILED,
            RunState.CANCELLED,
        }
        self.editor.setReadOnly(not editable)
        self.open_button.setEnabled(editable)
        self.save_button.setEnabled(editable)
        self.save_as_button.setEnabled(editable)
        self.validate_button.setEnabled(
            state
            in {
                RunState.LOADED,
                RunState.VALIDATED,
                RunState.PASSED,
                RunState.FAILED,
                RunState.CANCELLED,
            }
        )
        self.run_button.setEnabled(state is RunState.VALIDATED)
        self.pause_button.setEnabled(state is RunState.RUNNING)
        self.step_button.setEnabled(state is RunState.PAUSED)
        self.stop_button.setEnabled(
            state in {RunState.RUNNING, RunState.PAUSED}
        )
        self.reset_button.setEnabled(
            state
            in {
                RunState.VALIDATED,
                RunState.PASSED,
                RunState.FAILED,
                RunState.CANCELLED,
            }
        )

    def closeEvent(self, event) -> None:
        if self.controller.state in {
            RunState.RUNNING,
            RunState.PAUSED,
        }:
            self.controller.cancel()
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        event.accept()
```

Long-running robot work remains in `StudentProgramController`; the Qt slots
above only start, signal or render that controller and must not wait for the
child process。

Button matrix:

| State | Open/Save | Validate | Run | Pause | Step | Stop | Reset |
|---|---:|---:|---:|---:|---:|---:|---:|
| EMPTY | Yes | No | No | No | No | No | No |
| LOADED | Yes | Yes | No | No | No | No | No |
| VALIDATED | Yes | Yes | Yes | No | No | No | Yes |
| RUNNING | No | No | No | Yes | No | Yes | No |
| PAUSED | No | No | No | No | Yes | Yes | No |
| PASSED/FAILED/CANCELLED | Yes | Yes | No | No | No | No | Yes |
| RESETTING | No | No | No | No | No | No | No |

- [ ] **Step 4: 集成现有主窗口**

In `vision_platform/ui/pyqt_app.py`, import:

```python
from vision_platform.student.runner import StudentProgramController
from vision_platform.student.safety import StudentExecutionPolicy
from vision_platform.ui.student_program_panel import StudentProgramPanel
```

After the existing classification tab is created, construct:

```python
student = self.application.config.student
speed_range = student["speed_range"]
student_policy = StudentExecutionPolicy(
    min_speed=float(speed_range[0]),
    max_speed=float(speed_range[1]),
    max_runtime_s=float(student["max_runtime_s"]),
    max_commands=int(student["max_commands"]),
    command_timeout_s=float(student["command_timeout_s"]),
    max_sleep_s=float(student["max_sleep_s"]),
    tool_on_max_z_mm=float(student["tool_on_max_z_mm"]),
)
student_output = self.application._resolve_project_path(
    self.application.config,
    student["output"],
)
self.student_controller = StudentProgramController(
    session=self.session,
    execution_policy=student_policy,
    output_root=student_output,
)
self.student_program_panel = StudentProgramPanel(
    controller=self.student_controller,
)
self.tabs.addTab(self.student_program_panel, "学生编程")
```

If `VisionLabWindow` is constructed without a session or without a full
configuration in legacy smoke tests, add a disabled `学生编程` tab containing
`当前窗口未配置学生程序会话` instead of constructing a controller。Do not alter
existing calibration and classification tab behavior。

Update subtitle to:

```text
仿真标定 · 视觉识别 · 学生编程 · 六轴机械臂分类闭环
```

- [ ] **Step 5: 运行 PyQt 测试**

Run:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
python -m pytest `
  tests/test_vision_platform/test_student_program_panel.py `
  tests/test_vision_platform/test_pyqt_smoke.py `
  -q
```

Expected: 全部 PASS。

- [ ] **Step 6: 提交 PyQt 页面**

```powershell
git add `
  vision_platform/ui/student_program_panel.py `
  vision_platform/ui/pyqt_app.py `
  tests/test_vision_platform/test_student_program_panel.py `
  tests/test_vision_platform/test_pyqt_smoke.py
git commit -m "feat(ui): add student program workspace"
```

### Task 11: 增加真实 CoppeliaSim 在线验收

**Files:**
- Create: `tests/test_acceptance/test_coppeliasim_student_program.py`
- Modify: `tools/vision_lab/run_acceptance.ps1`
- Modify: `docs/视觉仿真实训平台使用说明.md`
- Create: `docs/学生自编程实验说明.md`

- [ ] **Step 1: 写在线验收测试**

Create `tests/test_acceptance/test_coppeliasim_student_program.py`:

```python
from __future__ import annotations

import json
from math import dist
from pathlib import Path

import pytest

from vision_platform.student.runner import StudentProgramController
from vision_platform.student.safety import StudentExecutionPolicy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROGRAM = (
    PROJECT_ROOT
    / "student_programs"
    / "templates"
    / "basic_motion.py"
)


class BoundSession:
    def __init__(self, application):
        self.application = application

    def reset_simulation(self):
        raise AssertionError("online test must not reset before assertions")


@pytest.mark.coppeliasim
def test_student_program_moves_live_openr6_and_records_evidence(
    running_vision_scene,
    tmp_path,
):
    application = running_vision_scene
    application.open()
    student = application.config.student
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
    controller = StudentProgramController(
        session=BoundSession(application),
        execution_policy=policy,
        output_root=tmp_path / "student-runs",
    )
    controller.load(PROGRAM)
    assert controller.validate().ok is True

    controller.start()
    result = controller.wait(timeout_s=15)

    assert result.status == "PASS"
    assert controller.command_count >= 5
    assert controller.process_is_alive is False
    for name in (
        "source.py",
        "source.sha256",
        "manifest.json",
        "commands.jsonl",
        "events.jsonl",
        "summary.json",
    ):
        assert (result.evidence_dir / name).is_file()
    summary = json.loads(
        result.summary_path.read_text(encoding="utf-8")
    )
    assert summary["status"] == "PASS"
    assert summary["hardware_status"] == "PENDING_HARDWARE"

    expected = tuple(
        float(value)
        for value in application.scene_spec["robot_visuals"][
            "teaching_ready_pose"
        ]["tcp_expected_mm"]
    )
    actual = application.robot.current_world_pose()
    assert dist(actual, expected) <= 2.0
```

- [ ] **Step 2: 添加验收脚本步骤**

In `tools/vision_lab/run_acceptance.ps1`, add an explicit step:

```powershell
Invoke-CheckedPython -Name "student_program_online" -Arguments @(
    "-m", "pytest",
    "tests/test_acceptance/test_coppeliasim_student_program.py",
    "-m", "coppeliasim",
    "--junitxml", (Join-Path $OutputDir "student-program.xml"),
    "-q"
)
```

Do not count a skip as PASS。

- [ ] **Step 3: 完成学生实验说明**

`docs/学生自编程实验说明.md` must include:

- learning objectives；
- template copy workflow；
- SDK API；
- safe Z explanation；
- PyQt loading steps；
- CLI launch steps；
- pause, step, stop and reset；
- evidence directory；
- five intentional failure exercises；
- simulation-to-real limitations；
- student submission checklist；
- teacher acceptance checklist。

- [ ] **Step 4: Update the main usage guide**

Add a “学生自编程” section to `docs/视觉仿真实训平台使用说明.md`，link to the new experiment guide and show the PowerShell command。

- [ ] **Step 5: Run the static suite**

Run:

```powershell
python -m pytest -q
```

Expected: 0 failed；CoppeliaSim online tests may be skipped only in this static command。

- [ ] **Step 6: Run the live student test**

Run with CoppeliaSim available:

```powershell
python -m pytest `
  tests/test_acceptance/test_coppeliasim_student_program.py `
  -m coppeliasim `
  -q
```

Expected: PASS, not skip。

- [ ] **Step 7: Commit online acceptance and docs**

```powershell
git add `
  tests/test_acceptance/test_coppeliasim_student_program.py `
  tools/vision_lab/run_acceptance.ps1 `
  docs/视觉仿真实训平台使用说明.md `
  docs/学生自编程实验说明.md
git commit -m "test(student): verify live student program workflow"
```

### Task 12: 最终回归、发布白名单和交付

**Files:**
- Modify: `README.md`
- Modify: `RETAINED_FILES.txt`
- Modify: `docs/视觉仿真实训平台使用说明.md`
- Modify: `docs/视觉仿真实训平台自动验收报告.md`
- Evidence: `artifacts/vision_lab/student-program-v2-final/`

- [ ] **Step 1: 登记正式发布文件**

把本计划新增的全部 Python、测试、模板、脚本和文档逐项加入 `RETAINED_FILES.txt`。列表至少包含：

```text
docs/superpowers/plans/2026-07-30-student-program-runner-v2-plan.md
docs/superpowers/specs/2026-07-30-student-program-runner-v2-design.md
docs/学生自编程实验说明.md
docs/视觉仿真实训平台V2-开发端执行Prompt.md
student_programs/README.md
student_programs/templates/basic_motion.py
student_programs/templates/pick_and_place.py
tools/vision_lab/run_student_program.ps1
vision_platform/student/__init__.py
vision_platform/student/evidence.py
vision_platform/student/protocol.py
vision_platform/student/runner.py
vision_platform/student/safety.py
vision_platform/student/sdk.py
vision_platform/session.py
vision_platform/student/validator.py
vision_platform/student/worker.py
vision_platform/ui/student_program_panel.py
tests/test_acceptance/test_coppeliasim_student_program.py
tests/fixtures/student_programs/infinite_loop.py
tests/test_student_programs/test_cli.py
tests/test_student_programs/test_evidence.py
tests/test_student_programs/test_protocol.py
tests/test_student_programs/test_runner.py
tests/test_student_programs/test_sdk.py
tests/test_student_programs/test_student_safety.py
tests/test_student_programs/test_validator.py
tests/test_student_programs/test_worker.py
tests/test_vision_platform/test_session.py
tests/test_vision_platform/test_student_program_panel.py
```

如果实施中实际文件名与计划不同，先统一设计、计划和代码命名，再登记最终真实路径。不要把 `artifacts/`、`.venv-vision/`、缓存或学生个人文件加入白名单。

- [ ] **Step 2: 扩展发布合同**

Append to `tests/test_acceptance/test_delivery_contract.py`：

```python
def test_clean_release_whitelist_contains_student_runner_delivery():
    retained = {
        line.strip()
        for line in (ROOT / "RETAINED_FILES.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.startswith("#")
    }
    required = {
        "docs/superpowers/plans/2026-07-30-student-program-runner-v2-plan.md",
        "docs/superpowers/specs/2026-07-30-student-program-runner-v2-design.md",
        "docs/学生自编程实验说明.md",
        "docs/视觉仿真实训平台V2-开发端执行Prompt.md",
        "student_programs/templates/pick_and_place.py",
        "tools/vision_lab/run_student_program.ps1",
        "vision_platform/student/runner.py",
        "vision_platform/ui/student_program_panel.py",
        "tests/test_acceptance/test_coppeliasim_student_program.py",
    }
    assert required <= retained
    assert all((ROOT / path).is_file() for path in retained)


def test_clean_release_whitelist_excludes_generated_student_data():
    retained = (ROOT / "RETAINED_FILES.txt").read_text(encoding="utf-8")
    assert "artifacts/" not in retained
    assert ".venv-vision/" not in retained
    assert "__pycache__/" not in retained
```

- [ ] **Step 3: 更新入口文档**

在 `README.md` 和 `docs/视觉仿真实训平台使用说明.md` 中加入：

- 学生模板位置；
- PyQt “学生编程”页；
- `student-validate` 与 `run_student_program.ps1` 命令；
- 暂停、单步、停止、复位语义；
- 证据目录；
- 子进程不是恶意代码安全沙箱；
- 真机仍为 `PENDING_HARDWARE`。

- [ ] **Step 4: 运行格式与范围检查**

Run:

```powershell
git diff --check
git status --short
git diff --name-only origin/main...HEAD
```

Expected: 无空白错误；只有 V2.1 范围内文件发生变化；正式机器人资产未出现在 diff 中。

- [ ] **Step 5: 运行完整自动验收**

Run:

```powershell
powershell -ExecutionPolicy Bypass `
  -File tools\vision_lab\run_acceptance.ps1 `
  -OutputDir artifacts\vision_lab\student-program-v2-final
```

Expected:

- overall PASS；
- 原标定与六物体分类 PASS；
- 原 CoppeliaSim 在线测试 PASS；
- 学生程序在线测试 PASS；
- PyQt 离屏测试 PASS；
- 完整静态测试 PASS；
- simUI 和场景运行验证 PASS；
- hardware status 仍为 `PENDING_HARDWARE`。

- [ ] **Step 6: 验证正式资产和发布白名单**

Run:

```powershell
python -m pytest `
  tests/test_simulation/test_robot_asset_provenance.py `
  tests/test_acceptance/test_delivery_contract.py `
  -q
```

Expected: 两个测试文件全部 PASS；受保护资产匹配 `robot_assets_manifest.json` 及来源清单；白名单中的每个路径存在。

- [ ] **Step 7: 检查进程和端口清理**

Run:

```powershell
$listeners = @(
  Get-NetTCPConnection `
    -LocalPort 23000 `
    -State Listen `
    -ErrorAction SilentlyContinue
)
$listeners | Select-Object LocalAddress, LocalPort, OwningProcess
```

Expected: 验收脚本拥有的进程已退出；预先存在、由用户拥有的 CoppeliaSim 进程可以保留且不得被误杀。

- [ ] **Step 8: 更新验收报告**

在 `docs/视觉仿真实训平台自动验收报告.md` 记录实际命令、精确测试数、在线结果、证据路径、资产验证结果和 `PENDING_HARDWARE`。不得把结构测试或仿真 PASS 写成真实教学效果或真机验收通过。

- [ ] **Step 9: 自审设计、计划和 Prompt**

Run:

```powershell
$patterns = @(
  ('T' + 'BD'),
  ('TO' + 'DO'),
  ('implement' + ' later'),
  ('fill' + ' in'),
  ('待' + '定'),
  ('待' + '实现'),
  ('H:' + '\智能机械与机器人基础'),
  ('codex/' + 'coppeliasim-vision-mvp'),
  ('formal_asset_' + 'hashes.json'),
  ('tests/test_' + 'delivery'),
  ('HERMES' + '_BRIEF.md'),
  ('168' + ' passed')
)
Get-Content `
  docs/superpowers/specs/2026-07-30-student-program-runner-v2-design.md, `
  docs/superpowers/plans/2026-07-30-student-program-runner-v2-plan.md, `
  docs/视觉仿真实训平台V2-开发端执行Prompt.md `
  -Encoding UTF8 |
  Select-String -Pattern $patterns
git diff --check
```

Expected: 无失效引用、无未解决占位符、无 diff 错误。

- [ ] **Step 10: 提交最终发布集成**

```powershell
git add `
  README.md `
  RETAINED_FILES.txt `
  docs/视觉仿真实训平台使用说明.md `
  docs/视觉仿真实训平台自动验收报告.md `
  docs/学生自编程实验说明.md `
  docs/superpowers/specs/2026-07-30-student-program-runner-v2-design.md `
  docs/superpowers/plans/2026-07-30-student-program-runner-v2-plan.md `
  docs/视觉仿真实训平台V2-开发端执行Prompt.md
git commit -m "docs: complete student program v2 delivery"
```

- [ ] **Step 11: 保留开发分支供教师审核**

不要自动合并或推送到 `origin/main`。最终报告必须列出：

- 分支和提交列表；
- 精确静态与在线测试结果；
- 证据目录；
- 受保护资产验证结果；
- 发布白名单检查结果；
- 尚待教师人工完成的教学效果验收；
- 全部 `PENDING_HARDWARE` 项。

## 完成定义

The development side may claim “V2.1 software implementation complete” only when:

- every task above is checked；
- static suite has zero failures；
- live CoppeliaSim student program test passes, not skips；
- PyQt visual inspection confirms Chinese labels and code controls are readable；
- pause, one-step, stop, timeout and reset have evidence；
- protected assets match the recorded manifests；
- the development branch is clean；
- teacher teaching-effect acceptance remains explicitly pending；
- hardware remains explicitly `PENDING_HARDWARE`。
