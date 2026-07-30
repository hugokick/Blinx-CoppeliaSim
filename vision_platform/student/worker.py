from __future__ import annotations

import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import uuid4

from vision_platform.student.sdk import (
    StudentContext,
    _StudentProgramCancelled,
)


def _load_module(
    path: Path,
    *,
    source_bytes: bytes | None = None,
) -> ModuleType:
    name = f"_vision_platform_student_program_{uuid4().hex}"
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        if source_bytes is None:
            source = path.read_text(encoding="utf-8")
        else:
            if type(source_bytes) is not bytes:
                raise TypeError("captured student source must be bytes")
            source = source_bytes.decode("utf-8")
        code = compile(
            source,
            str(path),
            "exec",
            dont_inherit=True,
        )
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _cancelled_result() -> dict[str, Any]:
    return {
        "status": "CANCELLED",
        "error": {
            "code": "STUDENT_PROGRAM_CANCELLED",
            "message": "学生程序通信已取消",
        },
    }


def run_student_worker(
    path: str | Path,
    connection: Any,
    *,
    source_bytes: bytes | None = None,
) -> dict[str, Any]:
    selected = Path(path).expanduser().resolve()
    module_name: str | None = None
    try:
        module = _load_module(
            selected,
            source_bytes=source_bytes,
        )
        module_name = module.__name__
        entry = getattr(module, "main")
        entry(StudentContext(connection))
        return {"status": "PASS", "error": None}
    except _StudentProgramCancelled:
        return _cancelled_result()
    except (EOFError, BrokenPipeError):
        return _cancelled_result()
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
        if module_name is not None:
            sys.modules.pop(module_name, None)
        try:
            connection.close()
        except Exception:
            pass
