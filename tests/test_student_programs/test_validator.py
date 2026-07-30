from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from vision_platform.student.validator import (
    BLOCKED_IMPORTS,
    MAX_PROGRAM_BYTES,
    ValidationIssue,
    ValidationResult,
    validate_program,
    validate_program_bytes,
)


def _write(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path


def _codes(result: ValidationResult) -> list[str]:
    return [issue.code for issue in result.issues]


def test_captured_bytes_use_the_same_validation_rules_without_rereading(
    tmp_path,
):
    selected = tmp_path / "snapshot.py"
    selected.write_text("def main(ctx)\n    pass\n", encoding="utf-8")
    captured = (
        b"def main(ctx):\n"
        b"    import subprocess\n"
        b"    ctx.robot.home()\n"
    )

    result = validate_program_bytes(captured, selected)

    assert result.path == selected.resolve()
    assert result.ok is False
    assert _codes(result) == ["IMPORT_NOT_ALLOWED"]


def test_valid_main_contract_passes_and_resolves_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    selected = _write(
        tmp_path / "ok.py",
        "def main(ctx):\n"
        "    ctx.robot.home()\n"
        "    ctx.log('完成')\n",
    )

    result = validate_program(Path("ok.py"))

    assert result.path == selected.resolve()
    assert result.ok is True
    assert result.issues == ()


def test_syntax_error_reports_line_and_column(tmp_path):
    result = validate_program(
        _write(tmp_path / "bad.py", "def main(ctx)\n    pass\n")
    )

    assert result.ok is False
    assert result.issues[0].code == "SYNTAX_ERROR"
    assert result.issues[0].line == 1
    assert result.issues[0].column is not None


def test_missing_main_is_rejected(tmp_path):
    result = validate_program(_write(tmp_path / "missing.py", "VALUE = 1\n"))

    assert _codes(result) == ["MAIN_MISSING"]


def test_multiple_main_functions_report_second_definition(tmp_path):
    result = validate_program(
        _write(
            tmp_path / "duplicate.py",
            "def main(ctx):\n    pass\n"
            "def main(ctx):\n    pass\n",
        )
    )

    duplicated = result.issues[0]
    assert duplicated.code == "MAIN_DUPLICATED"
    assert duplicated.line == 3
    assert duplicated.column == 1


@pytest.mark.parametrize(
    "definition",
    [
        "def main(ctx, extra):",
        "def main(ctx, extra=None):",
        "def main(ctx=None):",
        "def main(*ctx):",
        "def main(**ctx):",
        "def main(ctx, *, option):",
        "async def main(ctx):",
    ],
)
def test_main_must_be_sync_with_exactly_one_required_positional_argument(
    tmp_path,
    definition,
):
    result = validate_program(
        _write(tmp_path / "signature.py", f"{definition}\n    pass\n")
    )

    issue = result.issues[0]
    assert issue.code == "MAIN_SIGNATURE_INVALID"
    assert issue.line == 1
    assert issue.column == 1


@pytest.mark.parametrize("prefix", BLOCKED_IMPORTS)
@pytest.mark.parametrize("style", ["import", "from"])
def test_blocked_imports_and_submodules_are_rejected_at_any_depth(
    tmp_path,
    prefix,
    style,
):
    module = f"{prefix}.private"
    statement = (
        f"import {module}"
        if style == "import"
        else f"from {module} import hidden"
    )
    result = validate_program(
        _write(
            tmp_path / "blocked.py",
            "def main(ctx):\n"
            f"    {statement}\n",
        )
    )

    issue = result.issues[0]
    assert issue.code == "IMPORT_NOT_ALLOWED"
    assert issue.line == 2
    assert issue.column == 5


@pytest.mark.parametrize("prefix", BLOCKED_IMPORTS)
def test_blocked_module_itself_is_rejected(tmp_path, prefix):
    result = validate_program(
        _write(
            tmp_path / "blocked.py",
            f"import {prefix}\n"
            "def main(ctx):\n"
            "    pass\n",
        )
    )

    assert "IMPORT_NOT_ALLOWED" in _codes(result)


@pytest.mark.parametrize(
    "statement",
    [
        "from vision_platform import application",
        "from vision_platform import application as app",
    ],
)
def test_import_from_equivalent_blocked_module_is_rejected(
    tmp_path,
    statement,
):
    result = validate_program(
        _write(
            tmp_path / "blocked_from.py",
            f"{statement}\n"
            "def main(ctx):\n"
            "    pass\n",
        )
    )

    assert _codes(result) == ["IMPORT_NOT_ALLOWED"]
    assert result.issues[0].line == 1
    assert result.issues[0].column == 1


def test_import_from_blocked_base_reports_one_issue(tmp_path):
    result = validate_program(
        _write(
            tmp_path / "blocked_base.py",
            "from socket import socket\n"
            "def main(ctx):\n"
            "    pass\n",
        )
    )

    assert _codes(result) == ["IMPORT_NOT_ALLOWED"]


def test_similarly_named_import_is_not_blocked(tmp_path):
    result = validate_program(
        _write(
            tmp_path / "allowed.py",
            "import socketserver\n"
            "def main(ctx):\n"
            "    pass\n",
        )
    )

    assert result.ok is True
    assert result.issues == ()


def test_top_level_call_is_rejected_but_module_docstring_is_allowed(tmp_path):
    called = validate_program(
        _write(
            tmp_path / "top_level.py",
            '"""学生程序。"""\n'
            "print('不能在载入时执行')\n"
            "def main(ctx):\n"
            "    pass\n",
        )
    )
    docstring_only = validate_program(
        _write(
            tmp_path / "docstring.py",
            '"""学生程序。"""\n'
            "def main(ctx):\n"
            "    pass\n",
        )
    )

    issue = next(
        item
        for item in called.issues
        if item.code == "TOP_LEVEL_CALL_NOT_ALLOWED"
    )
    assert issue.line == 2
    assert issue.column == 1
    assert docstring_only.ok is True


def test_non_py_and_large_files_are_rejected_before_parsing(tmp_path):
    text = _write(tmp_path / "program.txt", "def main(ctx):\n    pass\n")
    assert validate_program(text).issues[0].code == "FILE_EXTENSION_INVALID"

    large = tmp_path / "large.py"
    large.write_bytes(b"#" * (MAX_PROGRAM_BYTES + 1))
    assert validate_program(large).issues[0].code == "FILE_TOO_LARGE"


def test_missing_file_is_rejected(tmp_path):
    result = validate_program(tmp_path / "missing.py")

    assert result.issues[0].code == "FILE_NOT_FOUND"


def test_non_utf8_file_is_rejected(tmp_path):
    selected = tmp_path / "encoded.py"
    selected.write_bytes(b"\xff\xfe\x00")

    result = validate_program(selected)

    assert result.issues[0].code == "FILE_ENCODING_INVALID"


@pytest.mark.parametrize("error_type", [PermissionError, OSError])
def test_file_access_error_is_returned_instead_of_raised(
    tmp_path,
    monkeypatch,
    error_type,
):
    selected = _write(
        tmp_path / "unreadable.py",
        "def main(ctx):\n"
        "    pass\n",
    )
    original_open = Path.open

    def denied_open(path, *args, **kwargs):
        if path == selected:
            raise error_type("platform-specific access failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied_open)

    result = validate_program(selected)

    assert result.ok is False
    assert result.issues[0].code == "FILE_ACCESS_ERROR"
    assert result.issues[0].message == "无法读取学生程序文件"


def test_validation_records_are_immutable(tmp_path):
    issue = ValidationIssue("EXAMPLE", "example")
    result = ValidationResult(
        path=(tmp_path / "example.py").resolve(),
        ok=False,
        issues=(issue,),
    )

    with pytest.raises(FrozenInstanceError):
        issue.code = "CHANGED"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.ok = True  # type: ignore[misc]
