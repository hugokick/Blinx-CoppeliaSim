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


def _source_column(node: ast.AST) -> int | None:
    column = getattr(node, "col_offset", None)
    return column + 1 if isinstance(column, int) else None


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

    try:
        with selected.open("rb") as program_file:
            payload = program_file.read(MAX_PROGRAM_BYTES + 1)
    except OSError:
        issues.append(
            ValidationIssue(
                "FILE_ACCESS_ERROR",
                "无法读取学生程序文件",
            )
        )
        return ValidationResult(selected, False, tuple(issues))

    return _validate_program_payload(selected, payload)


def validate_program_bytes(
    payload: bytes,
    path: str | Path,
) -> ValidationResult:
    selected = Path(path).expanduser().resolve()
    if selected.suffix.lower() != ".py":
        return ValidationResult(
            selected,
            False,
            (
                ValidationIssue(
                    "FILE_EXTENSION_INVALID",
                    "学生程序必须使用 .py 扩展名",
                ),
            ),
        )
    if type(payload) is not bytes:
        raise TypeError("captured student source must be bytes")
    return _validate_program_payload(selected, payload)


def _validate_program_payload(
    selected: Path,
    payload: bytes,
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    if len(payload) > MAX_PROGRAM_BYTES:
        issues.append(
            ValidationIssue(
                "FILE_TOO_LARGE",
                "学生程序不得超过 256 KiB",
            )
        )
        return ValidationResult(selected, False, tuple(issues))

    try:
        source = payload.decode("utf-8")
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
        duplicate = main_nodes[1]
        issues.append(
            ValidationIssue(
                "MAIN_DUPLICATED",
                "只能定义一个顶层 main(ctx)",
                line=duplicate.lineno,
                column=_source_column(duplicate),
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
                    column=_source_column(main),
                )
            )

    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if _blocked(base):
                modules = [base]
            else:
                modules = [
                    f"{base}.{alias.name}" if base else alias.name
                    for alias in node.names
                ]
        for module in dict.fromkeys(modules):
            if _blocked(module):
                issues.append(
                    ValidationIssue(
                        "IMPORT_NOT_ALLOWED",
                        f"禁止导入：{module}",
                        line=getattr(node, "lineno", None),
                        column=_source_column(node),
                    )
                )

    for node in tree.body:
        if not isinstance(node, ast.Expr):
            continue
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
                    column=_source_column(node),
                )
            )

    return ValidationResult(selected, not issues, tuple(issues))
