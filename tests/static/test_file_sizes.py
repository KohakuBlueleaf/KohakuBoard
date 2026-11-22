"""Static guard enforcing per-file AST statement caps."""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY_SRC = ROOT / "src" / "kohakuboard"
SOFT_LIMIT = 500
HARD_LIMIT = 1000


def _count_statements(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            count += 1
    return count


def test_python_file_sizes() -> None:
    failures: list[str] = []
    for path in sorted(PY_SRC.rglob("*.py")):
        stmts = _count_statements(path)
        rel = path.relative_to(ROOT).as_posix()
        if stmts > HARD_LIMIT:
            failures.append(f"{rel}: {stmts} statements (> {HARD_LIMIT})")
        elif stmts > SOFT_LIMIT:
            warnings.warn(f"{rel}: {stmts} statements (> {SOFT_LIMIT})", stacklevel=1)
    assert not failures, "\n".join(failures)
