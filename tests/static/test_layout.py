"""Static contract tests for module layout and import hygiene."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUST_SRC = ROOT / "src" / "kohakuboard-rust"
PY_SRC = ROOT / "src" / "kohakuboard"

ALLOWED_LOCAL_IMPORTS: set[tuple[str, str]] = {
    ("client/types.py", "PIL"),
    ("client/types.py", "PIL.Image"),
    ("client/types.py", "torch"),
    ("client/board.py", "kohakuboard.tui.config"),
    ("client/board.py", "kohakuboard.tui.adapter"),
    ("cli/main.py", "kohakuboard_web"),
    ("cli/main.py", "kohakuboard_web.cli"),
    ("cli/main.py", "kohakuboard.tui"),
    ("cli/main.py", "kohakuboard.tui.cli"),
}


def test_rust_module_layout() -> None:
    """Every Rust submodule directory uses mod.rs with no sibling <name>.rs file."""
    for entry in RUST_SRC.rglob("*"):
        if entry.is_dir():
            assert (entry / "mod.rs").exists(), f"missing mod.rs in {entry}"
            sibling = entry.with_suffix(".rs")
            assert not sibling.exists(), f"sibling {sibling} beside directory {entry}"


def test_no_unlisted_in_function_imports() -> None:
    """No imports inside Python functions except allowlisted optional dependencies."""
    violations: list[str] = []
    for path in sorted(PY_SRC.rglob("*.py")):
        rel = path.relative_to(PY_SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Import):
                    violations.extend(
                        f"{rel}:{child.lineno}: import {alias.name}"
                        for alias in child.names
                        if (rel, alias.name) not in ALLOWED_LOCAL_IMPORTS
                    )
                elif (
                    isinstance(child, ast.ImportFrom)
                    and child.module
                    and (rel, child.module) not in ALLOWED_LOCAL_IMPORTS
                ):
                    violations.append(f"{rel}:{child.lineno}: from {child.module} import ...")
    assert not violations, "Disallowed in-function imports:\n" + "\n".join(violations)
