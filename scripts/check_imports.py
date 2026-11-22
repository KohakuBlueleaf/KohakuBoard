"""Enforce KohakuBoard layer import boundaries."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PKG = ROOT / "src" / "kohakuboard"

FORBIDDEN_BY_SUBPKG: dict[str, set[str]] = {
    "client": {"kohakuboard.tui", "kohakuboard_web", "fastapi", "uvicorn"},
    "reader": {"kohakuboard.client", "kohakuboard.tui", "kohakuboard_web", "fastapi", "uvicorn"},
    "tui": {"kohakuboard_web", "fastapi", "uvicorn"},
}

ALLOWLIST: set[tuple[str, str]] = {
    ("client/board.py", "kohakuboard.tui"),
    ("cli/main.py", "kohakuboard_web"),
}


def check_file(path: Path) -> list[str]:
    rel = path.relative_to(SRC_PKG).as_posix()
    subpkg = rel.split("/", 1)[0]
    forbidden = FORBIDDEN_BY_SUBPKG.get(subpkg, set())
    if not forbidden:
        return []

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    errors: list[str] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for mod in names:
            errors.extend(
                f"{rel}:{node.lineno}: forbidden import '{mod}' in '{subpkg}'"
                for bad in forbidden
                if (mod == bad or mod.startswith(f"{bad}.")) and (rel, bad) not in ALLOWLIST
            )
    return errors


def main() -> int:
    if not SRC_PKG.exists():
        return 0
    errors: list[str] = []
    for py_file in sorted(SRC_PKG.rglob("*.py")):
        errors.extend(check_file(py_file))
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
