"""Static contract verifying kohakuboard._rs runtime exports match _rs.pyi."""

from __future__ import annotations

import ast
from pathlib import Path

import kohakuboard._rs as rs

ROOT = Path(__file__).resolve().parents[2]
STUB = ROOT / "src" / "kohakuboard" / "_rs.pyi"


def test_rs_stub_matches_runtime() -> None:
    tree = ast.parse(STUB.read_text(encoding="utf-8"), filename=str(STUB))
    stub_exports: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            stub_exports.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            stub_exports.add(node.target.id)

    runtime_exports = {name for name in dir(rs) if not name.startswith("__")}
    missing_in_stub = runtime_exports - stub_exports
    missing_in_runtime = stub_exports - runtime_exports
    assert not missing_in_stub, f"Missing from _rs.pyi: {sorted(missing_in_stub)}"
    assert not missing_in_runtime, f"Stale entries in _rs.pyi: {sorted(missing_in_runtime)}"
