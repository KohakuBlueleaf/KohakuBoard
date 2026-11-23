"""Unit tests for client validation, rich payload types, CLI commands, and scalar hot-path speed."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from kohakuboard import Board, Histogram, KernelDensity, Media, RunReader, Table, errors
from kohakuboard.cli.main import _delegate_optional
from kohakuboard.cli.main import main as cli_main


class _FakeTorchTensor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def detach(self) -> _FakeTorchTensor:
        return self

    def cpu(self) -> _FakeTorchTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _FakePolarsFrame:
    columns = ["a", "b"]

    def to_dicts(self) -> list[dict[str, object]]:
        return [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]


def test_config_validation_and_guards(tmp_path: Path) -> None:
    with pytest.raises(errors.ConfigError):
        Board(project="../escape", dir=tmp_path)
    with pytest.raises(errors.ConfigError):
        Board(run_id="bad/id", dir=tmp_path)
    with pytest.raises(errors.ConfigError):
        Board(tui="invalid_mode", dir=tmp_path)
    with pytest.raises(errors.ConfigError):
        Board(mode="remote", dir=tmp_path)
    with pytest.raises(errors.ConfigError):
        Board(overflow="invalid", dir=tmp_path)  # type: ignore[arg-type]


def test_rich_type_duck_typing_and_validation() -> None:
    ft = _FakeTorchTensor(np.zeros((3, 8, 8), dtype=np.float32))
    kind, payload = Media(ft).prepare()
    assert kind == "pixels"
    pixels, width, height, channels, fmt, _, _, _ = payload
    assert (width, height, channels, fmt) == (8, 8, 3, "png")
    assert len(pixels) == 8 * 8 * 3

    tbl = Table(_FakePolarsFrame())
    assert tbl.columns == ["a", "b"]
    assert tbl.rows == [[1, "x"], [2, "y"]]

    with pytest.raises(errors.ConfigError):
        Histogram()
    with pytest.raises(errors.ConfigError):
        KernelDensity()


def test_cli_commands_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with Board(
        run_id="cli1",
        name="cli-demo",
        project="demo",
        config={"lr": 2e-4},
        dir=tmp_path,
        system_metrics=False,
        capture_output=True,
    ) as board:
        for i in range(5):
            board.log({"train/loss": 1.0 / (i + 1)})
        print("cli captured hello world line")

    # Drain captured training stdout before asserting JSON CLI output.
    assert "cli captured hello world line" in capsys.readouterr().out
    run_dir = tmp_path / "demo" / "cli1"

    assert cli_main(["ls", str(tmp_path), "--json"]) == 0
    ls_data = json.loads(capsys.readouterr().out)
    assert len(ls_data) == 1 and ls_data[0]["run_id"] == "cli1"

    assert cli_main(["info", str(run_dir), "--json"]) == 0
    info_data = json.loads(capsys.readouterr().out)
    assert info_data["info"]["name"] == "cli-demo"
    assert info_data["storage"]["run_db_bytes"] > 0

    assert cli_main(["logs", str(run_dir), "--search", "captured", "--json"]) == 0
    log_hits = json.loads(capsys.readouterr().out)
    assert len(log_hits) == 1 and "hello world" in log_hits[0]["text"]

    csv_out = tmp_path / "out.csv"
    json_out = tmp_path / "out.json"
    npz_out = tmp_path / "out.npz"
    assert cli_main(["export", str(run_dir), "-f", "csv", "-o", str(csv_out)]) == 0
    assert "train/loss" in csv_out.read_text(encoding="utf-8")
    assert cli_main(["export", str(run_dir), "-f", "json", "-o", str(json_out)]) == 0
    assert (
        len(json.loads(json_out.read_text(encoding="utf-8"))["scalars"]["train/loss"]["value"]) == 5
    )
    assert cli_main(["export", str(run_dir), "-f", "npz", "-o", str(npz_out)]) == 0
    with np.load(npz_out) as data:
        assert len(data["train__loss__value"]) == 5

    assert cli_main(["doctor", str(tmp_path), "--reindex"]) == 0
    doc_out = capsys.readouterr().out
    assert "all checks passed" in doc_out

    # Missing optional extras report a clean install hint.
    assert _delegate_optional("_kohakuboard_missing_opt_pkg", "run_tui", "tui", []) == 2
    assert "kohakuboard[tui]" in capsys.readouterr().err

    # `kobo tui` validates missing target directories without touching the terminal.
    assert cli_main(["tui", str(tmp_path / "does_not_exist")]) == 1
    assert "directory not found" in capsys.readouterr().err


def test_cli_info_and_table_handle_non_finite_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Human-readable `kobo info` and `Table` JSON must not crash on NaN/Inf values."""
    with Board(
        run_id="nan1",
        project="edge",
        dir=tmp_path,
        system_metrics=False,
        capture_output=False,
    ) as board:
        board.log(
            {
                "only_nan": float("nan"),
                "only_inf": float("inf"),
                "tbl": Table(columns=["v"], rows=[[float("nan")], [float("inf")], [1.25]]),
            }
        )

    run_dir = tmp_path / "edge" / "nan1"
    assert cli_main(["info", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "only_nan" in out and "NaN/Inf" in out

    # Table JSON is strict RFC-8259 (no bare NaN/Infinity tokens).
    with RunReader(run_dir) as reader:
        rows = reader.tables("tbl")[0]["rows"]
        assert rows == [[None], [None], [1.25]]


@pytest.mark.serial
def test_scalar_logging_hot_path_latency(tmp_path: Path) -> None:
    """Measure scalar `board.log()` hot-path latency and throughput."""
    n = 10_000
    with Board(
        run_id="perf1",
        project="bench",
        dir=tmp_path,
        system_metrics=False,
        capture_output=False,
        flush_interval_ms=250,
    ) as board:
        # Warm up column creation and JIT/cache.
        for i in range(100):
            board.log({"train/loss": float(i), "train/lr": 1e-4, "train/grad_norm": 1.0})

        t0 = time.perf_counter()
        for i in range(n):
            board.log({"train/loss": float(i), "train/lr": 1e-4, "train/grad_norm": 1.0})
        t_log = time.perf_counter() - t0
        board.flush()

    us_per_call = (t_log / n) * 1e6
    # Generous CI-safe upper bound (< 50 us/call; release builds typically hit < 8 us/call).
    assert us_per_call < 50.0, f"scalar log hot path too slow: {us_per_call:.2f} us/call"
