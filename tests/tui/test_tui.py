"""Headless end-to-end tests for KohakuTUI live monitoring (`Board(tui=...)`) and `kobo tui`."""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from kohakuboard import Board, ChartSpec, Histogram, KernelDensity, Media, Table, TuiConfig
from kohakuboard.tui.viewer import BoardTuiViewer


def _wait_until(
    sync_fn: Callable[[], bool],
    pred: Callable[[], bool],
    timeout: float = 4.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sync_fn()
        if pred():
            return
        time.sleep(0.03)
    sync_fn()
    assert pred(), "condition did not become true before timeout"


def _make_rgb(size: int, seed: int) -> np.ndarray:
    xs = np.linspace(0.0, 2.0 * np.pi, size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, xs)
    r = ((np.sin(xx + seed) * 0.5 + 0.5) * 255).astype(np.uint8)
    g = ((np.cos(yy - seed) * 0.5 + 0.5) * 255).astype(np.uint8)
    b = ((np.sin(xx + yy) * 0.5 + 0.5) * 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def test_board_live_tui_inline_high_cardinality_and_seal(tmp_path: Path) -> None:
    """Inline live TUI handles 120 metrics, NaN gaps, stdio capture, and seals into scrollback."""
    orig_stdout = sys.stdout
    cfg = TuiConfig(
        mode="inline",
        backend="test",
        size=(110, 32),
        refresh_hz=30.0,
        total_steps=30,
        watch=["train/lr", "val/loss"],
        hide=["grad/layer_*"],
        log_height=4,
        on_finish="seal",
    )
    board = Board(
        run_id="tui_in1",
        name="dit-inline-demo",
        project="tui-test",
        config={"lr": 3e-4, "batch": 128},
        dir=tmp_path,
        tui=cfg,
        system_metrics=False,
        capture_output=True,
    )
    assert board.tui is not None and board.tui.active

    for s in range(30):
        payload: dict[str, float] = {
            "train/loss": float("nan") if s == 15 else 2.0 * math.exp(-s / 12.0),
            "val/loss": 2.2 * math.exp(-s / 14.0),
            "train/lr": 3e-4 * (1.0 - s / 40.0),
        }
        # High-cardinality 120 per-layer norms (must stay bounded without flooding the screen)
        for layer in range(120):
            payload[f"grad/layer_{layer:03d}"] = 0.5 + 0.01 * layer
        board.log(payload)
        board.progress(s + 1, total=30, label="train")
        if s == 10:
            print("sentinel TUI_CAPTURED_LINE_42")

    board.flush()
    _wait_until(
        board.tui.sync,
        lambda: "train/loss" in "\n".join(board.tui.screen())
        and "TUI_CAPTURED_LINE_42" in "\n".join(board.tui.screen()),
    )
    screen_text = "\n".join(board.tui.screen())
    assert "train/loss" in screen_text
    assert "train/lr" in screen_text
    assert "more metrics" in screen_text

    # Capture history before finish() closes the app (board.tui.finish seals into transcript)
    tui_handle = board.tui
    board.finish()

    # stdout is deterministically restored after finish()
    assert sys.stdout is orig_stdout
    assert not tui_handle.active
    history_text = "\n".join(tui_handle.history())
    assert "dit-inline-demo" in history_text
    assert "finished" in history_text


def test_board_live_tui_fullscreen_runtime_api_and_safe_detach(tmp_path: Path) -> None:
    """Fullscreen TUI supports pin_chart/watch/notify/tab-switch and `q` detaches safely."""
    cfg = TuiConfig(
        mode="fullscreen",
        backend="test",
        size=(120, 34),
        refresh_hz=30.0,
        total_steps=20,
        charts=[ChartSpec(["train/loss", "val/loss"], label="Loss", height=8)],
        on_finish="clear",
    )
    board = Board(
        run_id="tui_fs1",
        name="dit-fs-demo",
        project="tui-test",
        config={"layers": 24},
        dir=tmp_path,
        tui=cfg,
        system_metrics=False,
    )
    assert board.tui is not None and board.tui.active

    for s in range(10):
        board.log({"train/loss": 1.5 / (s + 1), "val/loss": 1.8 / (s + 1), "train/grad_norm": 0.9})
    board.tui.pin_chart("train/grad_norm", label="Grad Norm", height=5)
    board.tui.watch("train/grad_norm")
    board.tui.notify("Checkpoint saved", level="ok")
    board.flush()

    _wait_until(
        board.tui.sync,
        lambda: "Loss" in "\n".join(board.tui.screen())
        and "Grad Norm" in "\n".join(board.tui.screen()),
    )

    # Switch to Tab 2 (Metrics explorer) via hotkey `2`
    board.tui.inject({"text": "2"})
    _wait_until(
        board.tui.sync,
        lambda: "All Logged Metrics" in "\n".join(board.tui.screen()),
    )

    # Press `q` -> detaches TUI cleanly WITHOUT raising or closing the Board!
    board.tui.inject({"text": "q"})
    deadline = time.monotonic() + 3.0
    while board.tui.active and time.monotonic() < deadline:
        time.sleep(0.03)
    assert not board.tui.active
    assert not board.closed

    # Training loop keeps logging cleanly after TUI detach
    board.log({"train/loss": 0.05})
    board.finish()


def test_board_tui_auto_degrades_on_non_tty_and_ddp_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`tui=True` degrades safely when stdout is not a TTY or on non-zero DDP rank."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    with Board(
        run_id="notty1",
        project="tui-test",
        dir=tmp_path,
        tui=True,
        system_metrics=False,
        capture_output=False,
    ) as b1:
        assert b1.tui is not None
        assert not b1.tui.active
        b1.log({"loss": 0.5})

    monkeypatch.setenv("RANK", "1")
    with Board(
        run_id="rank1",
        project="tui-test",
        dir=tmp_path,
        tui=TuiConfig(backend="test", size=(100, 30)),
        system_metrics=False,
        capture_output=False,
    ) as b2:
        assert b2.disabled
        assert b2.tui is None


def test_standalone_kobo_tui_viewer_multimodal_and_diff(tmp_path: Path) -> None:
    """`BoardTuiViewer` renders Overview, Metrics, Tables, Media, Distributions, Logs, and Diff."""
    rng = np.random.default_rng(19)
    for idx, (rid, name, lr) in enumerate(
        [("runA", "amber-dit", 3e-4), ("runB", "iolite-dit", 1e-4)]
    ):
        with Board(
            run_id=rid,
            name=name,
            project="ssh-demo",
            config={"lr": lr, "batch_size": 128 if idx == 0 else 256},
            dir=tmp_path,
            system_metrics=False,
            capture_output=True,
        ) as b:
            for s in range(25):
                b.log(
                    {
                        "train/loss": (1.0 + 0.2 * idx) * math.exp(-s / 10.0),
                        "val/acc": 0.5 + 0.4 * (s / 25.0),
                    }
                )
            img = _make_rgb(32, idx)
            b.log(
                {
                    "samples/preview": Media(img, caption=f"{name} img", format="png"),
                    "dist/weights": Histogram(rng.normal(0.0, 1.0, size=512), bins=24),
                    "dist/kde": KernelDensity(rng.normal(0.0, 1.0, size=512), num_points=48),
                    "eval/table": Table(
                        columns=["prompt", "score"],
                        rows=[[f"terrarium_{rid}", 0.95]],
                    ),
                }
            )
            print(f"SSH_VIEWER_LOG_TOKEN_{rid}")

    viewer = BoardTuiViewer(tmp_path, backend="test", size=(120, 36), refresh_sec=10.0)
    viewer.start()
    try:
        # Tab 1: Overview (multi-run overlay + summary table)
        _wait_until(
            viewer.sync,
            lambda: "amber-dit" in "\n".join(viewer.screen())
            and "iolite-dit" in "\n".join(viewer.screen()),
        )

        # Tab 2: Metrics
        viewer.switch_tab("metrics")
        _wait_until(viewer.sync, lambda: "val/acc" in "\n".join(viewer.screen()))

        # Tab 3: Tables
        viewer.switch_tab("tables")
        _wait_until(viewer.sync, lambda: "terrarium_" in "\n".join(viewer.screen()))

        # Tab 4: Media (half-block image + caption)
        viewer.switch_tab("media")
        _wait_until(viewer.sync, lambda: "samples/preview" in "\n".join(viewer.screen()))

        # Tab 5: Distributions (Bars)
        viewer.switch_tab("dist")
        _wait_until(viewer.sync, lambda: "dist/weights" in "\n".join(viewer.screen()))

        # Tab 6: Logs
        viewer.switch_tab("logs")
        _wait_until(viewer.sync, lambda: "SSH_VIEWER_LOG_TOKEN_" in "\n".join(viewer.screen()))

        # Tab 7: Diff (config + summary diff between runA and runB)
        viewer.switch_tab("diff")
        _wait_until(
            viewer.sync,
            lambda: "Config Diff:" in "\n".join(viewer.screen())
            and "batch_size" in "\n".join(viewer.screen()),
        )
    finally:
        viewer.close()
