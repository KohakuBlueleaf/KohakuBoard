"""Playwright browser end-to-end tests against a live `kohakuboard-web` Uvicorn server."""

from __future__ import annotations

import math
import os
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import uvicorn
from playwright.sync_api import Browser, Playwright, sync_playwright

from kohakuboard import Artifact, Board, Histogram, KernelDensity, Media, Table
from kohakuboard_web import create_app


def _launch_chromium(pw: Playwright) -> Browser:
    try:
        return pw.chromium.launch()
    except Exception:
        cache_root = Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"
        if cache_root.exists():
            for pattern in (
                "chromium_headless_shell-*/**/chrome-headless-shell.exe",
                "chromium-*/**/chrome.exe",
            ):
                matches = sorted(cache_root.glob(pattern), reverse=True)
                if matches:
                    return pw.chromium.launch(executable_path=str(matches[0]))
        raise


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _make_rgb_image(size: int, seed: int) -> np.ndarray:
    xs = np.linspace(0.0, 2.0 * np.pi, size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, xs)
    r = ((np.sin(xx + seed) * 0.5 + 0.5) * 255).astype(np.uint8)
    g = ((np.cos(yy - seed) * 0.5 + 0.5) * 255).astype(np.uint8)
    b = ((np.sin(xx + yy) * 0.5 + 0.5) * 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


@pytest.fixture()
def live_server_url(tmp_path: Path) -> Iterator[str]:
    rng = np.random.default_rng(77)
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"weights-demo" * 32)

    specs = [
        ("r001", "dit-xl-amber", 3e-4, 1.0, 12_000),
        ("r002", "dit-xl-iolite", 1e-4, 0.75, 4_000),
        ("r003", "dit-b-aqua", 5e-4, 1.25, 2_000),
    ]
    for idx, (rid, name, lr, scale, steps) in enumerate(specs):
        b = Board(
            run_id=rid,
            name=name,
            project="vision",
            group="lr-sweep",
            tags=["v2", "bf16"],
            notes=f"Benchmark run {name}",
            config={"lr": lr, "batch_size": 128, "layers": 24 if idx < 2 else 12},
            dir=tmp_path,
            system_metrics=False,
            capture_output=True,
        )
        # Fast bulk scalar logging (r001 has 12,000 rows -> exercises multi-chunk streaming).
        for s in range(steps):
            val = (
                float("nan")
                if (rid == "r001" and s == steps // 2)
                else scale * math.exp(-s / (steps * 0.35)) + 0.04 * math.sin(s / 40.0)
            )
            b.log({"train/loss": val, "train/lr": lr, "eval/acc": 1.0 - 0.8 * math.exp(-s / 600.0)})

        # Log multiple multimodal snapshots under the same "eval/" namespace + "dist/"
        # to exercise unified multimodal auto-grouping and 2D flow heatmaps.
        for k in range(5):
            img = _make_rgb_image(48, idx * 10 + k)
            b.log(
                {
                    "eval/preview": Media(img, caption=f"{name} snap {k}", format="png"),
                    "eval/weights": Histogram(
                        rng.normal(loc=k * 0.25, scale=1.0 - k * 0.1, size=1024), bins=32
                    ),
                    "eval/table": Table(
                        columns=["prompt", "score", "img"],
                        rows=[[f"amber terrarium #{k}", round(0.85 + k * 0.02, 2), Media(img)]],
                    ),
                    "dist/kde": KernelDensity(
                        rng.normal(loc=k * 0.2, scale=0.8, size=1024), num_points=64
                    ),
                }
            )
        b.log({"ckpt": Artifact("step-final", path=ckpt)})
        b.alert("Checkpoint", f"Saved checkpoint for {name}", level="info")
        print(f"[train] {name} converged smoothly; sentinel token: GEMSTONE_{rid}")
        b.finish()

    port = _free_port()
    app = create_app(tmp_path, enforce_loopback=True)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 10.0
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "Uvicorn failed to start"

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def test_multimodal_board_tabs_sections_and_run_view(live_server_url: str) -> None:
    console_errors: list[str] = []
    page_errors: list[str] = []

    shots_env = os.environ.get("KOBO_SHOTS_DIR")
    shots_dir = Path(shots_env) if shots_env else None
    if shots_dir is not None:
        shots_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = _launch_chromium(pw)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on(
            "console",
            lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
        )
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        page.goto(live_server_url, wait_until="networkidle")

        # 0. Confirm old siloed modality buttons and Edit Layout UI are gone.
        body_text = page.locator("body").inner_text()
        assert "Edit Layout" not in body_text
        page.wait_for_selector('[data-testid="project-tab-view"]')
        page.wait_for_selector('[data-testid="runs-sidebar"]')
        assert "dit-xl-amber" in page.locator('[data-testid="runs-sidebar"]').inner_text()

        # 1. Auto-created namespace tabs across ALL modalities + built-in Runs Table.
        strip = page.locator('[data-testid="board-tab-strip"]')
        strip.locator('[data-testid="board-tab-eval"]').wait_for(state="visible", timeout=10_000)
        strip.locator('[data-testid="board-tab-train"]').wait_for(state="visible")
        strip.locator('[data-testid="board-tab-dist"]').wait_for(state="visible")
        strip.locator('[data-testid="builtin-tab-project"]').wait_for(state="visible")

        # 2. "eval" tab groups scalar + media + histogram + table TOGETHER.
        strip.locator('[data-testid="board-tab-eval"]').click()
        page.wait_for_selector('[data-testid="board-card-scalar"]')
        page.wait_for_selector('[data-testid="board-card-media"]')
        page.wait_for_selector('[data-testid="board-card-distribution"]')
        page.wait_for_selector('[data-testid="board-card-table"]')

        # Verify scalar uPlot, media <img>, distribution canvas, and table row all render.
        page.wait_for_function(
            """() => {
                const sc = document.querySelector('[data-testid="board-card-scalar"] canvas');
                const img = document.querySelector('[data-testid="board-card-media"] img');
                const d = document.querySelector('[data-testid="board-card-distribution"] canvas');
                const tbl = document.querySelector('[data-testid="board-card-table"]');
                const okImg = img && img.complete && img.naturalWidth > 0;
                const okTbl = tbl && tbl.innerText.includes('amber terrarium');
                return Boolean(sc && okImg && d && okTbl);
            }""",
            timeout=10_000,
        )

        # Toggle distribution card to 2D Flow Heatmap (uPlot hooks.draw Canvas2D).
        dist_card = page.locator('[data-testid="board-card-distribution"]').first
        dist_card.get_by_role("button", name="Flow").click()
        dist_card.locator('[data-testid="dist-flow-plot"] canvas').wait_for(
            state="visible", timeout=5_000
        )

        page.wait_for_timeout(150)
        if shots_dir is not None:
            page.screenshot(path=str(shots_dir / "v3_01_project_multimodal_eval_light.png"))

        # 3. Create a custom collapsible Section inside the active tab.
        page.locator('[data-testid="add-section-btn"]').click()
        sec_modal = page.locator('[data-testid="add-section-modal"]')
        sec_modal.wait_for(state="visible")
        sec_modal.locator("input").fill("Qualitative Check")
        sec_modal.get_by_role("button", name="Add Section").click()
        custom_sec = page.locator('[data-testid="board-section-Qualitative Check"]')
        custom_sec.wait_for(state="visible")

        # Collapse & expand section.
        custom_sec.get_by_text("Qualitative Check").click()
        custom_sec.get_by_text("Qualitative Check").click()

        # 4. Create a custom Board Tab & drag a card onto that Tab pill.
        page.locator('[data-testid="add-board-tab-btn"]').click()
        tab_modal = page.locator('[data-testid="add-tab-modal"]')
        tab_modal.wait_for(state="visible")
        tab_modal.locator("input").fill("Paper Figs")
        tab_modal.get_by_role("button", name="Create Tab").click()

        # Switch to "train" tab, verify NaN badge + canvas, drag card onto "Paper Figs" tab.
        strip.locator('[data-testid="board-tab-train"]').click()
        loss_card = page.locator('[data-card-id="auto:scalar:train/loss"]')
        loss_card.wait_for(state="visible", timeout=10_000)
        page.wait_for_function(
            """() => {
                const el = document.querySelector('[data-card-id="auto:scalar:train/loss"]');
                return el && el.querySelector('canvas') && el.innerText.includes('NaN');
            }""",
            timeout=10_000,
        )

        drag_header = loss_card.locator('[draggable="true"]').first
        paper_tab = strip.locator('[data-testid="board-tab-Paper Figs"]')
        drag_header.drag_to(paper_tab)
        paper_tab.click()
        page.locator('[data-card-id="auto:scalar:train/loss"]').wait_for(
            state="visible", timeout=5_000
        )

        # 5. Built-in "Runs Table" tab inside Project View.
        strip.locator('[data-testid="builtin-tab-project"]').click()
        runs_tbl = page.locator('[data-testid="runs-table"]')
        runs_tbl.wait_for(state="visible")
        assert "cfg.lr" in runs_tbl.inner_text()

        # 6. Open First-Class Run View directly from RailPane's Projects -> Runs tree.
        page.locator('[data-testid="rail-run-r001"]').click()
        run_view = page.locator('[data-testid="run-tab-view"]')
        run_view.wait_for(state="visible", timeout=10_000)
        assert "#/p/vision/r/r001" in page.url

        # Verify per-run auto-created multimodal tabs + sections exist inside Run View.
        run_strip = run_view.locator('[data-testid="board-tab-strip"]')
        run_eval_tab = run_strip.locator('[data-testid="board-tab-eval"]')
        run_eval_tab.wait_for(state="visible", timeout=10_000)
        run_eval_tab.click()
        run_view.locator('[data-testid="board-card-scalar"]').wait_for(state="visible")
        run_view.locator('[data-testid="board-card-media"]').wait_for(state="visible")
        run_view.locator('[data-testid="board-card-distribution"]').wait_for(state="visible")
        run_view.locator('[data-testid="board-card-table"]').wait_for(state="visible")

        # Open collapsible Run Dossier & verify alert/config/storage.
        run_view.locator('[data-testid="toggle-dossier-btn"]').click()
        dossier = run_view.locator('[data-testid="run-overview"]')
        dossier.wait_for(state="visible")
        dossier.get_by_text("Checkpoint").first.wait_for(state="visible", timeout=5_000)

        page.wait_for_timeout(200)
        if shots_dir is not None:
            page.screenshot(path=str(shots_dir / "v3_02_run_view_multimodal_light.png"))

        # Verify built-in "Logs" tab inside Run View with FTS5 search.
        run_strip.locator('[data-testid="builtin-tab-run"]').click()
        logs_panel = run_view.locator('[data-testid="console-logs"]')
        logs_panel.wait_for(state="visible")
        logs_panel.locator("input").fill("GEMSTONE_r001")
        page.wait_for_function(
            """() => {
                const el = document.querySelector('[data-testid="console-logs"]');
                return el && el.innerText.includes('GEMSTONE_r001');
            }""",
            timeout=5_000,
        )

        # 7. Switch to Warm Obsidian Dark Mode & inspect Project View multimodal eval tab.
        page.locator('[data-testid="theme-toggle"]').click()
        assert page.evaluate("document.documentElement.classList.contains('dark')") is True
        page.locator('[data-testid="tab-strip"]').get_by_text("vision").click()
        page.locator('[data-testid="project-tab-view"] [data-testid="board-tab-eval"]').click()
        flow_sel = '[data-testid="project-tab-view"] [data-testid="dist-flow-plot"] canvas'
        page.wait_for_selector(flow_sel)
        page.wait_for_timeout(200)
        if shots_dir is not None:
            page.screenshot(path=str(shots_dir / "v3_03_project_multimodal_dark.png"))

        # 8. Command Palette via Ctrl+K.
        page.keyboard.press("Control+k")
        page.locator('[data-testid="command-palette"]').wait_for(state="visible")
        page.keyboard.press("Escape")

        browser.close()

    assert not console_errors, f"Browser console errors: {console_errors}"
    assert not page_errors, f"Uncaught page errors: {page_errors}"
