"""Interactive live training TUI example using `Board(tui=TuiConfig(...))`.

Run:
    python examples/tui_training_demo.py
    python examples/tui_training_demo.py --mode fullscreen
    python examples/tui_training_demo.py --steps 400 --high-cardinality --speed 0.5

Hotkeys while running:
    ctrl+t  toggle Inline <-> Fullscreen workstation
    1-3     switch tabs in Fullscreen (Overview / Metrics / Logs)
    p       open fuzzy Metric Picker (pin any metric as a live chart)
    e       toggle EMA smoothing overlay
    q       hide/detach the TUI without stopping training
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np

from kohakuboard import Board, ChartSpec, Histogram, Media, Table, TuiConfig


def _make_sample_image(step: int, size: int = 48) -> np.ndarray:
    xs = np.linspace(0.0, 2.0 * np.pi, size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, xs)
    phase = step * 0.03
    r = ((np.sin(xx + phase) * 0.5 + 0.5) * 255).astype(np.uint8)
    g = ((np.cos(yy - phase) * 0.5 + 0.5) * 255).astype(np.uint8)
    b = ((np.sin(xx + yy + phase) * 0.5 + 0.5) * 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="KohakuBoard live TUI training demo.")
    parser.add_argument("--dir", default="./kohakuboard_examples", help="Output data directory")
    parser.add_argument(
        "--mode",
        choices=("inline", "fullscreen"),
        default="inline",
        help="Initial TUI mode (ctrl+t toggles live)",
    )
    parser.add_argument("--steps", type=int, default=240, help="Total simulated training steps")
    parser.add_argument("--speed", type=float, default=1.0, help="Step sleep scale (0 = instant)")
    parser.add_argument("--theme", default="kohaku-dark", help="KohakuTUI color theme")
    parser.add_argument(
        "--high-cardinality",
        action="store_true",
        help="Log 64 extra per-layer gradient norms to showcase bounded TUI & Picker",
    )
    parser.add_argument(
        "--on-finish",
        choices=("seal", "clear", "hold"),
        default="seal",
        help="Behavior when training finishes (default: seal curve to scrollback)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    random.seed(42)

    tui_cfg = TuiConfig(
        mode=args.mode,
        theme=args.theme,
        refresh_hz=12.0,
        total_steps=args.steps,
        charts=[
            ChartSpec(["train/loss", "val/loss"], label="Loss", height=9, ema=0.95),
            ChartSpec("train/grad_norm", label="Grad Norm", height=5, digits=3, ema=0.9),
        ],
        watch=["train/lr", "eval/acc"],
        hide=["grad/layer_*"],
        log_height=4,
        on_finish=args.on_finish,
    )

    board = Board(
        name="tiny-dit-tui",
        project="tui-demo",
        group="cosine-lr",
        tags=["demo", "bf16"],
        config={
            "model": "tiny-dit",
            "lr": 3e-4,
            "batch_size": 256,
            "steps": args.steps,
        },
        dir=Path(args.dir),
        tui=tui_cfg,
        system_metrics=1.0,
    )

    loss = 2.4
    val_loss = 2.6
    for s in range(1, args.steps + 1):
        loss = max(0.06, loss * 0.992 + random.gauss(0.0, 0.025))
        grad_norm = abs(random.gauss(0.85, 0.18)) * (3.5 if s % 90 == 0 else 1.0)
        lr = 3e-4 * 0.5 * (1.0 + math.cos(math.pi * s / args.steps))
        is_nan = s == args.steps // 2

        payload: dict[str, object] = {
            "train/loss": float("nan") if is_nan else loss,
            "train/grad_norm": grad_norm,
            "train/lr": lr,
        }
        if args.high_cardinality:
            for layer in range(64):
                payload[f"grad/layer_{layer:02d}"] = abs(random.gauss(0.4 + layer * 0.01, 0.08))

        if s % 10 == 0 or s == 1:
            val_loss = max(0.09, val_loss * 0.96 + random.gauss(0.0, 0.015))
            payload["val/loss"] = val_loss
            payload["eval/acc"] = min(0.99, 1.0 - 0.85 * math.exp(-s / (args.steps * 0.35)))

        if s % 60 == 0 or s == args.steps:
            img = _make_sample_image(s)
            payload["samples/preview"] = Media(img, caption=f"step {s} sample", format="png")
            payload["dist/weights"] = Histogram(rng.normal(0.0, 1.0 / math.sqrt(1 + s / 80), 1024))
            payload["eval/summary"] = Table(
                columns=["step", "loss", "val_loss", "grad_norm"],
                rows=[[s, round(loss, 4), round(val_loss, 4), round(grad_norm, 3)]],
            )
            print(f"[eval] step {s}/{args.steps}: loss={loss:.4f} val_loss={val_loss:.4f}")

        if is_nan:
            print(f"[warn] step {s}: NaN loss detected, batch skipped")
            board.alert("NaN batch", f"Skipped NaN batch at step {s}", level="warn")

        board.log(payload, global_step=s)
        board.progress(s, total=args.steps, label="train")

        if args.speed > 0:
            time.sleep(0.02 * args.speed)

    board.finish()


if __name__ == "__main__":
    main()
