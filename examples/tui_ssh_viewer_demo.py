"""Generate a 3-run multimodal project and launch the standalone SSH TUI viewer (`kobo tui`).

Run:
    python examples/tui_ssh_viewer_demo.py
    python examples/tui_ssh_viewer_demo.py --no-launch   # generate data only

Inside the viewer:
    1-7   switch tabs (Overview / Metrics / Tables / Media / Dist / Logs / Diff)
    c     toggle runs to compare/overlay
    m     fuzzy-pick a scalar metric to inspect
    /     FTS5 search across console logs
    r     refresh from disk (safe while runs are actively logging!)
    q     quit
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from kohakuboard import Board, Histogram, KernelDensity, Media, Table
from kohakuboard.cli.main import main as kobo_cli_main


def _make_rgb(size: int, seed: int) -> np.ndarray:
    xs = np.linspace(0.0, 2.0 * np.pi, size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, xs)
    r = ((np.sin(xx + seed) * 0.5 + 0.5) * 255).astype(np.uint8)
    g = ((np.cos(yy - seed) * 0.5 + 0.5) * 255).astype(np.uint8)
    b = ((np.sin(xx + yy) * 0.5 + 0.5) * 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def generate_demo_project(out_dir: Path) -> None:
    rng = np.random.default_rng(2026)
    runs = [
        ("dit-xl-amber", 3e-4, 256, 1.00, 180),
        ("dit-xl-iolite", 1e-4, 256, 0.82, 180),
        ("dit-b-aqua", 5e-4, 128, 1.18, 140),
    ]
    for idx, (name, lr, batch, scale, steps) in enumerate(runs):
        with Board(
            name=name,
            project="ssh-viewer-demo",
            group="lr-sweep",
            tags=["demo", f"bs{batch}"],
            config={"lr": lr, "batch_size": batch, "layers": 24 if idx < 2 else 12},
            dir=out_dir,
            system_metrics=False,
            capture_output=True,
        ) as board:
            for s in range(1, steps + 1):
                loss = scale * math.exp(-s / 60.0) + 0.03 * math.sin(s / 8.0)
                val_acc = min(0.98, 1.0 - 0.8 * math.exp(-s / 55.0))
                board.log(
                    {
                        "train/loss": loss,
                        "train/lr": lr * 0.5 * (1.0 + math.cos(math.pi * s / steps)),
                        "eval/acc": val_acc,
                    },
                    global_step=s,
                )
            for k in range(3):
                img = _make_rgb(48, idx * 10 + k)
                board.log(
                    {
                        "samples/preview": Media(img, caption=f"{name} snap #{k}", format="png"),
                        "dist/weights": Histogram(rng.normal(k * 0.15, 0.9, size=1024), bins=24),
                        "dist/kde": KernelDensity(
                            rng.normal(k * 0.15, 0.9, size=1024), num_points=48
                        ),
                        "eval/table": Table(
                            columns=["prompt", "fid", "clip", "preview"],
                            rows=[
                                [
                                    f"warm stone #{k}",
                                    round(11.5 - k * 1.1 + idx, 2),
                                    round(0.31 + k * 0.02, 3),
                                    Media(img),
                                ]
                            ],
                        ),
                    }
                )
            print(f"[done] {name} finished with lr={lr} batch={batch}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 3-run project and launch `kobo tui`.")
    parser.add_argument("--dir", default="./kohakuboard_examples", help="Output data directory")
    parser.add_argument("--theme", default="kohaku-dark", help="KohakuTUI theme")
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Only generate runs without launching the interactive TUI",
    )
    args = parser.parse_args()

    out_dir = Path(args.dir)
    generate_demo_project(out_dir)
    print(f"Generated 3 runs in {out_dir / 'ssh-viewer-demo'}")
    print(f"  Terminal viewer: kobo tui {out_dir} --project ssh-viewer-demo")
    print(f"  Web UI viewer:   kobo open {out_dir}")

    if not args.no_launch:
        kobo_cli_main(["tui", str(out_dir), "--project", "ssh-viewer-demo", "--theme", args.theme])


if __name__ == "__main__":
    main()
