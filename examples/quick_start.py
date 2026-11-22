"""Minimal KohakuBoard v2 quick start: drop-in `kobo.init()` / `kobo.log()` without `with`."""

from __future__ import annotations

import math
from pathlib import Path

import kohakuboard as kobo
from kohakuboard import ProjectReader, RunReader

OUT_DIR = Path("./kohakuboard_examples")


def train_loop() -> None:
    """Any function can log directly to the active board without passing a handle around."""
    loss = 0.0
    for step in range(200):
        loss = 2.0 * math.exp(-step / 60.0) + 0.05 * math.sin(step / 5.0)
        lr = 3e-4 * 0.5 * (1.0 + math.cos(math.pi * step / 200.0))

        # commit=False lets multiple subsystems log onto the same step cleanly.
        kobo.log({"train/loss": loss}, commit=False)
        kobo.log({"train/lr": lr, "train/acc": 1.0 - 0.5 * math.exp(-step / 80.0)}, commit=True)

        if step % 50 == 0:
            print(f"step={step:03d} loss={loss:.4f} lr={lr:.2e}")

    kobo.summary_update({"best_loss": loss})


def main() -> None:
    # Drop-in W&B-style init — no `with` block required (atexit + chained excepthook
    # flushes and records finished/interrupted/crashed automatically on exit).
    board = kobo.init(
        name="cosine-demo",
        project="quickstart",
        group="baseline",
        tags=["v2", "demo"],
        config={"lr": 3e-4, "batch_size": 128, "optimizer": "adamw"},
        dir=OUT_DIR,
        system_metrics=False,
        capture_output=True,
    )
    run_dir = board.run_dir
    print(f"Started run '{board.name}' (id={board.run_id}) at {run_dir}")

    train_loop()
    # Explicit finish() is optional at script end; call it here before reading the final state.
    kobo.finish()

    # Inspect runs offline via ProjectReader + RunReader (no server required).
    with ProjectReader(OUT_DIR, project="quickstart") as proj:
        runs = proj.list_runs()
        print(f"\nIndexed {len(runs)} run(s) in project 'quickstart':")
        for r in runs:
            last_loss = r.summary.get("train/loss", {}).get("last")
            print(
                f"  - {r.run_id} ({r.name}) state={r.state} "
                f"steps={r.last_step + 1} loss={last_loss:.4f}"
            )

    assert run_dir is not None
    with RunReader(run_dir) as run:
        steps, gsteps, _, values = run.scalar("train/loss")
        print(f"\nRead {len(values)} points from {run_dir / 'run.db'}:")
        print(f"  step[0]={steps[0]} (gstep={gsteps[0]}): {values[0]:.4f}")
        print(f"  step[-1]={steps[-1]} (gstep={gsteps[-1]}): {values[-1]:.4f}")
        hits = run.search_logs("step=150")
        print(f"  FTS5 log search for 'step=150': {len(hits)} hit(s)")


if __name__ == "__main__":
    main()
