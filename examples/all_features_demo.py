"""Multimodal KohakuBoard v2 showcase: scalars, media, tables, distributions, and logs."""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

from kohakuboard import (
    Artifact,
    Board,
    Histogram,
    KernelDensity,
    Media,
    RunReader,
    Table,
    TensorLog,
)

OUT_DIR = Path("./kohakuboard_examples")


def _make_gradient_image(size: int, phase: float) -> np.ndarray:
    """Create a colorful synthetic HWC uint8 RGB gradient image."""
    xs = np.linspace(0.0, 2.0 * np.pi, size, dtype=np.float32)
    ys = np.linspace(0.0, 2.0 * np.pi, size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    r = ((np.sin(xx + phase) * 0.5 + 0.5) * 255).astype(np.uint8)
    g = ((np.cos(yy - phase) * 0.5 + 0.5) * 255).astype(np.uint8)
    b = ((np.sin(xx + yy + phase) * 0.5 + 0.5) * 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def main() -> None:
    rng = np.random.default_rng(2026)

    with tempfile.TemporaryDirectory() as tmp_ckpt_dir:
        ckpt_path = Path(tmp_ckpt_dir) / "checkpoint_step_50.safetensors"
        ckpt_path.write_bytes(b"synthetic-safetensors-payload" * 512)

        # Explicit Board handle — no `with` block required.
        board = Board(
            name="multimodal-showcase",
            project="demo",
            annotation="full",
            group="v2-tour",
            job_type="train",
            tags=["multimodal", "rust-core"],
            notes="Demonstrates multimodal logging, SWMR live reads, and run resume.",
            config={"model": {"arch": "dit-b", "layers": 12}, "lr": 1e-4, "precision": "bf16"},
            dir=OUT_DIR,
            system_metrics=0.5,
            capture_output=True,
            flush_interval_ms=100,
        )
        run_dir = board.run_dir
        run_id = board.run_id
        board.define_metric("eval/*", step_metric="global_step", summary="min")
        board.config_update({"param_count_m": 131.2})
        board.add_tags("verified")

        for i in range(50):
            loss = 1.8 * math.exp(-i / 18.0) + float(rng.normal(0.0, 0.03))
            board.log(
                {
                    "train/loss": loss,
                    "train/lr": 1e-4,
                    "nested": {"grad_norm": 1.0 / (1.0 + i * 0.05)},
                }
            )
            board.progress(i + 1, 50, label="train", unit="step")

            if i % 15 == 0:
                img = _make_gradient_image(64, phase=i * 0.25)
                # Identical duplicate image verifies blake3 dedup in blobs.db while
                # keeping both timeline events in kb_media_events.
                board.log(
                    {
                        "samples/grid": [
                            Media(img, caption=f"step {i} main", format="png"),
                            Media(img, caption=f"step {i} dedup-copy", format="png"),
                        ],
                        "samples/webp": Media(img, caption="lossless webp", format="webp"),
                        "dist/weights_hist": Histogram(
                            rng.normal(0.0, 1.0 / (1.0 + i * 0.02), size=4096),
                            bins=64,
                            precision="exact",
                        ),
                        "dist/weights_kde": KernelDensity(
                            rng.normal(0.0, 1.0, size=8192),
                            num_points=128,
                            approximate=True,
                        ),
                    },
                    commit=False,
                )
                print(f"[train] logged media & distributions at step={i}")

        # Non-finite IEEE-754 scalars preserved verbatim without poisoning min/max.
        board.log({"train/loss": float("nan"), "train/overflow_check": float("inf")})
        print("WARNING: injected synthetic NaN/Inf sentinel at step 50", file=sys.stderr)
        board.alert("Sentinel NaN", "Injected NaN to verify IEEE-754 preservation", level="warn")

        # Tables (embedded Media cell + TensorLog + Artifact).
        preview_img = _make_gradient_image(32, phase=1.0)
        eval_table = Table(
            columns=["prompt", "score", "preview"],
            rows=[
                ["amber crystal terrarium", 0.94, Media(preview_img, caption="top-1")],
                ["mossy stone garden", 0.89, Media(preview_img, caption="top-2")],
            ],
        )
        tensor = rng.standard_normal((8, 16), dtype=np.float32)
        board.log(
            {
                "eval/samples": eval_table,
                "eval/fid": 4.21,
                "debug/attn_map": TensorLog(tensor, compress=True, metadata={"head": 0}),
                "checkpoints/latest": Artifact("step-50", kind="checkpoint", path=ckpt_path),
            }
        )
        board.flush()

        # Demonstrate live SWMR read while the Board is still open and running.
        assert run_dir is not None
        with RunReader(run_dir) as live:
            print(
                f"\n[SWMR live read] state={live.info['state']} "
                f"train/loss_rows={live.scalar_len('train/loss')} "
                f"media_events={len(live.media('samples/grid'))}"
            )

        board.finish()

    # Demonstrate clean resume="must" continuing steps seamlessly.
    resumed = Board(
        run_id=run_id,
        project="demo",
        dir=OUT_DIR,
        resume="must",
        config={"resumed_stage": 2},
        system_metrics=False,
        capture_output=False,
    )
    for extra in range(5):
        resumed.log({"train/loss": 0.15 - extra * 0.01})
    resumed.finish()

    # Final verification summary via offline RunReader.
    assert run_dir is not None
    with RunReader(run_dir) as reader:
        info = reader.info
        _, _, _, loss_vals = reader.scalar("train/loss")
        grid_events = reader.media("samples/grid")
        unique_blobs = {ev["blob_key"] for ev in grid_events}
        t_rec = reader.tensors("debug/attn_map")[0]
        t_back = reader.read_tensor(t_rec)
        log_hits = reader.search_logs("NaN")

        run_kib = (run_dir / "run.db").stat().st_size / 1024.0
        blobs_kib = (run_dir / "blobs.db").stat().st_size / 1024.0
        max_err = float(np.max(np.abs(t_back - tensor)))
        arts = reader.artifacts()

        print("\n=== Final Run Summary ===")
        print(f"  run_dir:          {run_dir}")
        print(
            f"  state:            {info['state']} "
            f"(last_step={info['last_step']}, gstep={info['last_global_step']})"
        )
        print(
            f"  scalars:          {len(reader.scalar_names)} metrics "
            f"({len(loss_vals)} loss pts, NaN preserved={bool(np.isnan(loss_vals[50]))})"
        )
        print(
            f"  media dedup:      {len(grid_events)} events -> "
            f"{len(unique_blobs)} unique blob(s) in blobs.db"
        )
        print(
            f"  distributions:    {len(reader.histograms('dist/weights_hist'))} histograms, "
            f"{len(reader.kdes('dist/weights_kde'))} KDEs"
        )
        print(f"  tables:           {len(reader.tables('eval/samples'))} snapshot(s)")
        print(
            f"  tensor roundtrip: shape={t_back.shape} dtype={t_back.dtype} "
            f"max_err={max_err:.1e}"
        )
        print(f"  artifacts:        {len(arts)} (digest={arts[0]['digest'][:12]}...)")
        print(f"  alerts:           {len(reader.alerts())}")
        print(f"  FTS5 log hits:    {len(log_hits)} hit(s) for 'NaN'")
        print(f"  storage files:    run.db={run_kib:.1f} KiB, blobs.db={blobs_kib:.1f} KiB")


if __name__ == "__main__":
    main()
