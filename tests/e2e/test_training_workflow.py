"""End-to-end training workflows covering all data types, SWMR live reads, resume, and KV parity."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import kohakuboard as kobo
from kohakuboard import (
    Artifact,
    Board,
    Histogram,
    KernelDensity,
    Media,
    ProjectReader,
    RunReader,
    Table,
    TensorLog,
    errors,
)


def test_full_multimodal_training_workflow(tmp_path: Path) -> None:
    ckpt_file = tmp_path / "model.bin"
    ckpt_file.write_bytes(b"checkpoint-weights-v2" * 64)

    with Board(
        name="dit-xl-e2e",
        project="hdm",
        run_id="e2e1",
        annotation="stage1",
        group="lr-sweep",
        job_type="train",
        tags=["baseline"],
        notes="initial run",
        config={"lr": 3e-4, "model": {"depth": 24}},
        dir=tmp_path,
        system_metrics=False,
        capture_output=True,
        flush_interval_ms=50,
    ) as board:
        assert board.run_dir == tmp_path / "hdm" / "e2e1_stage1"
        board.define_metric("eval/*", step_metric="epoch", summary="min")
        board.config_update({"params_m": 675})
        board.add_tags("fp16", "baseline")
        board.set_notes("updated notes")

        for step_idx in range(20):
            board.step()
            # commit=False accumulates multiple calls onto the same internal step.
            board.log({"train/loss": 2.0 / (step_idx + 1)}, commit=False)
            board.log({"nested": {"lr": 3e-4}}, grad_norm=float(step_idx) * 0.5, commit=True)
            board.progress(step_idx + 1, 20, label="train")

        # Non-finite IEEE-754 scalars preserved verbatim.
        board.log({"train/loss": float("nan"), "train/inf": float("inf")})

        # Rich types: Media (HWC u8, CHW float [-1,1], PIL Image, list + dedup).
        img_hwc = np.full((8, 8, 3), 120, dtype=np.uint8)
        img_chw = np.linspace(-1.0, 1.0, 3 * 6 * 6, dtype=np.float32).reshape(3, 6, 6)
        pil_img = Image.new("RGB", (6, 6), color=(10, 200, 30))
        board.log(
            {
                "samples": [
                    Media(img_hwc, caption="a", format="png"),
                    Media(img_hwc, caption="dup", format="png"),
                ],
                "chw": Media(img_chw, format="webp"),
                "pil": Media(pil_img, format="jpg", quality=85),
            }
        )

        # Table with embedded Media cell + large table spilling to blobs.db.
        small_tbl = Table(
            columns=["prompt", "score", "preview"],
            rows=[["cat", 0.95, Media(img_hwc, caption="cell")], ["dog", 0.88, None]],
        )
        big_rows = [[i, "x" * 128] for i in range(700)]
        big_tbl = Table(columns=["idx", "payload"], rows=big_rows)
        board.log({"eval/preds": small_tbl, "eval/big": big_tbl})

        # Histogram (raw + precomputed) & KernelDensity (raw FFT + precomputed).
        rng = np.random.default_rng(42)
        samples = rng.normal(0.0, 1.0, size=2048).astype(np.float32)
        board.log(
            {
                "dist/hist_raw": Histogram(samples, bins=32, precision="exact"),
                "dist/hist_pre": Histogram(
                    edges=[0.0, 1.0, 2.0], counts=[3, 7], precision="compact"
                ),
                "dist/kde_raw": KernelDensity(samples, num_points=64, approximate=True),
                "dist/kde_pre": KernelDensity(grid=[0.0, 0.5, 1.0], density=[0.1, 0.8, 0.1]),
            }
        )

        # TensorLog (compressed & raw) + Artifact + Alert + console logs.
        t_arr = np.arange(24, dtype=np.float32).reshape(4, 6)
        board.log(
            {
                "tensors/comp": TensorLog(t_arr, compress=True, metadata={"layer": "attn"}),
                "tensors/raw": TensorLog(t_arr, compress=False),
                "ckpt": Artifact("step-20", kind="checkpoint", path=ckpt_file),
            }
        )
        board.alert("NaN spike", "train/loss hit NaN at step 20", level="warn")
        board.summary_update({"best_fid": 1.72})
        print("Epoch 1 completed cleanly without divergence")
        print("CRITICAL NaN gradient alert on layer 12", file=sys.stderr)

        # Live SWMR check before finish()!
        board.flush()
        snap = board.snapshot()
        assert snap["metrics"]["train/loss"]["count"] == 21
        assert snap["progress"]["train"]["current"] == 20

        with RunReader(board.run_dir) as live:
            assert live.info["state"] == "running"
            assert live.scalar_len("train/loss") == 21

    # Post-finish verification.
    run_dir = tmp_path / "hdm" / "e2e1_stage1"
    with RunReader(run_dir) as reader:
        assert reader.info["state"] == "finished"
        assert reader.info["exit_code"] == 0
        assert reader.info["tags"] == ["baseline", "fp16"]
        assert reader.info["notes"] == "updated notes"
        assert reader.config["params_m"] == 675
        assert reader.config["model"]["depth"] == 24
        assert reader.summary["best_fid"]["last"] == pytest.approx(1.72)
        assert reader.metric_definitions[0]["pattern"] == "eval/*"

        steps, gsteps, _, vals = reader.scalar("train/loss")
        assert len(vals) == 21
        assert math.isnan(vals[-1])
        assert steps[0] == 0 and steps[19] == 19
        assert gsteps[0] == 1 and gsteps[19] == 20
        _, _, _, inf_vals = reader.scalar("train/inf")
        assert math.isinf(inf_vals[0]) and inf_vals[0] > 0
        assert reader.scalar_len("nested/lr") == 20
        assert reader.scalar_len("grad_norm") == 20

        # Media deduplication: two identical images share one blob_key, two events recorded.
        sample_events = reader.media("samples")
        assert len(sample_events) == 2
        assert sample_events[0]["blob_key"] == sample_events[1]["blob_key"]
        blob_bytes = reader.blob(sample_events[0]["blob_key"])
        assert blob_bytes is not None and blob_bytes.startswith(b"\x89PNG")
        assert len(reader.media("chw")) == 1
        assert len(reader.media("pil")) == 1

        # Tables (embedded media + spilled large table).
        preds = reader.tables("eval/preds")
        assert len(preds) == 1
        media_cell = preds[0]["rows"][0][2]
        assert "$media" in media_cell
        assert reader.blob(media_cell["$media"]) is not None
        big = reader.tables("eval/big")
        assert big[0]["row_count"] == 700
        assert len(big[0]["rows"]) == 700

        # Histograms + KDEs.
        h_raw = reader.histograms("dist/hist_raw")
        assert len(h_raw) == 1 and len(h_raw[0]["counts"]) == 32
        h_pre = reader.histograms("dist/hist_pre")
        assert len(h_pre) == 1 and len(h_pre[0]["counts"]) == 2
        k_raw = reader.kdes("dist/kde_raw")
        assert len(k_raw) == 1 and len(k_raw[0]["density"]) == 64
        k_pre = reader.kdes("dist/kde_pre")
        assert len(k_pre) == 1 and len(k_pre[0]["grid"]) == 3

        # Tensors round-trip.
        t_comp_rec = reader.tensors("tensors/comp")[0]
        assert t_comp_rec["compressed"] is True
        np.testing.assert_array_equal(reader.read_tensor(t_comp_rec), t_arr)
        t_raw_rec = reader.tensors("tensors/raw")[0]
        assert t_raw_rec["compressed"] is False
        np.testing.assert_array_equal(reader.read_tensor(t_raw_rec), t_arr)

        # Artifacts + Alerts + Console Logs FTS5 search.
        arts = reader.artifacts()
        assert len(arts) == 1
        assert arts[0]["name"] == "step-20"
        assert arts[0]["size_bytes"] == ckpt_file.stat().st_size
        assert arts[0]["digest"] is not None and len(arts[0]["digest"]) == 32
        alerts = reader.alerts()
        assert len(alerts) == 1 and alerts[0]["title"] == "NaN spike"
        hits = reader.search_logs("divergence")
        assert len(hits) == 1 and "Epoch 1 completed" in hits[0]["text"]
        err_hits = reader.search_logs("layer 12")
        assert len(err_hits) == 1 and err_hits[0]["stream"] == "stderr"


def test_resume_and_exit_states(tmp_path: Path) -> None:
    with pytest.raises(errors.RunNotFoundError):
        Board(run_id="miss", project="p", dir=tmp_path, resume="must", system_metrics=False)

    # Simulate crashed run via exception in context manager.
    with (
        pytest.raises(RuntimeError, match="boom"),
        Board(
            run_id="rsm1",
            project="p",
            config={"lr": 1e-3},
            dir=tmp_path,
            system_metrics=False,
            capture_output=False,
        ) as b1,
    ):
        b1.step(value=10)
        b1.log({"loss": 1.5})
        b1.log({"loss": 1.2})
        raise RuntimeError("boom")

    run_dir = tmp_path / "p" / "rsm1"
    with RunReader(run_dir) as r:
        assert r.info["state"] == "crashed"
        assert r.info["exit_code"] == 1
        assert r.scalar_len("loss") == 2

    # Collision with resume="never" raises RunExistsError.
    with pytest.raises(errors.RunExistsError):
        Board(run_id="rsm1", project="p", dir=tmp_path, resume="never", system_metrics=False)

    # Resume with resume="must" continues step & global_step seamlessly.
    with Board(
        run_id="rsm1",
        project="p",
        config={"wd": 0.01},
        dir=tmp_path,
        resume="must",
        system_metrics=False,
        capture_output=False,
    ) as b2:
        assert b2.current_step == 1
        assert b2.global_step == 10
        b2.step()
        b2.log({"loss": 0.9})

    with RunReader(run_dir) as r:
        assert r.info["state"] == "finished"
        assert r.config == {"lr": 1e-3, "wd": 0.01}
        steps, gsteps, _, vals = r.scalar("loss")
        assert steps.tolist() == [0, 1, 2]
        assert gsteps.tolist() == [10, 10, 11]
        assert vals.tolist() == pytest.approx([1.5, 1.2, 0.9])

    # KeyboardInterrupt marks state as interrupted.
    with (
        pytest.raises(KeyboardInterrupt),
        Board(
            run_id="int1",
            project="p",
            dir=tmp_path,
            system_metrics=False,
            capture_output=False,
        ) as b3,
    ):
        b3.log({"x": 1.0})
        raise KeyboardInterrupt

    with RunReader(tmp_path / "p" / "int1") as r:
        assert r.info["state"] == "interrupted"
        assert r.info["exit_code"] == 130


def test_kohakuvault_python_interoperability(tmp_path: Path) -> None:
    """Verify `kohakuvault` Python package can directly inspect `run.db` created by `Board`."""
    kv_mod = pytest.importorskip("kohakuvault")
    with Board(
        run_id="kv01",
        project="interop",
        dir=tmp_path,
        system_metrics=False,
        capture_output=False,
    ) as board:
        for i in range(5):
            board.log({"train/loss": float(i)})

    db_path = str(tmp_path / "interop" / "kv01" / "run.db")
    kv = kv_mod.KVault(db_path, table="kb_meta")
    assert kv.get(b"run/info") is not None
    kv.close()

    with ProjectReader(tmp_path, project="interop") as pr:
        runs = pr.list_runs()
        assert len(runs) == 1 and runs[0].run_id == "kv01"


def test_module_api_and_default_global_step(tmp_path: Path) -> None:
    """Verify `kobo.init()` / `kobo.log()`, strict uninitialized error, and smart `global_step`."""
    kobo.finish()
    assert kobo.current() is None

    with pytest.raises(errors.ConfigError, match="not been initialized"):
        kobo.log({"loss": 1.0})

    board = kobo.init(
        run_id="mod1",
        name="module-run",
        project="modproj",
        dir=tmp_path,
        system_metrics=False,
        capture_output=False,
    )
    assert kobo.current() is board

    # Second init without reinit=True raises clearly.
    with pytest.raises(errors.ConfigError, match="already initialized"):
        kobo.init(project="modproj", dir=tmp_path, system_metrics=False, capture_output=False)

    # Without calling step(), global_step defaults to matching step.
    kobo.log({"loss": 1.0})
    kobo.log({"loss": 0.5})
    kobo.log({"loss": 0.25}, step=10)
    kobo.alert("milestone", "reached step 10")

    # reinit=True finishes mod1 and starts mod2 cleanly.
    board2 = kobo.init(
        run_id="mod2",
        project="modproj",
        dir=tmp_path,
        reinit=True,
        system_metrics=False,
        capture_output=False,
    )
    assert board.closed is True
    assert kobo.current() is board2
    kobo.log({"loss": 0.1})
    kobo.finish()
    assert kobo.current() is None

    with RunReader(tmp_path / "modproj" / "mod1") as r1:
        steps, gsteps, _, vals = r1.scalar("loss")
        assert steps.tolist() == [0, 1, 10]
        assert gsteps.tolist() == [0, 1, 10]
        assert vals.tolist() == pytest.approx([1.0, 0.5, 0.25])
        assert len(r1.alerts()) == 1

    with RunReader(tmp_path / "modproj" / "mod2") as r2:
        assert r2.info["state"] == "finished"
        assert r2.scalar_len("loss") == 1


def test_no_with_normal_exit_and_crash_in_subprocess(tmp_path: Path) -> None:
    """Plain `kobo.init()` / `Board()` without `with` or `finish()` flushes & sets exact state."""
    ok_script = (
        "import kohakuboard as kobo\n"
        f"kobo.init(run_id='sub1', project='sub', dir={str(tmp_path)!r}, "
        "system_metrics=False, capture_output=False)\n"
        "for i in range(10):\n"
        "    kobo.log({'loss': float(i)})\n"
    )
    res_ok = subprocess.run([sys.executable, "-c", ok_script], capture_output=True, text=True)
    assert res_ok.returncode == 0, res_ok.stderr

    with RunReader(tmp_path / "sub" / "sub1") as r_ok:
        assert r_ok.info["state"] == "finished"
        assert r_ok.info["exit_code"] == 0
        steps, gsteps, _, vals = r_ok.scalar("loss")
        assert len(vals) == 10
        assert steps.tolist() == gsteps.tolist() == list(range(10))

    crash_script = (
        "from kohakuboard import Board\n"
        f"b = Board(run_id='sub2', project='sub', dir={str(tmp_path)!r}, "
        "system_metrics=False, capture_output=False)\n"
        "for i in range(5):\n"
        "    b.log({'loss': float(i)})\n"
        "raise RuntimeError('simulated training crash')\n"
    )
    res_crash = subprocess.run([sys.executable, "-c", crash_script], capture_output=True, text=True)
    assert res_crash.returncode != 0
    assert "simulated training crash" in res_crash.stderr

    with RunReader(tmp_path / "sub" / "sub2") as r_crash:
        assert r_crash.info["state"] == "crashed"
        assert r_crash.info["exit_code"] == 1
        assert r_crash.scalar_len("loss") == 5
