"""Integration tests for storage scalability, catalog queries, binary chunking, and modes."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from kohakuboard import Board, ProjectReader, RunReader, errors


def _decode_frame(frame: bytes) -> tuple[dict, bytes]:
    (hdr_len,) = struct.unpack_from("<I", frame, 0)
    offset = 4 + hdr_len
    assert offset % 8 == 0, f"payload offset {offset} must be 8-byte aligned"
    header = json.loads(frame[4:offset].decode("utf-8"))
    return header, frame[offset:]


def test_thousand_metrics_single_run_db_file(tmp_path: Path) -> None:
    """1,000 distinct scalar metrics live inside one `run.db` (zero FD explosion)."""
    with Board(
        run_id="m1k0",
        project="scale",
        dir=tmp_path,
        system_metrics=False,
        capture_output=False,
        flush_interval_ms=100,
    ) as board:
        payload = {f"layer_{i // 10}/metric_{i}": float(i) for i in range(1_000)}
        board.log(payload)
        board.log({k: v + 1.0 for k, v in payload.items()})

    run_dir = tmp_path / "scale" / "m1k0"
    db_files = [p.name for p in run_dir.iterdir() if p.suffix == ".db"]
    assert db_files == ["run.db"], f"expected single run.db (no lazy blobs.db), got {db_files}"

    with RunReader(run_dir) as reader:
        assert len(reader.scalar_names) == 1_000
        _, _, _, vals = reader.scalar("layer_42/metric_427")
        assert vals.tolist() == [427.0, 428.0]


def test_newest_first_binary_chunk_frames_lossless(tmp_path: Path) -> None:
    n_rows = 20_000
    with Board(
        run_id="chk1",
        project="chunks",
        dir=tmp_path,
        system_metrics=False,
        capture_output=False,
    ) as board:
        for i in range(n_rows):
            val = float("nan") if i == 12345 else float(i) * 0.5
            board.log({"loss": val})

    run_dir = tmp_path / "chunks" / "chk1"
    with RunReader(run_dir) as reader:
        plan = reader.scalar_newest_first_plan("loss", first_chunk=4096, max_chunk=8192)
        assert plan[0] == (n_rows - 4096, n_rows)
        assert plan[-1][0] == 0

        reconstructed = np.full(n_rows, -1.0, dtype=np.float64)
        for start, end in plan:
            frame = reader.scalar_range_frame("chk1", "loss", start, end, x_dtype="f64")
            hdr, payload = _decode_frame(frame)
            assert hdr["start"] == start and hdr["end"] == end
            assert hdr["total_rows"] == n_rows
            assert hdr["done"] == (start == 0)
            count = end - start
            cols = np.frombuffer(payload, dtype="<f8").reshape(4, count)
            np.testing.assert_array_equal(cols[0], np.arange(start, end, dtype=np.float64))
            reconstructed[start:end] = cols[3]

        assert np.isnan(reconstructed[12345])
        assert reconstructed[0] == 0.0
        assert reconstructed[-1] == (n_rows - 1) * 0.5

        # Forward tail query (`after=19990`).
        tail_frame = reader.scalar_page_frame("chk1", "loss", after=19990, limit=100, x_dtype="i64")
        hdr, payload = _decode_frame(tail_frame)
        assert hdr["start"] == 19990 and hdr["end"] == 20000
        assert hdr["dtypes"] == ["i64", "i64", "i64", "f64"]
        assert len(payload) == 10 * 4 * 8


def test_project_reader_catalog_filtering_and_sorting(tmp_path: Path) -> None:
    specs = [
        ("alpha", "r001", "lr-sweep", 1e-4, 0.8),
        ("alpha", "r002", "lr-sweep", 3e-4, 0.4),
        ("alpha", "r003", "ablation", 1e-3, 0.6),
        ("beta", "b001", None, 5e-4, 0.2),
    ]
    for proj, rid, grp, lr, loss in specs:
        with Board(
            run_id=rid,
            name=f"run-{rid}",
            project=proj,
            group=grp,
            config={"lr": lr},
            dir=tmp_path,
            system_metrics=False,
            capture_output=False,
        ) as b:
            b.log({"val/loss": loss})

    with ProjectReader(tmp_path) as pr:
        projects = pr.list_projects()
        assert {p.name for p in projects} == {"alpha", "beta"}

        sweep_runs = pr.list_runs(project="alpha", group="lr-sweep", sort="summary.val/loss")
        assert [r.run_id for r in sweep_runs] == ["r002", "r001"]

        desc_runs = pr.list_runs(project="alpha", sort="-summary.val/loss")
        assert [r.run_id for r in desc_runs] == ["r001", "r003", "r002"]

        search_runs = pr.list_runs(search="r003")
        assert len(search_runs) == 1 and search_runs[0].run_id == "r003"

        with pr.open_run("b001", project="beta") as rr:
            assert rr.info["name"] == "run-b001"


def test_memory_disabled_and_rank_zero_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Memory mode writes zero files to disk.
    with Board(mode="memory", system_metrics=False, capture_output=False) as mem:
        assert mem.run_dir is None
        mem.log({"loss": 0.5})
        mem.flush()
        assert mem.snapshot()["metrics"]["loss"]["last"] == 0.5
    assert list(tmp_path.iterdir()) == []

    # Disabled mode is a pure no-op.
    with Board(mode="disabled", dir=tmp_path) as dis:
        assert dis.disabled is True
        dis.step()
        dis.log({"loss": 1.0})
        dis.flush()
    assert list(tmp_path.iterdir()) == []

    # Non-zero DDP rank becomes a no-op when rank_zero_only=True.
    monkeypatch.setenv("RANK", "2")
    with Board(dir=tmp_path, rank_zero_only=True) as ddp:
        assert ddp.disabled is True
        ddp.log({"loss": 1.0})
    assert list(tmp_path.iterdir()) == []


def test_closed_board_and_drop_oldest_overflow(tmp_path: Path) -> None:
    b = Board(
        run_id="ovf1",
        project="p",
        dir=tmp_path,
        overflow="drop_oldest",
        queue_capacity=32,
        system_metrics=False,
        capture_output=False,
    )
    for i in range(500):
        b.log({"x": float(i)})
    b.finish()
    b.finish()  # idempotent

    with pytest.raises(errors.ClosedBoardError):
        b.log({"x": 1.0})
    with pytest.raises(errors.ClosedBoardError):
        b.flush()
