"""End-to-end API tests for `kohakuboard-web` FastAPI server."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kohakuboard import Artifact, Board, Histogram, KernelDensity, Media, Table, TensorLog
from kohakuboard_web import create_app


def _decode_stream(buf: bytes) -> list[tuple[dict, bytes]]:
    frames: list[tuple[dict, bytes]] = []
    pos = 0
    while pos < len(buf):
        (hdr_len,) = struct.unpack_from("<I", buf, pos)
        payload_start = pos + 4 + hdr_len
        assert (4 + hdr_len) % 8 == 0
        hdr = json.loads(buf[pos + 4 : payload_start].decode("utf-8"))
        n = hdr["end"] - hdr["start"]
        payload_len = n * 4 * 8
        frames.append((hdr, buf[payload_start : payload_start + payload_len]))
        pos = payload_start + payload_len
    return frames


@pytest.fixture()
def populated_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(123)
    img = np.full((12, 12, 3), 180, dtype=np.uint8)
    ckpt = tmp_path / "ckpt.bin"
    ckpt.write_bytes(b"weights" * 32)

    for rid, scale in [("run1", 1.0), ("run2", 0.5)]:
        b = Board(
            run_id=rid,
            name=f"exp-{rid}",
            project="vision",
            group="sweep-1",
            tags=["v2"],
            config={"lr": 1e-3 * scale, "layers": 12},
            dir=tmp_path,
            system_metrics=False,
            capture_output=True,
        )
        for i in range(150):
            val = float("nan") if (rid == "run1" and i == 75) else (scale / (i + 1.0))
            b.log({"train/loss": val, "train/lr": 1e-3 * scale})
        b.log(
            {
                "samples": Media(img, caption="preview", format="png"),
                "dist/h": Histogram(rng.normal(0.0, 1.0, size=512), bins=16),
                "dist/k": KernelDensity(rng.normal(0.0, 1.0, size=512), num_points=32),
                "eval/tbl": Table(columns=["k", "v"], rows=[["a", 1], ["b", 2]]),
                "debug/t": TensorLog(np.arange(16, dtype=np.float32).reshape(4, 4)),
                "ckpt": Artifact("final", path=ckpt),
            }
        )
        b.alert("done", f"{rid} finished", level="info")
        print(f"training log marker for {rid} NaN check")
        b.finish()

    return tmp_path


def test_projects_runs_detail_patch_and_delete(populated_dir: Path) -> None:
    app = create_app(populated_dir, allow_testserver=True)
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        sys_info = client.get("/api/system/info").json()
        assert sys_info["mode"] == "local"

        projs = client.get("/api/projects").json()
        assert len(projs) == 1 and projs[0]["name"] == "vision" and projs[0]["run_count"] == 2

        runs = client.get("/api/projects/vision/runs").json()
        assert {r["run_id"] for r in runs} == {"run1", "run2"}

        detail = client.get("/api/projects/vision/runs/run1").json()
        assert detail["info"]["name"] == "exp-run1"
        assert "train/loss" in detail["scalar_names"]
        assert "samples" in detail["media_names"]
        assert "eval/tbl" in detail["table_names"]
        assert "dist/h" in detail["histogram_names"]
        assert "dist/k" in detail["kde_names"]

        batch = client.post(
            "/api/projects/vision/runs/batch/meta",
            json={"run_ids": ["run1", "run2", "missing"]},
        ).json()
        assert set(batch.keys()) == {"run1", "run2"}

        # PATCH notes/tags/name
        patched = client.patch(
            "/api/projects/vision/runs/run1",
            json={"name": "renamed-run1", "tags": ["pinned", "best"], "notes": "great run"},
        ).json()
        assert patched["name"] == "renamed-run1"
        assert patched["tags"] == ["pinned", "best"]
        assert patched["notes"] == "great run"

        # DELETE run2
        assert client.delete("/api/projects/vision/runs/run2").json() == {"deleted": True}
        assert len(client.get("/api/projects/vision/runs").json()) == 1


def test_newest_first_stream_paging_blobs_and_rich_endpoints(populated_dir: Path) -> None:
    app = create_app(populated_dir, allow_testserver=True)
    with TestClient(app) as client:
        # Round-robin newest-first streaming across 2 runs x 1 metric.
        resp = client.get(
            "/api/projects/vision/runs/scalars/stream",
            params={"runs": "run1,run2", "metrics": "train/loss", "x_dtype": "f64"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/vnd.kobo.chunks"
        frames = _decode_stream(resp.content)
        assert len(frames) == 2
        assert {hdr["run_id"] for hdr, _ in frames} == {"run1", "run2"}
        for hdr, payload in frames:
            assert hdr["start"] == 0 and hdr["end"] == 150 and hdr["done"] is True
            cols = np.frombuffer(payload, dtype="<f8").reshape(4, 150)
            if hdr["run_id"] == "run1":
                assert np.isnan(cols[3, 75])

        # Explicit backward & forward page endpoints.
        p1 = client.get(
            "/api/projects/vision/runs/run1/scalars/train/loss/page",
            params={"limit": 50, "x_dtype": "f64"},
        )
        f1 = _decode_stream(p1.content)
        assert f1[0][0]["start"] == 100 and f1[0][0]["end"] == 150
        assert f1[0][0]["next_before"] == 100

        p_tail = client.get(
            "/api/projects/vision/runs/run1/scalars/train/loss/page",
            params={"after": 140, "limit": 50},
        )
        ft = _decode_stream(p_tail.content)
        assert ft[0][0]["start"] == 140 and ft[0][0]["end"] == 150

        # Media + immutable ETag + HTTP Range 206 blob endpoint.
        media = client.get("/api/projects/vision/runs/run1/media/samples").json()
        assert len(media) == 1
        blob_key = media[0]["blob_key"]
        b_full = client.get(f"/api/projects/vision/runs/run1/blobs/{blob_key}")
        assert b_full.status_code == 200
        assert "immutable" in b_full.headers["cache-control"]
        assert b_full.content.startswith(b"\x89PNG")

        b_304 = client.get(
            f"/api/projects/vision/runs/run1/blobs/{blob_key}",
            headers={"If-None-Match": b_full.headers["etag"]},
        )
        assert b_304.status_code == 304

        b_range = client.get(
            f"/api/projects/vision/runs/run1/blobs/{blob_key}",
            headers={"Range": "bytes=0-7"},
        )
        assert b_range.status_code == 206
        assert b_range.content == b_full.content[:8]

        # Distributions (histogram + KDE), tables, tensors, artifacts, alerts, logs & FTS5 search.
        h_page = client.get(
            "/api/projects/vision/runs/run1/distributions/histogram/dist/h/page"
        ).json()
        assert h_page["total"] == 1 and len(h_page["items"]) == 1
        k_page = client.get("/api/projects/vision/runs/run1/distributions/kde/dist/k/page").json()
        assert k_page["total"] == 1 and len(k_page["items"]) == 1
        assert len(client.get("/api/projects/vision/runs/run1/tables/eval/tbl").json()) == 1
        assert len(client.get("/api/projects/vision/runs/run1/tensors/debug/t").json()) == 1
        assert len(client.get("/api/projects/vision/runs/run1/artifacts").json()) == 1
        assert len(client.get("/api/projects/vision/runs/run1/alerts").json()) == 1
        assert len(client.get("/api/projects/vision/runs/run1/logs").json()) >= 1
        hits = client.get(
            "/api/projects/vision/runs/run1/logs/search", params={"q": "marker"}
        ).json()
        assert len(hits) == 1 and "run1" in hits[0]["text"]


def test_loopback_guard_blocks_rebinding_and_cross_origin(populated_dir: Path) -> None:
    app = create_app(populated_dir, enforce_loopback=True, allow_testserver=True)
    with TestClient(app) as client:
        bad_host = client.get("/api/projects", headers={"Host": "evil.example.com"})
        assert bad_host.status_code == 403

        bad_origin = client.delete(
            "/api/projects/vision/runs/run1",
            headers={"Origin": "https://attacker.example"},
        )
        assert bad_origin.status_code == 403

        ok_local = client.get("/api/projects", headers={"Host": "127.0.0.1:48889"})
        assert ok_local.status_code == 200
