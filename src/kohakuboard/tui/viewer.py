"""Standalone SSH/offline KohakuTUI viewer for `kobo tui` and `kobo watch`."""

from __future__ import annotations

import contextlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from kohakutui import (
    KV,
    App,
    Bars,
    Chart,
    Diff,
    Divider,
    Event,
    Footer,
    Image,
    Log,
    Pane,
    Picker,
    Prompt,
    Stack,
    Table,
    TabStrip,
    Text,
    Tree,
)

from kohakuboard.errors import KohakuBoardError
from kohakuboard.reader import ProjectReader, RunReader
from kohakuboard.tui.helpers import (
    auto_select_charts,
    format_bytes,
    format_distribution_bars,
    format_duration,
    format_num,
    status_tone,
)


class BoardTuiViewer:
    """Interactive multi-run terminal viewer backed by `ProjectReader` + `RunReader`."""

    TAB_DEFS = [
        ("overview", "1 Overview"),
        ("metrics", "2 Metrics"),
        ("tables", "3 Tables"),
        ("media", "4 Media"),
        ("dist", "5 Dist"),
        ("logs", "6 Logs"),
        ("diff", "7 Diff"),
    ]

    def __init__(
        self,
        data_dir: str | Path,
        *,
        project: str | None = None,
        run_id: str | None = None,
        theme: str = "kohaku-dark",
        refresh_sec: float = 2.0,
        backend: str = "auto",
        size: tuple[int, int] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.theme = theme
        self.refresh_sec = max(0.25, float(refresh_sec))
        self.backend = backend
        self.size = size

        self._reader = ProjectReader(self.data_dir)
        self._stop_evt = threading.Event()
        self._lock = threading.RLock()

        self._active_project: str | None = project
        self._selected_runs: list[tuple[str, str]] = []
        self._initial_run_id = run_id
        self._active_tab = "overview"
        self._selected_metric: str | None = None
        self._log_query: str = ""

        self._app: App | None = None
        self._tab_strip: Any = None
        self._tree: Any = None
        self._footer: Any = None

        self._docs: dict[str, Any] = {}
        self._tab_roots: dict[str, Any] = {}

    def _init_selection(self) -> None:
        projects = self._reader.list_projects(refresh=True)
        if not projects:
            return
        if self._active_project is None or not any(
            p.name == self._active_project for p in projects
        ):
            self._active_project = projects[0].name
        runs = self._reader.list_runs(project=self._active_project, refresh=False)
        if not runs:
            return
        if self._initial_run_id is not None:
            match = [r for r in runs if r.run_id == self._initial_run_id]
            if match:
                self._selected_runs = [(match[0].project, match[0].run_id)]
                return
        self._selected_runs = [(r.project, r.run_id) for r in runs[:3]]

    def start(self) -> App:
        """Build and start the KohakuTUI App (non-blocking; call `run()` for interactive loop)."""
        self._init_selection()
        interrupt = "event" if self.backend == "test" else "auto"
        app = App(
            mode="fullscreen",
            backend=self.backend,
            size=self.size,
            interrupt=interrupt,
            theme=self.theme,
            max_fps=30,
        )
        self._app = app
        self._build_ui(app)
        app.start()
        self.refresh_all()
        return app

    def run(self) -> None:
        """Start viewer and block until the user presses `q` or Ctrl+C."""
        app = self.start()
        try:
            while not self._stop_evt.is_set() and app.running:
                self._stop_evt.wait(self.refresh_sec)
                if not self._stop_evt.is_set() and app.running:
                    self.refresh_all()
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        self._stop_evt.set()
        if self._app is not None:
            self._app.stop()

    def _build_ui(self, app: App) -> None:
        with app.batch():
            self._tab_strip = app.head.add(
                TabStrip(
                    tabs=[{"label": lbl, "value": val} for val, lbl in self.TAB_DEFS],
                    active=0,
                )
            )
            rail = app.panels.add(Pane(side="left", width=28, title="Projects & Runs"))
            self._tree = rail.add(Tree(items=[], height=28))

            self._docs["overview"] = app.transcript
            for key, _ in self.TAB_DEFS[1:]:
                self._docs[key] = app.add_document()

            self._footer = app.bottom.add(
                Footer(
                    left=(
                        "1-7 tabs · c compare runs · m pick metric · "
                        "/ search logs · r refresh · q quit"
                    ),
                    right=str(self.data_dir.name),
                )
            )

        app.bind("q", "quit")
        app.bind("r", "refresh")
        app.bind("c", "compare_runs")
        app.bind("m", "pick_metric")
        app.bind("/", "search_logs")
        for idx, (val, _) in enumerate(self.TAB_DEFS, start=1):
            app.bind(str(idx), f"tab_{val}")

        app.on("action")(self._on_action)
        app.on("select", node=self._tab_strip)(self._on_tab_select)
        app.on("select", node=self._tree)(self._on_tree_select)

    def _reset_tab_stack(self, tab_key: str) -> Any:
        old = self._tab_roots.get(tab_key)
        if old is not None:
            with contextlib.suppress(Exception):
                old.remove()
        stack = self._docs[tab_key].add(Stack(gap=1))
        self._tab_roots[tab_key] = stack
        return stack

    def _on_tab_select(self, ev: Event) -> None:
        self.switch_tab(str(ev.data.get("value") or "overview"))

    def switch_tab(self, tab_value: str) -> None:
        app = self._app
        if app is None or tab_value not in self._docs:
            return
        self._active_tab = tab_value
        idx = next((i for i, (v, _) in enumerate(self.TAB_DEFS) if v == tab_value), 0)
        self._tab_strip.update(active=idx)
        app.show_document(self._docs[tab_value], top=True)
        self._refresh_active_tab_content()

    def _on_tree_select(self, ev: Event) -> None:
        val = str(ev.data.get("value") or "")
        if val.startswith("run:"):
            _, proj, rid = val.split(":", 2)
            with self._lock:
                self._active_project = proj
                self._selected_runs = [(proj, rid)]
            self.refresh_all()
        elif val.startswith("proj:"):
            proj = val.split(":", 1)[1]
            with self._lock:
                self._active_project = proj
                runs = self._reader.list_runs(project=proj, refresh=False)
                self._selected_runs = [(r.project, r.run_id) for r in runs[:3]]
            self.refresh_all()

    def _on_action(self, ev: Event) -> None:
        name = ev.name
        if name == "quit":
            self._stop_evt.set()
        elif name == "refresh":
            self.refresh_all()
            if self._app is not None:
                self._app.notify("Refreshed catalog", level="info", ttl=1.5)
        elif name == "compare_runs":
            threading.Thread(target=self._open_run_picker, daemon=True).start()
        elif name == "pick_metric":
            threading.Thread(target=self._open_metric_picker, daemon=True).start()
        elif name == "search_logs":
            threading.Thread(target=self._open_log_search, daemon=True).start()
        elif name.startswith("tab_"):
            self.switch_tab(name[4:])

    def _open_run_picker(self) -> None:
        app = self._app
        if app is None or not app.running or not self._active_project:
            return
        runs = self._reader.list_runs(project=self._active_project, refresh=False)
        selected_ids = {rid for _, rid in self._selected_runs}
        items = [
            {
                "label": f"{r.name} ({r.run_id})",
                "value": r.run_id,
                "detail": f"{r.state} · {r.last_step + 1} st",
                "checked": r.run_id in selected_ids,
            }
            for r in runs
        ]
        picked = app.run_dialog(
            Picker(
                items=items,
                title=f"Compare Runs ({self._active_project})",
                placeholder="Space to toggle runs, Enter to apply...",
                multi=True,
                max_visible=14,
            )
        )
        if isinstance(picked, list) and picked:
            with self._lock:
                self._selected_runs = [(self._active_project, str(rid)) for rid in picked[:6]]
            self.refresh_all()

    def _open_metric_picker(self) -> None:
        app = self._app
        if app is None or not app.running or not self._selected_runs:
            return
        all_metrics: set[str] = set()
        for proj, rid in self._selected_runs:
            with self._reader.open_run(rid, project=proj) as rr:
                all_metrics.update(rr.scalar_names)
        if not all_metrics:
            app.notify("No scalar metrics found", level="info", ttl=2.0)
            return
        picked = app.run_dialog(
            Picker(
                items=[{"label": m, "value": m} for m in sorted(all_metrics)],
                title="Inspect Scalar Metric",
                placeholder="Filter metrics...",
                max_visible=14,
            )
        )
        if isinstance(picked, str) and picked:
            self._selected_metric = picked
            self.switch_tab("metrics")

    def _open_log_search(self) -> None:
        app = self._app
        if app is None or not app.running:
            return
        query = app.run_dialog(
            Prompt(
                question="FTS5 Log Search (empty = latest tail):",
                value=self._log_query,
                placeholder="e.g. NaN OR checkpoint",
            )
        )
        if query is not None:
            self._log_query = str(query).strip()
            self.switch_tab("logs")

    def refresh_all(self) -> None:
        """Refresh catalog tree and currently active tab."""
        app = self._app
        if app is None or not app.running:
            return
        with self._lock:
            projects = self._reader.list_projects(refresh=True)
            selected_set = set(self._selected_runs)
            tree_items: list[dict[str, Any]] = []
            for proj in projects:
                runs = self._reader.list_runs(project=proj.name, refresh=False)
                children = [
                    {
                        "label": f"{'● ' if (r.project, r.run_id) in selected_set else ''}{r.name}",
                        "value": f"run:{r.project}:{r.run_id}",
                        "detail": f"{r.last_step + 1}st",
                        "tone": status_tone(r.state),
                    }
                    for r in runs[:40]
                ]
                tree_items.append(
                    {
                        "label": proj.name,
                        "value": f"proj:{proj.name}",
                        "detail": f"{proj.run_count}",
                        "open": proj.name == self._active_project,
                        "children": children,
                    }
                )
            self._tree.update(items=tree_items)
            self._refresh_active_tab_content()

    def _open_selected_readers(self) -> list[tuple[str, str, RunReader]]:
        opened: list[tuple[str, str, RunReader]] = []
        for proj, rid in self._selected_runs[:6]:
            try:
                opened.append((proj, rid, self._reader.open_run(rid, project=proj)))
            except (KohakuBoardError, OSError):
                continue
        return opened

    def _refresh_active_tab_content(self) -> None:
        app = self._app
        if app is None or not app.running:
            return
        readers = self._open_selected_readers()
        try:
            with app.batch():
                if self._active_tab == "overview":
                    self._render_overview(readers)
                elif self._active_tab == "metrics":
                    self._render_metrics(readers)
                elif self._active_tab == "tables":
                    self._render_tables(readers)
                elif self._active_tab == "media":
                    self._render_media(readers)
                elif self._active_tab == "dist":
                    self._render_dist(readers)
                elif self._active_tab == "logs":
                    self._render_logs(readers)
                elif self._active_tab == "diff":
                    self._render_diff(readers)
        finally:
            for _, _, rr in readers:
                rr.close()

    def _feed_multi_run_chart(
        self,
        chart_node: Any,
        metric: str,
        readers: list[tuple[str, str, RunReader]],
        *,
        max_points: int = 120,
    ) -> None:
        series_vals: dict[str, list[float]] = {}
        max_len = 0
        for _, rid, rr in readers:
            if metric not in rr.scalar_names:
                continue
            vals = rr.scalar_slice(metric).values
            if len(vals) == 0:
                continue
            if len(vals) > max_points:
                stride = max(1, len(vals) // max_points)
                sampled = [float(x) for x in vals[::stride]]
                if float(vals[-1]) != sampled[-1]:
                    sampled.append(float(vals[-1]))
            else:
                sampled = [float(x) for x in vals]
            label = rr.info.get("name") or rid
            series_vals[label] = sampled
            max_len = max(max_len, len(sampled))

        if not series_vals:
            return
        for i in range(max_len):
            pt = {
                lbl: (arr[i] if i < len(arr) else float("nan")) for lbl, arr in series_vals.items()
            }
            chart_node.append(pt)

    def _render_overview(self, readers: list[tuple[str, str, RunReader]]) -> None:
        root = self._reset_tab_stack("overview")
        root.add(
            KV(
                fields={
                    "project": self._active_project or "-",
                    "selected_runs": ", ".join(rr.info.get("name", rid) for _, rid, rr in readers)
                    or "none",
                }
            )
        )
        root.add(Divider(label="Overlay Charts"))
        all_metrics: set[str] = set()
        for _, _, rr in readers:
            all_metrics.update(rr.scalar_names)
        chosen = auto_select_charts(sorted(all_metrics), max_charts=2)
        for m in chosen:
            ch = root.add(Chart(label=m, height=8, unit="step"))
            self._feed_multi_run_chart(ch, m, readers)
        if not chosen:
            root.add(Text("No scalar metrics in selected runs.", dim=True))

        root.add(Divider(label="Selected Runs Summary"))
        rows: list[list[str]] = []
        now_ms = int(time.time() * 1000)
        for _, rid, rr in readers:
            info = rr.info
            summ = rr.summary
            loss_key = next((k for k in summ if "loss" in k.lower()), None)
            loss_str = format_num(summ[loss_key].get("last")) if loss_key else "-"
            cfg_str = ", ".join(f"{k}={v}" for k, v in list(rr.config.items())[:3]) or "-"
            dur = format_duration(info.get("created_at_ms"), info.get("finished_at_ms"), now_ms)
            rows.append(
                [
                    f"{info.get('name', rid)} ({rid})",
                    str(info.get("state", "-")),
                    f"{int(info.get('last_step', -1)) + 1} · {dur}",
                    loss_str,
                    cfg_str,
                ]
            )
        root.add(
            Table(columns=["run", "state", "steps", "last_loss", "config"], rows=rows, max_rows=12)
        )

    def _render_metrics(self, readers: list[tuple[str, str, RunReader]]) -> None:
        root = self._reset_tab_stack("metrics")
        all_metrics: set[str] = set()
        rows: list[list[str]] = []
        for _, rid, rr in readers:
            rname = rr.info.get("name", rid)
            summ = rr.summary
            for m in rr.scalar_names:
                all_metrics.add(m)
                s = summ.get(m) or {}
                rows.append(
                    [
                        m,
                        rname,
                        format_num(s.get("last")),
                        format_num(s.get("min"), 3),
                        format_num(s.get("max"), 3),
                        str(s.get("count", rr.scalar_len(m))),
                    ]
                )
        if all_metrics and (self._selected_metric not in all_metrics):
            picked = auto_select_charts(sorted(all_metrics), max_charts=1)
            self._selected_metric = picked[0] if picked else sorted(all_metrics)[0]
        if self._selected_metric:
            ch = root.add(Chart(label=self._selected_metric, height=10, unit="step"))
            self._feed_multi_run_chart(ch, self._selected_metric, readers)
        root.add(Divider(label=f"All Scalar Metrics ({len(all_metrics)}) · press m to pick"))
        root.add(
            Table(
                columns=["metric", "run", "last", "min", "max", "count"],
                rows=rows[:80],
                max_rows=20,
            )
        )

    def _render_tables(self, readers: list[tuple[str, str, RunReader]]) -> None:
        root = self._reset_tab_stack("tables")
        if not readers:
            root.add(Text("No run selected.", dim=True))
            return
        _, rid, rr = readers[0]
        names = rr.table_names
        if not names:
            root.add(Text(f"No tables logged for {rr.info.get('name', rid)}.", dim=True))
            return
        tname = names[0]
        snaps = rr.tables(tname, limit=1)
        if not snaps:
            root.add(Text(f"Table {tname} is empty.", dim=True))
            return
        snap = snaps[0]
        cols = [str(c) for c in snap.get("columns") or []]
        out_rows = [
            [
                (
                    f"[media:{cell['$media'][:8]}]"
                    if isinstance(cell, dict) and "$media" in cell
                    else str(cell)
                )
                for cell in row
            ]
            for row in (snap.get("rows") or [])[:40]
        ]
        rname = rr.info.get("name", rid)
        root.add(
            Text(
                f"{rname} · {tname} (step {snap.get('step')}, {snap.get('row_count')} rows)",
                bold=True,
            )
        )
        root.add(Table(columns=cols, rows=out_rows, max_rows=20))

    def _render_media(self, readers: list[tuple[str, str, RunReader]]) -> None:
        root = self._reset_tab_stack("media")
        if not readers:
            root.add(Text("No run selected.", dim=True))
            return
        _, rid, rr = readers[0]
        names = rr.media_names
        if not names:
            root.add(Text(f"No media logged for {rr.info.get('name', rid)}.", dim=True))
            return
        for mname in names[:2]:
            evs = rr.media(mname, limit=1)
            if not evs:
                continue
            ev = evs[0]
            caption = ev.get("caption") or f"{mname} (step {ev.get('step')})"
            root.add(Divider(label=f"{rr.info.get('name', rid)} · {mname} · step {ev.get('step')}"))
            blob = rr.blob(ev["blob_key"])
            if blob is not None and ev.get("media_type") == "image":
                root.add(Image.from_file(blob, width=48, alt=caption, caption=True))
            else:
                root.add(
                    KV(
                        fields={
                            "media_type": str(ev.get("media_type")),
                            "mime": str(ev.get("mime")),
                            "size": format_bytes(ev.get("size_bytes")),
                            "caption": str(caption),
                        }
                    )
                )

    def _render_dist(self, readers: list[tuple[str, str, RunReader]]) -> None:
        root = self._reset_tab_stack("dist")
        if not readers:
            root.add(Text("No run selected.", dim=True))
            return
        _, rid, rr = readers[0]
        hnames = rr.histogram_names
        knames = rr.kde_names
        rec: dict[str, Any] | None = None
        label = ""
        if hnames:
            items = rr.histograms(hnames[0])
            if items:
                rec = items[-1]
                label = f"[histogram] {hnames[0]} ({len(items)} snapshots)"
        elif knames:
            items = rr.kdes(knames[0])
            if items:
                rec = items[-1]
                label = f"[kde] {knames[0]} ({len(items)} snapshots)"
        if rec is None:
            root.add(Text(f"No distributions logged for {rr.info.get('name', rid)}.", dim=True))
            return
        labels, series = format_distribution_bars(rec, max_bins=14)
        root.add(Text(f"{rr.info.get('name', rid)} · {label} · step {rec.get('step')}", bold=True))
        root.add(
            KV(
                fields={
                    "step": str(rec.get("step")),
                    "min": format_num(rec.get("min_val", rec.get("range_min")), 4),
                    "max": format_num(rec.get("max_val", rec.get("range_max")), 4),
                }
            )
        )
        root.add(Bars(labels=labels, series=series))

    def _render_logs(self, readers: list[tuple[str, str, RunReader]]) -> None:
        root = self._reset_tab_stack("logs")
        if not readers:
            root.add(Text("No run selected.", dim=True))
            return
        _, rid, rr = readers[0]
        rname = rr.info.get("name", rid)
        if self._log_query:
            lines = rr.search_logs(self._log_query, limit=100)
            hdr = f"{rname} · FTS5 '{self._log_query}' ({len(lines)} hits · press /)"
        else:
            lines = rr.logs(limit=100)
            hdr = f"{rname} · latest {len(lines)} lines (press / to search FTS5)"
        root.add(Text(hdr, dim=True))
        log_node = root.add(Log(title="logs", height=24, max_lines=500))
        if lines:
            text = (
                "\n".join(
                    f"[{ln.get('stream', 'out')}] {ln.get('text', ln.get('line', ''))}"
                    for ln in lines
                )
                + "\n"
            )
            log_node.append(text)

    def _render_diff(self, readers: list[tuple[str, str, RunReader]]) -> None:
        root = self._reset_tab_stack("diff")
        if len(readers) < 2:
            root.add(Text("Select at least 2 runs (press c) to diff config & summary.", dim=True))
            return
        _, rid_a, rr_a = readers[0]
        _, rid_b, rr_b = readers[1]
        name_a = rr_a.info.get("name", rid_a)
        name_b = rr_b.info.get("name", rid_b)

        cfg_a = json.dumps(rr_a.config, indent=2, sort_keys=True)
        cfg_b = json.dumps(rr_b.config, indent=2, sort_keys=True)
        root.add(Divider(label=f"Config Diff: {name_a} vs {name_b}"))
        root.add(Diff(old=cfg_a, new=cfg_b, path=f"config ({name_a} -> {name_b})"))

        sum_a = json.dumps(
            {k: round(float(v.get("last", 0.0)), 6) for k, v in sorted(rr_a.summary.items())},
            indent=2,
            sort_keys=True,
        )
        sum_b = json.dumps(
            {k: round(float(v.get("last", 0.0)), 6) for k, v in sorted(rr_b.summary.items())},
            indent=2,
            sort_keys=True,
        )
        root.add(Divider(label=f"Summary Diff: {name_a} vs {name_b}"))
        root.add(Diff(old=sum_a, new=sum_b, path=f"summary ({name_a} -> {name_b})"))

    # ---- Headless Test Helpers ----

    def sync(self, timeout: float = 2.0) -> bool:
        return bool(self._app.sync(timeout=timeout)) if self._app is not None else False

    def screen(self) -> list[str]:
        return list(self._app.screen()) if self._app is not None else []

    def inject(self, *events: dict[str, Any]) -> None:
        if self._app is not None:
            self._app.inject(*events)
