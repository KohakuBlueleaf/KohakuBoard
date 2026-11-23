"""Live in-training KohakuTUI monitor (`Board(tui=...)`) polling Rust snapshots at `refresh_hz`."""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from kohakutui import (
    KV,
    App,
    Cell,
    Chart,
    Divider,
    Event,
    Footer,
    Log,
    Pane,
    Picker,
    Progress,
    Stack,
    Table,
    TabStrip,
    Text,
    Tree,
)

from kohakuboard.tui.config import ChartSpec, TuiConfig
from kohakuboard.tui.helpers import (
    auto_select_charts,
    compute_ema,
    extract_system_fields,
    format_num,
    group_by_namespace,
    is_hidden,
)

if TYPE_CHECKING:
    from kohakuboard.client.board import Board


class _ChartSlot:
    """Tracks one rendered KohakuTUI Chart node and its incremental cursor."""

    def __init__(self, spec: ChartSpec, node: Any) -> None:
        self.spec = spec
        self.node = node
        self.last_step_by_metric: dict[str, int] = {}
        self.ema_state: dict[str, tuple[float, float]] = {}

    def push_from_snapshot(self, metrics: dict[str, Any], *, show_ema: bool) -> None:
        names = self.spec.metric_names()
        # Collect new (step, value) samples from preview rings across all overlaid series.
        new_steps: set[int] = set()
        per_metric_points: dict[str, dict[int, float]] = {}
        for m in names:
            info = metrics.get(m)
            if not info:
                continue
            prev_step = self.last_step_by_metric.get(m, -1)
            pts: dict[int, float] = {}
            for pair in info.get("preview") or ():
                st, val = int(pair[0]), float(pair[1])
                if st > prev_step:
                    pts[st] = val
                    new_steps.add(st)
            # Also capture non-finite latest sample (not stored in finite preview ring)
            last_st = int(info.get("last_step", -1))
            last_v = info.get("last")
            if last_st > prev_step and last_st not in pts and last_v is not None:
                pts[last_st] = float(last_v)
                new_steps.add(last_st)
            if pts:
                per_metric_points[m] = pts

        if not new_steps:
            return

        ordered = sorted(new_steps)
        # Keep frame latency bounded during huge bursts while preserving final point.
        if len(ordered) > 64:
            stride = max(1, len(ordered) // 64)
            ordered = ordered[::stride] + (
                [ordered[-1]] if ordered[-1] != ordered[::stride][-1] else []
            )

        alpha = self.spec.ema if (show_ema and self.spec.ema) else None
        for st in ordered:
            payload: dict[str, float] = {}
            for m in names:
                pts = per_metric_points.get(m)
                if pts and st in pts:
                    val = pts[st]
                    payload[m] = val
                    self.last_step_by_metric[m] = st
                    if alpha is not None:
                        acc, denom = self.ema_state.get(m, (0.0, 0.0))
                        if val == val and abs(val) != float("inf"):
                            acc = acc * alpha + val * (1.0 - alpha)
                            denom = denom * alpha + (1.0 - alpha)
                            self.ema_state[m] = (acc, denom)
                            payload[f"{m} (ema)"] = acc / denom if denom > 0 else val
                        else:
                            payload[f"{m} (ema)"] = float("nan")
                else:
                    payload[m] = float("nan")
                    if alpha is not None:
                        payload[f"{m} (ema)"] = float("nan")
            self.node.append(payload)


class LiveTuiHandle:
    """Thread-safe runtime controller attached to `Board.tui` during training."""

    def __init__(self, board: Board, config: TuiConfig) -> None:
        self._board = board
        self.config = config
        self._lock = threading.RLock()
        self._stop_evt = threading.Event()
        self._detached = False
        self._active = False
        self._show_ema = True
        self._last_seq = -1
        self._last_alert_count = 0
        self._active_tab = "overview"

        self._app: App | None = None
        self._driver_thread: threading.Thread | None = None

        # Runtime customizable specs & watched metrics
        self._chart_specs: list[ChartSpec] = config.normalized_charts()
        self._auto_charts_done = len(self._chart_specs) > 0
        self._watched: list[str] = list(config.watch)
        self._extra_panels: list[tuple[str, Callable[[Any], None]]] = []

        # Live widget handles
        self._chart_slots: list[_ChartSlot] = []
        self._charts_container: Any = None
        self._progress_node: Any = None
        self._metrics_table: Any = None
        self._overflow_text: Any = None
        self._log_node: Any = None

        # Fullscreen workstation handles
        self._tab_strip: Any = None
        self._doc_metrics: Any = None
        self._doc_logs: Any = None
        self._fs_metrics_tree: Any = None
        self._fs_metrics_table: Any = None
        self._fs_log_node: Any = None
        self._side_run_kv: Any = None
        self._side_watch_kv: Any = None
        self._side_sys_kv: Any = None
        self._side_alerts_log: Any = None
        self._footer: Any = None

    @property
    def active(self) -> bool:
        return self._active and not self._detached

    def start(self) -> bool:
        """Start the KohakuTUI session and background refresh thread (False if auto-degraded)."""
        if (
            self.config.backend == "auto"
            and not self.config.force
            and not getattr(sys.stdout, "isatty", lambda: False)()
        ):
            return False

        mode = "fullscreen" if self.config.mode == "fullscreen" else "inline"
        interrupt = "event" if self.config.backend == "test" else "auto"
        app = App(
            mode=mode,
            backend=self.config.backend,
            size=self.config.size,
            interrupt=interrupt,
            theme=self.config.theme,
            max_fps=max(10, int(self.config.refresh_hz * 2)),
        )
        self._app = app
        self._build_layout(app, mode=mode)
        app.start()
        self._active = True

        self._driver_thread = threading.Thread(
            target=self._poll_loop,
            name=f"kobo-tui-{self._board.run_id}",
            daemon=True,
        )
        self._driver_thread.start()
        return True

    def _build_layout(self, app: App, *, mode: str) -> None:
        cfg_fields: dict[str, str] = {
            "project": self._board.project,
            "run_id": self._board.run_id,
            "mode": self._board.mode,
        }
        if self._board.group:
            cfg_fields["group"] = self._board.group
        if self._board.tags:
            cfg_fields["tags"] = ", ".join(self._board.tags[:6])
        for k, v in list(self._board.config.items())[:6]:
            cfg_fields[f"cfg.{k}"] = str(v)

        with app.batch():
            if mode == "inline":
                app.transcript.add(
                    Cell(
                        role="system",
                        title=f"{self._board.name} ({self._board.run_id})",
                        children=[KV(fields=cfg_fields)],
                    )
                ).seal()

                self._charts_container = app.live.add(Stack(gap=0))
                for spec in self._chart_specs:
                    self._add_chart_slot(spec)

                if self.config.total_steps is not None:
                    self._progress_node = app.live.add(
                        Progress(total=self.config.total_steps, label="train", unit="step")
                    )

                self._metrics_table = app.live.add(
                    Table(columns=["metric", "last", "ema", "min..max"], max_rows=5)
                )
                self._overflow_text = app.live.add(
                    Text(
                        "ctrl+t fullscreen · p pick metrics · e toggle ema · q hide",
                        dim=True,
                    )
                )
                if self.config.log_height > 0:
                    self._log_node = app.live.add(
                        Log(title="output", height=self.config.log_height, max_lines=500)
                    )
            else:
                self._tab_strip = app.head.add(
                    TabStrip(
                        tabs=[
                            {"label": "1 Overview", "value": "overview"},
                            {"label": "2 Metrics", "value": "metrics", "badge": 0},
                            {"label": "3 Logs", "value": "logs"},
                        ],
                        active=0,
                    )
                )
                main_stack = app.transcript.add(Stack(gap=1))
                if self.config.total_steps is not None:
                    self._progress_node = main_stack.add(
                        Progress(total=self.config.total_steps, label="train", unit="step")
                    )
                self._charts_container = main_stack.add(Stack(gap=1))
                for spec in self._chart_specs:
                    self._add_chart_slot(spec)
                self._metrics_table = main_stack.add(
                    Table(columns=["metric", "last", "ema", "min..max"], max_rows=8)
                )

                # Tab 2: High-cardinality Metrics explorer
                self._doc_metrics = app.add_document()
                m_stack = self._doc_metrics.add(Stack(gap=1))
                m_stack.add(Divider(label="Namespaces"))
                self._fs_metrics_tree = m_stack.add(Tree(items=[], height=8))
                m_stack.add(Divider(label="All Logged Metrics"))
                self._fs_metrics_table = m_stack.add(
                    Table(columns=["metric", "last", "ema", "min", "max", "nans"], max_rows=20)
                )

                # Tab 3: Full-height Logs
                self._doc_logs = app.add_document()
                self._fs_log_node = self._doc_logs.add(
                    Log(title="console output", height=24, max_lines=2000)
                )

                # Right sidebar pane
                side = app.panels.add(Pane(side="right", width=32, title="Run Info"))
                self._side_run_kv = side.add(KV(fields=cfg_fields))
                side.add(Divider(label="watched"))
                self._side_watch_kv = side.add(KV(fields={"step": "0"}))
                if self.config.show_system:
                    side.add(Divider(label="system"))
                    self._side_sys_kv = side.add(KV(fields={"status": "sampling..."}))
                side.add(Divider(label="alerts"))
                self._side_alerts_log = side.add(Log(height=5, max_lines=64))

                self._footer = app.bottom.add(
                    Footer(
                        left="ctrl+t inline · 1-3 tabs · p pick · e ema · q hide",
                        right=f"{self._board.project}/{self._board.name}",
                    )
                )

        # Hotkeys (never kills training on q!)
        app.bind("q", "detach_tui")
        app.bind("p", "pick_metric")
        app.bind("e", "toggle_ema")
        app.bind("1", "tab_overview")
        app.bind("2", "tab_metrics")
        app.bind("3", "tab_logs")
        app.on("action")(self._on_action)
        if self._tab_strip is not None:
            app.on("select", node=self._tab_strip)(self._on_tab_select)

    def _add_chart_slot(self, spec: ChartSpec) -> _ChartSlot:
        assert self._charts_container is not None
        node = self._charts_container.add(
            Chart(
                label=spec.display_label(),
                height=spec.height,
                unit="step",
                digits=spec.digits,
                y_min=spec.y_min,
                y_max=spec.y_max,
            )
        )
        slot = _ChartSlot(spec, node)
        self._chart_slots.append(slot)
        return slot

    def _on_tab_select(self, ev: Event) -> None:
        val = str(ev.data.get("value") or "overview")
        self._switch_tab(val)

    def _switch_tab(self, tab_value: str) -> None:
        app = self._app
        if app is None or self._tab_strip is None:
            return
        self._active_tab = tab_value
        idx_map = {"overview": 0, "metrics": 1, "logs": 2}
        self._tab_strip.update(active=idx_map.get(tab_value, 0))
        if tab_value == "overview":
            app.show_document(app.transcript, top=True)
        elif tab_value == "metrics" and self._doc_metrics is not None:
            app.show_document(self._doc_metrics, top=True)
        elif tab_value == "logs" and self._doc_logs is not None:
            app.show_document(self._doc_logs, top=False)

    def _on_action(self, ev: Event) -> None:
        name = ev.name
        if name == "detach_tui":
            self.detach()
        elif name == "toggle_ema":
            with self._lock:
                self._show_ema = not self._show_ema
            if self._app is not None:
                self._app.notify(
                    f"EMA overlay {'enabled' if self._show_ema else 'disabled'}",
                    level="info",
                    ttl=2.0,
                )
        elif name == "tab_overview":
            self._switch_tab("overview")
        elif name == "tab_metrics":
            self._switch_tab("metrics")
        elif name == "tab_logs":
            self._switch_tab("logs")
        elif name == "pick_metric":
            threading.Thread(target=self._open_metric_picker, daemon=True).start()

    def _open_metric_picker(self) -> None:
        app = self._app
        if app is None or not app.running:
            return
        snap = self._board.snapshot()
        metrics = snap.get("metrics") or {}
        if not metrics:
            app.notify("No scalar metrics logged yet", level="info", ttl=2.0)
            return
        grouped = group_by_namespace(metrics.keys())
        items: list[dict[str, Any]] = []
        for ns, names in grouped.items():
            for m in names:
                last_v = format_num(metrics[m].get("last"))
                items.append({"label": m, "value": m, "group": ns, "detail": f"last={last_v}"})
        chosen = app.run_dialog(
            Picker(
                items=items,
                title="Pin Metric Chart",
                placeholder="Filter metrics...",
                sidebar=True,
                max_visible=14,
            )
        )
        if isinstance(chosen, str) and chosen:
            self.pin_chart(chosen)
            app.notify(f"Pinned chart: {chosen}", level="ok", ttl=2.0)

    def _poll_loop(self) -> None:
        interval = 1.0 / max(1.0, self.config.refresh_hz)
        while not self._stop_evt.is_set():
            with contextlib.suppress(Exception):
                self.refresh_once()
            self._stop_evt.wait(interval)

    def refresh_once(self) -> None:
        """Pull the latest Rust snapshot and push a single coalesced batch to KohakuTUI."""
        app = self._app
        if app is None or not app.running or self._detached:
            return
        snap = self._board.snapshot()
        seq = int(snap.get("seq", 0))
        with self._lock:
            if seq == self._last_seq:
                return
            self._last_seq = seq
            metrics: dict[str, Any] = snap.get("metrics") or {}
            progress_map: dict[str, Any] = snap.get("progress") or {}
            alerts: list[dict[str, Any]] = snap.get("recent_alerts") or []

            with app.batch():
                # Auto-pick 1-2 default charts once metrics appear if user didn't specify any
                if not self._auto_charts_done and metrics:
                    max_c = 2 if self.config.mode == "fullscreen" else 1
                    chosen = auto_select_charts(
                        sorted(metrics.keys()),
                        self.config.hide,
                        max_charts=max_c,
                    )
                    for m in chosen:
                        spec = ChartSpec(m, height=8)
                        self._chart_specs.append(spec)
                        self._add_chart_slot(spec)
                    self._auto_charts_done = True

                # Push incremental points to every chart slot
                for slot in self._chart_slots:
                    slot.push_from_snapshot(metrics, show_ema=self._show_ema)

                # Progress bar (explicit board.progress() or total_steps from config)
                if progress_map:
                    first_prog = next(iter(progress_map.values()))
                    cur = int(first_prog.get("current", 0))
                    tot = first_prog.get("total")
                    lbl = str(first_prog.get("label", "train"))
                    unit = str(first_prog.get("unit", "step"))
                    if self._progress_node is None and self._charts_container is not None:
                        self._progress_node = app.live.add(
                            Progress(value=cur, total=tot, label=lbl, unit=unit)
                        )
                    elif self._progress_node is not None:
                        self._progress_node.update(label=lbl, total=tot, unit=unit)
                        self._progress_node.append(cur)
                elif self._progress_node is not None and self.config.total_steps:
                    step_val = max(0, int(snap.get("last_step", 0)) + 1)
                    self._progress_node.append(step_val)

                # Compact metrics strip (watched first, then non-hidden non-sys metrics)
                visible_names = [
                    m
                    for m in sorted(metrics.keys())
                    if not is_hidden(m, self.config.hide) and not m.startswith("sys/")
                ]
                ordered_strip: list[str] = []
                for w in self._watched:
                    if w in metrics and w not in ordered_strip:
                        ordered_strip.append(w)
                for m in visible_names:
                    if m not in ordered_strip:
                        ordered_strip.append(m)

                rows: list[list[str]] = []
                for m in ordered_strip[:12]:
                    st = metrics[m]
                    rows.append(
                        [
                            m,
                            format_num(st.get("last")),
                            format_num(st.get("ema")),
                            f"{format_num(st.get('min'), 3)}..{format_num(st.get('max'), 3)}",
                        ]
                    )
                if self._metrics_table is not None:
                    self._metrics_table.update(rows=rows)

                hidden_more = max(0, len(metrics) - min(5, len(rows)))
                if self._overflow_text is not None:
                    hint = (
                        f"+{hidden_more} more metrics · " if hidden_more > 0 else ""
                    ) + "ctrl+t fullscreen · p pick · e ema · q hide"
                    self._overflow_text.update(value=hint)

                # Fullscreen workstation panels & high-cardinality metrics tab
                if self._tab_strip is not None:
                    self._tab_strip.update(
                        tabs=[
                            {"label": "1 Overview", "value": "overview"},
                            {"label": "2 Metrics", "value": "metrics", "badge": len(metrics)},
                            {"label": "3 Logs", "value": "logs"},
                        ]
                    )
                if self._fs_metrics_tree is not None:
                    grouped = group_by_namespace(metrics.keys())
                    tree_items = [
                        {
                            "label": f"{ns}/",
                            "value": f"ns:{ns}",
                            "detail": f"{len(names)} metrics",
                            "open": f"{ns}/" not in self.config.collapse_namespaces,
                            "children": [
                                {
                                    "label": m,
                                    "value": m,
                                    "detail": format_num(metrics[m].get("last")),
                                }
                                for m in names[:64]
                            ],
                        }
                        for ns, names in grouped.items()
                    ]
                    self._fs_metrics_tree.update(items=tree_items)
                if self._fs_metrics_table is not None:
                    all_rows = [
                        [
                            m,
                            format_num(metrics[m].get("last")),
                            format_num(metrics[m].get("ema")),
                            format_num(metrics[m].get("min"), 3),
                            format_num(metrics[m].get("max"), 3),
                            str(metrics[m].get("nan_count", 0)),
                        ]
                        for m in sorted(metrics.keys())[:100]
                    ]
                    self._fs_metrics_table.update(rows=all_rows)
                if self._side_watch_kv is not None:
                    watch_fields: dict[str, str] = {
                        "step": str(max(0, int(snap.get("last_step", 0)) + 1)),
                        "global_step": str(snap.get("last_global_step", 0)),
                    }
                    for m in ordered_strip[:6]:
                        watch_fields[m] = format_num(metrics[m].get("last"))
                    self._side_watch_kv.update(fields=watch_fields)
                if self._side_sys_kv is not None:
                    sys_fields = extract_system_fields(metrics)
                    if sys_fields:
                        self._side_sys_kv.update(fields=sys_fields)

                # New alerts -> side log + toast
                if len(alerts) > self._last_alert_count:
                    for al in alerts[self._last_alert_count :]:
                        lvl_s = al.get("level", "info")
                        title_s = al.get("title", "")
                        text_s = al.get("text", "")
                        msg = f"[{lvl_s}] {title_s}: {text_s}"
                        if self._side_alerts_log is not None:
                            self._side_alerts_log.append(msg + "\n")
                        if self._log_node is not None:
                            self._log_node.append(msg + "\n")
                        lvl = "warn" if al.get("level") == "warn" else "info"
                        app.notify(msg, level=lvl, ttl=3.0)
                    self._last_alert_count = len(alerts)

    # ---- Thread-safe Runtime Customization API ----

    def write_log(self, stream_name: str, text: str) -> None:
        """Route captured stdout/stderr text into the bounded TUI Log widget."""
        if not self.active or not text:
            return
        prefix = "[err] " if stream_name == "stderr" else ""
        payload = f"{prefix}{text}" if prefix and not text.startswith("\r") else text
        if self._log_node is not None:
            self._log_node.append(payload)
        if self._fs_log_node is not None:
            self._fs_log_node.append(payload)

    def pin_chart(
        self,
        metrics: str | list[str],
        *,
        label: str | None = None,
        height: int = 8,
        ema: float | None = 0.97,
    ) -> None:
        """Dynamically pin a new chart into the active TUI from training code or Picker."""
        spec = ChartSpec(metrics=metrics, label=label, height=height, ema=ema)
        with self._lock:
            self._chart_specs.append(spec)
            self._auto_charts_done = True
            if self._app is not None and self._app.running and self._charts_container is not None:
                slot = self._add_chart_slot(spec)
                snap = self._board.snapshot()
                slot.push_from_snapshot(snap.get("metrics") or {}, show_ema=self._show_ema)

    def watch(self, metric: str) -> None:
        """Pin a scalar metric to the top of the compact metrics strip / sidebar."""
        with self._lock:
            if metric not in self._watched:
                self._watched.append(metric)

    def notify(self, text: str, *, level: str = "info", ttl: float = 3.0) -> None:
        """Show a transient notification toast inside the TUI."""
        if self._app is not None and self._app.running:
            self._app.notify(text, level=level, ttl=ttl)

    def add_panel(self, title: str, builder_fn: Callable[[Any], None]) -> None:
        """Escape hatch: mount a custom KohakuTUI container into the active workspace."""
        with self._lock:
            self._extra_panels.append((title, builder_fn))
            if self._app is not None and self._app.running:
                target = self._app.panels if self.config.mode == "fullscreen" else self._app.live
                container = target.add(Stack(title=title, gap=0))
                builder_fn(container)

    def detach(self) -> None:
        """Hide/stop the TUI cleanly without interrupting the running training script."""
        with self._lock:
            if self._detached:
                return
            self._detached = True
            self._stop_evt.set()
        if self._board._capture is not None:
            self._board._capture.set_sink(passthrough=True, on_raw=None)
        if self._app is not None:
            with contextlib.suppress(Exception):
                self._app.stop()
        self._active = False

    def finish(self, *, state: str = "finished") -> None:
        """Seal final charts/summary into scrollback (on_finish='seal') and stop the TUI."""
        self._stop_evt.set()
        if (
            self._driver_thread is not None
            and self._driver_thread is not threading.current_thread()
        ):
            self._driver_thread.join(timeout=2.0)

        app = self._app
        if app is None or not app.running or self._detached:
            self._active = False
            return

        # One final synchronous refresh so the very last logged points are drawn.
        with contextlib.suppress(Exception):
            self.refresh_once()

        snap = self._board.snapshot()
        metrics = snap.get("metrics") or {}
        step_count = max(0, int(snap.get("last_step", 0)) + 1)

        if self.config.on_finish == "seal" and self.config.mode != "fullscreen":
            with app.batch():
                for transient in (self._progress_node, self._overflow_text, self._log_node):
                    if transient is not None:
                        transient.remove()
                for slot in self._chart_slots:
                    slot.node.move(app.transcript)
                    slot.node.seal()
                if self._metrics_table is not None:
                    self._metrics_table.update(max_rows=None)
                    self._metrics_table.move(app.transcript)
                    self._metrics_table.seal()
                summary_parts = [
                    f"{self._board.name} ({self._board.run_id})",
                    state,
                    f"{step_count} steps",
                ]
                if self._chart_specs:
                    first_m = self._chart_specs[0].metric_names()[0]
                    if first_m in metrics:
                        summary_parts.append(
                            f"{first_m}={format_num(metrics[first_m].get('last'))}"
                        )
                app.print(" · ".join(summary_parts), role="system")
        elif self.config.on_finish == "hold":
            app.notify(f"Run {state} · press q to exit", level="ok", ttl=60.0)
            while not self._detached and app.running:
                time.sleep(0.05)

        app.sync(timeout=2.0)
        app.stop()
        self._active = False

    # ---- Headless Test Helpers (for `backend="test"`) ----

    def sync(self, timeout: float = 2.0) -> bool:
        """Flush pending snapshot + wait until KohakuTUI renders all ops."""
        self.refresh_once()
        if self._app is None:
            return False
        return bool(self._app.sync(timeout=timeout))

    def screen(self) -> list[str]:
        return list(self._app.screen()) if self._app is not None else []

    def history(self) -> list[str]:
        return list(self._app.history()) if self._app is not None else []

    def inject(self, *events: dict[str, Any]) -> None:
        if self._app is not None:
            self._app.inject(*events)


__all__ = ["LiveTuiHandle", "compute_ema"]
