"""Histogram / KDE visualization view with single-step bars and flow heatmap."""

from __future__ import annotations

import math
import threading
from typing import Any, Iterable

import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from kohakuboard.inspector.widgets import TreeExplorer


class HistogramsView(ctk.CTkFrame):
    """View that renders histogram and KDE logs with Matplotlib."""

    MAX_FLOW_COLUMNS = 220  # keep flow heatmap fast even on long runs

    def __init__(self, parent, data_service, font_scale: float = 1.0):
        super().__init__(parent)
        self.data_service = data_service
        self.font_scale = font_scale

        # State
        self.current_histogram: str | None = None
        self.pending_histogram: str | None = None
        self.histogram_data: list[dict[str, Any]] = []
        self.current_index: int = 0
        self.view_mode: str = "flow"

        self.fig = None
        self.ax = None
        self.canvas = None

        self.build_ui()
        self.load_histogram_tree()

    # UI ------------------------------------------------------------------ #
    def build_ui(self):
        """Compose the view layout."""
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        # --- Left Tree --------------------------------------------------- #
        tree_frame = ctk.CTkFrame(self, width=260)
        tree_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        tree_frame.grid_rowconfigure(2, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_propagate(False)

        ctk.CTkLabel(
            tree_frame,
            text="Histogram Logs",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=15, pady=(15, 5))

        info = ctk.CTkLabel(
            tree_frame,
            text="Select a log to load.\nKDE tracks appear here as well.",
            justify="left",
            text_color="gray",
            font=ctk.CTkFont(size=max(10, int(11 * self.font_scale))),
        )
        info.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 8))

        self.tree = TreeExplorer(
            tree_frame,
            height=520,
            command=self.on_tree_clicked,
            font_scale=self.font_scale,
        )
        self.tree.grid(row=2, column=0, sticky="nsew", padx=15, pady=(0, 10))

        ctk.CTkButton(
            tree_frame,
            text="Refresh",
            command=self.load_histogram_tree,
        ).grid(row=3, column=0, sticky="ew", padx=15, pady=(0, 15))

        # --- Right Plot Area --------------------------------------------- #
        plot_container = ctk.CTkFrame(self)
        plot_container.grid(row=0, column=1, sticky="nsew")
        plot_container.grid_rowconfigure(2, weight=1)
        plot_container.grid_columnconfigure(0, weight=1)

        controls = ctk.CTkFrame(plot_container)
        controls.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 10))
        controls.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            controls,
            text="Select a histogram log",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.title_label.grid(row=0, column=0, sticky="w")

        self.mode_selector = ctk.CTkSegmentedButton(
            controls,
            values=["Single", "Distribution Flow"],
            command=self.on_mode_changed,
        )
        self.mode_selector.grid(row=0, column=1, padx=(10, 0))
        self.mode_selector.set("Distribution Flow")

        self.status_label = ctk.CTkLabel(
            controls,
            text="",
            text_color="gray",
            anchor="w",
        )
        self.status_label.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        # Flow-only options (hidden for single mode)
        self.flow_options = ctk.CTkFrame(plot_container)
        self.flow_options.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 10))
        self.flow_options.grid_columnconfigure(6, weight=1)

        ctk.CTkLabel(self.flow_options, text="X-Axis:").grid(
            row=0, column=0, padx=(0, 5)
        )
        self.xaxis_combo = ctk.CTkComboBox(
            self.flow_options,
            values=["step", "global_step"],
            width=140,
            command=lambda _: self.update_plot(),
        )
        self.xaxis_combo.set("step")
        self.xaxis_combo.grid(row=0, column=1, padx=(0, 15))

        ctk.CTkLabel(self.flow_options, text="Normalization:").grid(
            row=0, column=2, padx=(0, 5)
        )
        self.norm_combo = ctk.CTkComboBox(
            self.flow_options,
            values=["per-step", "global"],
            width=140,
            command=lambda _: self.update_plot(),
        )
        self.norm_combo.set("per-step")
        self.norm_combo.grid(row=0, column=3, padx=(0, 15))

        ctk.CTkLabel(self.flow_options, text="Colormap:").grid(
            row=0, column=4, padx=(0, 5)
        )
        self.cmap_combo = ctk.CTkComboBox(
            self.flow_options,
            values=["viridis", "plasma", "magma", "cividis", "turbo", "inferno"],
            width=140,
            command=lambda _: self.update_plot(),
        )
        self.cmap_combo.set("viridis")
        self.cmap_combo.grid(row=0, column=5, padx=(0, 15))

        self.flow_info_label = ctk.CTkLabel(
            self.flow_options,
            text="",
            text_color="gray",
            anchor="w",
        )
        self.flow_info_label.grid(
            row=1, column=0, columnspan=6, sticky="ew", pady=(6, 0)
        )

        # Plot frame with toolbar
        plot_frame = ctk.CTkFrame(plot_container)
        plot_frame.grid(row=2, column=0, sticky="nsew", padx=15, pady=(0, 10))
        plot_frame.grid_rowconfigure(1, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.setup_style()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=0, column=0, sticky="ew")

        # Slider for single mode
        self.slider_frame = ctk.CTkFrame(plot_container)
        self.slider_frame.grid(row=3, column=0, sticky="ew", padx=15, pady=(0, 15))
        self.slider_frame.grid_columnconfigure(1, weight=1)

        self.step_value_label = ctk.CTkLabel(
            self.slider_frame,
            text="Step: -",
        )
        self.step_value_label.grid(row=0, column=0, padx=(0, 15))

        self.step_slider = ctk.CTkSlider(
            self.slider_frame,
            from_=0,
            to=1,
            number_of_steps=1,
            state="disabled",
            command=self.on_slider_changed,
        )
        self.step_slider.grid(row=0, column=1, sticky="ew")

        self.show_empty("Select a histogram log to begin")
        self.update_mode_visibility()

    # Data loading -------------------------------------------------------- #
    def load_histogram_tree(self):
        """Load histogram names from the data service asynchronously."""

        def load():
            try:
                names = self.data_service.get_available_histogram_names()
                self.after(0, lambda: self.populate_tree(names))
            except Exception as exc:
                self.after(
                    0, lambda: self.set_status(f"Failed to list histograms: {exc}")
                )

        threading.Thread(target=load, daemon=True).start()

    def populate_tree(self, names: Iterable[str]):
        """Populate the namespace-aware tree."""
        self.tree.clear()
        structure: dict[str, Any] = {}

        for name in sorted(names):
            parts = name.split("/")
            current = structure
            for idx, part in enumerate(parts):
                if idx == len(parts) - 1:
                    current[part] = {"_type": "histogram", "_name": name}
                else:
                    current = current.setdefault(part, {})

        self._populate_tree_recursive("", structure)

    def _populate_tree_recursive(self, parent_id: str, node: dict[str, Any]):
        for key, val in sorted(node.items()):
            if key.startswith("_"):
                continue

            if isinstance(val, dict) and val.get("_type") == "histogram":
                self.tree.insert(
                    parent_id, "end", text=f"📈 {key}", tags=(f"hist:{val['_name']}",)
                )
            else:
                folder = self.tree.insert(
                    parent_id, "end", text=f"📁 {key}", tags=("folder",)
                )
                self._populate_tree_recursive(folder, val)

    # Events -------------------------------------------------------------- #
    def on_tree_clicked(self, item_id: str):
        """Load histogram when a tree entry is selected."""
        if not item_id:
            return

        tags = self.tree.get_tags(item_id)
        for tag in tags:
            if tag.startswith("hist:"):
                hist_name = tag[5:]
                self.select_histogram(hist_name)
                break

    def on_mode_changed(self, value: str):
        """Switch between single-step and flow visualization."""
        normalized = "flow" if "flow" in value.lower() else "single"
        if normalized == self.view_mode:
            return
        self.view_mode = normalized
        self.update_mode_visibility()
        self.update_plot()

    def on_slider_changed(self, value: float):
        """Handle slider drag for single-step view."""
        if not self.histogram_data:
            return
        idx = int(round(value))
        idx = max(0, min(idx, len(self.histogram_data) - 1))
        if idx != self.current_index:
            self.current_index = idx
            self.update_single_step_ui()
            self.update_plot()

    # Histogram selection ------------------------------------------------- #
    def select_histogram(self, name: str):
        """Begin loading a histogram log."""
        if self.current_histogram == name and self.histogram_data:
            return

        self.current_histogram = name
        self.pending_histogram = name
        self.histogram_data = []
        self.current_index = 0
        self.title_label.configure(text=f"{name}")
        self.set_status("Loading histogram data...")
        self.show_empty("Loading histogram data...")

        def load():
            try:
                data = self.data_service.get_histogram_data(name)
            except Exception as exc:
                self.after(0, lambda: self.set_status(f"Failed to load data: {exc}"))
                return

            self.after(0, lambda: self.on_histogram_loaded(name, data))

        threading.Thread(target=load, daemon=True).start()

    def on_histogram_loaded(self, name: str, data: list[dict[str, Any]]):
        """Store loaded histogram data (if it matches the current request)."""
        if name != self.pending_histogram:
            return

        processed: list[dict[str, Any]] = []
        for entry in sorted(data, key=lambda d: d.get("step", 0)):
            bins, counts = self._ensure_bins_and_counts(entry)
            if not bins or not counts:
                continue
            processed.append({**entry, "bins": bins, "counts": counts})

        self.histogram_data = processed
        if not self.histogram_data:
            self.set_status("No histogram entries available.")
            self.show_empty("No histogram entries available.")
            self.step_slider.configure(state="disabled")
            return

        self.current_index = len(self.histogram_data) - 1
        self.update_slider_config()
        self.update_single_step_ui()

        log_kind = (
            "KDE"
            if any(p.get("precision") == "kde" for p in processed)
            else "Histogram"
        )
        bins = len(self.histogram_data[0]["counts"])
        self.set_status(
            f"{len(self.histogram_data)} entries | {log_kind} | {bins} bins"
        )
        self.update_plot()

    # Helpers ------------------------------------------------------------- #
    def _ensure_bins_and_counts(
        self, entry: dict[str, Any]
    ) -> tuple[list[float] | None, list[float] | None]:
        """Normalize entry to always expose bins + counts."""
        bins = entry.get("bins")
        counts = entry.get("counts")

        if bins and counts:
            bins = [float(b) for b in bins]
            counts = [float(c) for c in counts]
            if len(bins) >= 2 and len(counts) == len(bins) - 1:
                return bins, counts

        values = entry.get("values")
        if values:
            num_bins = entry.get("num_bins") or min(
                64, max(16, int(math.sqrt(len(values))))
            )
            counts_arr, bins_arr = np.histogram(values, bins=num_bins)
            return bins_arr.tolist(), counts_arr.astype(float).tolist()

        return None, None

    def update_slider_config(self):
        """Enable/disable slider based on data length."""
        steps = len(self.histogram_data)
        if steps <= 1:
            self.step_slider.configure(state="disabled", number_of_steps=1)
            self.step_slider.set(0)
        else:
            self.step_slider.configure(
                state="normal",
                from_=0,
                to=steps - 1,
                number_of_steps=steps - 1,
            )
            self.step_slider.set(self.current_index)

    def update_single_step_ui(self):
        """Update slider label with current step metadata."""
        if not self.histogram_data:
            self.step_value_label.configure(text="Step: —")
            return

        entry = self.histogram_data[self.current_index]
        step = entry.get("step")
        global_step = entry.get("global_step")
        g_label = f"{global_step}" if global_step is not None else "—"
        self.step_value_label.configure(text=f"Step: {step} | Global: {g_label}")

    def set_status(self, message: str):
        """Update status label."""
        self.status_label.configure(text=message)

    def update_mode_visibility(self):
        """Show/hide control groups based on current mode."""
        if self.view_mode == "single":
            self.flow_options.grid_remove()
            self.flow_info_label.configure(text="")
            self.slider_frame.grid()
        else:
            self.flow_options.grid()
            self.slider_frame.grid_remove()

    # Plotting ------------------------------------------------------------ #
    def setup_style(self):
        """Match Matplotlib colors to CustomTkinter theme."""
        mode = ctk.get_appearance_mode()
        bg = "#2B2B2B" if mode == "Dark" else "#FFFFFF"
        fg = "#DCE4EE" if mode == "Dark" else "#000000"
        grid = "#4A4A4A" if mode == "Dark" else "#CCCCCC"

        self.ax.set_facecolor(bg)
        self.fig.patch.set_facecolor(bg)
        self.ax.tick_params(colors=fg)
        for spine in self.ax.spines.values():
            spine.set_color(fg)
        self.ax.xaxis.label.set_color(fg)
        self.ax.yaxis.label.set_color(fg)
        self.ax.title.set_color(fg)
        self.ax.grid(True, alpha=0.3, color=grid)

    def show_empty(self, message: str):
        """Show placeholder text when nothing is selected."""
        self.ax.clear()
        self.setup_style()
        self.ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=self.ax.transAxes,
            fontsize=14,
            color="gray",
        )
        self.canvas.draw()

    def update_plot(self):
        """Render figure based on current state."""
        if not self.histogram_data:
            return

        if self.view_mode == "flow":
            drawn = self.render_flow_heatmap()
            if not drawn:
                self.show_empty("Not enough data to build flow heatmap")
        else:
            self.render_single_histogram()

    def render_single_histogram(self):
        """Render a bar chart for the selected step."""
        entry = self.histogram_data[self.current_index]
        bins = entry["bins"]
        counts = entry["counts"]
        bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        widths = np.diff(bins)

        self.ax.clear()
        self.setup_style()

        color = "#60a5fa" if ctk.get_appearance_mode() == "Dark" else "#3b82f6"
        self.ax.bar(
            bin_centers,
            counts,
            width=widths,
            align="center",
            color=color,
            edgecolor=color,
            alpha=0.85,
        )

        self.ax.set_xlabel("Value")
        self.ax.set_ylabel("Count")
        self.ax.set_title(f"{self.current_histogram} - Step {entry.get('step')}")
        self.fig.tight_layout()
        self.canvas.draw()

    def render_flow_heatmap(self) -> bool:
        """Render distribution flow heatmap. Returns True if drawn."""
        prepared = self._prepare_heatmap_data()
        if not prepared:
            self.flow_info_label.configure(text="")
            return False

        steps = prepared["steps"]
        bin_centers = prepared["bin_centers"]
        matrix = prepared["matrix"]
        stride = prepared["stride"]
        used = prepared["used"]

        self.ax.clear()
        self.setup_style()

        x_min = float(min(steps))
        x_max = float(max(steps))
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5

        y_min = float(min(bin_centers))
        y_max = float(max(bin_centers))
        if y_min == y_max:
            y_min -= 0.5
            y_max += 0.5

        extent = [x_min, x_max, y_min, y_max]
        cmap = self.cmap_combo.get()
        im = self.ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=cmap,
            interpolation="nearest",
        )
        im.set_clim(0.0, 1.0)

        axis_label = (
            "Global Step" if self.xaxis_combo.get() == "global_step" else "Step"
        )
        self.ax.set_xlabel(axis_label)
        self.ax.set_ylabel("Bin value")
        self.ax.set_title(f"{self.current_histogram} - Distribution Flow")
        self.fig.tight_layout()
        self.canvas.draw()

        total = len(self.histogram_data)
        if stride > 1:
            self.flow_info_label.configure(
                text=f"Downsampled to {used} / {total} steps (every {stride} entries)"
            )
        else:
            self.flow_info_label.configure(text=f"Using all {used} steps")
        return True

    def _prepare_heatmap_data(self) -> dict[str, Any] | None:
        """Build normalized matrix for flow mode."""
        if not self.histogram_data:
            return None

        base_entry = next(
            (
                entry
                for entry in self.histogram_data
                if entry["bins"] and entry["counts"]
            ),
            None,
        )
        if not base_entry:
            return None

        bins = base_entry["bins"]
        if len(bins) < 2:
            return None

        stride = 1
        total = len(self.histogram_data)
        if total > self.MAX_FLOW_COLUMNS:
            stride = math.ceil(total / self.MAX_FLOW_COLUMNS)

        selected_steps: list[float] = []
        selected_counts: list[list[float]] = []
        xaxis = self.xaxis_combo.get()

        for idx in range(0, total, stride):
            entry = self.histogram_data[idx]
            entry_bins = entry["bins"]
            entry_counts = entry["counts"]

            if len(entry_bins) != len(bins) or len(entry_counts) != len(bins) - 1:
                continue

            step_value = (
                entry.get("global_step")
                if xaxis == "global_step"
                else entry.get("step")
            )
            if step_value is None:
                step_value = entry.get("step") or idx
            selected_steps.append(float(step_value))
            selected_counts.append([float(c) for c in entry_counts])

        if not selected_counts:
            return None

        counts = np.asarray(selected_counts, dtype=float)  # shape (steps, bins-1)
        if counts.ndim != 2:
            return None

        if self.norm_combo.get() == "per-step":
            col_max = counts.max(axis=1, keepdims=True)
            col_max[col_max == 0] = 1.0
            normalized = (counts / col_max).T  # -> (bins-1, steps)
        else:
            global_max = counts.max()
            if global_max > 0:
                normalized = (counts / global_max).T
            else:
                normalized = counts.T

        bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

        return {
            "matrix": normalized,
            "steps": selected_steps,
            "bin_centers": bin_centers,
            "stride": stride,
            "used": len(selected_steps),
        }

    # Public API ---------------------------------------------------------- #
    def update_font_scale(self, scale: float):
        """Update fonts for tree control."""
        self.font_scale = scale
        self.tree.set_font_scale(scale)

    def cleanup(self):
        """Close Matplotlib figure."""
        if self.fig:
            plt.close(self.fig)
