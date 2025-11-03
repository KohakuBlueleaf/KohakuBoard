"""Plots view - visualize metrics with matplotlib and tree selector"""

import threading

import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from kohakuboard.inspector.widgets import TreeExplorer


class PlotsView(ctk.CTkFrame):
    """View for plotting metrics with tree explorer for multi-select"""

    def __init__(self, parent, data_service, font_scale: float = 1.0):
        super().__init__(parent)
        self.data_service = data_service
        self.font_scale = font_scale

        self.selected_metrics = []
        self.plot_data = {}

        # Grid: left tree + right plot
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        self.build_ui()
        self.load_metrics_tree()

    def build_ui(self):
        """Build UI with tree selector and plot"""
        # === LEFT: Metric Tree (resizable) ===
        tree_frame = ctk.CTkFrame(self, width=250)
        tree_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        tree_frame.grid_rowconfigure(1, weight=1)  # Tree gets space
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_propagate(False)

        # Header
        header = ctk.CTkFrame(tree_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 10))

        ctk.CTkLabel(
            header, text="Select Metrics", font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w")

        self.selected_label = ctk.CTkLabel(
            header, text="Selected: 0", text_color="gray"
        )
        self.selected_label.pack(anchor="w", pady=(2, 0))

        # Tree explorer (custom widget)
        self.tree = TreeExplorer(
            tree_frame,
            height=500,
            command=self.on_tree_clicked,
            font_scale=self.font_scale,
        )
        self.tree.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 10))

        # Clear button
        ctk.CTkButton(tree_frame, text="Clear All", command=self.clear_all).grid(
            row=2, column=0, sticky="ew", padx=15, pady=(0, 15)
        )

        # === RIGHT: Plot Area (70%) ===
        plot_container = ctk.CTkFrame(self)
        plot_container.grid(row=0, column=1, sticky="nsew")
        plot_container.grid_rowconfigure(1, weight=1)
        plot_container.grid_columnconfigure(0, weight=1)

        # Controls
        controls = ctk.CTkFrame(plot_container)
        controls.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 10))

        ctk.CTkLabel(controls, text="X-Axis:").pack(side="left", padx=5)
        self.xaxis = ctk.CTkComboBox(
            controls,
            values=["step", "global_step", "timestamp"],
            width=120,
            command=lambda _: self.update_plot(),
        )
        self.xaxis.set("step")
        self.xaxis.pack(side="left", padx=5)

        ctk.CTkLabel(controls, text="Smoothing:").pack(side="left", padx=(20, 5))
        self.smooth = ctk.CTkSlider(
            controls, from_=0, to=0.99, width=150, command=lambda _: self.update_plot()
        )
        self.smooth.set(0)
        self.smooth.pack(side="left", padx=5)

        self.smooth_label = ctk.CTkLabel(controls, text="0.00", width=40)
        self.smooth_label.pack(side="left")

        # Plot
        plot_frame = ctk.CTkFrame(plot_container)
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))
        plot_frame.grid_rowconfigure(1, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)

        # Matplotlib figure (created on main thread to avoid threading issues)
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.setup_style()

        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        # Toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=0, column=0, sticky="ew")

        self.show_empty()

    def setup_style(self):
        """Match matplotlib to CTk theme"""
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

    def show_empty(self):
        """Empty plot"""
        self.ax.clear()
        self.setup_style()
        self.ax.text(
            0.5,
            0.5,
            "Select metrics from tree\n\nDouble-click to add/remove",
            ha="center",
            va="center",
            transform=self.ax.transAxes,
            fontsize=14,
            color="gray",
        )
        self.canvas.draw()

    def load_metrics_tree(self):
        """Load metrics into tree"""

        def load():
            try:
                metrics = self.data_service.get_available_metrics()
                metrics = [
                    m for m in metrics if m not in ["step", "global_step", "timestamp"]
                ]
                self.after(0, lambda: self.build_tree(metrics))
            except Exception as e:
                print(f"Error: {e}")

        threading.Thread(target=load, daemon=True).start()

    def build_tree(self, metrics: list[str]):
        """Build tree"""
        self.tree.clear()

        # Build structure
        structure = {}
        for metric in metrics:
            parts = metric.split("/")
            cur = structure
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    cur[part] = {"_type": "metric", "_name": metric}
                else:
                    if part not in cur:
                        cur[part] = {}
                    cur = cur[part]

        self.populate("", structure)

    def populate(self, parent, structure):
        """Populate tree"""
        for key, val in sorted(structure.items()):
            if key.startswith("_"):
                continue

            if isinstance(val, dict) and val.get("_type") == "metric":
                self.tree.insert(
                    parent, "end", text=f"☐ {key}", tags=(f"metric:{val['_name']}",)
                )
            else:
                folder = self.tree.insert(
                    parent, "end", text=f"📁 {key}", tags=("folder",)
                )
                self.populate(folder, val)

    def on_tree_clicked(self, item_id):
        """Handle tree click - toggle metric selection

        Args:
            item_id: Clicked item ID
        """
        if not item_id:
            return

        tags = self.tree.get_tags(item_id)
        for tag in tags:
            if tag.startswith("metric:"):
                metric = tag[7:]

                # Get current text
                current_text = self.tree.item(item_id, "text")

                if metric in self.selected_metrics:
                    # Remove from plot
                    self.selected_metrics.remove(metric)
                    if metric in self.plot_data:
                        del self.plot_data[metric]

                    # Update checkbox
                    new_text = current_text.replace("☑", "☐")
                    self.tree.item(item_id, text=new_text)

                    self.update_plot()
                else:
                    # Add to plot
                    self.selected_metrics.append(metric)

                    # Update checkbox
                    new_text = current_text.replace("☐", "☑")
                    self.tree.item(item_id, text=new_text)

                    self.load_metric(metric)

                self.selected_label.configure(
                    text=f"Selected: {len(self.selected_metrics)}"
                )
                break

    def load_metric(self, metric):
        """Load metric data"""

        def load():
            try:
                data = self.data_service.get_metric_data(metric)
                self.after(0, lambda: self.add_to_plot(metric, data))
            except Exception as e:
                print(f"Error: {e}")

        threading.Thread(target=load, daemon=True).start()

    def add_to_plot(self, metric, data):
        """Add metric to plot"""
        xaxis = self.xaxis.get()
        x = (
            data["steps"]
            if xaxis == "step"
            else data["global_steps"] if xaxis == "global_step" else data["timestamps"]
        )
        y = data["values"]

        # Filter
        fx, fy = [], []
        for xi, yi in zip(x, y):
            if yi not in (None, "NaN", "Infinity", "-Infinity"):
                fx.append(xi)
                fy.append(float(yi))

        self.plot_data[metric] = {"x": fx, "y": fy}
        self.update_plot()

    def update_plot(self):
        """Update plot"""
        smooth = self.smooth.get()
        self.smooth_label.configure(text=f"{smooth:.2f}")

        if not self.plot_data:
            self.show_empty()
            return

        self.ax.clear()
        self.setup_style()

        for metric, data in self.plot_data.items():
            y = data["y"]
            if smooth > 0:
                y = self.ema(y, smooth)
            self.ax.plot(data["x"], y, label=metric, linewidth=2, alpha=0.8)

        xaxis = self.xaxis.get()
        self.ax.set_xlabel(xaxis.replace("_", " ").title())
        self.ax.set_ylabel("Value")
        self.ax.set_title(f"{len(self.plot_data)} Metrics")
        self.ax.legend(loc="best", framealpha=0.9)
        self.fig.tight_layout()
        self.canvas.draw()

    def ema(self, data, factor):
        """EMA smoothing"""
        if not data:
            return data
        smoothed = []
        last = data[0]
        for v in data:
            s = last * factor + v * (1 - factor)
            smoothed.append(s)
            last = s
        return smoothed

    def clear_all(self):
        """Clear all selections"""
        self.selected_metrics.clear()
        self.plot_data.clear()
        self.selected_label.configure(text="Selected: 0")

        # Reset checkboxes in tree
        for item in self.tree.get_children():
            self.reset_checks(item)

        self.show_empty()

    def reset_checks(self, item_id):
        """Reset checkboxes recursively

        Args:
            item_id: Item to reset
        """
        # Get text and reset checkbox
        text = self.tree.item(item_id, "text")
        if "☑" in text:
            self.tree.item(item_id, text=text.replace("☑", "☐"))

        # Recurse to children
        for child in self.tree.get_children(item_id):
            self.reset_checks(child)

    def update_font_scale(self, scale: float):
        """Update font scale for tree

        Args:
            scale: Font scale multiplier
        """
        self.font_scale = scale
        self.tree.set_font_scale(scale)

    def cleanup(self):
        """Cleanup matplotlib resources"""
        import matplotlib.pyplot as plt

        # Close this specific figure
        if self.fig:
            plt.close(self.fig)
