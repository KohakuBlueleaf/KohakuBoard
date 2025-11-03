"""Export view - export data to various formats"""

import threading
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from kohakuboard.inspector.services.export_service import ExportService
from kohakuboard.inspector.widgets import TreeExplorer


class ExportView(ctk.CTkFrame):
    """View for exporting board data to Parquet/CSV/JSON"""

    def __init__(self, parent, data_service, font_scale: float = 1.0):
        super().__init__(parent)
        self.data_service = data_service
        self.font_scale = font_scale
        self.selected_metrics = []

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        self.build_ui()
        self.load_metrics_tree()

    def build_ui(self):
        """Build UI"""
        # === LEFT: Metric Selector ===
        left_frame = ctk.CTkFrame(self, width=300)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_propagate(False)

        ctk.CTkLabel(
            left_frame,
            text="Select Metrics to Export",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=15, pady=(15, 10))

        # Tree selector
        self.tree = TreeExplorer(
            left_frame,
            height=500,
            command=self.on_metric_clicked,
            font_scale=self.font_scale,
        )
        self.tree.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 10))

        # Selected count
        self.selected_label = ctk.CTkLabel(left_frame, text="Selected: 0 metrics")
        self.selected_label.grid(row=2, column=0, sticky="w", padx=15, pady=(0, 10))

        ctk.CTkButton(left_frame, text="Clear All", command=self.clear_all).grid(
            row=3, column=0, sticky="ew", padx=15, pady=(0, 15)
        )

        # === RIGHT: Export Options ===
        right_frame = ctk.CTkFrame(self)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(2, 0))

        # Title
        ctk.CTkLabel(
            right_frame, text="Export Options", font=ctk.CTkFont(size=18, weight="bold")
        ).pack(padx=20, pady=(20, 10))

        # Format selector
        format_frame = ctk.CTkFrame(right_frame)
        format_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(format_frame, text="Format:", font=ctk.CTkFont(size=14)).pack(
            anchor="w", pady=(0, 5)
        )

        self.format_var = ctk.StringVar(value="Parquet")
        formats = ["Parquet", "CSV", "JSON"]

        for fmt in formats:
            ctk.CTkRadioButton(
                format_frame, text=fmt, variable=self.format_var, value=fmt
            ).pack(anchor="w", pady=3)

        # File path
        path_frame = ctk.CTkFrame(right_frame)
        path_frame.pack(fill="x", padx=20, pady=20)

        ctk.CTkLabel(path_frame, text="Output File:", font=ctk.CTkFont(size=14)).pack(
            anchor="w", pady=(0, 5)
        )

        path_entry_frame = ctk.CTkFrame(path_frame, fg_color="transparent")
        path_entry_frame.pack(fill="x")

        self.path_entry = ctk.CTkEntry(
            path_entry_frame, placeholder_text="Select output file..."
        )
        self.path_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        ctk.CTkButton(
            path_entry_frame, text="Browse...", width=100, command=self.browse_file
        ).pack(side="left")

        # Export button
        self.export_btn = ctk.CTkButton(
            right_frame,
            text="📥 Export Data",
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.do_export,
        )
        self.export_btn.pack(padx=20, pady=20)

        # Status
        self.status_label = ctk.CTkLabel(right_frame, text="", wraplength=400)
        self.status_label.pack(padx=20, pady=10)

    def load_metrics_tree(self):
        """Load metrics tree"""

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

    def on_metric_clicked(self, item_id):
        """Toggle metric selection"""
        if not item_id:
            return

        tags = self.tree.get_tags(item_id)
        for tag in tags:
            if tag.startswith("metric:"):
                metric = tag[7:]
                text = self.tree.item(item_id, "text")

                if metric in self.selected_metrics:
                    self.selected_metrics.remove(metric)
                    self.tree.item(item_id, text=text.replace("☑", "☐"))
                else:
                    self.selected_metrics.append(metric)
                    self.tree.item(item_id, text=text.replace("☐", "☑"))

                self.selected_label.configure(
                    text=f"Selected: {len(self.selected_metrics)} metrics"
                )
                break

    def clear_all(self):
        """Clear all selections"""
        self.selected_metrics.clear()
        self.selected_label.configure(text="Selected: 0 metrics")

        for item in self.tree.get_children():
            self.reset_checks(item)

    def reset_checks(self, item_id):
        """Reset checkboxes"""
        text = self.tree.item(item_id, "text")
        if "☑" in text:
            self.tree.item(item_id, text=text.replace("☑", "☐"))
        for child in self.tree.get_children(item_id):
            self.reset_checks(child)

    def browse_file(self):
        """Open file save dialog"""
        fmt = self.format_var.get()

        extensions = {
            "Parquet": [("Parquet files", "*.parquet"), ("All files", "*.*")],
            "CSV": [("CSV files", "*.csv"), ("All files", "*.*")],
            "JSON": [("JSON files", "*.json"), ("All files", "*.*")],
        }

        default_ext = {
            "Parquet": ".parquet",
            "CSV": ".csv",
            "JSON": ".json",
        }

        filepath = filedialog.asksaveasfilename(
            defaultextension=default_ext[fmt],
            filetypes=extensions[fmt],
            title=f"Export to {fmt}",
        )

        if filepath:
            self.path_entry.delete(0, "end")
            self.path_entry.insert(0, filepath)

    def do_export(self):
        """Perform export"""
        if not self.selected_metrics:
            self.status_label.configure(text="❌ No metrics selected", text_color="red")
            return

        output_path = self.path_entry.get()
        if not output_path:
            self.status_label.configure(
                text="❌ No output file selected", text_color="red"
            )
            return

        fmt = self.format_var.get()
        self.status_label.configure(text="", text_color="gray")
        self.export_btn.configure(state="disabled")

        def export():
            try:
                match fmt:
                    case "Parquet":
                        ExportService.export_metrics_to_parquet(
                            self.data_service, self.selected_metrics, output_path
                        )
                    case "CSV":
                        ExportService.export_metrics_to_csv(
                            self.data_service, self.selected_metrics, output_path
                        )
                    case "JSON":
                        ExportService.export_metrics_to_json(
                            self.data_service, self.selected_metrics, output_path
                        )

                self.after(
                    0,
                    lambda: self.status_label.configure(
                        text=f"✅ Exported {len(self.selected_metrics)} metrics to {Path(output_path).name}",
                        text_color="green",
                    ),
                )
                self.after(0, lambda: self.export_btn.configure(state="normal"))

            except Exception as e:
                self.after(
                    0,
                    lambda: self.status_label.configure(
                        text=f"❌ Export failed: {e}", text_color="red"
                    ),
                )
                self.after(0, lambda: self.export_btn.configure(state="normal"))

        threading.Thread(target=export, daemon=True).start()

    def update_font_scale(self, scale: float):
        """Update font scale"""
        self.font_scale = scale
        self.tree.set_font_scale(scale)
