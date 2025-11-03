"""Metrics view - tree explorer with high-performance table (no resize lag!)"""

import threading

import customtkinter as ctk

from kohakuboard.inspector.widgets import DataTable, TreeExplorer
from kohakuboard.inspector.widgets.step_range_dialog import StepRangeDialog


class MetricsView(ctk.CTkFrame):
    """Metrics view with tree explorer and fast ttk.Treeview-based table"""

    def __init__(self, parent, data_service, font_scale: float = 1.0):
        super().__init__(parent)
        self.data_service = data_service
        self.font_scale = font_scale

        self.current_metric = None
        self.current_page = 0
        self.page_size = 500  # ttk.Treeview handles this easily!
        self.all_data = None

        # Grid: left tree + right table
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        self.build_ui()
        self.load_metrics_tree()

    def build_ui(self):
        """Build UI"""
        # === LEFT: Tree Explorer ===
        tree_frame = ctk.CTkFrame(self, width=280)
        tree_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        tree_frame.grid_rowconfigure(2, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_propagate(False)

        ctk.CTkLabel(
            tree_frame,
            text="Metrics Explorer",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=15, pady=(15, 5))

        self.search_entry = ctk.CTkEntry(
            tree_frame, placeholder_text="🔍 Search...", height=32
        )
        self.search_entry.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 10))

        self.tree = TreeExplorer(
            tree_frame,
            height=500,
            command=self.on_metric_clicked,
            font_scale=self.font_scale,
        )
        self.tree.grid(row=2, column=0, sticky="nsew", padx=15, pady=(0, 15))

        # === RIGHT: Table ===
        table_container = ctk.CTkFrame(self)
        table_container.grid(row=0, column=1, sticky="nsew")
        table_container.grid_rowconfigure(1, weight=1)
        table_container.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(table_container)
        header.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 10))

        self.metric_title = ctk.CTkLabel(
            header, text="Select a metric", font=ctk.CTkFont(size=16, weight="bold")
        )
        self.metric_title.pack(side="left")

        # Delete button
        self.delete_btn = ctk.CTkButton(
            header,
            text="🗑️ Delete Metric",
            width=120,
            fg_color="red",
            hover_color="darkred",
            command=self.delete_metric,
        )
        self.delete_btn.pack(side="right", padx=5)

        self.refresh_btn = ctk.CTkButton(
            header, text="🔄", width=40, command=self.refresh_data
        )
        self.refresh_btn.pack(side="right")

        # Status bar
        self.status_bar = ctk.CTkLabel(
            table_container,
            text="",
            anchor="w",
            text_color="gray",
            font=ctk.CTkFont(size=10),
        )
        self.status_bar.grid(row=0, column=1, sticky="e", padx=15, pady=(15, 10))

        # Fast table (ttk.Treeview - NO resize lag!)
        self.data_table = DataTable(
            table_container,
            columns=["Step", "Global Step", "Timestamp", "Value"],
            height=550,
            font_scale=self.font_scale,
        )
        self.data_table.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 10))

        # Pagination
        pag = ctk.CTkFrame(table_container)
        pag.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 15))

        self.prev_btn = ctk.CTkButton(
            pag, text="◀ Prev", width=80, command=self.prev_page, state="disabled"
        )
        self.prev_btn.pack(side="left", padx=5)

        self.page_label = ctk.CTkLabel(pag, text="Page 1 / 1", width=90)
        self.page_label.pack(side="left", padx=10)

        self.next_btn = ctk.CTkButton(
            pag, text="Next ▶", width=80, command=self.next_page, state="disabled"
        )
        self.next_btn.pack(side="left", padx=5)

        ctk.CTkLabel(pag, text="Rows:").pack(side="left", padx=(30, 5))

        self.page_size_selector = ctk.CTkComboBox(
            pag,
            values=["100", "500", "1000", "All"],
            width=70,
            command=self.on_page_size_changed,
        )
        self.page_size_selector.set("500")
        self.page_size_selector.pack(side="left")

        # Delete range button
        ctk.CTkButton(
            pag,
            text="🗑️ Delete Range",
            width=110,
            fg_color="orange",
            hover_color="darkorange",
            command=self.delete_step_range,
        ).pack(side="right", padx=5)

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
        """Build tree from metrics"""
        self.tree.clear()

        # Build namespace structure
        structure = {}
        for metric in metrics:
            parts = metric.split("/")
            current = structure
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = {"_type": "metric", "_name": metric}
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

        self.populate_tree("", structure)

    def populate_tree(self, parent_id: str, structure: dict):
        """Populate tree recursively"""
        for key, val in sorted(structure.items()):
            if key.startswith("_"):
                continue

            if isinstance(val, dict) and val.get("_type") == "metric":
                self.tree.insert(
                    parent_id, "end", text=f"📊 {key}", tags=(f"metric:{val['_name']}",)
                )
            else:
                folder_id = self.tree.insert(
                    parent_id, "end", text=f"📁 {key}", tags=("folder",)
                )
                self.populate_tree(folder_id, val)

    def on_metric_clicked(self, item_id):
        """Handle metric click"""
        if not item_id:
            return

        tags = self.tree.get_tags(item_id)
        for tag in tags:
            if tag.startswith("metric:"):
                self.current_metric = tag[7:]
                self.current_page = 0
                self.load_metric_data()
                break

    def load_metric_data(self):
        """Load metric data"""
        if not self.current_metric:
            return

        self.metric_title.configure(text=f"{self.current_metric}")
        self.status_bar.configure(text="⏳ Loading data...")
        self.refresh_btn.configure(state="disabled")

        def load():
            try:
                data = self.data_service.get_metric_data(self.current_metric)
                self.after(0, lambda: self.display_data(data))
            except Exception as e:
                self.after(0, lambda: self.show_error(str(e)))

        threading.Thread(target=load, daemon=True).start()

    def display_data(self, data: dict):
        """Display data in table (FAST - no resize lag!)"""
        self.all_data = data

        # Clear table
        self.data_table.clear()

        steps = data["steps"]
        global_steps = data["global_steps"]
        timestamps = data["timestamps"]
        values = data["values"]

        total_rows = len(steps)
        if total_rows == 0:
            self.metric_title.configure(text=f"{self.current_metric}: No data")
            return

        # Pagination
        if self.page_size_selector.get() == "All":
            start, end = 0, total_rows
        else:
            start = self.current_page * self.page_size
            end = min(start + self.page_size, total_rows)

        # Insert rows (ttk.Treeview renders natively - FAST!)
        for i in range(start, end):
            ts = str(timestamps[i]) if timestamps[i] is not None else "N/A"
            val = values[i]
            val_str = (
                f"{val:.6f}"
                if val not in (None, "NaN", "Infinity", "-Infinity")
                else str(val)
            )

            self.data_table.insert(
                "end", [str(steps[i]), str(global_steps[i]), ts, val_str]
            )

        # Update controls
        if self.page_size_selector.get() == "All":
            total_pages = 1
            self.prev_btn.configure(state="disabled")
            self.next_btn.configure(state="disabled")
        else:
            total_pages = (total_rows + self.page_size - 1) // self.page_size
            self.prev_btn.configure(
                state="normal" if self.current_page > 0 else "disabled"
            )
            self.next_btn.configure(
                state="normal" if self.current_page < total_pages - 1 else "disabled"
            )

        self.page_label.configure(text=f"Page {self.current_page + 1} / {total_pages}")
        self.metric_title.configure(
            text=f"{self.current_metric} ({start + 1}-{end} of {total_rows})"
        )
        self.status_bar.configure(text="✅ Loaded")
        self.refresh_btn.configure(state="normal")

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.display_data(self.all_data)

    def next_page(self):
        total = len(self.all_data["steps"])
        pages = (total + self.page_size - 1) // self.page_size
        if self.current_page < pages - 1:
            self.current_page += 1
            self.display_data(self.all_data)

    def on_page_size_changed(self, size: str):
        if size == "All":
            self.page_size = 999999
        else:
            self.page_size = int(size)
        self.current_page = 0
        if self.all_data:
            self.display_data(self.all_data)

    def refresh_data(self):
        if self.current_metric:
            self.load_metric_data()

    def show_error(self, msg: str):
        self.metric_title.configure(text=f"Error: {msg}")
        self.refresh_btn.configure(state="normal")

    def delete_metric(self):
        """Delete entire metric with confirmation"""
        if not self.current_metric:
            self.status_bar.configure(text="❌ No metric selected")
            return

        from kohakuboard.inspector.widgets import ConfirmDialog

        dialog = ConfirmDialog(
            self,
            title="Delete Metric",
            message=f"Delete entire metric '{self.current_metric}'?\n\nThis will delete the metric file permanently.\nThis cannot be undone!",
            confirm_text="Delete Metric",
        )

        if dialog.get_result():
            # Delete the metric
            from kohakuboard.inspector.services.deletion_service import DeletionService

            try:
                board_path = self.data_service.board_path
                success = DeletionService.delete_entire_metric(
                    board_path, self.current_metric
                )

                if success:
                    self.status_bar.configure(text=f"✅ Deleted {self.current_metric}")
                    self.current_metric = None
                    self.data_table.clear()
                    self.metric_title.configure(text="Metric deleted - select another")
                    # Reload tree
                    self.load_metrics_tree()
                else:
                    self.status_bar.configure(text="❌ Failed to delete")

            except Exception as e:
                self.status_bar.configure(text=f"❌ Error: {e}")

    def delete_step_range(self):
        """Delete metric values in step range"""
        if not self.current_metric or not self.all_data:
            self.status_bar.configure(text="❌ No data loaded")
            return

        # Create dialog to get step range
        dialog = StepRangeDialog(self, self.all_data["steps"])

        if dialog.result:
            start_step, end_step = dialog.result

            from kohakuboard.inspector.widgets import ConfirmDialog

            confirm = ConfirmDialog(
                self,
                title="Delete Step Range",
                message=f"Delete steps {start_step} to {end_step} from '{self.current_metric}'?\n\nThis cannot be undone!",
                confirm_text="Delete Range",
            )

            if confirm.get_result():
                from kohakuboard.inspector.services.deletion_service import (
                    DeletionService,
                )

                try:
                    board_path = self.data_service.board_path
                    deleted = DeletionService.delete_metric_by_step_range(
                        board_path, self.current_metric, start_step, end_step
                    )

                    self.status_bar.configure(text=f"✅ Deleted {deleted} rows")
                    # Reload data
                    self.load_metric_data()

                except Exception as e:
                    self.status_bar.configure(text=f"❌ Error: {e}")

    def update_font_scale(self, scale: float):
        """Update font scale for tree and table

        Args:
            scale: Font scale multiplier
        """
        self.font_scale = scale
        self.tree.set_font_scale(scale)
        self.data_table.set_font_scale(scale)
