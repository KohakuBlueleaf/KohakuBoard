"""High-performance data table widget using ttk.Treeview"""

from tkinter import ttk

import customtkinter as ctk


class DataTable(ctk.CTkFrame):
    """Fast data table widget using ttk.Treeview (handles thousands of rows)

    Unlike CTkLabel-based tables, this uses ttk.Treeview which:
    - Renders on demand (not all rows at once)
    - Handles window resize smoothly
    - Supports thousands of rows without lag
    """

    def __init__(
        self,
        master,
        columns: list[str],
        height: int = 400,
        font_scale: float = 1.0,
        **kwargs,
    ):
        """Initialize data table

        Args:
            master: Parent widget
            columns: List of column names
            height: Height in pixels
            font_scale: Font scale multiplier (0.75-2.0)
            **kwargs: Additional CTkFrame arguments
        """
        super().__init__(master, **kwargs)

        self.columns = columns
        self.column_ids = [f"col{i}" for i in range(len(columns))]
        self.font_scale = font_scale

        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Create ttk.Treeview configured as table
        self.tree = ttk.Treeview(
            self,
            columns=self.column_ids,
            show="headings",  # Only show column headers, not tree column
            selectmode="browse",
            height=height // 25,
        )
        self.tree.grid(row=0, column=0, sticky="nsew")

        # Configure columns
        for col_id, col_name in zip(self.column_ids, columns):
            self.tree.heading(col_id, text=col_name, anchor="w")
            self.tree.column(col_id, anchor="w", width=150, stretch=True)

        # Add scrollbars
        vsb = ctk.CTkScrollbar(self, orientation="vertical", command=self.tree.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=vsb.set)

        # Apply theme
        self.apply_theme()

    def apply_theme(self):
        """Apply CustomTkinter theme colors to table"""
        mode = ctk.get_appearance_mode()

        style = ttk.Style()

        if mode == "Dark":
            bg = "#2B2B2B"
            fg = "#DCE4EE"
            select_bg = "#1F538D"
            select_fg = "#FFFFFF"
            heading_bg = "#1a1a1a"
        else:
            bg = "#F0F0F0"
            fg = "#000000"
            select_bg = "#3B8ED0"
            select_fg = "#FFFFFF"
            heading_bg = "#E0E0E0"

        # Configure table style with scaled fonts
        base_font_size = 11
        base_heading_size = 12
        base_row_height = 30

        scaled_font_size = int(base_font_size * self.font_scale)
        scaled_heading_size = int(base_heading_size * self.font_scale)
        scaled_row_height = int(base_row_height * self.font_scale)

        style.theme_use("default")
        style.configure(
            "DataTable.Treeview",
            background=bg,
            foreground=fg,
            fieldbackground=bg,
            borderwidth=0,
            relief="flat",
            rowheight=scaled_row_height,
            font=("Arial", scaled_font_size),
        )
        style.configure(
            "DataTable.Treeview.Heading",
            background=heading_bg,
            foreground=fg,
            borderwidth=1,
            relief="flat",
            font=("Arial", scaled_heading_size, "bold"),
        )
        style.map(
            "DataTable.Treeview",
            background=[("selected", select_bg)],
            foreground=[("selected", select_fg)],
        )
        style.map(
            "DataTable.Treeview.Heading",
            background=[("active", heading_bg)],
        )

        self.tree.configure(style="DataTable.Treeview")

    def set_font_scale(self, scale: float):
        """Update font scale

        Args:
            scale: Font scale multiplier (0.75-2.0)
        """
        self.font_scale = scale
        self.apply_theme()

    def insert(self, index: str, values: list):
        """Insert a row

        Args:
            index: Insert position ("end" or integer)
            values: List of cell values
        """
        self.tree.insert("", index, values=values)

    def clear(self):
        """Clear all rows"""
        for item in self.tree.get_children():
            self.tree.delete(item)

    def get_selected(self) -> str | None:
        """Get selected row ID

        Returns:
            Selected item ID or None
        """
        selection = self.tree.selection()
        return selection[0] if selection else None

    def get_row_values(self, item_id: str) -> list:
        """Get values for a row

        Args:
            item_id: Row item ID

        Returns:
            List of cell values
        """
        return self.tree.item(item_id, "values")

    def set_data(self, rows: list[list]):
        """Set all data at once (replaces existing)

        Args:
            rows: List of row data (each row is a list of values)
        """
        # Clear existing
        self.clear()

        # Insert all rows
        for row in rows:
            self.tree.insert("", "end", values=row)

    def bind_select(self, callback):
        """Bind selection event

        Args:
            callback: Function called when row selected (receives item_id)
        """

        def on_select(event):
            selection = self.tree.selection()
            if selection:
                callback(selection[0])

        self.tree.bind("<<TreeviewSelect>>", on_select)
