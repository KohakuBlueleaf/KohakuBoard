"""SQL query view - query board metadata"""

import sqlite3
import threading

import customtkinter as ctk

from kohakuboard.inspector.widgets import DataTable


class SQLView(ctk.CTkFrame):
    """View for executing SQL queries on board metadata"""

    def __init__(self, parent, data_service, font_scale: float = 1.0):
        super().__init__(parent)
        self.data_service = data_service
        self.font_scale = font_scale

        # Grid config: Results table gets most space
        self.grid_rowconfigure(0, weight=0)  # Title
        self.grid_rowconfigure(1, weight=0)  # Tables
        self.grid_rowconfigure(2, weight=0)  # Samples
        self.grid_rowconfigure(3, weight=0)  # Query input
        self.grid_rowconfigure(4, weight=0)  # Status
        self.grid_rowconfigure(5, weight=1)  # Results (expandable!)
        self.grid_columnconfigure(0, weight=1)

        self.build_ui()

    def build_ui(self):
        """Build UI"""
        # Title
        ctk.CTkLabel(
            self, text="SQL Query Interface", font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(20, 10))

        # Tables list
        tables_frame = ctk.CTkFrame(self)
        tables_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 10))

        ctk.CTkLabel(
            tables_frame, text="Available tables:", font=ctk.CTkFont(weight="bold")
        ).pack(side="left", padx=5)

        self.tables_label = ctk.CTkLabel(
            tables_frame,
            text="Loading...",
            fg_color=("gray85", "gray25"),  # Background for better contrast
            corner_radius=6,
        )
        self.tables_label.pack(side="left", padx=5, pady=5, ipadx=10, ipady=4)

        # Sample queries
        samples_frame = ctk.CTkFrame(self)
        samples_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 10))

        ctk.CTkLabel(samples_frame, text="Sample Queries:").pack(side="left", padx=5)

        sample_queries = [
            ("Recent Steps", "SELECT * FROM steps ORDER BY step DESC LIMIT 100"),
            (
                "Media List",
                "SELECT step, name, format, width, height FROM media ORDER BY step DESC LIMIT 50",
            ),
            ("Tables", "SELECT * FROM tables ORDER BY step DESC LIMIT 20"),
        ]

        for label, query in sample_queries:
            ctk.CTkButton(
                samples_frame,
                text=label,
                width=100,
                command=lambda q=query: self.load_sample_query(q),
            ).pack(side="left", padx=3)

        # Query input
        query_frame = ctk.CTkFrame(self)
        query_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=10)
        query_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(query_frame, text="Query (SELECT only):").grid(
            row=0, column=0, sticky="w", pady=(0, 5)
        )

        self.query_text = ctk.CTkTextbox(query_frame, height=100, font=("Consolas", 11))
        self.query_text.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.query_text.insert("1.0", "SELECT * FROM steps LIMIT 10")

        # Buttons
        btn_frame = ctk.CTkFrame(query_frame, fg_color="transparent")
        btn_frame.grid(row=2, column=0, sticky="w")

        self.execute_btn = ctk.CTkButton(
            btn_frame, text="▶ Execute", width=100, command=self.execute_query
        )
        self.execute_btn.pack(side="left", padx=(0, 5))

        ctk.CTkButton(
            btn_frame,
            text="Clear",
            width=80,
            command=lambda: self.query_text.delete("1.0", "end"),
        ).pack(side="left")

        # Load tables list
        self.load_tables_list()

        # Status
        self.status_label = ctk.CTkLabel(self, text="Ready", anchor="w")
        self.status_label.grid(row=4, column=0, sticky="w", padx=20, pady=5)

        # Results table
        results_frame = ctk.CTkFrame(self)
        results_frame.grid(row=5, column=0, sticky="nsew", padx=20, pady=(0, 20))
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)

        self.results_table = None
        self.results_container = results_frame

    def load_tables_list(self):
        """Load list of tables in database"""

        def load():
            try:
                db_path = self.data_service.board_path / "data" / "metadata.db"
                if not db_path.exists():
                    self.after(
                        0,
                        lambda: self.tables_label.configure(
                            text="metadata.db not found"
                        ),
                    )
                    return

                conn = sqlite3.connect(str(db_path))
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
                tables = [row[0] for row in cursor.fetchall()]
                conn.close()

                table_list = ", ".join(tables) if tables else "none"
                # Good contrast for both light and dark modes
                self.after(0, lambda: self.tables_label.configure(text=table_list))
            except Exception as e:
                self.after(
                    0,
                    lambda: self.tables_label.configure(
                        text=f"Error: {e}", text_color="red"
                    ),
                )

        threading.Thread(target=load, daemon=True).start()

    def load_sample_query(self, query: str):
        """Load a sample query into textbox"""
        self.query_text.delete("1.0", "end")
        self.query_text.insert("1.0", query)

    def execute_query(self):
        """Execute SQL query"""
        query = self.query_text.get("1.0", "end-1c").strip()

        if not query:
            self.status_label.configure(text="❌ Empty query")
            return

        # Validate SELECT only
        if not query.upper().startswith("SELECT"):
            self.status_label.configure(text="❌ Only SELECT queries allowed")
            return

        self.status_label.configure(text="")
        self.execute_btn.configure(state="disabled")

        def execute():
            try:
                # Get metadata DB path
                db_path = self.data_service.board_path / "data" / "metadata.db"

                if not db_path.exists():
                    self.after(0, lambda: self.show_error("metadata.db not found"))
                    return

                # Execute query
                conn = sqlite3.connect(str(db_path))
                cursor = conn.execute(query)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                conn.close()

                self.after(0, lambda: self.display_results(columns, rows))

            except Exception as e:
                self.after(0, lambda: self.show_error(str(e)))

        threading.Thread(target=execute, daemon=True).start()

    def display_results(self, columns: list[str], rows: list[tuple]):
        """Display query results"""
        # Clear old table
        if self.results_table:
            self.results_table.destroy()

        # Create new table
        self.results_table = DataTable(
            self.results_container,
            columns=columns,
            height=400,
            font_scale=self.font_scale,
        )
        self.results_table.grid(row=0, column=0, sticky="nsew")

        # Insert rows
        for row in rows:
            self.results_table.insert("end", [str(val) for val in row])

        self.status_label.configure(text=f"✅ Query returned {len(rows)} rows")
        self.execute_btn.configure(state="normal")

    def show_error(self, error: str):
        """Show error"""
        self.status_label.configure(text=f"❌ Error: {error}")
        self.execute_btn.configure(state="normal")

    def update_font_scale(self, scale: float):
        """Update font scale"""
        self.font_scale = scale
        if self.results_table:
            self.results_table.set_font_scale(scale)
