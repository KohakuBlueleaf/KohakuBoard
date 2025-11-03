"""Board content view - shows data for a single board"""

import logging
from pathlib import Path

import customtkinter as ctk

logger = logging.getLogger("Inspector.BoardContent")


class BoardContentView(ctk.CTkFrame):
    """Content view for a single board (displayed in a tab)"""

    def __init__(self, parent, board_path: Path, app):
        super().__init__(parent)
        self.board_path = board_path
        self.app = app

        # Current view ('metrics', 'plots', etc.)
        self.current_view = None
        self.current_view_widget = None  # Store reference to current view

        # Grid configuration
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Initialize DataService for this board
        self.init_data_service()

        # Build UI
        self.build_ui()

    def init_data_service(self):
        """Initialize data service for this board"""
        try:
            from kohakuboard.inspector.services.data_service import DataService

            self.data_service = DataService(self.board_path)
            self.service_available = True
        except Exception as e:
            print(f"Failed to initialize DataService: {e}")
            self.service_available = False
            self.data_service = None

    def build_ui(self):
        """Build the UI"""
        if not self.service_available:
            # Show error
            error_label = ctk.CTkLabel(
                self,
                text=f"Failed to load board:\n{self.board_path}\n\n"
                "Check if board data is valid.",
                font=ctk.CTkFont(size=14),
                text_color="red",
            )
            error_label.grid(row=0, column=0)
            return

        # Content container (will hold different views)
        self.content_container = ctk.CTkFrame(self)
        self.content_container.grid(row=0, column=0, sticky="nsew")
        self.content_container.grid_rowconfigure(0, weight=1)
        self.content_container.grid_columnconfigure(0, weight=1)

        # Show default view (metrics)
        self.show_view("metrics")

    def show_view(self, view_id: str):
        """Switch to a different view

        Args:
            view_id: View identifier (metrics, plots, media, sql, export)
        """
        logger.info(f"BoardContent.show_view called: {view_id}")

        # Clear current content
        for widget in self.content_container.winfo_children():
            widget.destroy()

        self.current_view = view_id

        # Create appropriate view
        logger.debug(f"Creating {view_id} view...")
        match view_id:
            case "metrics":
                self.show_metrics_view()
            case "plots":
                self.show_plots_view()
            case "media":
                self.show_media_view()
            case "sql":
                self.show_sql_view()
            case "export":
                self.show_export_view()
            case _:
                self.show_placeholder(view_id)

        logger.debug(f"View {view_id} created successfully")

    def show_metrics_view(self):
        """Show metrics table view"""
        try:
            logger.debug("Importing MetricsView...")
            from kohakuboard.inspector.views.metrics_view import MetricsView

            logger.debug("Creating MetricsView instance...")
            # Get current font scale from app
            font_scale = getattr(self.app, "current_font_scale", 1.0)
            view = MetricsView(self.content_container, self.data_service, font_scale)
            view.grid(row=0, column=0, sticky="nsew")
            self.current_view_widget = view
            logger.info("MetricsView displayed successfully")
        except Exception as e:
            logger.error(f"Failed to show metrics view: {e}", exc_info=True)
            raise

    def show_plots_view(self):
        """Show plots view"""
        try:
            logger.debug("Importing PlotsView...")
            from kohakuboard.inspector.views.plots_view import PlotsView

            logger.debug("Creating PlotsView instance...")
            # Get current font scale from app
            font_scale = getattr(self.app, "current_font_scale", 1.0)
            view = PlotsView(self.content_container, self.data_service, font_scale)
            view.grid(row=0, column=0, sticky="nsew")
            self.current_view_widget = view
            logger.info("PlotsView displayed successfully")
        except Exception as e:
            logger.error(f"Failed to show plots view: {e}", exc_info=True)
            raise

    def update_font_scale(self, scale: float):
        """Update font scale for current view

        Args:
            scale: Font scale multiplier
        """
        if self.current_view_widget and hasattr(
            self.current_view_widget, "update_font_scale"
        ):
            self.current_view_widget.update_font_scale(scale)

    def cleanup(self):
        """Cleanup resources before closing"""
        # Cleanup current view if it has cleanup method
        if self.current_view_widget and hasattr(self.current_view_widget, "cleanup"):
            self.current_view_widget.cleanup()

    def show_media_view(self):
        """Show media view"""
        self.show_placeholder("Media")

    def show_sql_view(self):
        """Show SQL query view"""
        self.show_placeholder("SQL")

    def show_export_view(self):
        """Show export view"""
        self.show_placeholder("Export")

    def show_placeholder(self, view_name: str):
        """Show placeholder for not-yet-implemented view

        Args:
            view_name: Name of the view
        """
        label = ctk.CTkLabel(
            self.content_container,
            text=f"{view_name} View\n\n(Coming in future phases)",
            font=ctk.CTkFont(size=16),
        )
        label.grid(row=0, column=0)
