"""Main application window for KohakuBoard Inspector with multi-tab support"""

import logging
from pathlib import Path

import customtkinter as ctk

from kohakuboard.inspector.utils import detect_path_type, list_boards_in_container

# Setup logging for debugging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Inspector")

# Suppress noisy matplotlib font logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


class KohakuBoardInspector(ctk.CTk):
    """Main application window for KohakuBoard Inspector with multi-tab support"""

    def __init__(self, board_dir: str = "./kohakuboard"):
        super().__init__()

        self.board_dir = Path(board_dir)
        self.path_type = detect_path_type(self.board_dir)

        # Track open boards: {board_id: {'path': Path, 'service': DataService, 'view': ...}}
        self.open_boards = {}
        self.current_board_id = None  # Currently active board tab
        self.current_font_scale = 1.0  # Global font scale

        # Window configuration
        self.title("KohakuBoard Inspector")
        self.geometry("1400x900")
        self.minsize(1000, 600)

        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Create UI
        self.create_sidebar()
        self.create_tabbed_main_panel()

        # Initialize based on path type
        logger.info(f"Initializing inspector for: {self.board_dir}")
        logger.info(f"Path type detected: {self.path_type}")
        self.initialize_view()

        # Bind cleanup on window close
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_sidebar(self):
        """Create left sidebar with navigation"""
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(10, weight=1)

        # Logo/Title
        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="KohakuBoard\nInspector",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Current board/directory label
        self.dir_label = ctk.CTkLabel(
            self.sidebar,
            text=f"📁 {self.board_dir.name}",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self.dir_label.grid(row=1, column=0, padx=20, pady=(0, 20))

        # Navigation buttons (disabled initially until board is opened)
        # Note: No "Boards" needed - we have tabs for multiple boards
        nav_items = [
            ("Metrics", "metrics"),
            ("Plots", "plots"),
            ("Histograms", "histograms"),
            ("Media", "media"),
            ("SQL", "sql"),
            ("Export", "export"),
        ]

        self.nav_buttons = {}
        for i, (label, view_id) in enumerate(nav_items, start=2):
            btn = ctk.CTkButton(
                self.sidebar,
                text=label,
                command=lambda v=view_id: self.show_view(v),
                fg_color="transparent",
                hover_color=("gray70", "gray30"),
                anchor="w",
                state="disabled",  # Disabled until board opened
            )
            btn.grid(row=i, column=0, padx=20, pady=5, sticky="ew")
            self.nav_buttons[view_id] = btn

        # Font scale selector
        self.scale_label = ctk.CTkLabel(self.sidebar, text="Font Scale:", anchor="w")
        self.scale_label.grid(row=11, column=0, padx=20, pady=(20, 0))

        self.font_scale = ctk.CTkOptionMenu(
            self.sidebar,
            values=["0.75x", "1.0x", "1.25x", "1.5x", "2.0x"],
            command=self.change_font_scale,
        )
        self.font_scale.set("1.0x")
        self.font_scale.grid(row=12, column=0, padx=20, pady=(5, 10))

        # Appearance mode switch
        self.appearance_label = ctk.CTkLabel(
            self.sidebar, text="Appearance:", anchor="w"
        )
        self.appearance_label.grid(row=13, column=0, padx=20, pady=(10, 0))

        self.appearance_mode = ctk.CTkOptionMenu(
            self.sidebar,
            values=["System", "Light", "Dark"],
            command=self.change_appearance_mode,
        )
        self.appearance_mode.grid(row=14, column=0, padx=20, pady=(5, 20))

    def create_tabbed_main_panel(self):
        """Create main panel with tab support"""
        # Main panel container
        main_container = ctk.CTkFrame(self, corner_radius=0)
        main_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)

        # Tabview for multiple boards
        self.tabview = ctk.CTkTabview(main_container, command=self.on_tab_changed)
        self.tabview.grid(row=0, column=0, sticky="nsew")

        # Add "Board Selector" tab (always present)
        self.tabview.add("📋 Board Selector")

    def initialize_view(self):
        """Initialize view based on path type"""
        match self.path_type:
            case "container":
                # Show board selector in first tab
                self.show_board_selector()

            case "board":
                # Open this board directly
                self.open_board_in_tab(self.board_dir)
                # Close board selector tab since we have a board
                try:
                    self.tabview.delete("📋 Board Selector")
                except:
                    pass

            case "empty":
                self.show_empty_message()

            case "invalid":
                self.show_error_message(f"Invalid path: {self.board_dir}")

    def show_board_selector(self):
        """Show board selector in the Board Selector tab"""
        # Get the Board Selector tab frame
        selector_tab = self.tabview.tab("📋 Board Selector")

        # Clear existing content
        for widget in selector_tab.winfo_children():
            widget.destroy()

        # Create board selector view
        from kohakuboard.inspector.views.board_selector import BoardSelectorView

        self.board_selector_view = BoardSelectorView(selector_tab, self.board_dir, self)
        self.board_selector_view.pack(fill="both", expand=True)

    def show_empty_message(self):
        """Show message for empty directory"""
        selector_tab = self.tabview.tab("📋 Board Selector")

        message = ctk.CTkLabel(
            selector_tab,
            text=f"No boards found in:\n{self.board_dir}\n\n"
            "Create a board first using:\nfrom kohakuboard import Board",
            font=ctk.CTkFont(size=14),
        )
        message.pack(expand=True)

    def show_error_message(self, error: str):
        """Show error message"""
        selector_tab = self.tabview.tab("📋 Board Selector")

        message = ctk.CTkLabel(
            selector_tab,
            text=f"Error:\n{error}",
            font=ctk.CTkFont(size=14),
            text_color="red",
        )
        message.pack(expand=True)

    def open_board_in_tab(self, board_path: Path):
        """Open a board in a new tab

        Args:
            board_path: Path to board directory
        """
        board_id = board_path.name
        logger.info(f"Opening board in tab: {board_id}")

        # Check if already open
        if board_id in self.open_boards:
            logger.debug(f"Board already open, switching to tab: {board_id}")
            # Switch to existing tab
            self.tabview.set(self.open_boards[board_id]["tab_name"])
            return

        # Create new tab
        tab_name = board_id[:20]  # Truncate long names
        logger.debug(f"Creating new tab: {tab_name}")
        tab = self.tabview.add(tab_name)

        # Create board content view
        from kohakuboard.inspector.views.board_content import BoardContentView

        logger.debug(f"Creating BoardContentView for {board_id}")
        content_view = BoardContentView(tab, board_path, self)
        content_view.pack(fill="both", expand=True)

        # Store board data
        self.open_boards[board_id] = {
            "path": board_path,
            "view": content_view,
            "tab_name": tab_name,
            "current_view": "metrics",  # Default view
        }

        logger.info(f"Board opened successfully: {board_id}")

        # Switch to new tab
        self.tabview.set(tab_name)

        # IMPORTANT: Set current_board_id manually (tabview.set() doesn't always trigger callback)
        self.current_board_id = board_id
        logger.debug(f"Set current_board_id to: {self.current_board_id}")

        # Enable sidebar navigation and highlight Metrics button (default view)
        self.enable_navigation()
        self.nav_buttons["metrics"].configure(fg_color=("gray75", "gray25"))

    def close_board_tab(self, board_id: str):
        """Close a board tab

        Args:
            board_id: Board ID to close
        """
        if board_id not in self.open_boards:
            return

        tab_name = self.open_boards[board_id]["tab_name"]

        # Delete tab
        try:
            self.tabview.delete(tab_name)
        except:
            pass

        # Cleanup
        del self.open_boards[board_id]

        # If no boards open, disable navigation
        if not self.open_boards:
            self.disable_navigation()

    def on_tab_changed(self):
        """Handle tab change event"""
        current_tab = self.tabview.get()

        # Find which board is active
        for board_id, board_data in self.open_boards.items():
            if board_data["tab_name"] == current_tab:
                self.current_board_id = board_id
                self.enable_navigation()

                # Restore button states for this board's current view
                current_view = board_data.get("current_view", "metrics")
                for name, btn in self.nav_buttons.items():
                    if name == current_view:
                        btn.configure(fg_color=("gray75", "gray25"))
                    else:
                        btn.configure(fg_color="transparent")

                return

        # Board Selector tab active
        self.current_board_id = None
        self.disable_navigation()

    def enable_navigation(self):
        """Enable sidebar navigation buttons"""
        for btn in self.nav_buttons.values():
            btn.configure(state="normal")

    def disable_navigation(self):
        """Disable sidebar navigation buttons"""
        for btn in self.nav_buttons.values():
            btn.configure(state="disabled")

    def show_view(self, view_id: str):
        """Switch to a different view in current board tab

        Args:
            view_id: View identifier (metrics, plots, histograms, media, sql, export)
        """
        logger.debug(
            f"show_view called: view_id={view_id}, current_board={self.current_board_id}"
        )

        if not self.current_board_id:
            # No board open, can't switch views
            logger.warning("show_view called but no board is open")
            return

        # Get current board's view
        board_data = self.open_boards.get(self.current_board_id)
        if not board_data:
            logger.error(f"Board data not found for {self.current_board_id}")
            return

        logger.info(f"Switching to {view_id} view for board {self.current_board_id}")

        # Switch view in the board content
        board_data["view"].show_view(view_id)

        # Update sidebar button states to show active view
        for name, btn in self.nav_buttons.items():
            if name == view_id:
                btn.configure(fg_color=("gray75", "gray25"))
            else:
                btn.configure(fg_color="transparent")

        # Store current view in board data
        board_data["current_view"] = view_id
        logger.debug(f"View switched successfully to {view_id}")

    def change_font_scale(self, scale: str):
        """Change font scale for all views

        Args:
            scale: Font scale (0.75x, 1.0x, 1.25x, 1.5x, 2.0x)
        """
        # Extract scale factor
        scale_factor = float(scale.replace("x", ""))
        self.current_font_scale = scale_factor
        logger.info(f"Changing font scale to {scale_factor}")

        # Update all open board views
        for board_id, board_data in self.open_boards.items():
            if hasattr(board_data["view"], "update_font_scale"):
                board_data["view"].update_font_scale(scale_factor)

        logger.debug(f"Font scale updated for {len(self.open_boards)} boards")

    def change_appearance_mode(self, mode: str):
        """Change appearance mode

        Args:
            mode: Appearance mode (System, Light, Dark)
        """
        ctk.set_appearance_mode(mode.lower())

    def on_closing(self):
        """Handle window close - cleanup matplotlib figures"""
        logger.info("Closing inspector, cleaning up...")

        # Close all matplotlib figures in open boards
        for board_id, board_data in self.open_boards.items():
            try:
                if hasattr(board_data["view"], "cleanup"):
                    board_data["view"].cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up board {board_id}: {e}")

        # Close all matplotlib figures globally
        import matplotlib.pyplot as plt

        plt.close("all")

        # Destroy window
        self.destroy()
