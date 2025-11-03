"""Board selector view - shows available boards in container"""

from pathlib import Path

import customtkinter as ctk

from kohakuboard.inspector.utils import (
    list_boards_in_container,
    format_size,
    get_board_size,
)


class BoardSelectorView(ctk.CTkFrame):
    """View for selecting boards from a container directory"""

    def __init__(self, parent, container_path: Path, app):
        super().__init__(parent)
        self.container_path = container_path
        self.app = app

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.build_ui()
        self.load_boards()

    def build_ui(self):
        """Build the UI"""
        # Header
        header = ctk.CTkFrame(self)
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=20)

        title = ctk.CTkLabel(
            header,
            text="Available Boards",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        title.pack(side="left", padx=(0, 20))

        self.search_entry = ctk.CTkEntry(
            header, placeholder_text="Search boards...", width=250
        )
        self.search_entry.pack(side="left", padx=5)
        self.search_entry.bind("<KeyRelease>", lambda e: self.filter_boards())

        self.refresh_btn = ctk.CTkButton(
            header, text="🔄 Refresh", width=100, command=self.load_boards
        )
        self.refresh_btn.pack(side="left", padx=5)

        # Boards grid (scrollable)
        self.boards_frame = ctk.CTkScrollableFrame(self)
        self.boards_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))

        # Status bar
        self.status_label = ctk.CTkLabel(self, text="Loading boards...", anchor="w")
        self.status_label.grid(row=2, column=0, sticky="w", padx=20, pady=(0, 10))

    def load_boards(self):
        """Load boards from container"""
        # Clear existing
        for widget in self.boards_frame.winfo_children():
            widget.destroy()

        # Get boards
        boards = list_boards_in_container(self.container_path)

        if not boards:
            # No boards found
            no_boards_label = ctk.CTkLabel(
                self.boards_frame,
                text="No boards found in this directory\n\n"
                "Create a board first using:\nfrom kohakuboard import Board",
                font=ctk.CTkFont(size=14),
            )
            no_boards_label.pack(expand=True, pady=50)
            self.status_label.configure(text="No boards found")
            return

        # Store for filtering
        self.all_boards = boards
        self.display_boards(boards)

        self.status_label.configure(text=f"Found {len(boards)} boards")

    def display_boards(self, boards):
        """Display boards in grid (max 6 per page)

        Args:
            boards: List of board dicts
        """
        # Clear existing
        for widget in self.boards_frame.winfo_children():
            widget.destroy()

        # Limit to 6 boards (3 rows × 2 columns)
        max_per_page = 6
        boards_to_show = boards[:max_per_page]

        if len(boards) > max_per_page:
            # Show pagination message
            more_label = ctk.CTkLabel(
                self.boards_frame,
                text=f"Showing {max_per_page} of {len(boards)} boards\n"
                f"(Pagination coming in future update)",
                text_color="gray",
                font=ctk.CTkFont(size=11),
            )
            more_label.grid(row=0, column=0, columnspan=2, pady=10)
            start_row = 1
        else:
            start_row = 0

        # Create board cards in grid
        for i, board in enumerate(boards_to_show):
            card = self.create_board_card(board)
            # Grid layout: 2 columns, max 3 rows
            row = start_row + (i // 2)
            col = i % 2
            card.grid(row=row, column=col, padx=10, pady=10, sticky="ew")

        # Configure grid weights
        self.boards_frame.grid_columnconfigure(0, weight=1)
        self.boards_frame.grid_columnconfigure(1, weight=1)

    def create_board_card(self, board: dict):
        """Create a card widget for a board

        Args:
            board: Board info dict

        Returns:
            CTkFrame with board info
        """
        card = ctk.CTkFrame(self.boards_frame, corner_radius=10)

        # Board name
        name_label = ctk.CTkLabel(
            card, text=board["name"], font=ctk.CTkFont(size=16, weight="bold")
        )
        name_label.pack(anchor="w", padx=15, pady=(15, 5))

        # Board ID (smaller, gray)
        id_label = ctk.CTkLabel(
            card,
            text=board["board_id"],
            font=ctk.CTkFont(size=10),
            text_color="gray",
        )
        id_label.pack(anchor="w", padx=15, pady=(0, 10))

        # Info frame
        info_frame = ctk.CTkFrame(card, fg_color="transparent")
        info_frame.pack(fill="x", padx=15, pady=(0, 10))

        # Created date
        date_text = (
            board["created_at"][:19]
            if len(board["created_at"]) > 19
            else board["created_at"]
        )
        date_label = ctk.CTkLabel(
            info_frame,
            text=f"📅 {date_text}",
            font=ctk.CTkFont(size=11),
        )
        date_label.pack(anchor="w")

        # Size
        size = get_board_size(board["path"])
        size_label = ctk.CTkLabel(
            info_frame,
            text=f"💾 {format_size(size)}",
            font=ctk.CTkFont(size=11),
        )
        size_label.pack(anchor="w")

        # Open button
        open_btn = ctk.CTkButton(
            card,
            text="Open Board",
            command=lambda: self.app.open_board_in_tab(board["path"]),
            height=35,
        )
        open_btn.pack(fill="x", padx=15, pady=(5, 15))

        return card

    def filter_boards(self):
        """Filter boards based on search text"""
        search_text = self.search_entry.get().lower()

        if not search_text:
            # Show all boards
            self.display_boards(self.all_boards)
            return

        # Filter boards
        filtered = [
            b
            for b in self.all_boards
            if search_text in b["name"].lower() or search_text in b["board_id"].lower()
        ]

        self.display_boards(filtered)
        self.status_label.configure(
            text=f"Showing {len(filtered)} of {len(self.all_boards)} boards"
        )
