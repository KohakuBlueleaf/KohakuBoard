"""Custom tree explorer widget - wraps ttk.Treeview with CTk styling"""

from tkinter import ttk

import customtkinter as ctk


class TreeExplorer(ctk.CTkFrame):
    """Modern tree view widget with CustomTkinter styling

    Wraps ttk.Treeview in a CTkFrame with styled scrollbar and theme matching.
    """

    def __init__(
        self,
        master,
        height: int = 400,
        command=None,  # Click callback: function(item_id)
        font_scale: float = 1.0,
        **kwargs,
    ):
        """Initialize tree explorer

        Args:
            master: Parent widget
            height: Height in pixels
            command: Callback function called when item clicked (receives item_id)
            font_scale: Font scale multiplier (0.75-2.0)
            **kwargs: Additional CTkFrame arguments
        """
        super().__init__(master, **kwargs)

        self.click_callback = command
        self._item_tags = {}  # Store tags for items
        self.font_scale = font_scale

        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Create ttk.Treeview
        self.tree = ttk.Treeview(
            self,
            show="tree",  # Only show tree column, no headings
            selectmode="browse",
            height=height // 25,  # Approximate row height
        )
        self.tree.grid(row=0, column=0, sticky="nsew")

        # CTk scrollbar
        scrollbar = ctk.CTkScrollbar(self, command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Apply styling
        self.apply_theme()

        # Bind clicks
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", self._on_double_click)

    def apply_theme(self):
        """Apply CustomTkinter theme colors to ttk.Treeview"""
        mode = ctk.get_appearance_mode()

        # Create style
        style = ttk.Style()

        if mode == "Dark":
            # Dark theme colors
            bg = "#2B2B2B"
            fg = "#DCE4EE"
            select_bg = "#1F538D"
            select_fg = "#FFFFFF"
        else:
            # Light theme colors
            bg = "#F0F0F0"
            fg = "#000000"
            select_bg = "#3B8ED0"
            select_fg = "#FFFFFF"

        # Configure Treeview style with scaled font
        base_font_size = 12
        base_row_height = 32

        scaled_font_size = int(base_font_size * self.font_scale)
        scaled_row_height = int(base_row_height * self.font_scale)

        style.theme_use("default")
        style.configure(
            "Custom.Treeview",
            background=bg,
            foreground=fg,
            fieldbackground=bg,
            borderwidth=0,
            relief="flat",
            rowheight=scaled_row_height,
            font=("Arial", scaled_font_size),
        )
        style.map(
            "Custom.Treeview",
            background=[("selected", select_bg)],
            foreground=[("selected", select_fg)],
        )

        self.tree.configure(style="Custom.Treeview")

    def set_font_scale(self, scale: float):
        """Update font scale

        Args:
            scale: Font scale multiplier (0.75-2.0)
        """
        self.font_scale = scale
        self.apply_theme()

    def insert(self, parent: str, index: str, text: str = "", tags: tuple = ()):
        """Insert item into tree

        Args:
            parent: Parent item ID ("" for root)
            index: Insert position ("end", integer)
            text: Display text
            tags: Tags tuple for categorization

        Returns:
            Item ID
        """
        item_id = self.tree.insert(parent, index, text=text)

        # Store tags separately (ttk.Treeview tags behavior is quirky)
        if tags:
            self._item_tags[item_id] = tags
            self.tree.item(item_id, tags=tags)

        return item_id

    def delete(self, *items):
        """Delete items from tree

        Args:
            *items: Item IDs to delete
        """
        for item in items:
            if item in self._item_tags:
                del self._item_tags[item]
            self.tree.delete(item)

    def get_children(self, item: str = ""):
        """Get children of an item

        Args:
            item: Parent item ID ("" for root)

        Returns:
            Tuple of child item IDs
        """
        return self.tree.get_children(item)

    def item(self, item_id: str, option: str | None = None, **kw):
        """Get or set item properties

        Args:
            item_id: Item ID
            option: Property name to get
            **kw: Properties to set

        Returns:
            Property value if option specified, else property dict
        """
        return self.tree.item(item_id, option=option, **kw)

    def get_tags(self, item_id: str) -> tuple:
        """Get tags for an item

        Args:
            item_id: Item ID

        Returns:
            Tags tuple
        """
        return self._item_tags.get(item_id, ())

    def _on_select(self, event):
        """Handle selection event"""
        selection = self.tree.selection()
        if selection and self.click_callback:
            self.click_callback(selection[0])

    def _on_double_click(self, event):
        """Handle double-click event"""
        # Could add separate double-click callback if needed
        pass

    def clear(self):
        """Clear all items from tree"""
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._item_tags.clear()

    # Override geometry methods to apply to frame instead of tree
    def pack(self, **kwargs):
        """Pack the frame"""
        super().pack(**kwargs)

    def grid(self, **kwargs):
        """Grid the frame"""
        super().grid(**kwargs)

    def place(self, **kwargs):
        """Place the frame"""
        super().place(**kwargs)
