"""Confirmation dialog widget"""

import customtkinter as ctk


class ConfirmDialog(ctk.CTkToplevel):
    """Confirmation dialog for destructive operations"""

    def __init__(self, parent, title: str, message: str, confirm_text: str = "Delete"):
        super().__init__(parent)

        self.result = False

        # Window config
        self.title(title)
        self.geometry("500x250")
        self.resizable(False, False)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Center on parent
        self.after(100, lambda: self.lift())
        self.after(100, lambda: self.focus())

        # Configure grid for better layout control
        self.grid_rowconfigure(0, weight=1)  # Message (expandable)
        self.grid_rowconfigure(1, weight=0)  # Buttons (fixed)
        self.grid_columnconfigure(0, weight=1)

        # Message
        message_label = ctk.CTkLabel(
            self,
            text=message,
            wraplength=440,
            font=ctk.CTkFont(size=13),
            justify="center",
        )
        message_label.grid(row=0, column=0, pady=(30, 20), padx=30)

        # Buttons (fixed at bottom)
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=1, column=0, pady=(10, 30))

        ctk.CTkButton(
            button_frame,
            text="Cancel",
            width=170,
            height=50,
            font=ctk.CTkFont(size=14),
            command=self.cancel,
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            button_frame,
            text=confirm_text,
            width=170,
            height=50,
            font=ctk.CTkFont(size=14),
            fg_color="red",
            hover_color="darkred",
            command=self.confirm,
        ).pack(side="left", padx=10)

    def confirm(self):
        """User confirmed"""
        self.result = True
        self.destroy()

    def cancel(self):
        """User cancelled"""
        self.result = False
        self.destroy()

    def get_result(self) -> bool:
        """Wait for dialog and return result

        Returns:
            True if confirmed, False if cancelled
        """
        self.wait_window()
        return self.result
