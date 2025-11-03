"""Confirmation dialog widget"""

import customtkinter as ctk


class ConfirmDialog(ctk.CTkToplevel):
    """Confirmation dialog for destructive operations"""

    def __init__(self, parent, title: str, message: str, confirm_text: str = "Delete"):
        super().__init__(parent)

        self.result = False

        # Window config
        self.title(title)
        self.geometry("400x200")
        self.resizable(False, False)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Center on parent
        self.after(100, lambda: self.lift())
        self.after(100, lambda: self.focus())

        # Message
        message_label = ctk.CTkLabel(
            self,
            text=message,
            wraplength=350,
            font=ctk.CTkFont(size=13),
        )
        message_label.pack(pady=30, padx=20)

        # Buttons
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(pady=(0, 20))

        ctk.CTkButton(
            button_frame,
            text="Cancel",
            width=100,
            command=self.cancel,
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            button_frame,
            text=confirm_text,
            width=100,
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
