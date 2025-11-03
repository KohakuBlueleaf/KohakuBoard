"""Step range selection dialog"""

import customtkinter as ctk


class StepRangeDialog(ctk.CTkToplevel):
    """Dialog to select step range for deletion"""

    def __init__(self, parent, available_steps: list[int]):
        super().__init__(parent)

        self.result = None
        self.min_step = min(available_steps) if available_steps else 0
        self.max_step = max(available_steps) if available_steps else 0

        # Window config
        self.title("Select Step Range")
        self.geometry("500x320")
        self.resizable(False, False)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Configure grid
        self.grid_rowconfigure(0, weight=0)  # Title
        self.grid_rowconfigure(1, weight=0)  # Start
        self.grid_rowconfigure(2, weight=0)  # End
        self.grid_rowconfigure(3, weight=0)  # Info
        self.grid_rowconfigure(4, weight=1)  # Spacer
        self.grid_rowconfigure(5, weight=0)  # Buttons
        self.grid_columnconfigure(0, weight=1)

        # Title
        ctk.CTkLabel(
            self,
            text="Select step range to delete:",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, pady=(25, 20))

        # Start step
        start_frame = ctk.CTkFrame(self, fg_color="transparent")
        start_frame.grid(row=1, column=0, sticky="ew", padx=40, pady=8)

        ctk.CTkLabel(
            start_frame,
            text="Start Step:",
            width=100,
            anchor="w",
            font=ctk.CTkFont(size=12),
        ).pack(side="left")
        self.start_entry = ctk.CTkEntry(
            start_frame, width=240, height=40, font=ctk.CTkFont(size=12)
        )
        self.start_entry.insert(0, str(self.min_step))
        self.start_entry.pack(side="left", padx=10)

        # End step
        end_frame = ctk.CTkFrame(self, fg_color="transparent")
        end_frame.grid(row=2, column=0, sticky="ew", padx=40, pady=8)

        ctk.CTkLabel(
            end_frame,
            text="End Step:",
            width=100,
            anchor="w",
            font=ctk.CTkFont(size=12),
        ).pack(side="left")
        self.end_entry = ctk.CTkEntry(
            end_frame, width=240, height=40, font=ctk.CTkFont(size=12)
        )
        self.end_entry.insert(0, str(self.max_step))
        self.end_entry.pack(side="left", padx=10)

        # Info
        ctk.CTkLabel(
            self,
            text=f"Available range: {self.min_step} - {self.max_step}",
            text_color="gray",
            font=ctk.CTkFont(size=11),
        ).grid(row=3, column=0, pady=10)

        # Buttons at bottom (fixed position with grid)
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=5, column=0, pady=(0, 30))

        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            width=160,
            height=50,
            font=ctk.CTkFont(size=14),
            command=self.cancel,
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            btn_frame,
            text="OK",
            width=160,
            height=50,
            font=ctk.CTkFont(size=14),
            command=self.ok,
        ).pack(side="left", padx=8)

    def ok(self):
        """Validate and return result"""
        try:
            start = int(self.start_entry.get())
            end = int(self.end_entry.get())

            if start > end:
                # Swap if needed
                start, end = end, start

            self.result = (start, end)
            self.destroy()

        except ValueError:
            # Invalid input
            pass

    def cancel(self):
        """Cancel"""
        self.result = None
        self.destroy()
