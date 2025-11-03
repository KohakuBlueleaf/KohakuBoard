"""Media view - browse and view media files"""

import io
import threading

import customtkinter as ctk
from PIL import Image

from kohakuboard.inspector.widgets import TreeExplorer


class MediaView(ctk.CTkFrame):
    """View for browsing and viewing media files"""

    def __init__(self, parent, data_service, font_scale: float = 1.0):
        super().__init__(parent)
        self.data_service = data_service
        self.font_scale = font_scale
        self.current_media_name = None
        self.current_media_entries = []
        self.current_step_index = 0

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        self.build_ui()
        self.load_media_list()

    def build_ui(self):
        """Build UI"""
        # LEFT: Media list
        left_frame = ctk.CTkFrame(self, width=300)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_propagate(False)

        ctk.CTkLabel(
            left_frame, text="Media Logs", font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=15, pady=(15, 10))

        self.tree = TreeExplorer(
            left_frame,
            height=500,
            command=self.on_media_clicked,
            font_scale=self.font_scale,
        )
        self.tree.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

        # RIGHT: Preview area
        right_frame = ctk.CTkFrame(self)
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.grid_rowconfigure(0, weight=0)  # Title
        right_frame.grid_rowconfigure(1, weight=0)  # Slider controls
        right_frame.grid_rowconfigure(2, weight=1)  # Image preview
        right_frame.grid_columnconfigure(0, weight=1)

        # Title
        self.title_label = ctk.CTkLabel(
            right_frame,
            text="Select a media log to view",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.title_label.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 10))

        # Step slider controls
        slider_frame = ctk.CTkFrame(right_frame)
        slider_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 10))

        ctk.CTkLabel(slider_frame, text="Step:").pack(side="left", padx=5)

        self.step_slider = ctk.CTkSlider(
            slider_frame,
            from_=0,
            to=1,
            number_of_steps=1,
            command=self.on_step_changed,
        )
        self.step_slider.set(0)
        self.step_slider.pack(side="left", fill="x", expand=True, padx=10)

        self.step_label = ctk.CTkLabel(slider_frame, text="0 / 0", width=80)
        self.step_label.pack(side="left", padx=5)

        # Delete buttons
        ctk.CTkButton(
            slider_frame,
            text="🗑️ Delete This",
            width=100,
            fg_color="red",
            hover_color="darkred",
            command=self.delete_current_image,
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            slider_frame,
            text="🗑️ Delete All",
            width=100,
            fg_color="darkred",
            hover_color="#8B0000",
            command=self.delete_media_log,
        ).pack(side="left", padx=5)

        self.step_slider.configure(state="disabled")

        # Image preview area (fills available space)
        preview_container = ctk.CTkFrame(right_frame)
        preview_container.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
        preview_container.grid_rowconfigure(0, weight=1)
        preview_container.grid_columnconfigure(0, weight=1)

        # Label for image (will be updated with CTkImage)
        self.image_label = ctk.CTkLabel(
            preview_container,
            text="",
        )
        self.image_label.grid(row=0, column=0, sticky="nsew")

        # Info label below image
        self.info_label = ctk.CTkLabel(
            preview_container,
            text="Select a media log from the tree",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self.info_label.grid(row=1, column=0, pady=10)

        # Status bar at bottom
        self.status_bar = ctk.CTkLabel(
            right_frame,
            text="",
            anchor="w",
            text_color="gray",
            font=ctk.CTkFont(size=10),
        )
        self.status_bar.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 10))

    def load_media_list(self):
        """Load media list"""
        self.status_bar.configure(text="⏳ Loading media list...")

        def load():
            try:
                media_names = self.data_service.get_available_media_names()
                self.after(0, lambda: self.build_tree(media_names))
                self.after(0, lambda: self.status_bar.configure(text="✅ Loaded"))
            except Exception as e:
                self.after(0, lambda: self.status_bar.configure(text=f"❌ Error: {e}"))

        threading.Thread(target=load, daemon=True).start()

    def build_tree(self, media_names: list[str]):
        """Build tree"""
        self.tree.clear()

        if not media_names:
            self.tree.insert("", "end", text="(No media found)", tags=())
            return

        for name in media_names:
            self.tree.insert("", "end", text=f"🖼️ {name}", tags=(f"media:{name}",))

    def on_media_clicked(self, item_id):
        """Handle media click"""
        if not item_id:
            return

        tags = self.tree.get_tags(item_id)
        for tag in tags:
            if tag.startswith("media:"):
                media_name = tag[6:]
                self.load_media_entries(media_name)
                break

    def load_media_entries(self, media_name: str):
        """Load media entries for a media log"""
        self.current_media_name = media_name
        self.title_label.configure(text=f"{media_name}")
        self.step_slider.configure(state="disabled")

        def load():
            try:
                self.after(
                    0, lambda: self.status_bar.configure(text="⏳ Loading entries...")
                )
                entries = self.data_service.get_media_entries(
                    media_name
                )  # Get all entries
                self.after(0, lambda: self.setup_slider(media_name, entries))
                self.after(0, lambda: self.status_bar.configure(text="✅ Loaded"))
            except Exception as exc:
                error_msg = f"Failed to load: {exc}"
                self.after(0, lambda: self.show_error(error_msg))
                self.after(0, lambda: self.status_bar.configure(text=f"❌ {error_msg}"))

        threading.Thread(target=load, daemon=True).start()

    def setup_slider(self, media_name: str, entries: list[dict]):
        """Setup slider with loaded entries"""
        self.current_media_entries = entries
        self.current_step_index = 0

        if not entries:
            self.title_label.configure(text=f"{media_name}: No media")
            self.info_label.configure(text="No media entries found")
            return

        self.title_label.configure(text=f"{media_name} ({len(entries)} steps)")

        # Configure slider
        self.step_slider.configure(
            from_=0,
            to=len(entries) - 1,
            number_of_steps=len(entries) - 1 if len(entries) > 1 else 1,
            state="normal",
        )
        self.step_slider.set(0)
        self.step_label.configure(text=f"1 / {len(entries)}")

        # Show first entry
        self.display_current_entry()

    def on_step_changed(self, value):
        """Handle step slider change"""
        self.current_step_index = int(value)
        self.step_label.configure(
            text=f"{self.current_step_index + 1} / {len(self.current_media_entries)}"
        )
        self.display_current_entry()

    def display_current_entry(self):
        """Display the current step's image"""
        if not self.current_media_entries or self.current_step_index >= len(
            self.current_media_entries
        ):
            return

        entry = self.current_media_entries[self.current_step_index]

        # Update info
        info = f"Step {entry.get('step', 'N/A')}"
        if entry.get("caption"):
            info += f" - {entry['caption']}"
        self.info_label.configure(text=info)

        # Load image
        filename = entry.get("filename")
        if filename:
            self.image_label.configure(text="")

            def load_image():
                try:
                    self.after(
                        0, lambda: self.status_bar.configure(text="⏳ Loading image...")
                    )
                    binary_data = self.data_service.get_media_binary(filename)
                    if binary_data:
                        image = Image.open(io.BytesIO(binary_data))

                        # Get preview area size (approximate)
                        preview_width = 800
                        preview_height = 600

                        # Scale to fit while maintaining aspect ratio
                        image.thumbnail(
                            (preview_width, preview_height), Image.Resampling.LANCZOS
                        )

                        ctk_image = ctk.CTkImage(image, size=image.size)
                        self.after(
                            0, lambda img=ctk_image: self.show_current_image(img)
                        )
                        self.after(
                            0, lambda: self.status_bar.configure(text="✅ Image loaded")
                        )
                    else:
                        self.after(
                            0,
                            lambda: self.image_label.configure(
                                text=f"Image not found: {filename}", text_color="orange"
                            ),
                        )
                        self.after(
                            0,
                            lambda: self.status_bar.configure(
                                text="❌ Image not found"
                            ),
                        )
                except Exception as exc:
                    error_msg = str(exc)
                    self.after(
                        0,
                        lambda e=error_msg: self.image_label.configure(
                            text=f"Error: {e}", text_color="red"
                        ),
                    )
                    self.after(
                        0, lambda e=error_msg: self.status_bar.configure(text=f"❌ {e}")
                    )

            threading.Thread(target=load_image, daemon=True).start()
        else:
            self.image_label.configure(text="No filename available")

    def show_current_image(self, ctk_image):
        """Show image in label"""
        self.image_label.configure(image=ctk_image, text="")
        self.image_label.image = ctk_image  # Keep reference

    def show_error(self, msg: str):
        """Show error"""
        self.title_label.configure(text=f"Error: {msg}")

    def delete_current_image(self):
        """Delete current image"""
        if not self.current_media_entries or self.current_step_index >= len(
            self.current_media_entries
        ):
            return

        entry = self.current_media_entries[self.current_step_index]

        from kohakuboard.inspector.widgets import ConfirmDialog

        dialog = ConfirmDialog(
            self,
            title="Delete Image",
            message=f"Delete image at step {entry.get('step')}?\n\nThis cannot be undone!",
            confirm_text="Delete",
        )

        if dialog.get_result():
            from kohakuboard.inspector.services.deletion_service import DeletionService

            try:
                board_path = self.data_service.board_path
                DeletionService.delete_media_entry(
                    board_path, entry["media_hash"], entry["format"]
                )

                self.status_bar.configure(text="✅ Deleted image")

                # Remove from current list and reload
                self.current_media_entries.pop(self.current_step_index)

                if self.current_media_entries:
                    # Adjust index if at end
                    if self.current_step_index >= len(self.current_media_entries):
                        self.current_step_index = len(self.current_media_entries) - 1

                    # Reload slider
                    self.setup_slider(
                        self.current_media_name, self.current_media_entries
                    )
                else:
                    # No more images
                    self.title_label.configure(text=f"{self.current_media_name}: Empty")
                    self.image_label.configure(text="No more images")
                    self.step_slider.configure(state="disabled")

            except Exception as e:
                self.status_bar.configure(text=f"❌ Error: {e}")

    def delete_media_log(self):
        """Delete entire media log"""
        if not self.current_media_name:
            return

        from kohakuboard.inspector.widgets import ConfirmDialog

        dialog = ConfirmDialog(
            self,
            title="Delete Media Log",
            message=f"Delete entire media log '{self.current_media_name}'?\n\n"
            f"This will delete {len(self.current_media_entries)} images permanently!\n\n"
            f"This cannot be undone!",
            confirm_text="Delete All",
        )

        if dialog.get_result():
            from kohakuboard.inspector.services.deletion_service import DeletionService

            try:
                board_path = self.data_service.board_path
                deleted = DeletionService.delete_media_log(
                    board_path, self.current_media_name
                )

                self.status_bar.configure(text=f"✅ Deleted {deleted} images")
                self.current_media_name = None
                self.current_media_entries = []
                self.image_label.configure(text="Select a media log")
                self.step_slider.configure(state="disabled")

                # Reload media list
                self.load_media_list()

            except Exception as e:
                self.status_bar.configure(text=f"❌ Error: {e}")

    def update_font_scale(self, scale: float):
        """Update font scale"""
        self.font_scale = scale
        self.tree.set_font_scale(scale)
