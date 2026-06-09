import os
import customtkinter
import tkinter.filedialog
from align_app.core import natural_sort

class SortingView(customtkinter.CTkFrame):
    def __init__(self, parent, on_start_slideshow=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.on_start_slideshow = on_start_slideshow
        self.file_list = []
        self.selected_index = -1

        # Stub widgets
        self.select_button = customtkinter.CTkButton(self, text="📁 Select Files", command=self.select_files)
        self.select_button.pack(pady=5)

        self.sort_button = customtkinter.CTkButton(
            self, text="↕ Sort Files", command=self.sort_files,
            fg_color="#1F6AA5", hover_color="#165a8a",
            font=customtkinter.CTkFont(size=14, weight="bold"),
            height=36
        )
        self.sort_button.pack(pady=5)

        self.up_button = customtkinter.CTkButton(self, text="▲ Up", command=self.move_up, width=80)
        self.up_button.pack(pady=3)

        self.down_button = customtkinter.CTkButton(self, text="▼ Down", command=self.move_down, width=80)
        self.down_button.pack(pady=3)

        self.remove_button = customtkinter.CTkButton(
            self, text="✕ Remove", command=self.remove_file,
            width=80, fg_color="#aa3333", hover_color="#882222"
        )
        self.remove_button.pack(pady=3)

        self.start_button = customtkinter.CTkButton(
            self, text="▶ Start Slideshow", command=self.start_slideshow,
            fg_color="#2FA572", hover_color="#238a5a",
            font=customtkinter.CTkFont(size=16, weight="bold"),
            height=44
        )
        self.start_button.pack(pady=10)

        self.help_button = customtkinter.CTkButton(
            self, text="❓ Help / Guide", command=self.show_help,
            width=120, fg_color="#555", hover_color="#777"
        )
        self.help_button.pack(pady=5)

        self.scroll_frame = customtkinter.CTkScrollableFrame(self, label_text="Selected Files")
        self.scroll_frame.pack(pady=5, fill="both", expand=True)
        
        self.labels = []

    def show_help(self):
        """Open a help/guide dialog."""
        help_win = customtkinter.CTkToplevel(self)
        help_win.title("Spectroscopy Alignment — Quick Guide")
        help_win.geometry("620x560")
        help_win.attributes("-topmost", True)

        scroll = customtkinter.CTkScrollableFrame(help_win)
        scroll.pack(fill="both", expand=True, padx=10, pady=10)

        guide_sections = [
            ("Step 1: Load Files", (
                "Click '📁 Select Files' to choose your TIFF spectroscopy images.\n"
                "You can select multiple files at once."
            )),
            ("Step 2: Sort Files", (
                "Click '↕ Sort Files' to auto-sort by filename (natural sorting).\n"
                "Use '▲ Up' / '▼ Down' to manually reorder if needed.\n"
                "Frame 1 (top of list) is the REFERENCE frame — all other frames\n"
                "will be aligned to it."
            )),
            ("Step 3: Start Slideshow", (
                "Click '▶ Start Slideshow' to begin analysis.\n"
                "The app computes a PCA reference line from Frame 1 at the\n"
                "default threshold of 99.9%. The red line shows the detected\n"
                "spectroscopic peak center. Warp is ON by default."
            )),
            ("Step 4: Auto Optimization", (
                "'Auto' — optimizes the PCA threshold for the CURRENT frame (~2s).\n"
                "'Auto All' — optimizes ALL frames at once (~15-20s total).\n"
                "The search finds the threshold that gives the tightest line fit.\n"
                "Progress is shown on the button (e.g. '3/10...')."
            )),
            ("Step 5: Navigate & Inspect", (
                "Use ← → arrow keys or '◀ Prev' / 'Next ▶' to step through frames.\n"
                "Use the frame slider for quick jumping.\n"
                "'▶ Play' auto-cycles through all frames.\n\n"
                "Check that the red reference line aligns with the bright\n"
                "spectroscopic line on each frame. With Warp ON, frames\n"
                "should look aligned when flipping between them."
            )),
            ("Step 6: Zoom", (
                "'🔍+ Zoom In' steps through 1× → 2× → 4× → 8× → 16×.\n"
                "Zoom centers on the reference line centroid.\n"
                "'🔍- Zoom Out' reverses. '⟲ Reset View' returns to 1×.\n"
                "Use zoom to inspect line fit quality at high magnification."
            )),
            ("Step 7: Manual Line Correction", (
                "If the auto-fit doesn't align a frame well:\n\n"
                "1. Click '✏ Manual Line' (button turns orange)\n"
                "2. Click TWO points on the spectroscopic line, far apart\n"
                "   (zoom to 8× or 16× for precision)\n"
                "3. The app computes the midpoint between your clicks and\n"
                "   draws a line through it using Frame 1's reference slope\n"
                "4. The warp offset is recalculated automatically\n\n"
                "You can draw manual lines directly on the warped image —\n"
                "the app back-calculates the un-warped coordinates.\n\n"
                "'Clear Manual' removes the manual override for that frame."
            )),
            ("Tips", (
                "• Each frame stores its own PCA threshold independently\n"
                "• The PCA slider shows/controls the CURRENT frame's threshold\n"
                "• Type exact values in the threshold text box + press Enter\n"
                "• Warp is purely translational (no rotation or stretching)\n"
                "• Use viridis colormap for best visibility of faint features"
            )),
        ]

        for title, body in guide_sections:
            title_label = customtkinter.CTkLabel(
                scroll, text=title,
                font=customtkinter.CTkFont(size=15, weight="bold"),
                anchor="w"
            )
            title_label.pack(fill="x", padx=5, pady=(10, 2))

            body_label = customtkinter.CTkLabel(
                scroll, text=body,
                anchor="w", justify="left",
                wraplength=560
            )
            body_label.pack(fill="x", padx=15, pady=(0, 5))

    def select_files(self):
        # Allow mock override of file dialog
        files = tkinter.filedialog.askopenfilenames(
            title="Select TIFF files",
            filetypes=[("TIFF Files", "*.tif;*.tiff")]
        )
        if files:
            self.file_list.extend(list(files))
            self.update_listbox()

    def sort_files(self):
        self.file_list = natural_sort(self.file_list)
        self.update_listbox()

    def select_item(self, idx):
        self.selected_index = idx
        self.update_listbox()

    def move_up(self):
        idx = self.selected_index
        if 0 < idx < len(self.file_list):
            self.file_list[idx], self.file_list[idx-1] = self.file_list[idx-1], self.file_list[idx]
            self.selected_index = idx - 1
            self.update_listbox()

    def move_down(self):
        idx = self.selected_index
        if 0 <= idx < len(self.file_list) - 1:
            self.file_list[idx], self.file_list[idx+1] = self.file_list[idx+1], self.file_list[idx]
            self.selected_index = idx + 1
            self.update_listbox()

    def remove_file(self):
        idx = self.selected_index
        if 0 <= idx < len(self.file_list):
            self.file_list.pop(idx)
            if self.file_list:
                self.selected_index = min(idx, len(self.file_list) - 1)
            else:
                self.selected_index = -1
            self.update_listbox()

    def start_slideshow(self):
        if self.on_start_slideshow and self.file_list:
            self.on_start_slideshow(self.file_list)

    def update_listbox(self):
        for lbl in self.labels:
            lbl.destroy()
        self.labels.clear()

        for idx, filename in enumerate(self.file_list):
            bg = "blue" if idx == self.selected_index else "transparent"
            lbl = customtkinter.CTkLabel(
                self.scroll_frame,
                text=os.path.basename(filename),
                fg_color=bg
            )
            lbl.bind("<Button-1>", lambda event, i=idx: self.select_item(i))
            lbl.pack(fill="x", anchor="w")
            self.labels.append(lbl)
