import os
import customtkinter
import tkinter.filedialog
from align_app.core import natural_sort

class SortingView(customtkinter.CTkFrame):
    """
    A view that allows users to select, organize, and sort files for the slideshow.

    This UI component provides buttons to add files via a file dialog, sort them
    naturally, reorder them manually (up/down), and remove them. It acts as the
    entry point before launching the main image analysis slideshow.
    """
    def __init__(self, parent, on_start_slideshow=None, **kwargs):
        """
        Initialize the sorting view and set up the user interface.

        Args:
            parent: The parent widget that contains this frame.
            on_start_slideshow (callable, optional): Callback triggered when the
                start slideshow button is clicked.
            **kwargs: Additional keyword arguments passed to the CTkFrame constructor.
        """
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

        self.clear_button = customtkinter.CTkButton(
            self, text="🗑 Clear All", command=self.clear_all,
            width=80, fg_color="#883333", hover_color="#662222"
        )
        self.clear_button.pack(pady=3)

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
                "You can select multiple files at once.\n"
                "Use '🗑 Clear All' to remove all files and start over."
            )),
            ("Step 2: Sort Files", (
                "Click '↕ Sort Files' to auto-sort by filename (natural sorting).\n"
                "Use '▲ Up' / '▼ Down' to manually reorder if needed.\n"
                "Frame 1 (top of list) is the REFERENCE frame — all other frames\n"
                "will be aligned to it."
            )),
            ("Step 3: Start Slideshow", (
                "Click '▶ Start Slideshow' to begin analysis.\n"
                "The default alignment engine is ECC (Enhanced Correlation\n"
                "Coefficient), which works well for most datasets. Warp is\n"
                "ON by default — frames are translated to align with Frame 1."
            )),
            ("Alignment Engines", (
                "Switch engines using the dropdown in the top-right navbar:\n\n"
                "• ECC (default) — Iterative Enhanced Correlation Coefficient.\n"
                "  Uses a 2-stage coarse-to-fine Gaussian pyramid for robust\n"
                "  sub-pixel alignment. Best for general/diffuse datasets.\n"
                "  Click 'Precompute' to batch-align all frames.\n\n"
                "• PCA — Peak-Line Fitting via SVD + Phase Correlation.\n"
                "  Detects the spectral line via intensity thresholding and\n"
                "  fits a reference line through Frame 1. Best for datasets\n"
                "  with a sharp, well-defined spectral line. Use the threshold\n"
                "  slider to tune sensitivity, or 'Auto' / 'Auto All' to\n"
                "  optimize automatically.\n\n"
                "• Phase Correlation — Fast Fourier-domain translation\n"
                "  estimation. Very fast but less accurate on noisy data.\n"
                "  Also used internally as a fallback by PCA."
            )),
            ("Step 4: Navigate & Inspect", (
                "Use ← → arrow keys or '◀ Prev' / 'Next ▶' to step through frames.\n"
                "Use the frame slider for quick jumping.\n"
                "'▶ Play' auto-cycles through all frames.\n\n"
                "With Warp ON, frames should look aligned when flipping between\n"
                "them. Toggle Warp OFF to see the original unaligned frames."
            )),
            ("Step 5: Zoom", (
                "Click '🔍+ Zoom In' to toggle interactive zoom mode, then click\n"
                "anywhere on the image to zoom into that point.\n"
                "Zoom steps are 1× → 2× → 4× → 8× → 16×.\n"
                "'🔍- Zoom Out' zooms out from the current view center.\n"
                "'⟲ Reset View' returns to 1×."
            )),
            ("Step 6: Manual Line Correction (PCA only)", (
                "If PCA auto-fit doesn't align a frame well:\n\n"
                "1. Switch to PCA engine in the navbar dropdown\n"
                "2. Click '✏ Manual Line' (button turns orange)\n"
                "3. Click TWO points on the spectroscopic line, far apart\n"
                "   (zoom to 8× or 16× for precision)\n"
                "4. The app computes the midpoint between your clicks and\n"
                "   draws a line through it using Frame 1's reference slope\n"
                "5. The warp offset is recalculated automatically\n\n"
                "You can draw manual lines directly on the warped image —\n"
                "the app back-calculates the un-warped coordinates.\n\n"
                "'Clear Manual' removes the manual override for that frame."
            )),
            ("Step 7: Export", (
                "Click 'Export' to generate aligned and direct sum images.\n"
                "A comparison view shows both side-by-side so you can verify\n"
                "the alignment quality before saving."
            )),
            ("Tips", (
                "• ECC is recommended for most workflows\n"
                "• PCA threshold slider and red line overlay are only visible\n"
                "  when the PCA engine is selected\n"
                "• Each frame stores its own PCA threshold independently\n"
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
        """
        Open a file dialog to allow the user to select multiple TIFF images.

        The selected files are appended to the current file list and the
        display is updated.
        """
        # Allow mock override of file dialog
        files = tkinter.filedialog.askopenfilenames(
            title="Select TIFF files",
            filetypes=[("TIFF Files", "*.tif;*.tiff")]
        )
        if files:
            self.file_list.extend(list(files))
            self.update_listbox()

    def sort_files(self):
        """
        Sort the current list of files using a natural sorting algorithm.

        Natural sorting ensures that numbered files like 'file_2.tif' appear
        before 'file_10.tif'. Updates the display after sorting.
        """
        self.file_list = natural_sort(self.file_list)
        self.update_listbox()

    def select_item(self, idx):
        """
        Highlight and select an item in the file list.

        Args:
            idx (int): The index of the file to select.
        """
        self.selected_index = idx
        self.update_listbox()

    def move_up(self):
        """
        Move the currently selected file one position up in the list.

        Does nothing if the file is already at the top of the list or if
        no file is selected.
        """
        idx = self.selected_index
        if 0 < idx < len(self.file_list):
            self.file_list[idx], self.file_list[idx-1] = self.file_list[idx-1], self.file_list[idx]
            self.selected_index = idx - 1
            self.update_listbox()

    def move_down(self):
        """
        Move the currently selected file one position down in the list.

        Does nothing if the file is already at the bottom of the list or if
        no file is selected.
        """
        idx = self.selected_index
        if 0 <= idx < len(self.file_list) - 1:
            self.file_list[idx], self.file_list[idx+1] = self.file_list[idx+1], self.file_list[idx]
            self.selected_index = idx + 1
            self.update_listbox()

    def remove_file(self):
        """
        Remove the currently selected file from the list.

        After removal, the selection is automatically shifted to the nearest
        available item to maintain a valid selection state.
        """
        idx = self.selected_index
        if 0 <= idx < len(self.file_list):
            self.file_list.pop(idx)
            if self.file_list:
                self.selected_index = min(idx, len(self.file_list) - 1)
            else:
                self.selected_index = -1
            self.update_listbox()

    def clear_all(self):
        """
        Clear all selected files from the list and reset selection state.

        This provides a quick way to deselect all files instead of removing
        them one-by-one with the Remove button.
        """
        self.file_list.clear()
        self.selected_index = -1
        self.update_listbox()

    def start_slideshow(self):
        """
        Trigger the callback to start the slideshow with the current file list.

        Only executes if the file list is not empty and a callback is provided.
        """
        if self.on_start_slideshow and self.file_list:
            self.on_start_slideshow(self.file_list)

    def update_listbox(self):
        """
        Refresh the visual list of files displayed in the scrollable frame.

        Clears existing labels, creates new ones for the current file list,
        and applies highlighting to the selected item.
        """
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
