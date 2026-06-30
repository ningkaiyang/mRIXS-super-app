# rixs_app/ui/slideshow/comparison_view.py

import tkinter as tk
import customtkinter as ctk
import numpy as np
import tifffile

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class ExportComparisonView(ctk.CTkFrame):
    """In-app view for comparing aligned vs. unaligned sums before exporting.

    This view replaces the former standalone ExportComparisonDialog (CTkToplevel)
    and is swapped into the main window grid alongside SortingView and
    SlideshowView.  Each time the user triggers an export, ``load_comparison``
    recreates the matplotlib figure so image dimensions and scaling are always
    correct.

    Layout (top-to-bottom):
        1. Side-by-side matplotlib plots (Direct Sum | Aligned Sum).
        2. Matplotlib navigation toolbar.
        3. Action bar — "Cancel" and "💾 Export Aligned Sum" buttons, right-aligned.

    Args:
        parent: The parent widget (typically ``RixsApp.container``).
        on_back: Callback invoked when the user cancels or finishes exporting,
            to return to the slideshow view.
        **kwargs: Extra keyword arguments forwarded to ``CTkFrame``.
    """

    def __init__(self, parent, on_back=None, **kwargs):
        """Initialize the ExportComparisonView shell.

        The matplotlib figure is *not* created here — it is built lazily
        inside ``load_comparison`` so that each invocation gets a fresh
        canvas sized to the actual image data.

        Args:
            parent: The parent widget.
            on_back (callable, optional): Callback to return to the slideshow.
            **kwargs: Additional keyword arguments for CTkFrame.
        """
        super().__init__(parent, **kwargs)
        self.on_back = on_back

        # Populated per-load
        self.aligned_sum = None
        self.direct_sum = None
        self.default_save_dir = None

        # Matplotlib widgets (created/destroyed per load)
        self._figure = None
        self._mpl_canvas = None
        self._toolbar = None

        # Persistent container frames
        self._plot_frame = ctk.CTkFrame(self)
        self._plot_frame.pack(fill="both", expand=True, padx=10, pady=(10, 0))

        self._toolbar_frame = ctk.CTkFrame(self, height=40)
        self._toolbar_frame.pack(fill="x", padx=10, pady=(5, 0))

        self._action_frame = ctk.CTkFrame(self)
        self._action_frame.pack(fill="x", padx=10, pady=10)

        self._export_button = ctk.CTkButton(
            self._action_frame, text="💾 Export Aligned Sum",
            command=self._handle_export,
            width=200, height=35,
            fg_color="#2F72A5", hover_color="#1F5A85"
        )
        # Pack export button first with side="right" so it takes the absolute right edge.
        self._export_button.pack(side="right", padx=(5, 10))

        self._cancel_button = ctk.CTkButton(
            self._action_frame, text="Cancel", command=self._handle_cancel,
            fg_color="#888", hover_color="#666", width=120, height=35
        )
        # Pack cancel button second with side="right" so it sits to the left of export.
        self._cancel_button.pack(side="right", padx=5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_comparison(self, aligned_sum, direct_sum, default_save_dir):
        """Populate the view with a new pair of comparison images.

        Tears down any existing matplotlib widgets, creates a new Figure with
        side-by-side subplots, draws the images with independent per-image
        scaling (60th-percentile of active pixels), and attaches a navigation
        toolbar.

        Args:
            aligned_sum (np.ndarray): The drift-corrected summed image.
            direct_sum (np.ndarray): The naïve (unaligned) summed image.
            default_save_dir (str): Default directory for the save-file dialog.
        """
        self.aligned_sum = aligned_sum
        self.direct_sum = direct_sum
        self.default_save_dir = default_save_dir

        # Tear down previous matplotlib widgets if any
        self._teardown_mpl()

        # Independent intensity scaling ----------------------------------
        aligned_min = float(np.min(aligned_sum)) if aligned_sum.size > 0 else 0.0
        aligned_active = aligned_sum[aligned_sum > max(aligned_min, 0.0)]
        if aligned_active.size > 0:
            aligned_vmax = float(np.percentile(aligned_active, 60.0))
        else:
            aligned_vmax = 1.0
        aligned_vmax = max(1e-6, aligned_vmax)

        direct_min = float(np.min(direct_sum)) if direct_sum.size > 0 else 0.0
        direct_active = direct_sum[direct_sum > max(direct_min, 0.0)]
        if direct_active.size > 0:
            direct_vmax = float(np.percentile(direct_active, 60.0))
        else:
            direct_vmax = 1.0
        direct_vmax = max(1e-6, direct_vmax)

        # Build matplotlib figure ----------------------------------------
        self._figure = Figure(figsize=(10, 5), dpi=100)
        ax1 = self._figure.add_subplot(121)
        ax2 = self._figure.add_subplot(122, sharex=ax1, sharey=ax1)

        ax1.imshow(direct_sum, cmap="viridis", vmin=0, vmax=direct_vmax)
        ax1.set_title("Direct Sum (Unaligned)")
        ax1.axis("off")

        ax2.imshow(aligned_sum, cmap="viridis", vmin=0, vmax=aligned_vmax)
        ax2.set_title("Aligned Sum")
        ax2.axis("off")

        self._figure.tight_layout()

        # Embed canvas ---------------------------------------------------
        self._mpl_canvas = FigureCanvasTkAgg(self._figure, master=self._plot_frame)
        self._mpl_canvas.draw()
        self._mpl_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Toolbar ---------------------------------------------------------
        self._toolbar = NavigationToolbar2Tk(self._mpl_canvas, self._toolbar_frame)
        self._toolbar.update()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _teardown_mpl(self):
        """Destroy existing matplotlib canvas and toolbar widgets.

        Called before each ``load_comparison`` to prevent stale widgets from
        accumulating.
        """
        if self._toolbar is not None:
            self._toolbar.destroy()
            self._toolbar = None
        if self._mpl_canvas is not None:
            self._mpl_canvas.get_tk_widget().destroy()
            self._mpl_canvas = None
        if self._figure is not None:
            import matplotlib.pyplot as plt
            plt.close(self._figure)
            self._figure = None

    def _handle_cancel(self):
        """Return to the slideshow view without saving.

        Tears down matplotlib resources before navigating back.
        """
        self._teardown_mpl()
        if self.on_back:
            self.on_back()

    def _handle_export(self):
        """Open a save-file dialog, write the aligned sum TIFF, and return.

        On success an info dialog is shown and the view navigates back to the
        slideshow.  On failure an error dialog is shown and the user remains
        on the comparison view to retry or cancel.
        """
        save_path = tk.filedialog.asksaveasfilename(
            initialdir=self.default_save_dir,
            initialfile="aligned_sum.tif",
            defaultextension=".tif",
            filetypes=[("TIFF Files", "*.tif;*.tiff")],
            parent=self.winfo_toplevel()
        )
        if save_path:
            try:
                tifffile.imwrite(save_path, self.aligned_sum)
                tk.messagebox.showinfo(
                    "Export Successful",
                    f"Aligned sum saved to:\n{save_path}",
                    parent=self.winfo_toplevel()
                )
                self._teardown_mpl()
                if self.on_back:
                    self.on_back()
            except Exception as e:
                tk.messagebox.showerror(
                    "Export Failed",
                    f"Failed to save file:\n{e}",
                    parent=self.winfo_toplevel()
                )
