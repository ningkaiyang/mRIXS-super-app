"""Matplotlib canvas panel displaying 2D frame views and 1D sharpness profiles side-by-side."""

import tkinter as tk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SharpnessCanvasPanel(tk.Frame):
    """Frame containing an embedded side-by-side Matplotlib figure."""

    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller

        # Layout Matplotlib figure & canvas
        self.figure = Figure(figsize=(10, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.ax_2d = self.figure.add_subplot(121)
        self.ax_1d = self.figure.add_subplot(122)
        self.figure.tight_layout()

    def draw_plots(self, img_2d, profile_1d, stage, colormap, vmin, vmax, centroid=None, direction=None):
        """Redraws both subplots inside the canvas layout.

        Args:
            img_2d (np.ndarray): The 2D frame array (Raw, Denoised, or Masked).
            profile_1d (tuple): (P, u) representing the 1D intensity profile.
            stage (str): The active pipeline stage name.
            colormap (str): Active matplotlib colormap.
            vmin (float): Minimum display intensity limit.
            vmax (float): Maximum display intensity limit.
            centroid (np.ndarray, optional): (x, y) coordinate of line center.
            direction (np.ndarray, optional): (dx, dy) direction vector of line.
        """
        self.ax_2d.clear()
        self.ax_1d.clear()

        # Map grayscale to gray for matplotlib
        matplotlib_cmap = colormap
        if colormap == "grayscale":
            matplotlib_cmap = "gray"

        # 1. Plot 2D stage
        self.ax_2d.imshow(img_2d, cmap=matplotlib_cmap, vmin=vmin, vmax=vmax, aspect='auto')
        self.ax_2d.set_title(f"2D View: {stage}")
        self.ax_2d.axis("off")

        # Overlay centroid & peak line on Denoised and Masked views
        if stage in ("Denoised", "Masked") and centroid is not None and direction is not None:
            self.ax_2d.plot(centroid[0], centroid[1], 'ro', label="Centroid")
            dx, dy = direction
            if abs(dx) > 1e-5:
                slope = dy / dx
                self.ax_2d.axline((centroid[0], centroid[1]), slope=slope, color="red", linestyle="--")

        # Apply zoom if zoomed in
        zoom = getattr(self.controller, "zoom_factor", 1.0)
        if zoom > 1.0 and img_2d is not None:
            h, w = img_2d.shape[:2]
            cx, cy = w / 2.0, h / 2.0
            if centroid is not None:
                cx, cy = centroid[0], centroid[1]
            half_w = (w / 2.0) / zoom
            half_h = (h / 2.0) / zoom
            self.ax_2d.set_xlim(cx - half_w, cx + half_w)
            self.ax_2d.set_ylim(cy + half_h, cy - half_h)  # Keep standard image orientation (y-axis inverted)

        # 2. Plot 1D profile
        if profile_1d is not None:
            P, u = profile_1d
            self.ax_1d.plot(u, P, color="blue", linewidth=1.5)
            self.ax_1d.set_title("1D Project Profile")
            self.ax_1d.set_xlabel("Perpendicular Distance (u)")
            self.ax_1d.set_ylabel("Accumulated Intensity")
            self.ax_1d.grid(True, linestyle=":", alpha=0.6)

        self.figure.tight_layout()
        self.canvas.draw()
