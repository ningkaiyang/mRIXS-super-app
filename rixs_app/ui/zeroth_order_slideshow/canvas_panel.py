"""Matplotlib canvas panel displaying 2D frame views and 1D FWHM profiles side-by-side."""

import tkinter as tk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ZerothOrderCanvasPanel(tk.Frame):
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

        self.canvas.mpl_connect("button_press_event", self.on_click)

    def on_click(self, event):
        if event.inaxes == self.ax_2d:
            if event.xdata is not None and event.ydata is not None:
                if hasattr(self.controller, 'handle_canvas_click'):
                    self.controller.handle_canvas_click(event.xdata, event.ydata)

    def draw_plots(self, img_2d, profile_1d, stage, colormap, vmin, vmax,
                   centroid=None, direction=None,
                   fit_ok=False, candidates_xy=None, inliers_xy=None,
                   segment_endpoints=None, detected_support_y_range=None,
                   show_support_points=False, show_extrapolation=False,
                   evaluator_result=None, show_fitted_line=True):
        """Redraws both subplots inside the canvas layout."""
        self.ax_2d.clear()
        self.ax_1d.clear()

        # Map grayscale to gray for matplotlib
        matplotlib_cmap = colormap
        if colormap == "grayscale":
            matplotlib_cmap = "gray"

        # 1. Plot 2D stage
        if img_2d is not None:
            # Slicing: points below vmin are mapped to vmin (cutting background noise); points above vmin scale up to vmax
            img_display = np.where(img_2d >= vmin, np.minimum(img_2d, vmax), vmin)
        else:
            img_display = None
        self.ax_2d.imshow(img_display, cmap=matplotlib_cmap, vmin=vmin, vmax=vmax, aspect='auto')
        self.ax_2d.set_title(f"2D View: {stage}")
        self.ax_2d.axis("off")

        # Overlay elements if fit_ok is True
        if not fit_ok:
            self.ax_2d.text(0.5, 0.5, "No valid line fit",
                            color="red", fontsize=16, weight="bold",
                            ha="center", va="center", transform=self.ax_2d.transAxes)
        else:
            if centroid is not None:
                self.ax_2d.plot(centroid[0], centroid[1], 'w+', markersize=12, label="Centroid")

            if show_fitted_line:
                if segment_endpoints is not None:
                    (x1, y1), (x2, y2) = segment_endpoints
                    self.ax_2d.plot([x1, x2], [y1, y2], color="red", linestyle="-", linewidth=2)
                elif centroid is not None and direction is not None:
                    # Fallback to axline but only within detected_support_y_range
                    dx, dy = direction
                    if abs(dx) > 1e-5 and detected_support_y_range is not None:
                        slope = dy / dx
                        y_min, y_max = detected_support_y_range
                        x_min = centroid[0] + (y_min - centroid[1]) / slope
                        x_max = centroid[0] + (y_max - centroid[1]) / slope
                        self.ax_2d.plot([x_min, x_max], [y_min, y_max], color="red", linestyle="-", linewidth=2)

            if show_support_points:
                if candidates_xy is not None and len(candidates_xy) > 0:
                    self.ax_2d.scatter(candidates_xy[:,0], candidates_xy[:,1], c='yellow', s=4, alpha=0.5, zorder=3)
                if inliers_xy is not None and len(inliers_xy) > 0:
                    self.ax_2d.scatter(inliers_xy[:,0], inliers_xy[:,1], c='lime', s=8, alpha=0.8, zorder=4)

            if show_extrapolation and centroid is not None and direction is not None:
                dx, dy = direction
                if abs(dx) > 1e-5:
                    slope = dy / dx
                    self.ax_2d.axline((centroid[0], centroid[1]), slope=slope, color="red", linestyle="--", linewidth=1, alpha=0.7)

            # Compact status text
            dx, dy = (direction if direction is not None else (1, 0))
            angle_deg = np.degrees(np.arctan2(dy, dx)) if direction is not None else 0.0
            n_cand = len(candidates_xy) if candidates_xy is not None else 0
            n_inl = len(inliers_xy) if inliers_xy is not None else 0
            status_text = f"∠{angle_deg:.1f}° {n_cand}/{n_inl}"

            self.ax_2d.text(0.95, 0.95, status_text,
                            color="white", fontsize=10, weight="bold",
                            ha="right", va="top", transform=self.ax_2d.transAxes,
                            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))

        # Apply zoom if zoomed in
        zoom = getattr(self.controller, "zoom_factor", 1.0)
        zoom_center = getattr(self.controller, "zoom_center", None)
        if zoom > 1.0 and img_2d is not None:
            h, w = img_2d.shape[:2]
            cx, cy = w / 2.0, h / 2.0
            if zoom_center is not None:
                cx, cy = zoom_center
            half_w = (w / 2.0) / zoom
            half_h = (h / 2.0) / zoom
            self.ax_2d.set_xlim(cx - half_w, cx + half_w)
            self.ax_2d.set_ylim(cy + half_h, cy - half_h)  # Keep standard image orientation (y-axis inverted)

        # 2. Plot 1D profile
        if evaluator_result is not None and evaluator_result.score_valid:
            # Plot evaluator's intensity profile
            self.ax_1d.plot(evaluator_result.profile_u, evaluator_result.intensity_profile, 'b-', linewidth=1.5, label='Intensity')

            # Plot Gaussian fit
            u_fine = np.linspace(evaluator_result.profile_u[0], evaluator_result.profile_u[-1], 200)
            from rixs_app.core.zeroth_order_evaluator import _gaussian
            fit_y = _gaussian(
                u_fine, evaluator_result.gaussian_background, evaluator_result.gaussian_amplitude,
                evaluator_result.gaussian_center, evaluator_result.gaussian_sigma
            )
            self.ax_1d.plot(u_fine, fit_y, 'r--', linewidth=1.5, label='Gaussian Fit')

            # Add FWHM annotation
            fwhm = evaluator_result.fwhm_px
            score = evaluator_result.score
            self.ax_1d.text(0.05, 0.95, f"FWHM: {fwhm:.2f} px\nScore: {score:.4f}",
                            transform=self.ax_1d.transAxes, va='top', ha='left',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            self.ax_1d.legend(loc='upper right')

            # FWHM shaded bracket overlay
            u0 = evaluator_result.gaussian_center
            bg = evaluator_result.gaussian_background
            amp = evaluator_result.gaussian_amplitude
            half_fwhm = fwhm / 2
            half_max_y = bg + amp / 2
            self.ax_1d.axvspan(u0 - half_fwhm, u0 + half_fwhm, alpha=0.15, color='red')
            self.ax_1d.hlines(half_max_y, u0 - half_fwhm, u0 + half_fwhm,
                              colors='red', linestyles='--', linewidth=1.5)

            self.ax_1d.set_title("1D Intensity Profile (Fitted)")
            self.ax_1d.set_xlabel("Perpendicular Distance (u)")
            self.ax_1d.set_ylabel("Intensity")
            self.ax_1d.grid(True, linestyle=":", alpha=0.6)

        elif profile_1d is not None:
            P, u = profile_1d
            self.ax_1d.plot(u, P, color="blue", linewidth=1.5)
            self.ax_1d.set_title("1D Project Profile")
            self.ax_1d.set_xlabel("Perpendicular Distance (u)")
            self.ax_1d.set_ylabel("Accumulated Intensity")
            self.ax_1d.grid(True, linestyle=":", alpha=0.6)

            if evaluator_result is not None and not evaluator_result.score_valid:
                reason = evaluator_result.failure_reason or "Unknown reason"
                self.ax_1d.text(0.5, 0.5, f"Fit failed:\n{reason}", color="red",
                                transform=self.ax_1d.transAxes, va='center', ha='center',
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        self.figure.tight_layout()
        self.canvas.draw()
