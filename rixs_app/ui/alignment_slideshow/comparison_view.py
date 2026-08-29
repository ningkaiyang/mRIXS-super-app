"""Export comparison view — PySide6 port.

Replaces the former Tkinter ``ExportComparisonView`` built on CTkFrame.
Shows aligned vs. unaligned sums side-by-side via an embedded
Matplotlib figure (``FigureCanvasQTAgg``) and provides save/cancel buttons.
"""

from __future__ import annotations

import numpy as np
import tifffile

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from rixs_app.ui.widgets import SafeFigureCanvasQTAgg

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QMessageBox,
    QFrame,
)
from rixs_app.ui.theme import set_accent_btn, set_cancel_btn


class ExportComparisonView(QWidget):
    """Side-by-side comparison of aligned and direct (unaligned) sum images.

    Lazily creates the matplotlib figure each time ``load_comparison`` is
    called, so image dimensions and DPI always match the actual data.

    Args:
        parent: Parent widget (the main stacked-widget container).
        on_back: Callback invoked when the user cancels or finishes, to return
            to the slideshow.
    """

    def __init__(self, parent=None, *, on_back=None):
        """Initialise the comparison view shell without creating a figure.

        Args:
            parent: Parent QWidget.
            on_back: Callable invoked to navigate back to the slideshow.
        """
        super().__init__(parent)
        self.on_back = on_back
        self.aligned_sum: np.ndarray | None = None
        self.direct_sum: np.ndarray | None = None
        self.default_save_dir: str = ""

        self._figure: Figure | None = None
        self._mpl_canvas: FigureCanvasQTAgg | None = None
        self._toolbar: NavigationToolbar | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(6)

        # Top-right slot for Co-Pilot button
        self._top_row = QHBoxLayout()
        self._top_row.setContentsMargins(0, 0, 0, 0)
        self._top_row.addStretch()
        outer.addLayout(self._top_row)

        # Plot area placeholder
        self._plot_container = QFrame()
        self._plot_layout = QVBoxLayout(self._plot_container)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._plot_container, stretch=1)

        # Toolbar placeholder (populated per-load)
        self._toolbar_container = QFrame()
        self._toolbar_layout = QVBoxLayout(self._toolbar_container)
        self._toolbar_layout.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._toolbar_container)

        # Action bar
        action_row = QHBoxLayout()
        action_row.addStretch()

        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setFixedSize(120, 35)
        set_cancel_btn(self._cancel_button)
        self._cancel_button.clicked.connect(self._handle_cancel)
        action_row.addWidget(self._cancel_button)

        self._export_button = QPushButton("\U0001f4be Export Aligned Sum")
        self._export_button.setFixedSize(200, 35)
        set_accent_btn(self._export_button)
        self._export_button.clicked.connect(self._handle_export)
        action_row.addWidget(self._export_button)

        outer.addLayout(action_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_comparison(
        self,
        aligned_sum: np.ndarray,
        direct_sum: np.ndarray,
        default_save_dir: str,
    ) -> None:
        """Populate the view with a new pair of comparison images.

        Tears down any existing matplotlib canvas, creates a fresh figure,
        and attaches a navigation toolbar.

        Args:
            aligned_sum: Drift-corrected summed image (H×W float32).
            direct_sum: Naïve (unaligned) summed image (H×W float32).
            default_save_dir: Default directory for the save-file dialog.
        """
        self.aligned_sum = aligned_sum
        self.direct_sum = direct_sum
        self.default_save_dir = default_save_dir
        self._teardown_mpl()

        # Independent intensity scaling
        def _vmax(arr: np.ndarray) -> float:
            mn = float(np.min(arr)) if arr.size > 0 else 0.0
            active = arr[arr > max(mn, 0.0)]
            return max(1e-6, float(np.percentile(active, 60.0)) if active.size > 0 else 1.0)

        aligned_vmax = _vmax(aligned_sum)
        direct_vmax = _vmax(direct_sum)

        self._figure = Figure(figsize=(10, 5), dpi=100, facecolor='#14172b')
        ax1 = self._figure.add_subplot(121)
        ax2 = self._figure.add_subplot(122, sharex=ax1, sharey=ax1)

        ax1.set_facecolor('#14172b')
        ax2.set_facecolor('#14172b')

        ax1.imshow(direct_sum, cmap="viridis", vmin=0, vmax=direct_vmax)
        ax1.set_title("Direct Sum (Unaligned)", color='white', fontsize=12, fontweight='bold')
        ax1.axis("off")

        ax2.imshow(aligned_sum, cmap="viridis", vmin=0, vmax=aligned_vmax)
        ax2.set_title("Aligned Sum", color='white', fontsize=12, fontweight='bold')
        ax2.axis("off")

        self._figure.tight_layout()

        canvas_cls = SafeFigureCanvasQTAgg or FigureCanvasQTAgg
        self._mpl_canvas = canvas_cls(self._figure)
        self._mpl_canvas.setStyleSheet("background-color: #14172b;")
        self._plot_layout.addWidget(self._mpl_canvas)
        self._mpl_canvas.draw()

        self._toolbar = NavigationToolbar(self._mpl_canvas, self._toolbar_container)
        self._toolbar.setStyleSheet("""
            QToolBar {
                background-color: #1a1f36;
                border: 1px solid #2d3561;
                border-radius: 6px;
                padding: 4px;
            }
            QToolButton {
                background-color: #2d3558;
                border: 1px solid #3f4b78;
                border-radius: 4px;
                padding: 4px 6px;
                margin: 2px;
            }
            QToolButton:hover {
                background-color: #3d4875;
                border: 1px solid #5667a0;
            }
        """)

        # Invert black toolbar icons to crisp white for dark mode visibility
        from PySide6.QtWidgets import QToolButton
        from PySide6.QtGui import QIcon, QImage, QPixmap
        for btn in self._toolbar.findChildren(QToolButton):
            icon = btn.icon()
            if not icon.isNull():
                pix = icon.pixmap(24, 24)
                img = pix.toImage().convertToFormat(QImage.Format_ARGB32)
                for y in range(img.height()):
                    for x in range(img.width()):
                        c = img.pixelColor(x, y)
                        if c.alpha() > 0:
                            c.setRed(255 - c.red())
                            c.setGreen(255 - c.green())
                            c.setBlue(255 - c.blue())
                            img.setPixelColor(x, y, c)
                btn.setIcon(QIcon(QPixmap.fromImage(img)))

        self._toolbar_layout.addWidget(self._toolbar)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _teardown_mpl(self) -> None:
        """Destroy existing matplotlib canvas and toolbar to free C-backend resources."""
        if self._toolbar is not None:
            self._toolbar_layout.removeWidget(self._toolbar)
            self._toolbar.deleteLater()
            self._toolbar = None
        if self._mpl_canvas is not None:
            self._plot_layout.removeWidget(self._mpl_canvas)
            self._mpl_canvas.deleteLater()
            self._mpl_canvas = None
        if self._figure is not None:
            import matplotlib.pyplot as plt
            try:
                plt.close(self._figure)
            except Exception:
                pass
            self._figure = None

    def _handle_cancel(self) -> None:
        """Return to the slideshow without saving."""
        self._teardown_mpl()
        if self.on_back:
            self.on_back()

    def _handle_export(self) -> None:
        """Open a save-file dialog, write aligned TIFF, then return."""
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Aligned Sum",
            self.default_save_dir + "/aligned_sum.tif" if self.default_save_dir else "aligned_sum.tif",
            "TIFF Files (*.tif *.tiff)",
        )
        if save_path:
            try:
                tifffile.imwrite(save_path, self.aligned_sum)
                QMessageBox.information(
                    self, "Export Successful",
                    f"Aligned sum saved to:\n{save_path}"
                )
                self._teardown_mpl()
                if self.on_back:
                    self.on_back()
            except Exception as exc:
                QMessageBox.critical(
                    self, "Export Failed",
                    f"Failed to save file:\n{exc}"
                )

    # ------------------------------------------------------------------
    # Co-Pilot button integration
    # ------------------------------------------------------------------

    def set_copilot_button(self, btn) -> None:
        """Insert the Co-Pilot toggle button into the top-right slot.

        Args:
            btn: The Co-Pilot toggle QPushButton to reparent here.
        """
        self._top_row.addWidget(btn)
