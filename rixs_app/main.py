"""Main application window — PySide6 port.

Replaces the CustomTkinter ``RixsApp`` with a ``QMainWindow``.
Views are managed via a ``QStackedWidget`` for zero-cost hidden frames.
"""

from __future__ import annotations

import sys
import platform

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from rixs_app.ui.theme import FULL_QSS
from rixs_app.ui.sorting_view import SortingView
from rixs_app.ui.alignment_slideshow.slideshow_view import SlideshowView
from rixs_app.ui.alignment_slideshow.comparison_view import ExportComparisonView
from rixs_app.ui.zeroth_order_slideshow.slideshow_view import ZerothOrderSlideshowView


# View indices in the QStackedWidget
_IDX_SORTING = 0
_IDX_SLIDESHOW = 1
_IDX_COMPARISON = 2
_IDX_ZEROTH_ORDER = 3


class RixsApp(QMainWindow):
    """Main application window for the mRIXS Super-App.

    Manages navigation between the four primary views:
    - SortingView
    - SlideshowView (alignment)
    - ExportComparisonView
    - ZerothOrderSlideshowView

    Args:
        show_window: Whether to show the window after construction. Set
            False during headless testing.
    """

    def __init__(self, show_window: bool = True):
        """Initialise the main window and all sub-views.

        Args:
            show_window: Whether to call show() at the end of __init__.
        """
        super().__init__()
        self.setWindowTitle("mRIXS Super-App — Advanced X-ray Spectroscopy Suite")
        self.resize(1200, 800)

        # Apply global dark theme
        self.setStyleSheet(FULL_QSS)

        # Stacked container
        self._stack = QStackedWidget(self)
        self.setCentralWidget(self._stack)

        # Build views
        self.sorting_view = SortingView(
            on_start_slideshow=self.show_slideshow,
            on_zeroth_order=self.show_zeroth_order_calibration,
        )
        self.slideshow_view = SlideshowView(
            on_back_to_sorting=self.show_sorting,
            on_show_export_comparison=self.show_export_comparison,
        )
        self.export_comparison_view = ExportComparisonView(
            on_back=self.show_slideshow_from_comparison,
        )
        self.zeroth_order_view = ZerothOrderSlideshowView(
            on_back_to_sorting=self.show_sorting,
        )

        self._stack.insertWidget(_IDX_SORTING, self.sorting_view)
        self._stack.insertWidget(_IDX_SLIDESHOW, self.slideshow_view)
        self._stack.insertWidget(_IDX_COMPARISON, self.export_comparison_view)
        self._stack.insertWidget(_IDX_ZEROTH_ORDER, self.zeroth_order_view)

        self.show_sorting()

        if show_window:
            self._maximize_window()
            self.show()

        app = QApplication.instance()
        if app:
            app.installEventFilter(self)

    # ------------------------------------------------------------------
    # Global Event Filter for Keyboard Navigation
    # ------------------------------------------------------------------

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        """Intercept application-wide events for focus clearing and frame navigation.

        Clears focus from QLineEdit text inputs whenever the user clicks outside them
        or presses Return/Escape, ensuring Left/Right arrow keys navigate frames reliably.
        """
        from PySide6.QtCore import QEvent
        from PySide6.QtWidgets import QLineEdit, QTextEdit

        # Clear text box focus on mouse click outside the focused text input
        if event.type() == QEvent.MouseButtonPress:
            focused = self.focusWidget()
            if isinstance(focused, (QLineEdit, QTextEdit)):
                from PySide6.QtWidgets import QWidget
                if isinstance(watched, QWidget) and watched is not focused and not focused.isAncestorOf(watched):
                    focused.clearFocus()

        elif event.type() == QEvent.KeyPress:
            key = event.key()
            focused = self.focusWidget()

            if isinstance(focused, (QLineEdit, QTextEdit)):
                if key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape):
                    focused.clearFocus()
                return super().eventFilter(watched, event)

            if key in (Qt.Key_Left, Qt.Key_Right):
                current = self._stack.currentIndex()
                if key == Qt.Key_Left:
                    if current == _IDX_SLIDESHOW:
                        self.slideshow_view.prev_frame()
                        return True
                    elif current == _IDX_ZEROTH_ORDER:
                        self.zeroth_order_view.prev_frame()
                        return True
                elif key == Qt.Key_Right:
                    if current == _IDX_SLIDESHOW:
                        self.slideshow_view.next_frame()
                        return True
                    elif current == _IDX_ZEROTH_ORDER:
                        self.zeroth_order_view.next_frame()
                        return True

        return super().eventFilter(watched, event)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def show_sorting(self) -> None:
        """Display the file sorting view."""
        self._stack.setCurrentIndex(_IDX_SORTING)
        self.sorting_view.update_listbox()

    def show_slideshow(self, file_list: list[str]) -> None:
        """Display the alignment slideshow and start with the provided files.

        Args:
            file_list: Absolute paths to TIFF images.
        """
        self._stack.setCurrentIndex(_IDX_SLIDESHOW)
        self.slideshow_view.start(file_list)

    def show_zeroth_order_calibration(
        self, file_list: list[str], txt_path: str | None = None
    ) -> None:
        """Display the zeroth-order calibration view.

        Args:
            file_list: Absolute paths to TIFF images.
            txt_path: Optional path to a scan log TXT file.
        """
        self._stack.setCurrentIndex(_IDX_ZEROTH_ORDER)
        self.zeroth_order_view.start(file_list, txt_path=txt_path)

    def show_export_comparison(
        self, aligned_sum, direct_sum, initial_dir: str
    ) -> None:
        """Transition to the export comparison view.

        Args:
            aligned_sum: Drift-corrected summed image (H×W float32).
            direct_sum: Naïve unaligned summed image (H×W float32).
            initial_dir: Default directory for the save-file dialog.
        """
        self.export_comparison_view.load_comparison(aligned_sum, direct_sum, initial_dir)
        self._stack.setCurrentIndex(_IDX_COMPARISON)

    def show_slideshow_from_comparison(self) -> None:
        """Return from the comparison view back to the slideshow."""
        self._stack.setCurrentIndex(_IDX_SLIDESHOW)
        focused = self.focusWidget()
        if focused:
            focused.clearFocus()
        self.setFocus()

    # ------------------------------------------------------------------
    # Keyboard navigation
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """Handle global left/right arrow key presses for frame navigation.

        Skips interception when focus is on a text input, combo box, or
        slider — those widgets need arrow keys for their own cursor /
        selection navigation.

        Args:
            event: The ``QKeyEvent``.
        """
        current = self._stack.currentIndex()
        key = event.key()

        # Don't swallow events when an interactive input widget has focus
        focused = self.focusWidget()
        from PySide6.QtWidgets import QSlider, QLineEdit, QComboBox
        if isinstance(focused, (QSlider, QLineEdit, QComboBox)):
            super().keyPressEvent(event)
            return

        if key == Qt.Key_Left:
            if current == _IDX_SLIDESHOW:
                self.slideshow_view.prev_frame()
            elif current == _IDX_ZEROTH_ORDER:
                self.zeroth_order_view.prev_frame()
        elif key == Qt.Key_Right:
            if current == _IDX_SLIDESHOW:
                self.slideshow_view.next_frame()
            elif current == _IDX_ZEROTH_ORDER:
                self.zeroth_order_view.next_frame()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------

    def _maximize_window(self) -> None:
        """Maximise the window in a platform-aware manner."""
        os_name = platform.system()
        try:
            if os_name in ("Windows", "Darwin"):
                self.showMaximized()
            elif os_name == "Linux":
                self.showMaximized()
            else:
                self.showMaximized()
        except Exception:
            self.showMaximized()

    def closeEvent(self, event) -> None:  # noqa: N802
        """Clean up Matplotlib canvases and timers on window close.

        Args:
            event: The ``QCloseEvent``.
        """
        for view_name in ('zeroth_order_view', 'slideshow_view', 'export_comparison_view'):
            view = getattr(self, view_name, None)
            if view is not None and hasattr(view, '_teardown_mpl'):
                try:
                    view._teardown_mpl()
                except Exception:
                    pass
        event.accept()


# Backward-compatibility alias
MainApplication = RixsApp


def main() -> None:
    """Application entry point."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = RixsApp(show_window=True)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
