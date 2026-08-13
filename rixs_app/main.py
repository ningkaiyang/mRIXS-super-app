"""Main application window — PySide6 port.

Replaces the CustomTkinter ``RixsApp`` with a ``QMainWindow``.
Views are managed via a ``QStackedWidget`` for zero-cost hidden frames.
The LLM Agent Sidebar ("mRIXS Co-Pilot") is integrated as a collapsible
right pane via ``QSplitter``.
"""

from __future__ import annotations

import os
import sys
import platform

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget, QSplitter, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
)

from rixs_app.ui.theme import FULL_QSS, PALETTE, set_tool_btn


from rixs_app.ui.sorting_view import SortingView
from rixs_app.ui.alignment_slideshow.slideshow_view import SlideshowView
from rixs_app.ui.alignment_slideshow.comparison_view import ExportComparisonView
from rixs_app.ui.zeroth_order_slideshow.slideshow_view import ZerothOrderSlideshowView


# View indices in the QStackedWidget
_IDX_SORTING = 0
_IDX_SLIDESHOW = 1
_IDX_COMPARISON = 2
_IDX_ZEROTH_ORDER = 3

# View name map for GUI context
_VIEW_NAMES = {
    _IDX_SORTING: "SortingView",
    _IDX_SLIDESHOW: "SlideshowView",
    _IDX_COMPARISON: "ExportComparisonView",
    _IDX_ZEROTH_ORDER: "ZerothOrderSlideshowView",
}


class RixsApp(QMainWindow):
    """Main application window for the mRIXS Super-App.

    Manages navigation between the four primary views and hosts the
    collapsible LLM Agent Sidebar (mRIXS Co-Pilot).

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

        # ------------------------------------------------------------------
        # Central layout: QSplitter (views left | agent sidebar right)
        # ------------------------------------------------------------------
        self._splitter = QSplitter(Qt.Horizontal, self)
        self.setCentralWidget(self._splitter)

        # Left side: stack of views + toggle button overlay
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # Toggle button bar at top-right
        toggle_bar = QHBoxLayout()
        toggle_bar.setContentsMargins(0, 4, 8, 0)
        toggle_bar.addStretch()
        self._sidebar_toggle = QPushButton("🤖 Co-Pilot")
        self._sidebar_toggle.setFixedHeight(28)
        self._sidebar_toggle.setToolTip("Toggle mRIXS Co-Pilot sidebar")
        set_tool_btn(self._sidebar_toggle)
        self._sidebar_toggle.clicked.connect(self._toggle_sidebar)
        toggle_bar.addWidget(self._sidebar_toggle)
        left_layout.addLayout(toggle_bar)

        # Stacked container
        self._stack = QStackedWidget()
        left_layout.addWidget(self._stack)

        self._splitter.addWidget(left_container)

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

        # ------------------------------------------------------------------
        # Agent sidebar (right side, initially collapsed)
        # ------------------------------------------------------------------
        self._sidebar = None          # Lazy-initialised on first toggle
        self._bridge = None
        self._engine = None
        self._sidebar_cached_width = 380  # Default sidebar width
        self._sidebar_visible = False
        self._models_loaded = False

        self.show_sorting()

        if show_window:
            self._maximize_window()
            self.show()

    # ------------------------------------------------------------------
    # Agent sidebar management
    # ------------------------------------------------------------------

    def _init_sidebar(self) -> None:
        """Lazily initialise the agent sidebar, engine, and bridge.

        Called on first sidebar toggle. Checks for API key availability
        and shows the setup wizard if needed.
        """
        from rixs_app.agent.auth import resolve_api_key, fetch_model_list, CBORG_DEFAULT_MODEL
        from rixs_app.agent.tools import create_default_registry
        from rixs_app.agent.engine import CborgAgentEngine
        from rixs_app.agent.bridge import GuiAgentBridge
        from rixs_app.ui.agent_sidebar.sidebar_widget import AgentSidebarWidget

        # Resolve API key (may trigger setup wizard)
        api_key = resolve_api_key()
        if not api_key:
            from rixs_app.ui.agent_sidebar.setup_wizard import CBORGSetupWizard
            wizard = CBORGSetupWizard(self)
            if wizard.exec():
                api_key = wizard.api_key
            if not api_key:
                return  # User cancelled — don't create sidebar

        # Create agent infrastructure
        registry = create_default_registry()
        registry.set_gui_context(self)

        self._engine = CborgAgentEngine(
            api_key=api_key,
            registry=registry,
        )
        self._bridge = GuiAgentBridge(self._engine, parent=self)
        self._bridge.start_worker()

        # Create sidebar widget
        self._sidebar = AgentSidebarWidget(
            bridge=self._bridge,
            main_window_ref=self,
            parent=None,
        )
        self._sidebar.sidebar_close_requested.connect(self._hide_sidebar)

        # Override get_minimal_gui_context to provide real context
        self._sidebar.get_minimal_gui_context = self._get_gui_context

        # Add to splitter
        self._splitter.addWidget(self._sidebar)

        # Populate model list in background
        self._load_models_async(api_key)

    def _load_models_async(self, api_key: str) -> None:
        """Load available models in a background thread and update UI safely on main thread."""
        from PySide6.QtCore import QRunnable, QThreadPool, QObject, Signal

        class _ModelSignals(QObject):
            finished = Signal(list)

        class _ModelLoader(QRunnable):
            def __init__(self, key: str, signals: _ModelSignals):
                super().__init__()
                self._key = key
                self._signals = signals

            def run(self):
                from rixs_app.agent.auth import fetch_model_list
                models = fetch_model_list(self._key)
                try:
                    self._signals.finished.emit(models)
                except (RuntimeError, AttributeError):
                    pass

        self._model_signals = _ModelSignals(self)

        def _on_models_loaded(models: list):
            try:
                if self._sidebar is not None and hasattr(self._sidebar, 'populate_models'):
                    self._sidebar.populate_models(models)
                    from rixs_app.agent.auth import CBORG_DEFAULT_MODEL
                    idx = models.index(CBORG_DEFAULT_MODEL) if CBORG_DEFAULT_MODEL in models else 0
                    self._sidebar.model_combo.setCurrentIndex(idx)
                    self._models_loaded = True
            except Exception:
                pass

        self._model_signals.finished.connect(_on_models_loaded)
        loader = _ModelLoader(api_key, self._model_signals)
        QThreadPool.globalInstance().start(loader)

    def _toggle_sidebar(self) -> None:
        """Toggle the agent sidebar visibility."""
        if self._sidebar is None:
            self._init_sidebar()
            if self._sidebar is None:
                return  # Setup was cancelled
            self._show_sidebar()
        elif self._sidebar_visible:
            self._hide_sidebar()
        else:
            self._show_sidebar()

    def _show_sidebar(self) -> None:
        """Show the agent sidebar, restoring cached width."""
        if self._sidebar is None:
            return
        self._sidebar.show()
        total_width = self._splitter.width()
        sidebar_w = min(self._sidebar_cached_width, total_width // 3)
        self._splitter.setSizes([total_width - sidebar_w, sidebar_w])
        self._sidebar_visible = True
        self._sidebar_toggle.setText("🤖 ✕")
        self._sidebar_toggle.setToolTip("Close Co-Pilot sidebar")

    def _hide_sidebar(self) -> None:
        """Collapse the sidebar, caching its current width."""
        if self._sidebar is None:
            return
        sizes = self._splitter.sizes()
        if len(sizes) > 1 and sizes[1] > 50:
            self._sidebar_cached_width = sizes[1]
        self._sidebar.hide()
        self._sidebar_visible = False
        self._sidebar_toggle.setText("🤖 Co-Pilot")
        self._sidebar_toggle.setToolTip("Toggle mRIXS Co-Pilot sidebar")

    def _get_gui_context(self) -> str:
        """Build a compact GUI context string for the agent.

        Returns:
            A one-line string describing the current app state.
        """
        current_idx = self._stack.currentIndex()
        view_name = _VIEW_NAMES.get(current_idx, "Unknown")
        parts = [f"View={view_name}"]

        try:
            if current_idx == _IDX_SORTING:
                sv = self.sorting_view
                n_files = len(sv.file_list) if hasattr(sv, 'file_list') else 0
                parts.append(f"Files={n_files}")
                if n_files > 0:
                    import os
                    parts.append(f"Dir={os.path.dirname(sv.file_list[0])}")

            elif current_idx == _IDX_SLIDESHOW:
                sv = self.slideshow_view
                if hasattr(sv, '_manager') and sv._manager:
                    mgr = sv._manager
                    parts.append(f"Files={getattr(mgr, '_n_frames', '?')}")
                    parts.append(f"Frame={getattr(mgr, '_current_idx', '?')}")

            elif current_idx == _IDX_ZEROTH_ORDER:
                zv = self.zeroth_order_view
                if hasattr(zv, '_manager') and zv._manager:
                    mgr = zv._manager
                    parts.append(f"Files={getattr(mgr, '_n_frames', '?')}")
                    parts.append(f"Frame={getattr(mgr, '_current_idx', '?')}")
                    if getattr(mgr, '_txt_path', None):
                        parts.append("ScanLog=yes")
        except Exception:
            pass

        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Global Event Filter for Keyboard Navigation
    # ------------------------------------------------------------------



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
        """Clean up Matplotlib canvases, timers, and agent threads on window close.

        Args:
            event: The ``QCloseEvent``.
        """
        # Teardown Matplotlib canvases
        for view_name in ('zeroth_order_view', 'slideshow_view', 'export_comparison_view'):
            view = getattr(self, view_name, None)
            if view is not None and hasattr(view, '_teardown_mpl'):
                try:
                    view._teardown_mpl()
                except Exception:
                    pass

        # Shutdown agent bridge thread
        if self._bridge is not None:
            try:
                self._bridge.stop_worker()
            except Exception:
                pass

        event.accept()




def main() -> None:
    """Application entry point."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = RixsApp(show_window=True)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
