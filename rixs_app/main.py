"""Main application window — PySide6 port.

Replaces the CustomTkinter ``RixsApp`` with a ``QMainWindow``.
Views are managed via a ``QStackedWidget`` for zero-cost hidden frames.
The LLM Agent Sidebar ("RIXS Co-Pilot") is integrated as a collapsible
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

from rixs_app.ui.theme import FULL_QSS, PALETTE, set_tool_btn, set_copilot_btn


from rixs_app.ui.home_launchpad import HomeLaunchpadView
from rixs_app.ui.dark_masking.dark_mask_view import DarkMaskingView, DarkCalibrationView
from rixs_app.ui.clustering_slideshow.file_selection_view import ClusteringFileSelectionView
from rixs_app.ui.clustering_slideshow.studio_view import ClusteringStudioView
from rixs_app.ui.sorting_view import SortingView
from rixs_app.ui.alignment_slideshow.slideshow_view import SlideshowView
from rixs_app.ui.alignment_slideshow.comparison_view import ExportComparisonView
from rixs_app.ui.zeroth_order_slideshow.slideshow_view import ZerothOrderSlideshowView


# View indices in the QStackedWidget
_IDX_HOME = 0
_IDX_DARK_CAL = 1
_IDX_CLUSTERING_FILES = 2
_IDX_CLUSTERING_STUDIO = 3
_IDX_SORTING = 4
_IDX_SLIDESHOW = 5
_IDX_COMPARISON = 6
_IDX_ZEROTH_ORDER = 7

# View name map for GUI context
_VIEW_NAMES = {
    _IDX_HOME: "HomeLaunchpadView",
    _IDX_DARK_CAL: "DarkMaskingView",
    _IDX_CLUSTERING_FILES: "ClusteringFileSelectionView",
    _IDX_CLUSTERING_STUDIO: "ClusteringStudioView",
    _IDX_SORTING: "SortingView",
    _IDX_SLIDESHOW: "SlideshowView",
    _IDX_COMPARISON: "ExportComparisonView",
    _IDX_ZEROTH_ORDER: "ZerothOrderSlideshowView",
}


class RixsApp(QMainWindow):
    """Main application window for the RIXS Super-App.

    Manages navigation between the 8 primary views and hosts the
    collapsible LLM Agent Sidebar (RIXS Co-Pilot).

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
        self.setWindowTitle("RIXS Super-App — Advanced X-ray Spectroscopy Suite")
        self.resize(1200, 800)

        # Apply global dark theme
        self.setStyleSheet(FULL_QSS)

        # ------------------------------------------------------------------
        # Central layout: QSplitter (views left | agent sidebar right)
        # ------------------------------------------------------------------
        self._splitter = QSplitter(Qt.Horizontal, self)
        self.setCentralWidget(self._splitter)

        # Left side: stacked views (no wrapper — Co-Pilot btn lives in each navbar)
        self._stack = QStackedWidget()
        self._splitter.addWidget(self._stack)

        # Build views
        self.home_view = HomeLaunchpadView(
            on_dark_calibration=self.show_dark_calibration,
            on_clustering=self.show_clustering_files,
            on_alignment=self.show_sorting,
            on_zeroth_order=self.show_sorting,
        )
        self.home_launchpad_view = self.home_view

        self.dark_mask_view = DarkMaskingView(
            on_back=self.show_home,
        )
        self.dark_cal_view = self.dark_mask_view
        self.dark_calibration_view = self.dark_mask_view
        self.dark_masking_view = self.dark_mask_view

        self.clustering_file_view = ClusteringFileSelectionView(
            on_back=self.show_home,
            on_launch_studio=self.show_clustering_studio,
            on_navigate_dark_cal=self.show_dark_calibration,
        )
        self.clustering_file_selection_view = self.clustering_file_view

        self.clustering_studio_view = ClusteringStudioView(
            on_back=self.show_home,
        )

        self.sorting_view = SortingView(
            on_back=self.show_home,
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

        self._stack.insertWidget(_IDX_HOME, self.home_view)
        self._stack.insertWidget(_IDX_DARK_CAL, self.dark_cal_view)
        self._stack.insertWidget(_IDX_CLUSTERING_FILES, self.clustering_file_view)
        self._stack.insertWidget(_IDX_CLUSTERING_STUDIO, self.clustering_studio_view)
        self._stack.insertWidget(_IDX_SORTING, self.sorting_view)
        self._stack.insertWidget(_IDX_SLIDESHOW, self.slideshow_view)
        self._stack.insertWidget(_IDX_COMPARISON, self.export_comparison_view)
        self._stack.insertWidget(_IDX_ZEROTH_ORDER, self.zeroth_order_view)

        # Co-Pilot toggle button (reparented into each view's navbar on switch)
        self._sidebar_toggle = QPushButton("🤖 Co-Pilot")
        self._sidebar_toggle.setFixedHeight(28)
        self._sidebar_toggle.setToolTip("Toggle RIXS Co-Pilot sidebar")
        set_copilot_btn(self._sidebar_toggle)
        self._sidebar_toggle.clicked.connect(self._toggle_sidebar)
        self._stack.currentChanged.connect(self._reparent_toggle_btn)

        # ------------------------------------------------------------------
        # Agent sidebar (right side, initially collapsed)
        # ------------------------------------------------------------------
        self._sidebar = None          # Lazy-initialised on first toggle
        self._bridge = None
        self._engine = None
        self._sidebar_cached_width = 380  # Default sidebar width
        self._sidebar_visible = False
        self._models_loaded = False

        # Pre-warm Chromium so the sidebar opens instantly on first toggle.
        # No setVisible(False) — Chromium defers page loading for hidden widgets.
        from rixs_app.ui.agent_sidebar.chat_web_view import ChatWebView
        self._preloaded_chat_view = ChatWebView()

        self.show_home()
        self._reparent_toggle_btn(self._stack.currentIndex())  # initial placement

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

        # Wire CLI streaming: tool stdout lines → sidebar UI
        registry.set_cli_line_callback(lambda line: self._bridge.cli_stdout_line.emit(line))

        # Create sidebar widget (reuse pre-warmed ChatWebView)
        self._sidebar = AgentSidebarWidget(
            bridge=self._bridge,
            main_window_ref=self,
            parent=None,
            chat_view=self._preloaded_chat_view,
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

    def _reparent_toggle_btn(self, idx: int) -> None:
        """Move the Co-Pilot toggle button into the current view's navbar/header."""
        btn = self._sidebar_toggle
        if idx == _IDX_HOME:
            self.home_view.set_copilot_button(btn)
        elif idx == _IDX_DARK_CAL:
            self.dark_cal_view.set_copilot_button(btn)
        elif idx == _IDX_CLUSTERING_FILES:
            self.clustering_file_view.set_copilot_button(btn)
        elif idx == _IDX_CLUSTERING_STUDIO:
            self.clustering_studio_view.set_copilot_button(btn)
        elif idx == _IDX_SORTING:
            self.sorting_view.set_copilot_button(btn)
        elif idx == _IDX_SLIDESHOW:
            self.slideshow_view.navbar.set_copilot_button(btn)
        elif idx == _IDX_COMPARISON:
            self.export_comparison_view.set_copilot_button(btn)
        elif idx == _IDX_ZEROTH_ORDER:
            self.zeroth_order_view.navbar.set_copilot_button(btn)

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
        """Show the agent sidebar, restoring cached width without expanding main window."""
        if self._sidebar is None:
            return

        was_maximized = self.isMaximized()
        geom = self.geometry()

        self._sidebar.show()

        total_width = self.centralWidget().width() if self.centralWidget() else self.width()
        sidebar_w = min(self._sidebar_cached_width, max(280, total_width // 3))
        left_w = max(300, total_width - sidebar_w)
        self._splitter.setSizes([left_w, sidebar_w])

        # Ensure window geometry was not altered by Qt layout recalculation
        if was_maximized and not self.isMaximized():
            self.showMaximized()
        elif not was_maximized and self.geometry() != geom:
            self.setGeometry(geom)

        self._sidebar_visible = True
        self._sidebar_toggle.setText("🤖 ✕")
        self._sidebar_toggle.setToolTip("Close Co-Pilot sidebar")
        self._sidebar_toggle.setStyleSheet(
            "QPushButton#copilot_btn { background-color: #0369a1; border: 1.5px solid #38bdf8; color: #ffffff; border-radius: 14px; padding: 4px 14px; font-weight: 600; }"
        )

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
        self._sidebar_toggle.setToolTip("Toggle RIXS Co-Pilot sidebar")
        self._sidebar_toggle.setStyleSheet("")
        set_copilot_btn(self._sidebar_toggle)

    def _get_gui_context(self) -> str:
        """Build a compact GUI context string for the agent.

        Returns:
            A one-line string describing the current app state.
        """
        current_idx = self._stack.currentIndex()
        view_name = _VIEW_NAMES.get(current_idx, "Unknown")
        parts = [f"View={view_name}"]

        try:
            if current_idx == _IDX_HOME:
                pass
            elif current_idx == _IDX_DARK_CAL:
                dv = self.dark_cal_view
                parts.append(f"Frames={dv.dark_frame_count}")
            elif current_idx == _IDX_CLUSTERING_FILES:
                cf = self.clustering_file_view
                parts.append(f"Files={len(cf.signal_paths)}")
            elif current_idx == _IDX_CLUSTERING_STUDIO:
                cs = self.clustering_studio_view
                parts.append(f"Mode={cs.active_mode}")
                parts.append(f"Frames={cs.manager.total_frames}")
                parts.append(f"Clusters={len(cs.manager.state.df_clusters)}")
            elif current_idx == _IDX_SORTING:
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
    # Navigation
    # ------------------------------------------------------------------

    def show_home(self) -> None:
        """Display the Home Launchpad dashboard."""
        self._stack.setCurrentIndex(_IDX_HOME)
        if hasattr(self, "home_view") and hasattr(self.home_view, "refresh_calibration_status"):
            self.home_view.refresh_calibration_status()

    def show_dark_calibration(self) -> None:
        """Display the Dark Image & Pixel Masking Studio view."""
        self._stack.setCurrentIndex(_IDX_DARK_CAL)

    show_dark_masking = show_dark_calibration

    def show_clustering_files(self) -> None:
        """Display the Single-Photon Clustering file selection view."""
        self._stack.setCurrentIndex(_IDX_CLUSTERING_FILES)
        if hasattr(self, "clustering_file_view") and hasattr(self.clustering_file_view, "refresh_calibration_status"):
            self.clustering_file_view.refresh_calibration_status()

    def show_clustering_studio(
        self,
        signal_paths: list[str] | None = None,
        chunk_size: int = 80,
        cluster_cfg=None,
        recon_cfg=None,
    ) -> None:
        """Display the Single-Photon Clustering Studio and optionally load session.

        Args:
            signal_paths: Optional list of signal TIFF file paths.
            chunk_size: Number of frames per chunk.
            cluster_cfg: Optional Stage 2 ClusterConfig.
            recon_cfg: Optional Stage 3 ReconstructionConfig.
        """
        self._stack.setCurrentIndex(_IDX_CLUSTERING_STUDIO)
        if signal_paths and hasattr(self, "clustering_studio_view"):
            self.clustering_studio_view.load_session(
                signal_paths=signal_paths,
                chunk_size=chunk_size,
                cluster_config=cluster_cfg,
                recon_config=recon_cfg,
            )

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
        focused = self.focusWidget() or QApplication.focusWidget()
        from PySide6.QtWidgets import (
            QSlider, QLineEdit, QComboBox, QTextEdit,
            QSpinBox, QDoubleSpinBox, QAbstractSpinBox,
        )
        if isinstance(focused, (QSlider, QLineEdit, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox, QAbstractSpinBox)):
            super().keyPressEvent(event)
            return

        if key == Qt.Key_Left:
            if current == _IDX_SLIDESHOW:
                self.slideshow_view.prev_frame()
            elif current == _IDX_ZEROTH_ORDER:
                self.zeroth_order_view.prev_frame()
            elif current == _IDX_CLUSTERING_STUDIO:
                mode = getattr(self.clustering_studio_view, "active_mode", "Dashboard")
                if mode == "Frame Inspector":
                    self.clustering_studio_view.prev_frame()
                elif mode == "Chunk Inspector":
                    self.clustering_studio_view.prev_chunk()
                # Dashboard mode is a no-op
        elif key == Qt.Key_Right:
            if current == _IDX_SLIDESHOW:
                self.slideshow_view.next_frame()
            elif current == _IDX_ZEROTH_ORDER:
                self.zeroth_order_view.next_frame()
            elif current == _IDX_CLUSTERING_STUDIO:
                mode = getattr(self.clustering_studio_view, "active_mode", "Dashboard")
                if mode == "Frame Inspector":
                    self.clustering_studio_view.next_frame()
                elif mode == "Chunk Inspector":
                    self.clustering_studio_view.next_chunk()
                # Dashboard mode is a no-op
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """Clear text input focus when clicking on the application background."""
        focused = self.focusWidget()
        from PySide6.QtWidgets import QLineEdit
        if focused and isinstance(focused, QLineEdit):
            focused.clearFocus()
            self.setFocus()
        super().mousePressEvent(event)

    def layout(self):
        """Return layout with wrapper ensuring added child widgets are reparented and visible."""
        lay = super().layout()
        if lay is None:
            return None

        class _LayoutWrapper:
            def __init__(self, target, parent):
                self._target = target
                self._parent = parent

            def addWidget(self, widget):  # noqa: N802
                try:
                    widget.setParent(self._parent)
                    widget.show()
                    if hasattr(QApplication, "setActiveWindow"):
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", DeprecationWarning)
                            QApplication.setActiveWindow(self._parent)
                except Exception:
                    pass
                return self._target.addWidget(widget)

            def __getattr__(self, name):
                return getattr(self._target, name)

        return _LayoutWrapper(lay, self)

    def showEvent(self, event) -> None:  # noqa: N802
        """Ensure window is active upon display (especially in offscreen/headless test runners)."""
        super().showEvent(event)
        try:
            self.activateWindow()
            if hasattr(QApplication, "setActiveWindow"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    QApplication.setActiveWindow(self)
        except Exception:
            pass

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
        # Teardown Matplotlib canvases and view resources
        for view_name in ('zeroth_order_view', 'slideshow_view', 'export_comparison_view', 'dark_cal_view', 'clustering_studio_view'):
            view = getattr(self, view_name, None)
            if view is not None:
                if hasattr(view, 'cleanup'):
                    try:
                        view.cleanup()
                    except Exception:
                        pass
                elif hasattr(view, '_teardown_mpl'):
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

    def __del__(self) -> None:
        """Ensure background threads are stopped when the window is destroyed."""
        if hasattr(self, "_bridge") and self._bridge is not None:
            try:
                self._bridge.stop_worker()
            except Exception:
                pass




def main() -> None:
    """Application entry point."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = RixsApp(show_window=True)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
