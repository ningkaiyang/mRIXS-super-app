"""Global pytest configuration and session fixtures for RIXS Super-App test suite.

Configures Qt to run strictly in offscreen/headless mode to prevent UI windows from popping up
or stealing focus during test execution on macOS / Linux / Windows.
"""

from __future__ import annotations

import os

# Enforce Qt native offscreen QPA platform before any PySide6 modules or QApplication initialize
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication


def pytest_configure(config: pytest.Config) -> None:
    """Ensure QT_QPA_PLATFORM is set to offscreen as early as possible during pytest initialization."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture(scope="session")
def qapp():
    """Ensure a shared session-scoped headless QApplication instance exists for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    return app


def cleanup_app(app_win, qapp=None, qtbot=None) -> None:
    """Clean up RixsApp, tear down worker threads, break view cycles, and flush deferred events."""
    if app_win is None:
        return
    if hasattr(app_win, "_bridge") and app_win._bridge is not None:
        try:
            app_win._bridge.stop_worker()
        except Exception:
            pass
    if qtbot is not None and hasattr(qtbot, "_request") and hasattr(qtbot._request, "node"):
        qt_widgets = getattr(qtbot._request.node, "qt_widgets", None)
        if isinstance(qt_widgets, list):
            qt_widgets.clear()
    app_win.close()
    app_win.deleteLater()
    for attr in (
        "home_view",
        "dark_cal_view",
        "clustering_files_view",
        "clustering_studio_view",
        "sorting_view",
        "slideshow_view",
        "export_comparison_view",
        "zeroth_order_view",
    ):
        if hasattr(app_win, attr):
            try:
                setattr(app_win, attr, None)
            except Exception:
                pass
    import gc
    gc.collect()
    if qapp is not None:
        qapp.processEvents()
    from PySide6.QtCore import QCoreApplication, QEvent
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

