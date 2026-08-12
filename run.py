#!/usr/bin/env python3
"""Entry point for the mRIXS Super-App (PySide6 build)."""

import signal
import sys

from PySide6.QtWidgets import QApplication

from rixs_app.main import RixsApp


def handle_sigint(sig, frame):
    """Handle Ctrl-C gracefully by closing the main window."""
    app = QApplication.instance()
    if app is not None:
        app.quit()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_sigint)
    qt_app = QApplication.instance() or QApplication(sys.argv)
    window = RixsApp(show_window=True)
    sys.exit(qt_app.exec())
