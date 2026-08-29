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
