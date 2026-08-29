"""Backward compatibility wrapper for rixs_app.ui.dark_masking.workers.

Deprecated: Prefer importing directly from ``rixs_app.ui.dark_masking.workers``.
"""

from __future__ import annotations

from rixs_app.ui.dark_masking.workers import WorkerSignals, DarkDiagnosticsWorker

__all__ = ["WorkerSignals", "DarkDiagnosticsWorker"]
