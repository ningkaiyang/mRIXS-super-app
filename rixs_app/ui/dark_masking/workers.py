"""Background QRunnable workers for detector dark image & pixel masking diagnostics.

Executes heavy temporal median, standard deviation, and 93rd-percentile residual
calculations on background thread pool with Qt signal marshaling to the main GUI thread.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from rixs_app.core.photon_clustering import (
    DarkDiagnostics,
    compute_dark_diagnostics,
    export_dark_diagnostics,
)

logger = logging.getLogger(__name__)


class WorkerSignals(QObject):
    """Qt Signals emitted by QRunnable workers to safely update the GUI thread."""

    progress = Signal(int, int)        # current_frame, total_frames
    progress_msg = Signal(str)         # textual progress description
    result = Signal(object)            # payload (DarkDiagnostics or dict of export paths)
    error = Signal(str)                # error message
    finished = Signal()                # completion signal


class DarkDiagnosticsWorker(QRunnable):
    """Background worker for computing dark frame diagnostics asynchronously.

    Analyzes a sequence of dark frame TIFFs to compute:
    - Temporal median baseline image
    - Per-pixel standard deviation (sigma)
    - 93rd-percentile absolute excursion / residual map

    Args:
        dark_paths: Sequence of filesystem paths to dark frame TIFF files.
        tail_pct: Percentile ratio for residual evaluation (default 0.9333).
        max_frames: Maximum number of dark frames to process (0 = all).
    """

    def __init__(
        self,
        dark_paths: Sequence[Path | str],
        tail_pct: float = 0.9333,
        max_frames: int = 0,
    ) -> None:
        super().__init__()
        self.dark_paths = [Path(p) for p in dark_paths]
        self.tail_pct = float(tail_pct)
        self.max_frames = int(max_frames)
        self.is_canceled = False
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    def _safe_emit(self, signal, *args) -> None:
        """Safely emit Qt signals guarding against deleted underlying C++ QObjects."""
        try:
            signal.emit(*args)
        except (RuntimeError, AttributeError):
            pass

    def cancel(self) -> None:
        """Request cooperative cancellation of the background worker."""
        self.is_canceled = True

    def _stage_callback(self, stage: int, current: int, total: int, msg: str) -> None:
        """Stage-aware progress callback invoked from compute_dark_diagnostics."""
        if self.is_canceled:
            return
        if stage == 1:
            pct = int((current / total) * 50) if total > 0 else 0
            formatted_msg = f"[1/3] Ingesting dark frames ({current}/{total})..."
        elif stage == 2:
            pct = int(50 + (current / total) * 40) if total > 0 else 50
            formatted_msg = f"[2/3] Computing noise statistics (chunk {current}/{total})..."
        else:
            pct = int((current / total) * 100) if total > 0 else 0
            formatted_msg = msg

        self._safe_emit(self.signals.progress, pct, 100)
        self._safe_emit(self.signals.progress_msg, formatted_msg)

    @Slot()
    def run(self) -> None:
        """Execute diagnostic computations on worker thread."""
        try:
            if not self.dark_paths:
                raise ValueError("No dark frame paths provided to DarkDiagnosticsWorker.")

            if self.is_canceled:
                return

            total_frames = self.max_frames if 0 < self.max_frames < len(self.dark_paths) else len(self.dark_paths)
            self._safe_emit(self.signals.progress, 0, 100)
            self._safe_emit(self.signals.progress_msg, f"[1/3] Ingesting dark frames (0/{total_frames})...")
            diagnostics = compute_dark_diagnostics(
                dark_paths=self.dark_paths,
                tail_pct=self.tail_pct,
                stage_callback=self._stage_callback,
                max_frames=self.max_frames,
            )
            if not self.is_canceled:
                self._safe_emit(self.signals.result, diagnostics)
        except Exception as exc:
            if not self.is_canceled:
                logger.exception("DarkDiagnosticsWorker failed: %s", exc)
                self._safe_emit(self.signals.error, str(exc))
        finally:
            self._safe_emit(self.signals.finished)


class DarkExportWorker(QRunnable):
    """Background worker for exporting dark diagnostics data and publication plots asynchronously.

    Args:
        diagnostics: DarkDiagnostics instance to export.
        export_dir: Destination directory path.
        stddev_thresh: StdDev threshold (ADU).
        absdev_thresh: Excursion residual threshold (ADU).
        tail_ratio: Tail percentile ratio (default 0.9333).
        bins: Number of histogram bins (default 60).
        dpi: Output resolution for publication figures (default 300).
    """

    def __init__(
        self,
        diagnostics: DarkDiagnostics,
        export_dir: Path | str,
        stddev_thresh: float = 40.0,
        absdev_thresh: float = 60.0,
        tail_ratio: float = 0.9333,
        bins: int = 60,
        dpi: int = 300,
    ) -> None:
        super().__init__()
        self.diagnostics = diagnostics
        self.export_dir = Path(export_dir)
        self.stddev_thresh = float(stddev_thresh)
        self.absdev_thresh = float(absdev_thresh)
        self.tail_ratio = float(tail_ratio)
        self.bins = int(bins)
        self.dpi = int(dpi)
        self.is_canceled = False
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    def _safe_emit(self, signal, *args) -> None:
        """Safely emit Qt signals guarding against deleted underlying C++ QObjects."""
        try:
            signal.emit(*args)
        except (RuntimeError, AttributeError):
            pass

    def cancel(self) -> None:
        """Request cooperative cancellation of the background worker."""
        self.is_canceled = True

    def _progress_cb(self, current: int, total: int, msg: str) -> None:
        if self.is_canceled:
            return
        self._safe_emit(self.signals.progress, current, total)
        self._safe_emit(self.signals.progress_msg, msg)

    @Slot()
    def run(self) -> None:
        """Execute data and plot export in background thread."""
        try:
            if self.is_canceled:
                return

            self._safe_emit(self.signals.progress, 0, 4)
            self._safe_emit(self.signals.progress_msg, "Initializing export...")

            results = export_dark_diagnostics(
                diagnostics=self.diagnostics,
                export_dir=self.export_dir,
                stddev_thresh=self.stddev_thresh,
                absdev_thresh=self.absdev_thresh,
                tail_ratio=self.tail_ratio,
                bins=self.bins,
                dpi=self.dpi,
                progress_callback=self._progress_cb,
            )

            if not self.is_canceled:
                self._safe_emit(self.signals.result, results)
        except Exception as exc:
            if not self.is_canceled:
                logger.exception("DarkExportWorker failed: %s", exc)
                self._safe_emit(self.signals.error, str(exc))
        finally:
            self._safe_emit(self.signals.finished)

