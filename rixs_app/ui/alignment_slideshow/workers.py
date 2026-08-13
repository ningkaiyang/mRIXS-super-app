# rixs_app/ui/alignment_slideshow/workers.py

"""
PySide6 QRunnable background worker classes for spatial alignment tasks.
Replaces legacy thread queues with native Qt Signals and QThreadPool execution.
"""

from PySide6.QtCore import QObject, QRunnable, Signal, Slot
import numpy as np

from rixs_app.core import (
    find_best_threshold,
    generate_direct_sum,
    generate_aligned_sum,
)


class WorkerSignals(QObject):
    """
    Qt Signals emitted by QRunnable workers to safely update the GUI main thread.
    """
    progress = Signal(int, int)        # current, total
    progress_msg = Signal(str)         # string progress message
    result = Signal(object)            # payload result
    error = Signal(str)                # error message
    finished = Signal()                # task completion signal


class AutoSnapWorker(QRunnable):
    """
    Background worker for finding the optimal PCA threshold on a single frame.
    """
    def __init__(self, raw_data: np.ndarray):
        super().__init__()
        self.raw_data = raw_data
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            best_t = find_best_threshold(self.raw_data)
            self.signals.result.emit(best_t)
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class AutoSnapAllWorker(QRunnable):
    """
    Background worker for finding optimal PCA thresholds for all frames.
    """
    def __init__(self, file_list: list[str], get_raw_fn):
        super().__init__()
        self.file_list = file_list
        self.get_raw_fn = get_raw_fn
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            n_frames = len(self.file_list)
            results = {}
            for idx in range(n_frames):
                raw = self.get_raw_fn(self.file_list[idx])
                if raw is not None:
                    results[idx] = find_best_threshold(raw)
                self.signals.progress.emit(idx + 1, n_frames)
            self.signals.result.emit(results)
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class PrecomputeOffsetsWorker(QRunnable):
    """
    Background worker for precomputing sub-pixel translation offsets across all frames.
    """
    def __init__(self, n_frames: int, get_offset_fn):
        super().__init__()
        self.n_frames = n_frames
        self.get_offset_fn = get_offset_fn
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            offsets = {0: (0.0, 0.0)}
            for idx in range(1, self.n_frames):
                self.signals.progress.emit(idx, self.n_frames)
                offsets[idx] = self.get_offset_fn(idx)
            self.signals.progress.emit(self.n_frames, self.n_frames)
            self.signals.result.emit(offsets)
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class ExportSumsWorker(QRunnable):
    """
    Background worker for computing Direct Sum and Aligned Sum images for comparison/export.
    """
    def __init__(self, file_list: list[str], get_raw_fn, offsets: dict[int, tuple[float, float]], ref_shape: tuple[int, int]):
        super().__init__()
        self.file_list = file_list
        self.get_raw_fn = get_raw_fn
        self.offsets = offsets
        self.ref_shape = ref_shape
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            def _progress_direct(current, total):
                self.signals.progress_msg.emit(f"Computing Direct Sum: {current}/{total}...")

            direct_sum = generate_direct_sum(
                self.file_list, self.get_raw_fn, self.ref_shape, progress_callback=_progress_direct
            )

            def _progress_aligned(current, total):
                self.signals.progress_msg.emit(f"Computing Aligned Sum: {current}/{total}...")

            aligned_sum = generate_aligned_sum(
                self.file_list, self.get_raw_fn, self.offsets,
                self.ref_shape, progress_callback=_progress_aligned
            )

            self.signals.result.emit((aligned_sum, direct_sum))
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()
