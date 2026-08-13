# rixs_app/ui/zeroth_order_slideshow/workers.py

"""
PySide6 QRunnable background worker classes for zeroth-order mirror pitch calibration tasks.
Replaces legacy thread queues with native Qt Signals and QThreadPool execution.
"""

import os
import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class ZerothOrderWorkerSignals(QObject):
    """
    Qt Signals emitted by QRunnable workers to update the GUI main thread.
    """
    progress = Signal(int, int)        # current, total
    progress_msg = Signal(str)         # string progress message
    result = Signal(object)            # payload result
    error = Signal(str)                # error message
    finished = Signal()                # task completion signal


class PrecomputeFramesWorker(QRunnable):
    """
    Background worker for evaluating and caching all zeroth-order scan frames.
    """
    def __init__(self, manager, total: int):
        super().__init__()
        self.manager = manager
        self.total = total
        self.session_id = manager.session_id
        self.signals = ZerothOrderWorkerSignals()

    @Slot()
    def run(self):
        try:
            for idx in range(self.total):
                if self.manager.session_id is not self.session_id:
                    return
                data = self.manager.get_frame_pipeline_data(idx)
                if not data:
                    raise ValueError(f"Frame {idx} data is missing or corrupted.")
                if self.manager.session_id is not self.session_id:
                    return
                self.signals.progress.emit(idx + 1, self.total)

            if self.manager.session_id is self.session_id:
                self.signals.result.emit(True)
        except Exception as e:
            if self.manager.session_id is self.session_id:
                self.signals.error.emit(str(e))
        finally:
            if self.manager.session_id is self.session_id:
                self.signals.finished.emit()


class ExportDiagnosticWorker(QRunnable):
    """
    Background worker for rendering diagnostic PNG plots and exporting focus_curve.png.
    """
    def __init__(self, manager, export_dir: str, vmin: float, vmax: float):
        super().__init__()
        self.manager = manager
        self.export_dir = export_dir
        self.vmin = vmin
        self.vmax = vmax
        self.session_id = manager.session_id
        self.signals = ZerothOrderWorkerSignals()

    @Slot()
    def run(self):
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            total = len(self.manager.file_list)
            matplotlib_cmap = self.manager.colormap
            if matplotlib_cmap == "grayscale":
                matplotlib_cmap = "gray"

            for idx in range(total):
                if self.manager.session_id is not self.session_id:
                    return
                data = self.manager.get_frame_pipeline_data(idx)
                if not data or data.get("raw_img") is None:
                    if self.manager.session_id is not self.session_id:
                        return
                    self.signals.progress.emit(idx + 1, total)
                    continue

                fig = Figure(figsize=(10, 8))
                canvas = FigureCanvasAgg(fig)

                ax1 = fig.add_subplot(221)
                ax2 = fig.add_subplot(222)
                ax3 = fig.add_subplot(223)
                ax4 = fig.add_subplot(224)

                raw_disp = np.where((data["raw_img"] >= self.vmin) & (data["raw_img"] <= self.vmax), data["raw_img"], 0.0) if data.get("raw_img") is not None else None

                ax1.imshow(raw_disp, cmap=matplotlib_cmap, vmin=self.vmin, vmax=self.vmax, aspect='auto')
                ax1.set_title("Raw Image")

                dx, dy = data["direction"]
                if abs(dx) > 1e-5:
                    ax1.axline((data["centroid"][0], data["centroid"][1]), slope=dy/dx, color="red", linestyle="--")
                ax1.axis("off")

                denoised = data.get("denoised_img")
                if denoised is not None:
                    p99 = np.percentile(denoised, 99.5)
                    if p99 == 0:
                        p99 = 1.0
                    ax2.imshow(denoised, cmap=matplotlib_cmap, vmin=0, vmax=p99, aspect='auto')
                    if "direction" in data:
                        c = data["centroid"]
                        d = data["direction"]
                        ax2.axline((c[0], c[1]), slope=d[1]/d[0], color="white", linestyle="-")
                else:
                    ax2.text(0.5, 0.5, "No Denoised Image", ha='center', va='center')
                ax2.set_title("Denoised Image")
                ax2.axis("off")

                masked_img = data.get("masked_img")
                if masked_img is not None:
                    p99_g = np.percentile(masked_img, 99.9)
                    if p99_g == 0:
                        p99_g = 1.0
                    ax3.imshow(masked_img, cmap=matplotlib_cmap, vmin=0, vmax=p99_g, aspect='auto')
                else:
                    ax3.text(0.5, 0.5, "No Masked Image", ha='center', va='center')
                ax3.set_title("Masked Gradient")
                ax3.axis("off")

                P, u = data["1d_profile"]
                ax4.plot(u, P, 'k-', linewidth=2, label='1D Profile')
                ax4.set_title(f"1D Profile (Score: {data['score']:.2f})")
                ax4.set_xlabel("Perpendicular Distance (u)")
                ax4.set_ylabel("Gradient Sum")
                ax4.legend(fontsize=8)

                fig.tight_layout()
                save_path = os.path.join(self.export_dir, f"frame_{idx:03d}_diagnostic.png")
                fig.savefig(save_path, dpi=150)

                if self.manager.session_id is not self.session_id:
                    return
                self.signals.progress.emit(idx + 1, total)

            # Generate focus curve
            self.manager._export_focus_curve(
                self.export_dir,
                self.manager.txt_metadata,
                self.manager.energy_dispersion,
                self.manager.mono_energy_ev,
            )

            if self.manager.session_id is self.session_id:
                self.signals.result.emit(True)
        except Exception as e:
            if self.manager.session_id is self.session_id:
                self.signals.error.emit(str(e))
        finally:
            if self.manager.session_id is self.session_id:
                self.signals.finished.emit()
