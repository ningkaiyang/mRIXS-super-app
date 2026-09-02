"""Background QRunnable workers for single-photon clustering and export operations.

Includes:
- ClusterPipelineWorker: Multithreaded Stage 2 connected component analysis with
  progressive per-frame signal emission for real-time GUI canvas accumulation.
- ChunkSaveWorker: Background batch export of per-chunk event maps, total event map,
  TSV spreadsheet, and diagnostic IntDen histogram.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import tifffile
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from rixs_app.core.photon_clustering import (
    ClusterConfig,
    ReconstructionConfig,
    export_intden_histogram,
    process_single_frame_clusters,
)
from rixs_app.ui.clustering_slideshow.manager import ClusteringManager, CLUSTER_COLUMNS

logger = logging.getLogger(__name__)


# ============================================================================
# Stage 2 Progressive Pipeline Worker
# ============================================================================

class ClusterPipelineSignals(QObject):
    """Qt Signals emitted by ClusterPipelineWorker."""

    frame_started = Signal(int, int)       # current_frame (1-indexed), total_frames
    frame_result = Signal(int, object)     # current_frame (1-indexed), pd.DataFrame
    progress = Signal(int, int, int)       # current_frame, total_frames, total_clusters_so_far
    progress_msg = Signal(str)            # Text description of progress
    finished = Signal(object)              # Consolidated full pd.DataFrame
    error = Signal(str)                   # Error message on exception
    canceled = Signal()                   # Emitted if worker was canceled


class ClusterPipelineWorker(QRunnable):
    """Background worker executing Stage 2 single-photon cluster extraction.

    Processes frames in parallel using a ThreadPoolExecutor while progressively
    emitting per-frame cluster results to enable real-time canvas accumulation
    on the GUI thread with cooperative cancellation.
    """

    def __init__(
        self,
        signal_paths: Sequence[Path | str],
        med_dark: np.ndarray,
        final_mask: np.ndarray,
        config: ClusterConfig = ClusterConfig(),
        max_workers: int | None = None,
    ) -> None:
        super().__init__()
        self.signal_paths = [Path(p) for p in signal_paths]
        self.med_dark = np.asarray(med_dark, dtype=np.float32)
        self.final_mask = np.asarray(final_mask, dtype=np.float32)
        self.config = config
        self.max_workers = max_workers or min(8, os.cpu_count() or 4)
        self.signals = ClusterPipelineSignals()
        self._is_canceled = False
        self.setAutoDelete(True)

    def cancel(self) -> None:
        """Request cooperative cancellation of the worker."""
        self._is_canceled = True

    @property
    def is_canceled(self) -> bool:
        """True if cancellation was requested."""
        return self._is_canceled

    def _process_one_frame(self, item: tuple[int, Path]) -> tuple[int, pd.DataFrame]:
        """Process Stage 2 cluster extraction for a single raw TIFF frame.

        Args:
            item: Tuple of (1-indexed slice number, file path to frame).

        Returns:
            Tuple of (slice_idx, extracted clusters DataFrame).
        """
        slice_idx, path = item
        if self._is_canceled:
            return slice_idx, pd.DataFrame(columns=CLUSTER_COLUMNS)

        raw_frame = tifffile.imread(path)
        if self._is_canceled:
            return slice_idx, pd.DataFrame(columns=CLUSTER_COLUMNS)

        frame_df = process_single_frame_clusters(
            frame=raw_frame,
            med_dark=self.med_dark,
            final_mask=self.final_mask,
            config=self.config,
            slice_idx=slice_idx,
        )
        return slice_idx, frame_df

    @Slot()
    def run(self) -> None:
        """Process frames in parallel with ThreadPoolExecutor and emit progressive signals."""
        try:
            total_frames = len(self.signal_paths)
            if total_frames == 0:
                raise ValueError("No signal frame paths provided to ClusterPipelineWorker.")

            if self._is_canceled:
                logger.info("ClusterPipelineWorker canceled before starting execution")
                self.signals.canceled.emit()
                return

            all_dfs: list[pd.DataFrame] = []
            total_clusters = 0
            completed_frames = 0

            self.signals.progress_msg.emit(
                f"Starting cluster analysis across {total_frames} frames..."
            )

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                items = [(i + 1, path) for i, path in enumerate(self.signal_paths)]
                future_to_item = {
                    executor.submit(self._process_one_frame, item): item for item in items
                }

                try:
                    for future in concurrent.futures.as_completed(future_to_item):
                        if self._is_canceled:
                            logger.info("ClusterPipelineWorker canceled during execution")
                            executor.shutdown(wait=False, cancel_futures=True)
                            self.signals.canceled.emit()
                            return

                        frame_idx, frame_df = future.result()
                        completed_frames += 1

                        if frame_df is not None and not frame_df.empty:
                            all_dfs.append(frame_df)
                            total_clusters += len(frame_df)

                        self.signals.frame_result.emit(frame_idx, frame_df)
                        self.signals.progress.emit(completed_frames, total_frames, total_clusters)
                        self.signals.progress_msg.emit(
                            f"Extracted {completed_frames}/{total_frames} frames ({total_clusters:,} clusters)..."
                        )
                except Exception:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

            if self._is_canceled:
                self.signals.canceled.emit()
                return

            if all_dfs:
                all_dfs.sort(key=lambda d: int(d["Slice"].iloc[0]) if not d.empty else 0)
                df_all = pd.concat(all_dfs, ignore_index=True)
                df_all["ClusterNum"] = np.arange(len(df_all), dtype=np.int64)
            else:
                df_all = pd.DataFrame(columns=CLUSTER_COLUMNS)

            self.signals.progress_msg.emit(
                f"Cluster extraction complete: {total_clusters:,} clusters detected across {total_frames} frames."
            )
            self.signals.finished.emit(df_all)

        except Exception as exc:
            logger.exception("ClusterPipelineWorker failed: %s", exc)
            self.signals.error.emit(str(exc))


# ============================================================================
# Stage 3 Chunk & Artifact Save Worker
# ============================================================================

class ChunkSaveSignals(QObject):
    """Qt Signals emitted by ChunkSaveWorker."""

    chunk_saved = Signal(int, int, str)    # current_chunk (1-indexed), total_chunks, saved_filepath
    progress = Signal(int, int)            # current_step, total_steps
    progress_msg = Signal(str)             # Text description
    finished = Signal(str)                 # output_directory path
    error = Signal(str)                    # Error message on exception


class ChunkSaveWorker(QRunnable):
    """Background worker exporting per-chunk event maps, total event map, TSV table, and histogram.

    Exports:
    - Photon_Event_Map_frames_{start}-{end}.tif for each chunk
    - Photon_Event_Map_total.tif for full sum
    - Results_clusters.xls (tab-delimited TSV matching ImageJ standard)
    - IntDen_histogram.png (diagnostic 1D distribution plot)
    """

    def __init__(
        self,
        manager: ClusteringManager,
        output_dir: Path | str,
        recon_config: ReconstructionConfig | None = None,
        save_total_map: bool = True,
        save_spreadsheet: bool = True,
        save_histogram: bool = True,
    ) -> None:
        super().__init__()
        self.manager = manager
        self.output_dir = Path(output_dir)
        self.recon_config = recon_config or manager.state.recon_config
        self.save_total_map = save_total_map
        self.save_spreadsheet = save_spreadsheet
        self.save_histogram = save_histogram
        self.signals = ChunkSaveSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        """Execute chunk event map and product export."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            ranges = self.manager.get_chunk_frame_ranges()
            total_chunks = len(ranges)

            # Total steps = chunks + (1 for total map) + (1 for spreadsheet) + (1 for histogram)
            extra_steps = int(self.save_total_map) + int(self.save_spreadsheet) + int(self.save_histogram)
            total_steps = total_chunks + extra_steps
            current_step = 0

            # 1. Save per-chunk event maps
            for chunk_idx, (start, end) in enumerate(ranges):
                current_step += 1
                self.signals.progress_msg.emit(
                    f"Reconstructing chunk {chunk_idx + 1}/{total_chunks} (frames {start}-{end})..."
                )
                chunk_recon = self.manager.get_chunk_reconstruction(chunk_idx, self.recon_config)
                chunk_tif_name = f"Photon_Event_Map_frames_{start}-{end}.tif"
                chunk_tif_path = self.output_dir / chunk_tif_name
                tifffile.imwrite(chunk_tif_path, chunk_recon.event_map.astype(np.float32))

                self.signals.chunk_saved.emit(chunk_idx + 1, total_chunks, str(chunk_tif_path))
                self.signals.progress.emit(current_step, total_steps)

            # 2. Save total event map
            if self.save_total_map:
                current_step += 1
                self.signals.progress_msg.emit("Reconstructing total photon event map...")
                total_recon = self.manager.get_reconstruction(self.recon_config)
                total_tif_path = self.output_dir / "Photon_Event_Map_total.tif"
                tifffile.imwrite(total_tif_path, total_recon.event_map.astype(np.float32))
                self.signals.progress.emit(current_step, total_steps)

            # 3. Save tab-delimited Results_clusters.xls
            if self.save_spreadsheet:
                current_step += 1
                self.signals.progress_msg.emit("Exporting cluster spreadsheet (Results_clusters.xls)...")
                xls_path = self.output_dir / "Results_clusters.xls"
                df = self.manager.state.df_clusters
                df.to_csv(xls_path, sep="\t", index=False)
                self.signals.progress.emit(current_step, total_steps)

            # 4. Save diagnostic IntDen histogram PNG
            if self.save_histogram:
                current_step += 1
                self.signals.progress_msg.emit("Exporting IntDen distribution plot (IntDen_histogram.png)...")
                hist_png_path = self.output_dir / "IntDen_histogram.png"
                export_intden_histogram(
                    df_clusters=self.manager.state.df_clusters,
                    output_png=hist_png_path,
                    intden_low=self.recon_config.intden_low,
                    intden_high=self.recon_config.intden_high,
                )
                self.signals.progress.emit(current_step, total_steps)

            self.signals.progress_msg.emit(f"All products successfully exported to {self.output_dir}")
            self.signals.finished.emit(str(self.output_dir))

        except Exception as exc:
            logger.exception("ChunkSaveWorker failed: %s", exc)
            self.signals.error.emit(str(exc))
