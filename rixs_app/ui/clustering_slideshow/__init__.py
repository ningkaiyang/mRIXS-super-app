"""Single-Photon Clustering Studio & Slideshow UI Package.

Exposes:
- ClusteringFileSelectionView: Signal TIFF file selection with dark calibration verification banner.
- ClusteringStudioView: 3-mode studio (Dashboard, Frame Inspector, Chunk Inspector).
- ClusteringManager & ClusteringState: In-memory session manager with <50ms filtering.
- ClusterPipelineWorker & ChunkSaveWorker: Background asynchronous workers.
"""

from __future__ import annotations

from rixs_app.ui.clustering_slideshow.file_selection_view import (
    ClusteringFileSelectionView,
)
from rixs_app.ui.clustering_slideshow.manager import (
    ClusteringManager,
    ClusteringState,
)
from rixs_app.ui.clustering_slideshow.studio_view import (
    ClusteringStudioView,
)
from rixs_app.ui.clustering_slideshow.workers import (
    ChunkSaveSignals,
    ChunkSaveWorker,
    ClusterPipelineSignals,
    ClusterPipelineWorker,
)

__all__ = [
    "ClusteringFileSelectionView",
    "ClusteringStudioView",
    "ClusteringManager",
    "ClusteringState",
    "ClusterPipelineWorker",
    "ClusterPipelineSignals",
    "ChunkSaveWorker",
    "ChunkSaveSignals",
]
