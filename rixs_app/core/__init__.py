"""
Core algorithms and utilities for RIXS beamline frame alignment.

Modules:
  - alignment: Line fitting and sub-pixel translation calculations.
  - image_processing: Sub-pixel image warping, contrast mapping, and alignment summation.
  - io: Fast, memory-mapped TIFF loading with persistent disk caching.
  - math_utils: Intensity-weighted Principal Component Analysis (PCA) using SVD.
  - utils: Natural alphanumeric sorting.
"""

from .io import load_raw
from .utils import natural_sort
from .math_utils import _weighted_pca
from .alignment import (
    PCAFitFailure,
    _cross_section_center,
    find_peak_line,
    find_peak_line_fast,
    _dog_prefilter,
    _tukey_window_2d,
    phase_correlation_offset,
    compute_line_based_offset,
    ecc_maximization_offset,
    compute_alignment_priors,
    find_best_threshold,
    precompute_ecc_reference,
)
from .image_processing import (
    warp_image,
    preprocess_image,
    apply_colormap,
    generate_aligned_sum,
    generate_direct_sum,
)
from .preprocessing import denoise_image, prepare_frame, PreprocessingConfig, CropTransform, PreparedFrame
from .zeroth_order import evaluate_zeroth_order, run_zeroth_order_pipeline
from .cli_utils import discover_directories, glob_tifs, export_focus_curve, extract_frame_index
from .photon_clustering import (
    DarkDiagnostics,
    DarkMaskConfig,
    ClusterConfig,
    ReconstructionConfig,
    Stage1Result,
    ReconstructionResult,
    compute_dark_diagnostics,
    apply_dark_thresholds,
    compute_dark_mask,
    process_single_frame_clusters,
    process_signal_stack_clusters,
    reconstruct_photon_event_map,
    export_intden_histogram,
)
from .dark_mask_store import (
    DEFAULT_MASK_DIR,
    DARK_MASK_DIR,
    DarkMaskRecord,
    has_dark_mask,
    load_dark_mask,
    save_dark_mask,
    get_mask_summary,
    clear_dark_mask,
    get_dark_mask_dir,
    get_meta_file_path,
    # Backward compatibility aliases
    DEFAULT_CALIBRATION_DIR,
    DARK_CAL_DIR,
    CalibrationRecord,
    has_calibration,
    load_calibration,
    save_calibration,
    get_calibration_summary,
    clear_calibration,
    get_dark_cal_dir,
)

__all__ = [
    "PCAFitFailure",
    "load_raw",
    "natural_sort",
    "_weighted_pca",
    "_cross_section_center",
    "find_peak_line",
    "find_peak_line_fast",
    "_dog_prefilter",
    "_tukey_window_2d",
    "phase_correlation_offset",
    "compute_line_based_offset",
    "warp_image",
    "preprocess_image",
    "apply_colormap",
    "generate_aligned_sum",
    "generate_direct_sum",
    "ecc_maximization_offset",
    "compute_alignment_priors",
    "find_best_threshold",
    "precompute_ecc_reference",
    "denoise_image",
    "evaluate_zeroth_order",
    "run_zeroth_order_pipeline",
    "discover_directories",
    "glob_tifs",
    "export_focus_curve",
    "extract_frame_index",
    "DarkDiagnostics",
    "DarkMaskConfig",
    "ClusterConfig",
    "ReconstructionConfig",
    "Stage1Result",
    "ReconstructionResult",
    "compute_dark_diagnostics",
    "apply_dark_thresholds",
    "compute_dark_mask",
    "process_single_frame_clusters",
    "process_signal_stack_clusters",
    "reconstruct_photon_event_map",
    "export_intden_histogram",
    "DEFAULT_CALIBRATION_DIR",
    "DARK_CAL_DIR",
    "CalibrationRecord",
    "has_calibration",
    "load_calibration",
    "save_calibration",
    "get_calibration_summary",
    "clear_calibration",
    "get_dark_cal_dir",
    "get_meta_file_path",
]

