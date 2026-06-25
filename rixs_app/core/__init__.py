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
]
