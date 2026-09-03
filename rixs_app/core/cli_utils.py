"""Shared utilities for headless CLI tools (align_cli.py, zeroth_order_cli.py).

This module provides directory discovery helpers and focus-curve plot generation
that are consumed by both CLI scripts and by the GUI export worker. Keeping
shared logic here avoids duplication and keeps the CLI scripts thin.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# TIF directory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _has_tif_files(directory: Path, min_count: int = 2) -> bool:
    """Return True if *directory* contains at least *min_count* TIF/TIFF files.

    Args:
        directory: Path object pointing to the directory to check.
        min_count: Minimum number of ``.tif`` / ``.tiff`` files required before
            the directory is considered a valid scan directory.

    Returns:
        ``True`` when enough TIF files are found, ``False`` otherwise.
    """
    count = 0
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in ('.tif', '.tiff'):
            count += 1
            if count >= min_count:
                return True
    return False


def discover_directories(
    root_dir: str,
    recursive: bool,
    skip_dirs: set[str] | None = None,
) -> list[str]:
    """Return directories under *root_dir* that contain ≥ 2 TIF files.

    In non-recursive mode only *root_dir* itself is considered.  In recursive
    mode every subdirectory at any depth is scanned, excluding entries whose
    lowercased basename appears in *skip_dirs*.

    Args:
        root_dir: Absolute or relative path to the root scan directory.
        recursive: If ``True``, recurse into all subdirectories.
        skip_dirs: Set of lowercase directory names to skip during traversal
            (e.g. ``{'sum', 'tif-cache', 'clusters', 'zeroth_order_analysis', 'denoised'}``).
            Defaults to ``{'sum', 'tif-cache', 'clusters', 'zeroth_order_analysis', 'denoised'}``
            when ``None``.

    Returns:
        Sorted list of absolute directory path strings that contain sufficient
        TIF files and are not inside a skipped subdirectory.
    """
    if skip_dirs is None:
        skip_dirs = {'sum', 'tif-cache', 'clusters', 'zeroth_order_analysis', 'denoised'}

    root = Path(root_dir).resolve()
    result: list[str] = []

    if _has_tif_files(root):
        result.append(str(root))

    if recursive:
        for dirpath, dirnames, _filenames in os.walk(str(root)):
            # Prune traversal to skip configured directories
            dirnames[:] = [d for d in dirnames if d.lower() not in skip_dirs]
            dp = Path(dirpath).resolve()
            if dp == root:
                continue  # already checked above
            if _has_tif_files(dp):
                result.append(str(dp))

    result.sort()
    return result


def glob_tifs(directory: str) -> list[str]:
    """Return naturally sorted absolute paths of all TIF/TIFF files in *directory*.

    Args:
        directory: Absolute or relative path to the directory to scan.

    Returns:
        Naturally sorted list of absolute TIFF file paths found directly inside
        *directory* (non-recursive).
    """
    from rixs_app.core.utils import natural_sort

    d = Path(directory)
    files: list[str] = []
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in ('.tif', '.tiff'):
            files.append(str(p.resolve()))
    natural_sort(files)
    return files


# ─────────────────────────────────────────────────────────────────────────────
# Shared focus-curve plot
# ─────────────────────────────────────────────────────────────────────────────

def export_focus_curve(
    export_dir: str,
    x_values: list[float],
    fwhms: list[float],
    x_label: str,
    energy_dispersion: float = 0.0,
    mono_energy_ev: float = 0.0,
    resolving_powers: list[float | None] | None = None,
) -> str:
    """Generate ``focus_curve.png`` and save it to *export_dir*.

    The focus curve is a scatter plot of *x_values* vs FWHM in pixels, with an
    optional parabolic fit and resolving-power annotation when enough data points
    and physical parameters are available.

    This function is the single source of truth for focus-curve rendering.  Both
    the GUI export worker in ``manager.py`` and the headless ``zeroth_order_cli.py``
    call this function to ensure identical output.

    Args:
        export_dir: Absolute path to the directory where ``focus_curve.png``
            will be written.
        x_values: Ordered list of X-axis values (motor positions or frame
            indices).  Must have the same length as *fwhms*.
        fwhms: Ordered list of FWHM values in pixels corresponding to each
            X value.  Must have the same length as *x_values*.
        x_label: Human-readable X-axis label.  Typical values are the motor
            name (e.g. ``"SM3 Mirror Pitch"``) or ``"Frame Index"``.
        energy_dispersion: Energy dispersion in meV/px.  When > 0 and
            *mono_energy_ev* > 0, resolving power R is computed for each frame.
        mono_energy_ev: Monochromator energy in eV (E_mono).  Required for
            resolving power R = E_mono / (FWHM_meV × 1e-3).
        resolving_powers: Optional pre-computed list of resolving powers (floats
            or ``None`` for invalid frames).  When ``None``, resolving powers are
            computed internally from *energy_dispersion* and *mono_energy_ev*.

    Returns:
        Absolute path of the saved ``focus_curve.png`` file.

    Raises:
        ValueError: If ``len(x_values) != len(fwhms)``.
        RuntimeError: If *export_dir* cannot be created or written to.
    """
    if len(x_values) != len(fwhms):
        raise ValueError(
            f"x_values length ({len(x_values)}) must equal fwhms length ({len(fwhms)})."
        )
    if len(x_values) < 2:
        raise ValueError(
            "At least 2 data points are required to generate a focus curve."
        )

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    # Build resolving powers if not supplied
    if resolving_powers is None:
        resolving_powers = []
        for fwhm_px in fwhms:
            if energy_dispersion > 0 and mono_energy_ev > 0 and fwhm_px is not None:
                fwhm_mev = fwhm_px * energy_dispersion
                R = mono_energy_ev / (fwhm_mev * 1e-3)
                resolving_powers.append(R)
            else:
                resolving_powers.append(None)

    title = f"Zeroth-Order Focus Curve — {x_label} vs FWHM"

    fig = Figure(figsize=(10, 6))
    _canvas = FigureCanvasAgg(fig)  # required to attach renderer
    ax = fig.add_subplot(111)

    ax.scatter(x_values, fwhms, c='steelblue', s=40, zorder=5, label='Measured FWHM')

    optimal_x = None
    if len(x_values) >= 3:
        coeffs = np.polyfit(x_values, fwhms, 2)
        if abs(coeffs[0]) > 1e-12:  # guard against degenerate parabola
            x_smooth = np.linspace(min(x_values), max(x_values), 200)
            y_smooth = np.polyval(coeffs, x_smooth)
            ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Parabolic fit')
            optimal_x = -coeffs[1] / (2 * coeffs[0])
            ax.axvline(
                optimal_x,
                color='green',
                linestyle='--',
                alpha=0.7,
                label=f'Optimal {x_label}: {optimal_x:.4f}',
            )

    # Resolving-power annotation on the best (minimum FWHM) frame
    valid_r = [
        (r, x, f)
        for r, x, f in zip(resolving_powers, x_values, fwhms)
        if r is not None
    ]
    if valid_r:
        peak_R, peak_x, peak_fwhm = max(valid_r, key=lambda t: t[0])
        ax.annotate(
            f'Peak R = {peak_R:,.0f}\nat {x_label} = {peak_x:.4f}',
            xy=(peak_x, peak_fwhm),
            xytext=(0.05, 0.95),
            textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='purple'),
            fontsize=10,
            color='purple',
            va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='purple'),
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel('FWHM (px)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(export_dir, exist_ok=True)
    out_path = os.path.join(export_dir, 'focus_curve.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    return out_path
