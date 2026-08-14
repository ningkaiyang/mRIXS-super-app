#!/usr/bin/env python3
"""Headless CLI for zeroth-order line calibration and FWHM optimization.

Processes one or more directories of TIFF files acquired during a mirror-pitch
scan, runs the complete zeroth-order pipeline on each frame to measure FWHM in
pixels (and optionally meV), identifies the frame with minimum FWHM, and
exports a focus curve, per-frame diagnostic PNGs, and tabular summary reports.

This script has **zero GUI dependencies** — it uses ``matplotlib.use('Agg')``
and imports exclusively from ``rixs_app.core``.  It is safe to run on headless
HPC nodes and beamline servers.

Typical usage::

    # Single directory with TXT scan log, dispersion and mono energy
    python zeroth_order_cli.py -d "RIXS_ZeroOrderScan/Single Motor Scan 004202 Images" \\
        --dispersion 2.5 --mono-energy 850.0

    # Recursive batch with all output formats
    python zeroth_order_cli.py -d RIXS_ZeroOrderScan -r \\
        --dispersion 2.5 --mono-energy 850.0 --format all --export-plots all

    # Minimal run — terminal table only, no plots
    python zeroth_order_cli.py -d path/to/scan --export-plots none --no-focus-curve
"""

# ── Headless Matplotlib backend ───────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')  # MUST be before any pyplot import

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import csv
import json
import os
import sys
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import tifffile

# ── Project-internal (core only — no GUI) ────────────────────────────────────
from rixs_app.core import natural_sort, run_zeroth_order_pipeline
from rixs_app.core.cli_utils import discover_directories, glob_tifs, export_focus_curve
from rixs_app.core.txt_metadata_parser import parse_scan_log


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the zeroth-order CLI.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance ready for
        ``parse_args()`` or ``parse_known_args()``.
    """
    p = argparse.ArgumentParser(
        prog='zeroth_order_cli.py',
        description=(
            'Headless zeroth-order FWHM calibration CLI.  Processes TIFF '
            'scan directories, identifies the sharpest (minimum FWHM) frame, '
            'and exports a focus curve, diagnostic PNGs, and summary reports.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        '-d', '--dir',
        required=True,
        metavar='DIR',
        help='Path to target directory (or parent directory when combined with -r).',
    )
    p.add_argument(
        '-r', '--recursive',
        action='store_true',
        default=False,
        help='Recursively scan subdirectories for TIFF scan datasets.',
    )
    p.add_argument(
        '-t', '--txt',
        default=None,
        metavar='TXT',
        help=(
            'Explicit path to a .txt scan log file.  When omitted, the CLI '
            'auto-discovers the first .txt file (sorted) inside each scan '
            'directory.  The -t flag overrides auto-discovery for all '
            'directories when running in single-directory mode.'
        ),
    )
    p.add_argument(
        '-o', '--output-dir',
        default=None,
        metavar='OUTPUT',
        help=(
            'Output directory name or path.  Defaults to '
            '<scan_dir>/zeroth_order_analysis/.'
        ),
    )
    p.add_argument(
        '--dispersion',
        type=float,
        default=None,
        metavar='MEV_PX',
        help=(
            'Energy dispersion in meV/px (e.g. 2.5).  When supplied, FWHM is '
            'also reported in meV.  Required for resolving-power R calculation.'
        ),
    )
    p.add_argument(
        '--mono-energy',
        type=float,
        default=None,
        metavar='EV',
        help=(
            'Monochromator energy E_mono in eV (e.g. 850.0).  Together with '
            '--dispersion, enables resolving power R = E_mono / FWHM_eV.'
        ),
    )
    focus_group = p.add_mutually_exclusive_group()
    focus_group.add_argument(
        '--plot-focus-curve',
        dest='plot_focus_curve',
        action='store_true',
        default=True,
        help='Generate focus_curve.png (default: enabled).',
    )
    focus_group.add_argument(
        '--no-focus-curve',
        dest='plot_focus_curve',
        action='store_false',
        help='Skip focus_curve.png generation.',
    )
    p.add_argument(
        '--export-plots',
        choices=['best', 'all', 'none'],
        default='best',
        metavar='MODE',
        help=(
            'Diagnostic multi-plot export mode.  '
            '"best" (default) exports a 2×2 diagnostic PNG for the optimal '
            'FWHM frame only.  "all" exports one PNG for every frame.  '
            '"none" skips all per-frame diagnostic PNGs.'
        ),
    )
    p.add_argument(
        '--format',
        choices=['table', 'csv', 'json', 'all'],
        default='table',
        metavar='FMT',
        help=(
            'Summary report format.  "table" (default) prints a formatted '
            'terminal summary.  "csv" writes summary.csv to disk.  '
            '"json" writes summary.json.  "all" does all three.'
        ),
    )
    p.add_argument(
        '--overwrite',
        action='store_true',
        default=False,
        help='Overwrite existing zeroth_order_analysis/ output directory.',
    )
    p.add_argument(
        '-q', '--quiet',
        action='store_true',
        default=False,
        help='Suppress terminal summary output.  Errors are still printed to stderr.',
    )
    return p


# ─────────────────────────────────────────────────────────────────────────────
# TXT auto-discovery
# ─────────────────────────────────────────────────────────────────────────────

def _auto_discover_txt(scan_dir: str) -> str | None:
    """Find the first .txt file (sorted) inside *scan_dir*.

    Args:
        scan_dir: Absolute path to the scan directory.

    Returns:
        Absolute path to the first ``.txt`` file found, or ``None`` when no
        ``.txt`` files are present in *scan_dir*.
    """
    candidates = sorted(
        p for p in Path(scan_dir).iterdir()
        if p.is_file() and p.suffix.lower() == '.txt'
    )
    return str(candidates[0]) if candidates else None


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic multi-plot (2×2 grid, mirrors the GUI export worker layout)
# ─────────────────────────────────────────────────────────────────────────────

def _export_diagnostic_png(
    export_dir: str,
    idx: int,
    result: dict,
) -> str:
    """Render and save a 2×2 diagnostic PNG for a single frame.

    The four-panel layout is identical to the one produced by the GUI export
    worker in :class:`~rixs_app.ui.zeroth_order_slideshow.manager.ZerothOrderManager`:

    - **Top-Left**: Raw image with fitted line overlay.
    - **Top-Right**: Denoised image with line overlay.
    - **Bottom-Left**: Masked gradient image.
    - **Bottom-Right**: 1-D perpendicular-profile plot with score.

    Args:
        export_dir: Directory where the PNG file is written.
        idx: Zero-based frame index used in the output filename.
        result: Pipeline result dict from
            :func:`~rixs_app.core.zeroth_order.run_zeroth_order_pipeline`.

    Returns:
        Absolute path of the saved PNG file.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(10, 8))
    _canvas = FigureCanvasAgg(fig)

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    raw_img = result.get('raw_img')
    denoised = result.get('denoised_img')
    masked = result.get('masked_img')
    P, u = result.get('1d_profile', (np.zeros(1), np.zeros(1)))

    # ── Raw image ──────────────────────────────────────────────────────────
    if raw_img is not None:
        vmin_r = float(np.percentile(raw_img, 0.5))
        vmax_r = float(np.percentile(raw_img, 99.5))
        ax1.imshow(raw_img, cmap='viridis', vmin=vmin_r, vmax=vmax_r, aspect='auto')
        centroid = result.get('centroid')
        direction = result.get('direction')
        if centroid is not None and direction is not None:
            dx, dy = direction
            if abs(dx) > 1e-5:
                ax1.axline(
                    (float(centroid[0]), float(centroid[1])),
                    slope=float(dy / dx),
                    color='red', linestyle='--', linewidth=1.0,
                )
    ax1.set_title('Raw Image')
    ax1.axis('off')

    # ── Denoised image ─────────────────────────────────────────────────────
    if denoised is not None:
        p99 = float(np.percentile(denoised, 99.5))
        if p99 == 0:
            p99 = 1.0
        ax2.imshow(denoised, cmap='viridis', vmin=0, vmax=p99, aspect='auto')
        centroid = result.get('centroid')
        direction = result.get('direction')
        if centroid is not None and direction is not None:
            dx, dy = direction
            if abs(dx) > 1e-5:
                ax2.axline(
                    (float(centroid[0]), float(centroid[1])),
                    slope=float(dy / dx),
                    color='white', linestyle='-', linewidth=1.0,
                )
    else:
        ax2.text(0.5, 0.5, 'No Denoised Image', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Denoised Image')
    ax2.axis('off')

    # ── Masked gradient ────────────────────────────────────────────────────
    if masked is not None:
        p99g = float(np.percentile(masked, 99.9))
        if p99g == 0:
            p99g = 1.0
        ax3.imshow(masked, cmap='viridis', vmin=0, vmax=p99g, aspect='auto')
    else:
        ax3.text(0.5, 0.5, 'No Masked Image', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Masked Gradient')
    ax3.axis('off')

    # ── 1-D profile ────────────────────────────────────────────────────────
    score = result.get('score', float('nan'))
    er = result.get('evaluator_result')
    fwhm_str = ''
    if er is not None and er.fwhm_px is not None:
        fwhm_str = f'  FWHM={er.fwhm_px:.2f}px'
    ax4.plot(u, P, 'k-', linewidth=2, label='1D Profile')
    ax4.set_title(f'1D Profile (Score: {score:.2f}{fwhm_str})')
    ax4.set_xlabel('Perpendicular Distance (u)')
    ax4.set_ylabel('Gradient Sum')
    ax4.legend(fontsize=8)

    fig.tight_layout()
    save_path = os.path.join(export_dir, f'frame_{idx:03d}_diagnostic.png')
    fig.savefig(save_path, dpi=150)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# Single-directory processing
# ─────────────────────────────────────────────────────────────────────────────

def _process_directory(
    scan_dir: str,
    txt_path: str | None,
    output_dir: str | None,
    dispersion: float | None,
    mono_energy: float | None,
    plot_focus_curve: bool,
    export_plots: str,
    fmt: str,
    overwrite: bool,
    quiet: bool,
) -> bool:
    """Run the full zeroth-order calibration pipeline on a single directory.

    Discovers TIF files, optionally parses a scan-log TXT file, runs
    ``run_zeroth_order_pipeline()`` on every frame, and writes outputs to the
    ``zeroth_order_analysis/`` subdirectory.

    Args:
        scan_dir: Absolute path to the directory containing TIFF images.
        txt_path: Explicit path to a scan-log ``.txt`` file, or ``None`` to
            trigger auto-discovery.
        output_dir: Custom output directory path, or ``None`` to default to
            ``<scan_dir>/zeroth_order_analysis/``.
        dispersion: Energy dispersion in meV/px, or ``None`` to skip meV
            reporting and resolving-power calculation.
        mono_energy: Monochromator energy E_mono in eV, or ``None``.
        plot_focus_curve: When ``True``, generate ``focus_curve.png``.
        export_plots: One of ``"best"``, ``"all"``, or ``"none"``.
        fmt: One of ``"table"``, ``"csv"``, ``"json"``, or ``"all"``.
        overwrite: When ``True``, delete and recreate the output directory if
            it already exists.
        quiet: When ``True``, suppress terminal table output.

    Returns:
        ``True`` on success, ``False`` when the directory was skipped (e.g.
        insufficient TIF files, output directory already exists without
        ``--overwrite``).
    """
    dir_name = os.path.basename(scan_dir)

    # ── Discover TIF files ────────────────────────────────────────────────
    file_list = glob_tifs(scan_dir)
    if len(file_list) < 2:
        sys.stderr.write(
            f'[{dir_name}] Warning: fewer than 2 TIF files found — skipping.\n'
        )
        return False

    # ── Resolve output directory ──────────────────────────────────────────
    if output_dir is not None:
        out_dir = str(Path(output_dir).resolve())
    else:
        out_dir = os.path.join(scan_dir, 'zeroth_order_analysis')

    if os.path.exists(out_dir):
        if not overwrite:
            sys.stderr.write(
                f'[{dir_name}] Warning: output directory already exists. '
                f'Use --overwrite to overwrite: {out_dir}\n'
            )
            return False
        import shutil
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    # ── Resolve TXT scan log ──────────────────────────────────────────────
    if txt_path is not None:
        resolved_txt = txt_path
    else:
        resolved_txt = _auto_discover_txt(scan_dir)

    txt_metadata: dict | None = None
    if resolved_txt is not None:
        try:
            txt_metadata = parse_scan_log(resolved_txt)
        except Exception as exc:
            sys.stderr.write(
                f'[{dir_name}] Warning: failed to parse scan log '
                f'"{resolved_txt}": {exc}. Falling back to frame index.\n'
            )
            txt_metadata = None

    # ── Run pipeline on all frames ────────────────────────────────────────
    energy_dispersion = dispersion if dispersion is not None else 0.0
    mono_energy_ev = mono_energy if mono_energy is not None else 0.0

    if not quiet:
        print(f'\n{"=" * 60}')
        print(f'[{dir_name}] Processing {len(file_list)} frames…')
        if txt_metadata is not None:
            print(f'  Scan log : {resolved_txt}')
            print(f'  Motor    : {txt_metadata.get("motor_name", "?")}')
        else:
            print('  Scan log : (not found — using frame index)')
        if dispersion is not None:
            print(f'  Dispersion: {dispersion} meV/px')
        if mono_energy is not None:
            print(f'  E_mono    : {mono_energy} eV')

    frame_records: list[dict] = []
    results_by_idx: dict[int, dict] = {}

    for frame_idx, tif_path in enumerate(file_list):
        filename = os.path.basename(tif_path)
        try:
            raw_img = tifffile.imread(tif_path)
        except Exception as exc:
            sys.stderr.write(
                f'[{dir_name}] Warning: could not read {filename}: {exc}\n'
            )
            frame_records.append({
                'frame_index': frame_idx,
                'filename': filename,
                'motor_position': None,
                'fwhm_px': None,
                'fwhm_mev': None,
                'resolving_power': None,
                'score': None,
                'fit_ok': False,
            })
            continue

        # Ensure 2D
        if raw_img.ndim > 2:
            raw_img = raw_img[0]

        result = run_zeroth_order_pipeline(raw_img, energy_dispersion=energy_dispersion)
        results_by_idx[frame_idx] = result

        er = result.get('evaluator_result')
        fwhm_px = er.fwhm_px if (er is not None and er.fwhm_px is not None) else None
        fit_ok = bool(result.get('fit_ok', False))
        score = float(result.get('score', 0.0))

        # Motor position from TXT metadata
        motor_position: float | None = None
        if txt_metadata is not None:
            fm = txt_metadata['frames'].get(filename)
            if fm is not None:
                motor_position = fm['motor_goal']

        fwhm_mev: float | None = None
        resolving_power: float | None = None
        if fwhm_px is not None and dispersion is not None:
            fwhm_mev = fwhm_px * dispersion
            if mono_energy is not None:
                resolving_power = mono_energy / (fwhm_mev * 1e-3)

        frame_records.append({
            'frame_index': frame_idx,
            'filename': filename,
            'motor_position': motor_position,
            'fwhm_px': fwhm_px,
            'fwhm_mev': fwhm_mev,
            'resolving_power': resolving_power,
            'score': score,
            'fit_ok': fit_ok,
        })

        if not quiet:
            fwhm_disp = f'{fwhm_px:.3f}px' if fwhm_px is not None else 'N/A'
            motor_disp = f'{motor_position:.4f}' if motor_position is not None else f'idx={frame_idx}'
            print(f'  Frame {frame_idx:03d} ({motor_disp}): FWHM={fwhm_disp}  score={score:.1f}')

    # ── Identify best frame ───────────────────────────────────────────────
    valid_records = [r for r in frame_records if r['fwhm_px'] is not None]
    best_record: dict | None = None
    best_idx: int | None = None
    if valid_records:
        best_record = min(valid_records, key=lambda r: r['fwhm_px'])
        best_idx = best_record['frame_index']

    # ── Export focus curve ────────────────────────────────────────────────
    if plot_focus_curve:
        x_values: list[float] = []
        fwhm_list: list[float] = []
        rp_list: list[float | None] = []

        for rec in frame_records:
            if rec['fwhm_px'] is None:
                continue
            if txt_metadata is not None and rec['motor_position'] is not None:
                x_values.append(rec['motor_position'])
            elif txt_metadata is None:
                x_values.append(float(rec['frame_index']))
            else:
                # TXT present but this frame had no motor entry — skip
                continue
            fwhm_list.append(rec['fwhm_px'])
            rp_list.append(rec['resolving_power'])

        if len(x_values) >= 2:
            x_label = (
                txt_metadata.get('motor_name', 'Motor Pitch')
                if txt_metadata is not None
                else 'Frame Index'
            )
            try:
                export_focus_curve(
                    export_dir=out_dir,
                    x_values=x_values,
                    fwhms=fwhm_list,
                    x_label=x_label,
                    energy_dispersion=energy_dispersion,
                    mono_energy_ev=mono_energy_ev,
                    resolving_powers=rp_list,
                )
                if not quiet:
                    print(f'  → focus_curve.png saved.')
            except Exception as exc:
                sys.stderr.write(f'[{dir_name}] Warning: focus curve failed: {exc}\n')

    # ── Export diagnostic PNGs ────────────────────────────────────────────
    if export_plots != 'none':
        indices_to_plot: list[int] = []
        if export_plots == 'best' and best_idx is not None:
            indices_to_plot = [best_idx]
        elif export_plots == 'all':
            indices_to_plot = list(results_by_idx.keys())

        for fi in indices_to_plot:
            if fi in results_by_idx:
                try:
                    _export_diagnostic_png(out_dir, fi, results_by_idx[fi])
                    if not quiet and export_plots == 'best':
                        print(f'  → frame_{fi:03d}_diagnostic.png saved (best frame).')
                except Exception as exc:
                    sys.stderr.write(
                        f'[{dir_name}] Warning: diagnostic PNG for frame {fi} failed: {exc}\n'
                    )

    # ── Export summary ────────────────────────────────────────────────────
    write_csv = fmt in ('csv', 'all')
    write_json = fmt in ('json', 'all')
    print_table = (fmt in ('table', 'all')) and not quiet

    if write_csv:
        csv_path = os.path.join(out_dir, 'summary.csv')
        fieldnames = [
            'frame_index', 'filename', 'motor_position',
            'fwhm_px', 'fwhm_mev', 'resolving_power', 'score', 'fit_ok',
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in frame_records:
                # Write None as empty string for missing values
                row = {k: ('' if v is None else v) for k, v in rec.items()}
                writer.writerow(row)
        if not quiet:
            print(f'  → summary.csv saved.')

    if write_json:
        # Compute optimal motor position from parabolic fit if enough data
        optimal_motor: float | None = None
        if txt_metadata is not None:
            mp_fwhm = [
                (rec['motor_position'], rec['fwhm_px'])
                for rec in frame_records
                if rec['motor_position'] is not None and rec['fwhm_px'] is not None
            ]
            if len(mp_fwhm) >= 3:
                xs = [t[0] for t in mp_fwhm]
                ys = [t[1] for t in mp_fwhm]
                try:
                    coeffs = np.polyfit(xs, ys, 2)
                    if abs(coeffs[0]) > 1e-12:
                        optimal_motor = float(-coeffs[1] / (2 * coeffs[0]))
                except Exception:
                    pass

        best_fwhm_px = best_record['fwhm_px'] if best_record else None
        best_fwhm_mev = best_record['fwhm_mev'] if best_record else None
        best_rp = best_record['resolving_power'] if best_record else None

        summary = {
            'scan_dir': scan_dir,
            'txt_log': resolved_txt,
            'total_frames': len(frame_records),
            'valid_fwhm_count': len(valid_records),
            'best_frame_index': best_idx,
            'best_fwhm_px': best_fwhm_px,
            'best_fwhm_mev': best_fwhm_mev,
            'best_resolving_power': best_rp,
            'optimal_motor_position': optimal_motor,
            'energy_dispersion_mev_per_px': dispersion,
            'mono_energy_ev': mono_energy,
            'frames': frame_records,
        }
        json_path = os.path.join(out_dir, 'summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        if not quiet:
            print(f'  → summary.json saved.')

    # ── Terminal summary ──────────────────────────────────────────────────
    if print_table:
        n_valid = len(valid_records)
        n_total = len(frame_records)
        print(f'\n  ┌─ Summary: {dir_name} {"─" * max(0, 45 - len(dir_name))}┐')
        print(f'  │  Frames processed     : {n_total}')
        print(f'  │  Frames with valid FWHM: {n_valid}')
        if best_record is not None:
            print(f'  │  Best frame index     : {best_idx}')
            bfwhm = best_record["fwhm_px"]
            print(f'  │  Best FWHM            : {bfwhm:.3f} px', end='')
            if best_record["fwhm_mev"] is not None:
                print(f'  /  {best_record["fwhm_mev"]:.3f} meV', end='')
            print()
            if best_record["resolving_power"] is not None:
                print(f'  │  Resolving power R    : {best_record["resolving_power"]:,.0f}')
            if txt_metadata is not None and best_record["motor_position"] is not None:
                mname = txt_metadata.get("motor_name", "Motor")
                print(f'  │  Best motor position  : {best_record["motor_position"]:.4f} ({mname})')
        else:
            print('  │  No frames with valid FWHM found.')
        print(f'  │  Output directory     : {out_dir}')
        print(f'  └{"─" * 52}┘')

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main entry point for the zeroth-order calibration CLI.

    Parses command-line arguments, discovers scan directories (single or
    recursive), and runs :func:`_process_directory` on each one.  Exits with
    a non-zero status code when no directories are found or all fail.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = _build_parser()
    args = parser.parse_args()

    root_dir = str(Path(args.dir).resolve())
    if not os.path.isdir(root_dir):
        sys.stderr.write(f'Error: directory does not exist: {root_dir}\n')
        sys.exit(1)

    # ── Discover scan directories ─────────────────────────────────────────
    skip = {'sum', 'tif-cache', 'zeroth_order_analysis'}
    scan_dirs = discover_directories(root_dir, args.recursive, skip_dirs=skip)

    if not scan_dirs:
        sys.stderr.write(
            f'Error: No directories with ≥2 TIF files found under {root_dir}.\n'
        )
        sys.exit(1)

    if not args.quiet:
        print(f'Found {len(scan_dirs)} scan director{"y" if len(scan_dirs) == 1 else "ies"}.')

    # ── Process each directory ────────────────────────────────────────────
    n_ok = 0
    for scan_dir in scan_dirs:
        ok = _process_directory(
            scan_dir=scan_dir,
            txt_path=args.txt,
            output_dir=args.output_dir,
            dispersion=args.dispersion,
            mono_energy=args.mono_energy,
            plot_focus_curve=args.plot_focus_curve,
            export_plots=args.export_plots,
            fmt=args.format,
            overwrite=args.overwrite,
            quiet=args.quiet,
        )
        if ok:
            n_ok += 1

    if n_ok == 0:
        sys.stderr.write('Error: all directories were skipped or failed.\n')
        sys.exit(1)

    if not args.quiet:
        print(f'\nDone. Processed {n_ok}/{len(scan_dirs)} directories successfully.')


if __name__ == '__main__':
    main()
