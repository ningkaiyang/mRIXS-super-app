"""Unit tests for the new zeroth_order_cli.py.

Tests cover argument parsing, single-directory execution, TXT metadata
auto-discovery, recursive batch mode, output format flags, plot controls,
and overwrite protection.

All tests use temporary directories with synthetic 100×100 TIFF images and
mock the heavyweight pipeline call to keep the suite fast.
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import tifffile


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_scan_dir(parent: Path, name: str = 'scan', n_frames: int = 5) -> Path:
    """Create a scan directory with *n_frames* synthetic TIFF files.

    Args:
        parent: Parent directory under which the scan directory is created.
        name: Subdirectory name.
        n_frames: Number of TIFF files to create.

    Returns:
        Path to the created scan directory.
    """
    scan = parent / name
    scan.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        img = np.zeros((100, 100), dtype=np.float32)
        img[40:60, 48:52] = float(100 + i)  # synthetic line
        tifffile.imwrite(str(scan / f'Marana_{i:03d}.tiff'), img)
    return scan


def _make_txt_log(scan_dir: Path, n_frames: int = 5) -> Path:
    """Write a minimal RIXS scan log TXT file compatible with parse_scan_log().

    Args:
        scan_dir: Directory where the .txt file is placed.
        n_frames: Number of data rows to write.

    Returns:
        Path to the written .txt file.
    """
    txt_path = scan_dir / 'scan_log.txt'
    header = (
        'Single Motor Scan\n'
        'Date: 2024-01-01\n'
        'SM3 Mirror Pitch\n'          # line 3 (0-indexed) — motor name
        'Start: 7.80\n'
        'Stop: 8.20\n'
        'Increment: 0.10\n'
        'Description: test scan\n'
        '\n\n\n\n\n\n\n'             # lines 7-13 (padding to 14 header lines)
    )
    # Column header line (index 14)
    col_header = '\t'.join(
        [f'Col{i}' for i in range(2)]
        + ['SM3 Mirror Pitch Goal', 'SM3 Mirror Pitch Actual']
        + [f'Col{i}' for i in range(4, 45)]
        + ['Marana']
    )
    lines = [header.rstrip('\n')]
    lines.append(col_header)
    for i in range(n_frames):
        motor = 7.80 + i * 0.10
        row_cols = ['0.0', '0.0', str(motor), str(motor)]
        row_cols += ['0.0'] * 41
        row_cols.append(f'.\\scan\\Marana_{i:03d}.tiff')
        lines.append('\t'.join(row_cols))
    txt_path.write_text('\n'.join(lines), encoding='utf-8')
    return txt_path


def _make_mock_pipeline_result(fwhm_px: float = 3.5) -> dict:
    """Build a minimal mock result dict compatible with _process_directory."""
    er = MagicMock()
    er.score_valid = True
    er.fwhm_px = fwhm_px
    img = np.zeros((100, 100), dtype=np.float32)
    return {
        'raw_img': img,
        'denoised_img': img,
        'masked_img': img,
        'grad_img': img,
        'centroid': np.array([50.0, 50.0]),
        'direction': np.array([1.0, 0.0]),
        '1d_profile': (np.ones(81), np.arange(-40, 41, dtype=float)),
        'score': fwhm_px * 10,
        'evaluator_result': er,
        'fit_ok': True,
        'fwhm_mev': None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing tests
# ─────────────────────────────────────────────────────────────────────────────

class TestArgumentParsing(unittest.TestCase):
    """Validate that argparse defaults and flag combinations work correctly."""

    def _parse(self, args: list[str]):
        """Import the CLI's parser and parse *args*."""
        # Import from the top-level module
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from zeroth_order_cli import _build_parser
        return _build_parser().parse_args(args)

    def test_required_dir_flag(self):
        """-d / --dir is required; omitting it raises SystemExit."""
        from zeroth_order_cli import _build_parser
        parser = _build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_default_values(self):
        """Default flag values match the spec."""
        args = self._parse(['-d', '/tmp'])
        self.assertFalse(args.recursive)
        self.assertIsNone(args.txt)
        self.assertIsNone(args.output_dir)
        self.assertIsNone(args.dispersion)
        self.assertIsNone(args.mono_energy)
        self.assertTrue(args.plot_focus_curve)
        self.assertEqual(args.export_plots, 'best')
        self.assertEqual(args.format, 'table')
        self.assertFalse(args.overwrite)
        self.assertFalse(args.quiet)

    def test_no_focus_curve_flag(self):
        """--no-focus-curve disables focus curve generation."""
        args = self._parse(['-d', '/tmp', '--no-focus-curve'])
        self.assertFalse(args.plot_focus_curve)

    def test_all_format_flag(self):
        """--format all is accepted."""
        args = self._parse(['-d', '/tmp', '--format', 'all'])
        self.assertEqual(args.format, 'all')

    def test_export_plots_none(self):
        """--export-plots none is accepted."""
        args = self._parse(['-d', '/tmp', '--export-plots', 'none'])
        self.assertEqual(args.export_plots, 'none')

    def test_dispersion_and_mono_energy(self):
        """--dispersion and --mono-energy parse to floats."""
        args = self._parse(['-d', '/tmp', '--dispersion', '2.5', '--mono-energy', '850.0'])
        self.assertAlmostEqual(args.dispersion, 2.5)
        self.assertAlmostEqual(args.mono_energy, 850.0)

    def test_recursive_flag(self):
        """-r / --recursive sets recursive=True."""
        args = self._parse(['-d', '/tmp', '-r'])
        self.assertTrue(args.recursive)


# ─────────────────────────────────────────────────────────────────────────────
# Single-directory execution tests (with mocked pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleDirectory(unittest.TestCase):
    """Integration tests for _process_directory() with a mocked pipeline."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.scan_dir = _make_scan_dir(self.root, n_frames=5)

    def tearDown(self):
        self.tmp.cleanup()

    @patch('zeroth_order_cli.run_zeroth_order_pipeline')
    def test_creates_output_directory(self, mock_pipeline):
        """_process_directory() creates zeroth_order_analysis/ inside the scan dir."""
        from zeroth_order_cli import _process_directory
        mock_pipeline.side_effect = [_make_mock_pipeline_result(float(3 + i)) for i in range(5)]
        ok = _process_directory(
            scan_dir=str(self.scan_dir),
            txt_path=None, output_dir=None,
            dispersion=None, mono_energy=None,
            plot_focus_curve=False, export_plots='none',
            fmt='table', overwrite=False, quiet=True,
        )
        self.assertTrue(ok)
        self.assertTrue((self.scan_dir / 'zeroth_order_analysis').is_dir())

    @patch('zeroth_order_cli.run_zeroth_order_pipeline')
    def test_focus_curve_generated_without_txt(self, mock_pipeline):
        """focus_curve.png is generated using Frame Index when no .txt is present."""
        from zeroth_order_cli import _process_directory
        mock_pipeline.side_effect = [_make_mock_pipeline_result(float(5 - i)) for i in range(5)]
        _process_directory(
            scan_dir=str(self.scan_dir),
            txt_path=None, output_dir=None,
            dispersion=None, mono_energy=None,
            plot_focus_curve=True, export_plots='none',
            fmt='table', overwrite=False, quiet=True,
        )
        self.assertTrue((self.scan_dir / 'zeroth_order_analysis' / 'focus_curve.png').exists())

    @patch('zeroth_order_cli.run_zeroth_order_pipeline')
    def test_focus_curve_generated_with_txt(self, mock_pipeline):
        """focus_curve.png is generated using motor positions when .txt is present."""
        from zeroth_order_cli import _process_directory
        _make_txt_log(self.scan_dir, n_frames=5)
        mock_pipeline.side_effect = [_make_mock_pipeline_result(float(5 - i)) for i in range(5)]
        _process_directory(
            scan_dir=str(self.scan_dir),
            txt_path=None, output_dir=None,
            dispersion=None, mono_energy=None,
            plot_focus_curve=True, export_plots='none',
            fmt='table', overwrite=False, quiet=True,
        )
        self.assertTrue((self.scan_dir / 'zeroth_order_analysis' / 'focus_curve.png').exists())

    @patch('zeroth_order_cli.run_zeroth_order_pipeline')
    def test_best_frame_diagnostic_png_exported(self, mock_pipeline):
        """--export-plots best creates one diagnostic PNG for the best frame."""
        from zeroth_order_cli import _process_directory
        mock_pipeline.side_effect = [_make_mock_pipeline_result(float(5 - i)) for i in range(5)]
        _process_directory(
            scan_dir=str(self.scan_dir),
            txt_path=None, output_dir=None,
            dispersion=None, mono_energy=None,
            plot_focus_curve=False, export_plots='best',
            fmt='table', overwrite=False, quiet=True,
        )
        out = self.scan_dir / 'zeroth_order_analysis'
        pngs = list(out.glob('frame_*_diagnostic.png'))
        self.assertEqual(len(pngs), 1)

    @patch('zeroth_order_cli.run_zeroth_order_pipeline')
    def test_all_frames_diagnostic_pngs_exported(self, mock_pipeline):
        """--export-plots all creates one diagnostic PNG per frame."""
        from zeroth_order_cli import _process_directory
        n = 5
        mock_pipeline.side_effect = [_make_mock_pipeline_result(float(3 + i)) for i in range(n)]
        _process_directory(
            scan_dir=str(self.scan_dir),
            txt_path=None, output_dir=None,
            dispersion=None, mono_energy=None,
            plot_focus_curve=False, export_plots='all',
            fmt='table', overwrite=False, quiet=True,
        )
        out = self.scan_dir / 'zeroth_order_analysis'
        pngs = list(out.glob('frame_*_diagnostic.png'))
        self.assertEqual(len(pngs), n)

    @patch('zeroth_order_cli.run_zeroth_order_pipeline')
    def test_no_diagnostic_pngs_when_none(self, mock_pipeline):
        """--export-plots none creates no diagnostic PNGs."""
        from zeroth_order_cli import _process_directory
        mock_pipeline.side_effect = [_make_mock_pipeline_result(float(3 + i)) for i in range(5)]
        _process_directory(
            scan_dir=str(self.scan_dir),
            txt_path=None, output_dir=None,
            dispersion=None, mono_energy=None,
            plot_focus_curve=False, export_plots='none',
            fmt='table', overwrite=False, quiet=True,
        )
        out = self.scan_dir / 'zeroth_order_analysis'
        pngs = list(out.glob('frame_*_diagnostic.png'))
        self.assertEqual(len(pngs), 0)

    @patch('zeroth_order_cli.run_zeroth_order_pipeline')
    def test_no_focus_curve_when_disabled(self, mock_pipeline):
        """--no-focus-curve skips focus_curve.png generation."""
        from zeroth_order_cli import _process_directory
        mock_pipeline.side_effect = [_make_mock_pipeline_result(float(3 + i)) for i in range(5)]
        _process_directory(
            scan_dir=str(self.scan_dir),
            txt_path=None, output_dir=None,
            dispersion=None, mono_energy=None,
            plot_focus_curve=False, export_plots='none',
            fmt='table', overwrite=False, quiet=True,
        )
        out = self.scan_dir / 'zeroth_order_analysis'
        self.assertFalse((out / 'focus_curve.png').exists())


# ─────────────────────────────────────────────────────────────────────────────
# Output format tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputFormats(unittest.TestCase):
    """Tests for summary.csv and summary.json output."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.scan_dir = _make_scan_dir(self.root, n_frames=4)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, fmt: str, extra_kwargs: dict | None = None) -> Path:
        """Run _process_directory with --format *fmt*."""
        from zeroth_order_cli import _process_directory
        kw = dict(
            scan_dir=str(self.scan_dir),
            txt_path=None, output_dir=None,
            dispersion=2.5, mono_energy=850.0,
            plot_focus_curve=False, export_plots='none',
            fmt=fmt, overwrite=False, quiet=True,
        )
        if extra_kwargs:
            kw.update(extra_kwargs)

        n = 4
        with patch('zeroth_order_cli.run_zeroth_order_pipeline') as mock_p:
            mock_p.side_effect = [_make_mock_pipeline_result(float(4 - i)) for i in range(n)]
            _process_directory(**kw)
        return self.scan_dir / 'zeroth_order_analysis'

    def test_csv_file_created(self):
        """--format csv writes summary.csv with correct columns."""
        out = self._run('csv')
        csv_path = out / 'summary.csv'
        self.assertTrue(csv_path.exists())
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 4)
        for col in ('frame_index', 'filename', 'fwhm_px', 'fit_ok'):
            self.assertIn(col, rows[0])

    def test_json_file_created(self):
        """--format json writes summary.json with correct schema."""
        out = self._run('json')
        json_path = out / 'summary.json'
        self.assertTrue(json_path.exists())
        with open(json_path) as f:
            data = json.load(f)
        self.assertIn('total_frames', data)
        self.assertIn('valid_fwhm_count', data)
        self.assertIn('best_frame_index', data)
        self.assertIn('frames', data)
        self.assertEqual(data['total_frames'], 4)
        # All frames should have valid FWHM given our mock
        self.assertEqual(data['valid_fwhm_count'], 4)

    def test_all_format_creates_both_files(self):
        """--format all writes both summary.csv and summary.json."""
        out = self._run('all')
        self.assertTrue((out / 'summary.csv').exists())
        self.assertTrue((out / 'summary.json').exists())

    def test_table_format_no_csv_or_json(self):
        """--format table (default) does NOT write summary.csv or summary.json."""
        out = self._run('table')
        self.assertFalse((out / 'summary.csv').exists())
        self.assertFalse((out / 'summary.json').exists())

    def test_json_best_frame_index_is_correct(self):
        """summary.json best_frame_index points to the frame with the lowest FWHM."""
        # Mock gives fwhm [4.0, 3.0, 2.0, 1.0] so best_frame_index=3
        out = self._run('json')
        with open(out / 'summary.json') as f:
            data = json.load(f)
        self.assertEqual(data['best_frame_index'], 3)


# ─────────────────────────────────────────────────────────────────────────────
# Overwrite protection test
# ─────────────────────────────────────────────────────────────────────────────

class TestOverwriteProtection(unittest.TestCase):
    """Tests for --overwrite flag behavior."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.scan_dir = _make_scan_dir(self.root, n_frames=3)

    def tearDown(self):
        self.tmp.cleanup()

    def test_skip_when_output_exists_without_overwrite(self):
        """_process_directory() returns False if output_dir exists and --overwrite is not set."""
        from zeroth_order_cli import _process_directory
        out = self.scan_dir / 'zeroth_order_analysis'
        out.mkdir()  # pre-create
        ok = _process_directory(
            scan_dir=str(self.scan_dir),
            txt_path=None, output_dir=None,
            dispersion=None, mono_energy=None,
            plot_focus_curve=False, export_plots='none',
            fmt='table', overwrite=False, quiet=True,
        )
        self.assertFalse(ok)

    @patch('zeroth_order_cli.run_zeroth_order_pipeline')
    def test_overwrites_when_flag_set(self, mock_pipeline):
        """_process_directory() succeeds when output_dir exists and --overwrite is set."""
        from zeroth_order_cli import _process_directory
        out = self.scan_dir / 'zeroth_order_analysis'
        out.mkdir()
        # place a sentinel file inside
        (out / 'old_file.txt').write_text('stale')
        mock_pipeline.side_effect = [_make_mock_pipeline_result(float(3 - i)) for i in range(3)]
        ok = _process_directory(
            scan_dir=str(self.scan_dir),
            txt_path=None, output_dir=None,
            dispersion=None, mono_energy=None,
            plot_focus_curve=False, export_plots='none',
            fmt='table', overwrite=True, quiet=True,
        )
        self.assertTrue(ok)
        # Sentinel file should have been removed
        self.assertFalse((out / 'old_file.txt').exists())


# ─────────────────────────────────────────────────────────────────────────────
# Recursive batch test
# ─────────────────────────────────────────────────────────────────────────────

class TestRecursiveBatch(unittest.TestCase):
    """Integration test for CLI recursive mode via subprocess."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # Two scan subdirectories
        self.scan_a = _make_scan_dir(self.root, 'scan_a', n_frames=3)
        self.scan_b = _make_scan_dir(self.root, 'scan_b', n_frames=3)

    def tearDown(self):
        self.tmp.cleanup()

    def test_recursive_creates_outputs_in_both_dirs(self):
        """Running with -r creates zeroth_order_analysis/ in each scan subdirectory."""
        cli = str(Path(__file__).parent.parent / 'zeroth_order_cli.py')
        cmd = [
            sys.executable, cli,
            '-d', str(self.root),
            '-r',
            '--export-plots', 'none',
            '--no-focus-curve',
            '--format', 'json',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        # The CLI must exit cleanly
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        # Both scan directories should have their output
        self.assertTrue((self.scan_a / 'zeroth_order_analysis' / 'summary.json').exists())
        self.assertTrue((self.scan_b / 'zeroth_order_analysis' / 'summary.json').exists())


if __name__ == '__main__':
    unittest.main()
