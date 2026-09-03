"""Unit tests for rixs_app/core/cli_utils.py.

Tests cover directory discovery, TIF globbing, and focus-curve plot generation.
All tests are independent of local git-untracked files and use temporary
directories with synthetic TIFF images.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile


class TestDiscoverDirectories(unittest.TestCase):
    """Tests for discover_directories()."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_tifs(self, directory: Path, count: int = 3) -> None:
        """Create *count* dummy TIF files inside *directory*."""
        directory.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            tifffile.imwrite(str(directory / f'frame_{i:03d}.tif'),
                             np.zeros((10, 10), dtype=np.uint16))

    def test_single_directory_mode_root_has_tifs(self):
        """Non-recursive mode: returns root when root contains TIF files."""
        from rixs_app.core.cli_utils import discover_directories
        self._make_tifs(self.root)
        result = discover_directories(str(self.root), recursive=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], str(self.root.resolve()))

    def test_single_directory_mode_no_tifs(self):
        """Non-recursive mode: returns empty list when root has no TIF files."""
        from rixs_app.core.cli_utils import discover_directories
        result = discover_directories(str(self.root), recursive=False)
        self.assertEqual(result, [])

    def test_recursive_discovers_subdirectories(self):
        """Recursive mode: discovers subdirectories with TIF files."""
        from rixs_app.core.cli_utils import discover_directories
        sub1 = self.root / 'scan_001'
        sub2 = self.root / 'scan_002'
        self._make_tifs(sub1)
        self._make_tifs(sub2)
        result = discover_directories(str(self.root), recursive=True)
        self.assertIn(str(sub1.resolve()), result)
        self.assertIn(str(sub2.resolve()), result)

    def test_recursive_skips_default_dirs(self):
        """Recursive mode: skips 'sum', 'tif-cache', 'clusters', 'zeroth_order_analysis', and 'denoised' by default."""
        from rixs_app.core.cli_utils import discover_directories
        for dname in ['sum', 'tif-cache', 'clusters', 'zeroth_order_analysis', 'denoised']:
            self._make_tifs(self.root / dname)
        self._make_tifs(self.root / 'valid_scan')
        result = discover_directories(str(self.root), recursive=True)
        basenames = [os.path.basename(p) for p in result]
        for dname in ['sum', 'tif-cache', 'clusters', 'zeroth_order_analysis', 'denoised']:
            self.assertNotIn(dname, basenames)
        self.assertIn('valid_scan', basenames)

    def test_recursive_skips_custom_dirs(self):
        """Recursive mode: skips directories listed in skip_dirs parameter."""
        from rixs_app.core.cli_utils import discover_directories
        self._make_tifs(self.root / 'zeroth_order_analysis')
        self._make_tifs(self.root / 'real_scan')
        result = discover_directories(
            str(self.root),
            recursive=True,
            skip_dirs={'sum', 'tif-cache', 'zeroth_order_analysis'},
        )
        basenames = [os.path.basename(p) for p in result]
        self.assertNotIn('zeroth_order_analysis', basenames)
        self.assertIn('real_scan', basenames)

    def test_result_is_sorted(self):
        """discover_directories() returns results in sorted order."""
        from rixs_app.core.cli_utils import discover_directories
        for name in ['scan_c', 'scan_a', 'scan_b']:
            self._make_tifs(self.root / name)
        result = discover_directories(str(self.root), recursive=True)
        self.assertEqual(result, sorted(result))

    def test_directory_with_only_one_tif_is_excluded(self):
        """Directories with fewer than 2 TIFs are not returned."""
        from rixs_app.core.cli_utils import discover_directories
        lonely = self.root / 'lone'
        lonely.mkdir()
        tifffile.imwrite(str(lonely / 'single.tif'),
                         np.zeros((10, 10), dtype=np.uint16))
        result = discover_directories(str(self.root), recursive=True)
        self.assertNotIn(str(lonely.resolve()), result)


class TestGlobTifs(unittest.TestCase):
    """Tests for glob_tifs()."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_only_tif_files(self):
        """glob_tifs() returns only .tif/.tiff files, ignoring others."""
        from rixs_app.core.cli_utils import glob_tifs
        tifffile.imwrite(str(self.root / 'a.tif'), np.zeros((5, 5), dtype=np.uint16))
        tifffile.imwrite(str(self.root / 'b.tiff'), np.zeros((5, 5), dtype=np.uint16))
        (self.root / 'readme.txt').write_text('not a tif')
        result = glob_tifs(str(self.root))
        self.assertEqual(len(result), 2)
        for p in result:
            self.assertTrue(p.lower().endswith(('.tif', '.tiff')))

    def test_natural_sort_order(self):
        """glob_tifs() returns files in natural alphanumeric order."""
        from rixs_app.core.cli_utils import glob_tifs
        for name in ['frame_9.tif', 'frame_10.tif', 'frame_2.tif']:
            tifffile.imwrite(str(self.root / name), np.zeros((5, 5), dtype=np.uint16))
        result = glob_tifs(str(self.root))
        basenames = [os.path.basename(p) for p in result]
        self.assertEqual(basenames, ['frame_2.tif', 'frame_9.tif', 'frame_10.tif'])

    def test_returns_empty_for_empty_directory(self):
        """glob_tifs() returns an empty list for a directory with no TIFs."""
        from rixs_app.core.cli_utils import glob_tifs
        result = glob_tifs(str(self.root))
        self.assertEqual(result, [])


class TestExportFocusCurve(unittest.TestCase):
    """Tests for export_focus_curve()."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.export_dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _make_x_fwhm(self, n: int = 5):
        x = list(np.linspace(7.8, 8.2, n))
        fwhm = [5.0 - 0.5 * abs(v - 8.0) for v in x]
        return x, fwhm

    def test_generates_focus_curve_png_with_motor(self):
        """export_focus_curve() creates focus_curve.png for motor-position data."""
        from rixs_app.core.cli_utils import export_focus_curve
        x, fwhm = self._make_x_fwhm()
        out = export_focus_curve(
            export_dir=self.export_dir,
            x_values=x,
            fwhms=fwhm,
            x_label='SM3 Mirror Pitch',
        )
        self.assertTrue(os.path.exists(out))
        self.assertTrue(out.endswith('focus_curve.png'))

    def test_generates_focus_curve_png_with_frame_index(self):
        """export_focus_curve() creates focus_curve.png for frame-index data."""
        from rixs_app.core.cli_utils import export_focus_curve
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        fwhm = [6.0, 5.0, 4.0, 5.5, 6.5]
        out = export_focus_curve(
            export_dir=self.export_dir,
            x_values=x,
            fwhms=fwhm,
            x_label='Frame Index',
        )
        self.assertTrue(os.path.exists(out))

    def test_focus_curve_png_nonzero_size(self):
        """The generated PNG should have non-zero file size."""
        from rixs_app.core.cli_utils import export_focus_curve
        x, fwhm = self._make_x_fwhm()
        out = export_focus_curve(self.export_dir, x, fwhm, 'Motor')
        self.assertGreater(os.path.getsize(out), 1000)

    def test_raises_on_fewer_than_two_points(self):
        """export_focus_curve() raises ValueError when fewer than 2 data points."""
        from rixs_app.core.cli_utils import export_focus_curve
        with self.assertRaises(ValueError):
            export_focus_curve(self.export_dir, [7.9], [5.0], 'Motor')

    def test_raises_on_mismatched_lengths(self):
        """export_focus_curve() raises ValueError when x_values and fwhms differ in length."""
        from rixs_app.core.cli_utils import export_focus_curve
        with self.assertRaises(ValueError):
            export_focus_curve(self.export_dir, [1.0, 2.0], [5.0], 'Motor')

    def test_resolving_power_annotation_with_dispersion(self):
        """Focus curve is generated without error when dispersion and mono_energy are provided."""
        from rixs_app.core.cli_utils import export_focus_curve
        x, fwhm = self._make_x_fwhm()
        out = export_focus_curve(
            export_dir=self.export_dir,
            x_values=x,
            fwhms=fwhm,
            x_label='SM3 Mirror Pitch',
            energy_dispersion=2.5,
            mono_energy_ev=850.0,
        )
        self.assertTrue(os.path.exists(out))

    def test_two_point_fallback_no_parabolic_fit(self):
        """With exactly 2 data points, focus_curve.png is generated (no parabolic fit)."""
        from rixs_app.core.cli_utils import export_focus_curve
        out = export_focus_curve(
            export_dir=self.export_dir,
            x_values=[7.9, 8.1],
            fwhms=[5.0, 4.5],
            x_label='Frame Index',
        )
        self.assertTrue(os.path.exists(out))

    def test_precomputed_resolving_powers_accepted(self):
        """Pre-computed resolving_powers list is accepted and used correctly."""
        from rixs_app.core.cli_utils import export_focus_curve
        x, fwhm = self._make_x_fwhm()
        rp = [100000.0 + i * 5000 for i in range(len(x))]
        out = export_focus_curve(
            export_dir=self.export_dir,
            x_values=x,
            fwhms=fwhm,
            x_label='Motor',
            resolving_powers=rp,
        )
        self.assertTrue(os.path.exists(out))


if __name__ == '__main__':
    unittest.main()
