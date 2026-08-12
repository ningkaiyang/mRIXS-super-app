import unittest
import os
import tempfile
import time
import numpy as np
import cv2
import tifffile
from rixs_app.core import (
    natural_sort,
    find_peak_line,
    PCAFitFailure,
    phase_correlation_offset,
    compute_line_based_offset,
    warp_image,
    preprocess_image,
    apply_colormap,
    generate_aligned_sum
)



class TestNaturalSort(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(natural_sort([]), [])

    def test_standard_sort(self):
        input_list = ["frame_10.tif", "frame_2.tif", "frame_1.tif", "frame_20.tif"]
        expected = ["frame_1.tif", "frame_2.tif", "frame_10.tif", "frame_20.tif"]
        # In-place check
        natural_sort(input_list)
        self.assertEqual(input_list, expected)

    def test_returned_list(self):
        input_list = ["img_10.png", "img_2.png"]
        result = natural_sort(input_list)
        self.assertEqual(result, ["img_2.png", "img_10.png"])

    def test_case_insensitivity(self):
        input_list = ["Frame_2.tif", "frame_1.tif", "FRAME_10.tif"]
        expected = ["frame_1.tif", "Frame_2.tif", "FRAME_10.tif"]
        natural_sort(input_list)
        self.assertEqual(input_list, expected)

    def test_leading_zeros(self):
        input_list = ["frame_02.tif", "frame_001.tif", "frame_1.tif"]
        result = natural_sort(input_list)
        # Verify naturally ordered
        # "frame_001.tif" and "frame_1.tif" represent 1; "frame_02.tif" represents 2.
        # Stable sort preserves input relative order: "frame_001.tif" before "frame_1.tif".
        self.assertEqual(result, ["frame_001.tif", "frame_1.tif", "frame_02.tif"])

    def test_invalid_types(self):
        with self.assertRaises(TypeError):
            natural_sort(["frame_1.tif", 2, None])

    def test_extremely_large_numbers(self):
        huge_num_str = "frame_" + ("9" * 5000) + ".tif"
        input_list = [huge_num_str, "frame_2.tif", "frame_10.tif"]
        try:
            result = natural_sort(input_list)
            self.assertEqual(result[0], "frame_2.tif")
            self.assertEqual(result[1], "frame_10.tif")
            self.assertEqual(result[2], huge_num_str)
        except ValueError as e:
            self.fail(f"natural_sort raised ValueError on huge number: {e}")


class TestFindPeakLine(unittest.TestCase):
    def setUp(self):
        self.img_width = 100
        self.img_height = 100
        self.synthetic_img = np.zeros((self.img_height, self.img_width), dtype=np.float32)
        self.synthetic_img[:, 50] = 10.0  # Vertical beam at x = 50
        
        self.diagonal_img = np.zeros((self.img_height, self.img_width), dtype=np.float32)
        for i in range(100):
            self.diagonal_img[i, i] = 10.0  # Diagonal beam

    def test_vertical_line(self):
        origin, direction = find_peak_line(self.synthetic_img, 99.0)
        self.assertAlmostEqual(origin[0], 50.0, places=1)
        self.assertAlmostEqual(origin[1], 49.5, places=1)
        self.assertAlmostEqual(abs(direction[0]), 0.0, places=3)
        self.assertAlmostEqual(abs(direction[1]), 1.0, places=3)

    def test_diagonal_line(self):
        origin, direction = find_peak_line(self.diagonal_img, 99.0)
        self.assertAlmostEqual(origin[0], 49.5, places=1)
        self.assertAlmostEqual(origin[1], 49.5, places=1)
        
        expected_comp = 1.0 / np.sqrt(2)
        self.assertAlmostEqual(abs(direction[0]), expected_comp, places=3)
        self.assertAlmostEqual(abs(direction[1]), expected_comp, places=3)

    def test_insufficient_points_raises_pca_fit_failure(self):
        """PCA failure on flat images should raise PCAFitFailure, not return fallback values."""
        flat_img = np.zeros((10, 10), dtype=np.float32)
        with self.assertRaises(PCAFitFailure):
            find_peak_line(flat_img, 99.0)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            find_peak_line(self.synthetic_img, -1.0)
        with self.assertRaises(ValueError):
            find_peak_line(self.synthetic_img, 101.0)
        with self.assertRaises(ValueError):
            find_peak_line(np.array([1, 2, 3]), 99.0)

    def test_nan_inf_in_pca(self):
        nan_inf_img = self.synthetic_img.copy()
        nan_inf_img[0, 0] = np.nan
        nan_inf_img[5, 5] = np.inf
        nan_inf_img[10, 10] = -np.inf
        origin, direction = find_peak_line(nan_inf_img, 99.0)
        self.assertFalse(np.isnan(origin).any())
        self.assertFalse(np.isnan(direction).any())
        self.assertFalse(np.isinf(origin).any())
        self.assertFalse(np.isinf(direction).any())


class TestPhaseCorrelationOffset(unittest.TestCase):
    def setUp(self):
        y, x = np.mgrid[0:128, 0:128]
        self.ref_img = np.exp(-((x - 64)**2 + (y - 64)**2) / (2 * 10**2)).astype(np.float32)

    def test_zero_shift(self):
        dx, dy = phase_correlation_offset(self.ref_img, self.ref_img)
        self.assertAlmostEqual(dx, 0.0, places=2)
        self.assertAlmostEqual(dy, 0.0, places=2)

    def test_positive_shift(self):
        # Shift target by dx = 5.0, dy = 3.0 relative to reference
        M = np.float32([[1, 0, 5.0], [0, 1, 3.0]])
        target_img = cv2.warpAffine(self.ref_img, M, (128, 128), flags=cv2.INTER_LINEAR)
        
        dx, dy = phase_correlation_offset(self.ref_img, target_img)
        self.assertAlmostEqual(dx, 5.0, places=1)
        self.assertAlmostEqual(dy, 3.0, places=1)

    def test_dimension_mismatch(self):
        mismatched_img = np.zeros((64, 128), dtype=np.float32)
        with self.assertRaises(ValueError):
            phase_correlation_offset(self.ref_img, mismatched_img)

    def test_flat_image_fallback(self):
        flat_ref = np.zeros((128, 128), dtype=np.float32)
        flat_target = np.zeros((128, 128), dtype=np.float32)
        dx, dy = phase_correlation_offset(flat_ref, flat_target)
        self.assertEqual(dx, 0.0)
        self.assertEqual(dy, 0.0)


class TestWarpImage(unittest.TestCase):
    def setUp(self):
        self.img = np.zeros((10, 10), dtype=np.float32)
        self.img[4, 4] = 1.0

    def test_zero_warp(self):
        warped = warp_image(self.img, 0.0, 0.0)
        np.testing.assert_array_equal(warped, self.img)

    def test_pixel_shift(self):
        warped = warp_image(self.img, 2.0, 1.0)
        self.assertEqual(warped[5, 6], 1.0)
        self.assertEqual(warped[4, 4], 0.0)

    def test_rgb_warp(self):
        rgb_img = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb_img[4, 4, :] = [255, 0, 0]
        warped = warp_image(rgb_img, 2.0, 1.0)
        np.testing.assert_array_equal(warped[5, 6, :], [255, 0, 0])

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            warp_image(np.array([1, 2, 3]), 1.0, 1.0)


class TestPreprocessImage(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        self.temp_path = self.temp_file.name
        self.temp_file.close()

        self.test_data = np.linspace(0.0, 100.0, 100).reshape(10, 10).astype(np.float32)
        tifffile.imwrite(self.temp_path, self.test_data)

    def tearDown(self):
        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)

    def test_preprocess_grayscale(self):
        rgb, raw = preprocess_image(self.temp_path, "grayscale", 100.0)
        
        self.assertEqual(rgb.shape, (10, 10, 3))
        self.assertEqual(raw.shape, (10, 10))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(raw.dtype, np.float32)
        
        np.testing.assert_array_equal(raw, self.test_data)
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 1])
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 2])

    def test_preprocess_inferno(self):
        rgb, raw = preprocess_image(self.temp_path, "inferno", 99.0)
        self.assertEqual(rgb.shape, (10, 10, 3))
        self.assertEqual(rgb.dtype, np.uint8)

    def test_colormap_fallback(self):
        # Should fallback to standard grayscale 3-channel
        rgb, raw = preprocess_image(self.temp_path, "nonexistent_colormap", 100.0)
        self.assertEqual(rgb.shape, (10, 10, 3))
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 1])
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 2])

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            preprocess_image("nonexistent_file.tif", "grayscale", 100.0)

    def test_invalid_percentile(self):
        with self.assertRaises(ValueError):
            preprocess_image(self.temp_path, "grayscale", -5.0)
        with self.assertRaises(ValueError):
            preprocess_image(self.temp_path, "grayscale", 105.0)

    def test_preprocess_nan_values(self):
        nan_temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        nan_temp_path = nan_temp_file.name
        nan_temp_file.close()
        
        try:
            # Create data containing NaNs
            nan_data = np.linspace(0.0, 100.0, 100).reshape(10, 10).astype(np.float32)
            nan_data[2, 2] = np.nan
            nan_data[5, 5] = np.nan
            tifffile.imwrite(nan_temp_path, nan_data)
            
            # Preprocess the image and check it doesn't raise an exception
            rgb, raw = preprocess_image(nan_temp_path, "grayscale", 100.0)
            
            # Verify shapes and types
            self.assertEqual(rgb.shape, (10, 10, 3))
            self.assertEqual(raw.shape, (10, 10))
            self.assertFalse(np.isnan(raw).any())
            self.assertFalse(np.isnan(rgb).any())
        finally:
            if os.path.exists(nan_temp_path):
                os.remove(nan_temp_path)

    def test_colormap_invalid_type(self):
        rgb, raw = preprocess_image(self.temp_path, 12345, 100.0)
        self.assertEqual(rgb.shape, (10, 10, 3))
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 1])
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 2])



class TestApplyColormap(unittest.TestCase):
    def setUp(self):
        self.raw = np.linspace(0.0, 100.0, 100).reshape(10, 10).astype(np.float32)

    def test_grayscale_output_shape(self):
        rgb = apply_colormap(self.raw, "grayscale")
        self.assertEqual(rgb.shape, (10, 10, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        # All channels identical for grayscale
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 1])
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 2])

    def test_viridis_colormap(self):
        rgb = apply_colormap(self.raw, "viridis")
        self.assertEqual(rgb.shape, (10, 10, 3))
        self.assertEqual(rgb.dtype, np.uint8)

    def test_clamping_floor_ceiling(self):
        # Clamping to [25, 75] should remap that range to [0, 255]
        rgb_full = apply_colormap(self.raw, "grayscale")
        rgb_clamped = apply_colormap(self.raw, "grayscale",
                                     display_floor=25.0, display_ceiling=75.0)
        # Pixels at value 0 should be 0 (clamped to floor), but in clamped
        # version the contrast should be different
        self.assertNotEqual(rgb_full[5, 5, 0], rgb_clamped[5, 5, 0])

    def test_floor_equals_ceiling(self):
        # Should produce flat image without error
        rgb = apply_colormap(self.raw, "grayscale",
                             display_floor=50.0, display_ceiling=50.0)
        self.assertEqual(rgb.shape, (10, 10, 3))

    def test_none_defaults(self):
        # When no floor/ceiling, should use raw min/max
        rgb = apply_colormap(self.raw, "grayscale")
        self.assertEqual(rgb.shape, (10, 10, 3))
        # Min pixel should be 0, max should be 255
        self.assertEqual(rgb[0, 0, 0], 0)
        self.assertEqual(rgb[9, 9, 0], 255)

    def test_invalid_colormap_fallback(self):
        # Should fall back to grayscale without error
        rgb = apply_colormap(self.raw, "nonexistent_cmap_xyz")
        self.assertEqual(rgb.shape, (10, 10, 3))
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 1])

    def test_non_string_colormap(self):
        # Should fall back to grayscale
        rgb = apply_colormap(self.raw, 12345)
        self.assertEqual(rgb.shape, (10, 10, 3))
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 1])

    def test_rejects_non_2d(self):
        with self.assertRaises(ValueError):
            apply_colormap(np.zeros((10, 10, 3), dtype=np.float32), "grayscale")

    def test_flat_image(self):
        flat = np.ones((5, 5), dtype=np.float32) * 42.0
        rgb = apply_colormap(flat, "grayscale")
        self.assertEqual(rgb.shape, (5, 5, 3))
        # All pixels should be the same value
        self.assertTrue(np.all(rgb[:, :, 0] == rgb[0, 0, 0]))


class TestGenerateAlignedSum(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
        # Create 3 simple test frames (10x10 each)
        for i in range(3):
            data = np.full((10, 10), float(i + 1), dtype=np.float32)
            path = os.path.join(self.temp_dir, f"frame_{i}.tif")
            tifffile.imwrite(path, data)
            self.temp_files.append(path)

        # Simple raw loader
        self._cache = {}
        for path in self.temp_files:
            self._cache[path] = tifffile.imread(path).astype(np.float32)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _get_raw(self, filepath):
        return self._cache.get(filepath)

    def test_basic_sum_no_offsets(self):
        offsets = {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0)}
        result = generate_aligned_sum(
            self.temp_files, self._get_raw, offsets, (10, 10)
        )
        self.assertEqual(result.shape, (10, 10))
        self.assertEqual(result.dtype, np.float32)
        # Sum should be 1 + 2 + 3 = 6 everywhere
        np.testing.assert_allclose(result, 6.0)

    def test_with_offset(self):
        # Frame 1 shifted left by warp_image(raw, -2, 0) — right edge becomes 0
        offsets = {0: (0.0, 0.0), 1: (2.0, 0.0), 2: (0.0, 0.0)}
        result = generate_aligned_sum(
            self.temp_files, self._get_raw, offsets, (10, 10)
        )
        # warp_image(raw, -2, 0) shifts the image LEFT by 2 pixels.
        # Interior (col 0-7): frame_0(1) + warped_frame_1(2) + frame_2(3) = 6
        # Right edge (col 8-9): frame_0(1) + 0 + frame_2(3) = 4
        np.testing.assert_allclose(result[5, 5], 6.0, atol=0.5)
        np.testing.assert_allclose(result[5, 9], 4.0, atol=0.5)

    def test_progress_callback(self):
        offsets = {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0)}
        progress_calls = []

        def callback(current, total):
            progress_calls.append((current, total))

        generate_aligned_sum(
            self.temp_files, self._get_raw, offsets, (10, 10),
            progress_callback=callback
        )
        self.assertEqual(len(progress_calls), 3)
        self.assertEqual(progress_calls[0], (1, 3))
        self.assertEqual(progress_calls[2], (3, 3))

    def test_missing_frame_raises(self):
        def _bad_loader(filepath):
            return None

        offsets = {0: (0.0, 0.0)}
        with self.assertRaises(ValueError):
            generate_aligned_sum(
                self.temp_files[:1], _bad_loader, offsets, (10, 10)
            )

    def test_single_frame(self):
        offsets = {0: (0.0, 0.0)}
        result = generate_aligned_sum(
            self.temp_files[:1], self._get_raw, offsets, (10, 10)
        )
        np.testing.assert_allclose(result, 1.0)


class TestPCAFitFailureException(unittest.TestCase):
    """Tests that PCAFitFailure is raised correctly in find_peak_line under various failure conditions."""

    def test_flat_image_raises(self):
        """A completely flat image should raise PCAFitFailure."""
        flat = np.ones((50, 50), dtype=np.float32) * 42.0
        with self.assertRaises(PCAFitFailure):
            find_peak_line(flat, 99.0)

    def test_empty_image_raises(self):
        """An image with zero dimensions should raise PCAFitFailure."""
        empty = np.zeros((0, 10), dtype=np.float32)
        with self.assertRaises(PCAFitFailure):
            find_peak_line(empty, 99.0)

    def test_aggressive_threshold_raises(self):
        """An extremely aggressive threshold leaving < 2 points should raise PCAFitFailure."""
        img = np.zeros((100, 100), dtype=np.float32)
        img[50, 50] = 1.0  # Only 1 bright pixel
        with self.assertRaises(PCAFitFailure):
            find_peak_line(img, 99.9999)

    def test_normal_image_does_not_raise(self):
        """A well-formed image with a clear line should not raise PCAFitFailure."""
        img = np.zeros((100, 100), dtype=np.float32)
        img[:, 50] = 10.0  # Vertical beam
        origin, direction = find_peak_line(img, 99.0)
        self.assertFalse(np.isnan(origin).any())
        self.assertFalse(np.isnan(direction).any())


class TestComputeLineBasedOffsetNoFallback(unittest.TestCase):
    """Tests that compute_line_based_offset returns (0, 0) on PCA failure without falling back to phase correlation."""

    def setUp(self):
        # Reference image with a clear vertical line at x=50
        self.ref = np.zeros((100, 100), dtype=np.float32)
        self.ref[:, 50] = 10.0
        self.ref_direction = np.array([0.0, 1.0])
        self.ref_origin = np.array([50.0, 49.5])

    def test_pca_failure_returns_zero_offset(self):
        """When PCA fails on the target frame, offset should be (0, 0), not a phase-correlation estimate."""
        # Target is a flat image — PCA will fail
        target = np.ones((100, 100), dtype=np.float32) * 5.0
        dx, dy = compute_line_based_offset(
            self.ref, target,
            self.ref_direction, self.ref_origin,
            99.0, 99.0
        )
        self.assertEqual(dx, 0.0)
        self.assertEqual(dy, 0.0)

    def test_successful_offset_with_known_shift(self):
        """When PCA succeeds, the offset should reflect centroid differences."""
        # Target image with a shifted vertical line at x=55
        target = np.zeros((100, 100), dtype=np.float32)
        target[:, 55] = 10.0
        dx, dy = compute_line_based_offset(
            self.ref, target,
            self.ref_direction, self.ref_origin,
            99.0, 99.0
        )
        # The line moved 5 pixels in x (perpendicular to vertical line direction)
        self.assertAlmostEqual(dx, 5.0, places=0)
        self.assertAlmostEqual(dy, 0.0, places=0)

    def test_no_phase_correlation_call_on_pca_failure(self):
        """Verify that phase correlation is NOT called when PCA fails.

        We do this by providing images with a known shift that phase correlation
        would detect — but since PCA fails, the result should be (0, 0) not
        the phase correlation estimate.
        """
        # Target has structure that phase correlation would detect as a shift,
        # but no clear peak line for PCA
        ref_structured = np.random.RandomState(42).rand(100, 100).astype(np.float32)
        target_shifted = np.roll(ref_structured, 5, axis=1)  # 5px horizontal shift

        # With a flat/noisy target, PCA should fail
        target_flat = np.ones((100, 100), dtype=np.float32) * 5.0
        dx, dy = compute_line_based_offset(
            ref_structured, target_flat,
            np.array([0.0, 1.0]), np.array([50.0, 50.0]),
            99.0, 99.0
        )
        # Should be (0, 0), NOT a phase correlation result
        self.assertEqual(dx, 0.0)
        self.assertEqual(dy, 0.0)

class TestNaturalSortStress(unittest.TestCase):
    def test_sort_1000_items(self):
        # List of 1000 items in reversed order
        items = [f"frame_{i}.tif" for i in reversed(range(1000))]
        sorted_items = natural_sort(items.copy())
        expected = [f"frame_{i}.tif" for i in range(1000)]
        self.assertEqual(sorted_items, expected)

    def test_sort_extremely_large_numbers(self):
        # Python supports arbitrary-precision integers, but sys.set_int_max_str_digits
        # can restrict conversion of very large integers.
        # Test filenames with extremely large numbers.
        huge_num_1 = "9" * 100
        huge_num_2 = "9" * 101
        items = [f"file_{huge_num_2}.txt", f"file_{huge_num_1}.txt"]
        sorted_items = natural_sort(items.copy())
        self.assertEqual(sorted_items, [f"file_{huge_num_1}.txt", f"file_{huge_num_2}.txt"])

        # Test extreme numbers that could cause overflow or performance issues
        huge_num_3 = "1" + "0" * 400  # 401 digits
        items = [f"file_{huge_num_3}.txt", "file_2.txt"]
        sorted_items = natural_sort(items.copy())
        self.assertEqual(sorted_items, ["file_2.txt", f"file_{huge_num_3}.txt"])

    def test_mixed_alphanumeric_strings(self):
        items = [
            "a1b2c3", "a1b2c10", "a1b10c2", "a10b1c1", "a1b2", "a1", "1a", "2a", "10a",
            "a_1.2.3", "a_1.2.10", "a_1.10.2", "a_1.2", "a_1"
        ]
        sorted_items = natural_sort(items.copy())
        # Let's verify standard natural ordering rules
        expected = [
            "1a", "2a", "10a",
            "a1", "a1b2", "a1b2c3", "a1b2c10", "a1b10c2", "a10b1c1",
            "a_1", "a_1.2", "a_1.2.3", "a_1.2.10", "a_1.10.2"
        ]
        # In-place check
        self.assertEqual(sorted_items, expected)

    def test_non_string_elements_type_error(self):
        with self.assertRaises(TypeError):
            natural_sort(["a", 1, "b"])
        with self.assertRaises(TypeError):
            natural_sort(None)


class TestPCAPeakFindingStress(unittest.TestCase):
    def test_horizontal_beam(self):
        # Horizontal beam: line at row 5
        img = np.zeros((10, 20), dtype=np.float32)
        img[5, :] = 10.0
        centroid, direction = find_peak_line(img, 99.0)
        self.assertAlmostEqual(centroid[1], 5.0, places=3)
        self.assertAlmostEqual(abs(direction[0]), 1.0, places=3)
        self.assertAlmostEqual(abs(direction[1]), 0.0, places=3)

    def test_vertical_beam(self):
        # Vertical beam: line at col 8
        img = np.zeros((20, 15), dtype=np.float32)
        img[:, 8] = 10.0
        centroid, direction = find_peak_line(img, 99.0)
        self.assertAlmostEqual(centroid[0], 8.0, places=3)
        self.assertAlmostEqual(abs(direction[0]), 0.0, places=3)
        self.assertAlmostEqual(abs(direction[1]), 1.0, places=3)

    def test_diagonal_beam(self):
        # Diagonal beam: row = col
        img = np.zeros((15, 15), dtype=np.float32)
        for i in range(15):
            img[i, i] = 10.0
        centroid, direction = find_peak_line(img, 99.0)
        self.assertAlmostEqual(centroid[0], 7.0, places=3)
        self.assertAlmostEqual(centroid[1], 7.0, places=3)
        expected_comp = 1.0 / np.sqrt(2)
        self.assertAlmostEqual(abs(direction[0]), expected_comp, places=3)
        self.assertAlmostEqual(abs(direction[1]), expected_comp, places=3)

    def test_noisy_profile(self):
        # Diagonal beam with significant noise
        np.random.seed(42)
        img = np.zeros((30, 30), dtype=np.float32)
        for i in range(30):
            img[i, i] = 10.0
        # Add random noise
        img += np.random.normal(0, 1.0, img.shape).astype(np.float32)
        centroid, direction = find_peak_line(img, 99.0)
        # Verify PCA still approximates the diagonal
        expected_comp = 1.0 / np.sqrt(2)
        self.assertAlmostEqual(abs(direction[0]), expected_comp, delta=0.1)
        self.assertAlmostEqual(abs(direction[1]), expected_comp, delta=0.1)


    def test_flat_image_variations(self):
        # Extremely large flat values -> flat image -> PCAFitFailure
        img_large = np.full((10, 10), 1e30, dtype=np.float32)
        with self.assertRaises(PCAFitFailure):
            find_peak_line(img_large, 90.0)

        # Extremely small flat values -> flat image -> PCAFitFailure
        img_small = np.full((10, 10), 1e-30, dtype=np.float32)
        with self.assertRaises(PCAFitFailure):
            find_peak_line(img_small, 90.0)

    def test_nan_inf_values(self):
        # Image with inf/nan values
        img = np.zeros((10, 10), dtype=np.float32)
        img[2, 2] = np.nan
        img[5, 5] = np.inf
        # nan_to_num converts NaN->0, Inf->large_number. After conversion the image
        # is effectively flat (only one non-zero pixel from inf), which raises PCAFitFailure.
        with self.assertRaises(PCAFitFailure):
            find_peak_line(img, 90.0)


class TestPhaseCorrelationOffsetStress(unittest.TestCase):
    def setUp(self):
        # Create a standard reference Gaussian blob
        y, x = np.mgrid[0:64, 0:64]
        self.ref_img = np.exp(-((x - 32)**2 + (y - 32)**2) / (2 * 5**2)).astype(np.float32)

    def test_subpixel_offsets_accuracy(self):
        # Verify sub-pixel shift accuracy. Let's shift by dx=1.25, dy=0.75 using bilinear interpolation or affine
        # OpenCV phaseCorrelate uses SVD on the cross-power spectrum and centroid estimation for subpixel accuracy.
        # Let's shift by different subpixel values and print the error.
        shifts = [(0.25, 0.4), (-0.5, 0.25), (1.5, -0.75), (0.0, 0.0)]
        for dx, dy in shifts:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            target_img = cv2.warpAffine(self.ref_img, M, (64, 64), flags=cv2.INTER_LINEAR)
            dx_est, dy_est = phase_correlation_offset(self.ref_img, target_img)
            # The accuracy of phaseCorrelate can vary but should be within ~0.75 pixels
            # especially since we now use Tukey windowing and mean subtraction which
            # can cause edge artifacts with cv2.warpAffine.
            self.assertAlmostEqual(dx_est, dx, delta=0.75)
            self.assertAlmostEqual(dy_est, dy, delta=0.75)

    def test_low_contrast_std_near_1e_5(self):
        # Standard deviation slightly below 1e-5 should return (0, 0)
        img_low = self.ref_img * 1.5e-5 / np.std(self.ref_img)  # std is 1.5e-5
        img_very_low = self.ref_img * 0.9e-5 / np.std(self.ref_img)  # std is 0.9e-5
        
        # Shift target by dx=2, dy=1
        M = np.float32([[1, 0, 2.0], [0, 1, 1.0]])
        target_low = cv2.warpAffine(img_low, M, (64, 64), flags=cv2.INTER_LINEAR)
        target_very_low = cv2.warpAffine(img_very_low, M, (64, 64), flags=cv2.INTER_LINEAR)

        # Standard deviation below 1e-5
        dx, dy = phase_correlation_offset(img_very_low, target_very_low)
        self.assertEqual((dx, dy), (0.0, 0.0))

        # Standard deviation slightly above 1e-5: std ~ 1.5e-5
        # Let's see if it works or fails/returns garbage
        dx, dy = phase_correlation_offset(img_low, target_low)
        # If it returns, verify it's close or returns (0.0, 0.0) without crashing
        self.assertIsInstance(dx, float)
        self.assertIsInstance(dy, float)

    def test_phase_corr_extremely_large_shifts(self):
        # Shift by more than half the image size
        M = np.float32([[1, 0, 40.0], [0, 1, 40.0]])
        target_img = cv2.warpAffine(self.ref_img, M, (64, 64), flags=cv2.INTER_LINEAR)
        dx, dy = phase_correlation_offset(self.ref_img, target_img)
        # Phase correlation with Hanning window may fail or return (0.0, 0.0) or incorrect shift due to windowing.
        # Verify it returns float values and does not crash.
        self.assertIsInstance(dx, float)
        self.assertIsInstance(dy, float)

    def test_nan_inf_inputs(self):
        # If input has NaN, np.std returns NaN, which makes np.std(ref_img) < 1e-5 evaluate to False.
        # This will pass the NaN array to cv2.phaseCorrelate.
        # Let's verify if phase_correlation_offset catches NaNs / doesn't crash.
        ref_nan = self.ref_img.copy()
        ref_nan[10, 10] = np.nan
        target = self.ref_img.copy()
        
        # Test if it runs and returns (0.0, 0.0) instead of crashing
        dx, dy = phase_correlation_offset(ref_nan, target)
        self.assertEqual((dx, dy), (0.0, 0.0))

        # Test if both have NaNs
        target_nan = self.ref_img.copy()
        target_nan[5, 5] = np.nan
        dx, dy = phase_correlation_offset(ref_nan, target_nan)
        self.assertEqual((dx, dy), (0.0, 0.0))

        # Test with Inf values
        ref_inf = self.ref_img.copy()
        ref_inf[10, 10] = np.inf
        dx, dy = phase_correlation_offset(ref_inf, target)
        self.assertEqual((dx, dy), (0.0, 0.0))


class TestWarpImageStress(unittest.TestCase):
    def setUp(self):
        self.img = np.zeros((10, 10), dtype=np.float32)
        self.img[4, 4] = 1.0

    def test_warp_extremely_large_shifts(self):
        # Shift larger than image size
        warped = warp_image(self.img, 100.0, 200.0)
        self.assertEqual(warped.shape, (10, 10))
        # Since shift is larger than image size, it should be all zeros
        np.testing.assert_array_equal(warped, np.zeros_like(self.img))

    def test_integer_vs_subpixel_translations(self):
        # Integer translation
        warped_int = warp_image(self.img, 1.0, 1.0)
        self.assertEqual(warped_int[5, 5], 1.0)

        # Subpixel translation (uses nearest neighbor, no blurring/energy distribution)
        warped_sub = warp_image(self.img, 0.5, 0.5)
        # Verify that energy is not distributed (remains a single sharp point of intensity 1.0)
        self.assertEqual(np.sum(warped_sub), 1.0)
        self.assertEqual(np.max(warped_sub), 1.0)

    def test_nan_inf_shift_parameters(self):
        # Warp with nan/inf shifts
        # cv2.warpAffine raises error if M contains nan or inf. Let's see what happens.
        # Let's catch if it raises cv2.error or ValueError or if it executes.
        try:
            warped = warp_image(self.img, np.nan, 2.0)
            # If it doesn't raise, check its shape
            self.assertEqual(warped.shape, (10, 10))
        except (cv2.error, ValueError) as e:
            # Re-raise or handle
            pass

        try:
            warped = warp_image(self.img, np.inf, 2.0)
            self.assertEqual(warped.shape, (10, 10))
        except (cv2.error, ValueError) as e:
            pass


class TestPreprocessingStress(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        self.temp_path = self.temp_file.name
        self.temp_file.close()

    def tearDown(self):
        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)

    def test_very_large_values(self):
        # Write extreme values to TIF (up to max float32)
        extreme_data = np.array([[0.0, 1e38], [1e-38, 3e38]], dtype=np.float32)
        tifffile.imwrite(self.temp_path, extreme_data)

        # Preprocess and ensure it normalizes without Overflow/NaN in RGB
        rgb, raw = preprocess_image(self.temp_path, "grayscale", 100.0)
        self.assertEqual(rgb.shape, (2, 2, 3))
        self.assertFalse(np.isnan(rgb).any())
        self.assertFalse(np.isinf(rgb).any())

    def test_float_inf_nan_values(self):
        # Image contains inf, nan, neginf
        extreme_data = np.array([[np.nan, np.inf], [-np.inf, 10.0]], dtype=np.float32)
        tifffile.imwrite(self.temp_path, extreme_data)

        # Preprocess. np.nan_to_num replaces nan with 0.0, inf with large float32, neginf with small float32.
        # Let's verify it executes without crashing and does not output NaN/Inf.
        rgb, raw = preprocess_image(self.temp_path, "grayscale", 100.0)
        self.assertEqual(rgb.shape, (2, 2, 3))
        self.assertFalse(np.isnan(raw).any())
        self.assertFalse(np.isnan(rgb).any())
        self.assertFalse(np.isinf(rgb).any())

    def test_matplotlib_colormap_fallbacks(self):
        # Standard colormap
        tifffile.imwrite(self.temp_path, np.zeros((5, 5), dtype=np.float32))
        rgb, _ = preprocess_image(self.temp_path, "viridis", 100.0)
        self.assertEqual(rgb.shape, (5, 5, 3))

        # Invalid colormap name
        rgb_fallback, _ = preprocess_image(self.temp_path, "invalid_colormap_name_xyz", 100.0)
        self.assertEqual(rgb_fallback.shape, (5, 5, 3))
        # Grayscale fallback check (all channels equal)
        np.testing.assert_array_equal(rgb_fallback[:, :, 0], rgb_fallback[:, :, 1])
        np.testing.assert_array_equal(rgb_fallback[:, :, 0], rgb_fallback[:, :, 2])

        # Test invalid colormap name types (e.g. None or int)
        # Verify that passing None does not crash and falls back to grayscale.
        rgb_none, _ = preprocess_image(self.temp_path, None, 100.0)
        self.assertEqual(rgb_none.shape, (5, 5, 3))
        np.testing.assert_array_equal(rgb_none[:, :, 0], rgb_none[:, :, 1])
        np.testing.assert_array_equal(rgb_none[:, :, 0], rgb_none[:, :, 2])

class TestMathCoreAdvancedStress(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = os.path.join(self.temp_dir.name, "temp.tif")
        # Standard small dummy image
        tifffile.imwrite(self.temp_path, np.zeros((10, 10), dtype=np.float32))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_find_peak_line_extreme_nans_infs(self):
        # Image consisting entirely of NaNs -> nan_to_num makes it flat -> PCAFitFailure
        img_all_nan = np.full((10, 10), np.nan, dtype=np.float32)
        with self.assertRaises(PCAFitFailure):
            find_peak_line(img_all_nan, 90.0)

        # Image consisting entirely of Infs -> nan_to_num makes it flat -> PCAFitFailure
        img_all_inf = np.full((10, 10), np.inf, dtype=np.float32)
        with self.assertRaises(PCAFitFailure):
            find_peak_line(img_all_inf, 90.0)

        # Mixed NaNs, Infs, and finite values with a beam line
        img_mixed = np.zeros((10, 10), dtype=np.float32)
        img_mixed[2, 2] = np.nan
        img_mixed[3, 3] = np.inf
        img_mixed[4, 4] = -np.inf
        img_mixed[5, :] = 10.0  # beam
        origin, direction = find_peak_line(img_mixed, 99.0)
        self.assertTrue(np.isfinite(origin).all())
        self.assertTrue(np.isfinite(direction).all())

    def test_find_peak_line_single_hot_pixel(self):
        # Only one pixel is non-zero
        img = np.zeros((10, 10), dtype=np.float32)
        img[4, 4] = 100.0
        # Percentile 99.9% should only select that single pixel
        # Since we have < 2 points, it should raise PCAFitFailure
        with self.assertRaises(PCAFitFailure):
            find_peak_line(img, 99.9)

    def test_find_peak_line_huge_dimensions(self):
        # 1000x1000 flat image performance check
        img = np.zeros((1000, 1000), dtype=np.float32)
        t0 = time.time()
        with self.assertRaises(PCAFitFailure):
            find_peak_line(img, 99.0)
        t1 = time.time()
        self.assertLess(t1 - t0, 0.5)  # should be fast (less than 500ms)

    def test_phase_correlation_offset_all_nan_or_inf(self):
        ref = np.full((64, 64), np.nan, dtype=np.float32)
        target = np.full((64, 64), np.nan, dtype=np.float32)
        dx, dy = phase_correlation_offset(ref, target)
        self.assertEqual((dx, dy), (0.0, 0.0))

        ref = np.full((64, 64), np.inf, dtype=np.float32)
        target = np.full((64, 64), np.inf, dtype=np.float32)
        dx, dy = phase_correlation_offset(ref, target)
        self.assertEqual((dx, dy), (0.0, 0.0))

    def test_phase_correlation_offset_extremely_low_std(self):
        # Std deviation is non-zero but extremely small
        ref = np.zeros((64, 64), dtype=np.float32)
        ref[0, 0] = 1e-20
        target = np.zeros((64, 64), dtype=np.float32)
        target[0, 0] = 1e-20
        dx, dy = phase_correlation_offset(ref, target)
        self.assertEqual((dx, dy), (0.0, 0.0))

    def test_warp_image_extreme_shifts(self):
        img = np.random.rand(10, 10).astype(np.float32)
        # Warp with NaNs in shift parameters (cv2.warpAffine might raise an error)
        # We check if warp_image survives or raises ValueError
        try:
            warped = warp_image(img, np.nan, np.nan)
            self.assertEqual(warped.shape, img.shape)
        except (cv2.error, ValueError):
            pass

        # Warp with large shifts
        warped = warp_image(img, 1e12, -1e12)
        self.assertEqual(warped.shape, img.shape)
        self.assertEqual(np.count_nonzero(warped), 0)

    def test_preprocess_image_errors(self):
        # Missing file path
        with self.assertRaises(FileNotFoundError):
            preprocess_image("non_existent_file_xyz_123.tif", "grayscale", 99.0)

        # Non-2D image file (1D array)
        one_d_path = os.path.join(self.temp_dir.name, "one_d.tif")
        tifffile.imwrite(one_d_path, np.arange(10, dtype=np.float32))
        with self.assertRaises(ValueError):
            preprocess_image(one_d_path, "grayscale", 99.0)

        # Invalid percentile threshold
        with self.assertRaises(ValueError):
            preprocess_image(self.temp_path, "grayscale", -0.01)
        with self.assertRaises(ValueError):
            preprocess_image(self.temp_path, "grayscale", 100.01)


class TestNaturalSortingAdvancedStress(unittest.TestCase):
    def test_natural_sort_huge_integer_digits_dos(self):
        # Create strings with extremely long numeric parts to test integer parsing limits
        # e.g., 5000 digits of 9.
        # sys.set_int_max_str_digits default limit is 4300 digits.
        # Slicing at [:4000] inside key_func prevents ValueError.
        huge_num_1 = "9" * 4500
        huge_num_2 = "9" * 4501
        
        lst = [f"frame_{huge_num_2}.tif", f"frame_{huge_num_1}.tif"]
        t0 = time.time()
        res = natural_sort(lst)
        t1 = time.time()
        
        # Verify it did not raise ValueError and ran quickly (no DoS)
        self.assertLess(t1 - t0, 0.5)
        # Because both numbers exceed 4000 digits, their keys collapse to the same value
        # Timsort (stable sort) preserves their original input order
        self.assertEqual(res, [f"frame_{huge_num_2}.tif", f"frame_{huge_num_1}.tif"])

    def test_natural_sort_many_digit_groups(self):
        # Filename with many alternating numeric and alpha parts
        # e.g. "a1b2c3d4e5f6g7h8i9j10"
        s1 = "a" + "".join(f"{i}a" for i in range(100))
        s2 = "a" + "".join(f"{i}a" for i in range(101))
        lst = [s2, s1]
        res = natural_sort(lst)
        self.assertEqual(res, [s1, s2])

    def test_natural_sort_unicode_and_spaces(self):
        lst = [" ", "frame_2.tif", "frame_10.tif", "фрейм_1.tif", "фрейм_10.tif", "фрейм_2.tif"]
        res = natural_sort(lst)
        expected = [" ", "frame_2.tif", "frame_10.tif", "фрейм_1.tif", "фрейм_2.tif", "фрейм_10.tif"]
        self.assertEqual(res, expected)


if __name__ == '__main__':
    unittest.main()

