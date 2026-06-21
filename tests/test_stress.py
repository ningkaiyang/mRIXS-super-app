#!/usr/bin/env python3
import unittest
import os
import tempfile
import numpy as np
import cv2
import tifffile

from align_app.core import (
    natural_sort,
    find_peak_line,
    phase_correlation_offset,
    warp_image,
    preprocess_image
)

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
        # Extremely large flat values
        img_large = np.full((10, 10), 1e30, dtype=np.float32)
        centroid, direction = find_peak_line(img_large, 90.0)
        self.assertEqual(list(centroid), [5.0, 5.0])
        self.assertEqual(list(direction), [1.0, 0.0])

        # Extremely small flat values
        img_small = np.full((10, 10), 1e-30, dtype=np.float32)
        centroid, direction = find_peak_line(img_small, 90.0)
        self.assertEqual(list(centroid), [5.0, 5.0])
        self.assertEqual(list(direction), [1.0, 0.0])

    def test_nan_inf_values(self):
        # Image with inf/nan values
        img = np.zeros((10, 10), dtype=np.float32)
        img[2, 2] = np.nan
        img[5, 5] = np.inf
        # Test if find_peak_line crashes
        # NumPy's np.percentile with np.nan returns np.nan, which causes image_data >= threshold_val
        # to return all False. That triggers fallback. Let's verify fallback is returned.
        centroid, direction = find_peak_line(img, 90.0)
        self.assertEqual(list(centroid), [5.0, 5.0])
        self.assertEqual(list(direction), [1.0, 0.0])


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
            # The accuracy of phaseCorrelate can vary but should be within ~0.2 pixels
            self.assertAlmostEqual(dx_est, dx, delta=0.25)
            self.assertAlmostEqual(dy_est, dy, delta=0.25)

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

    def test_extremely_large_shifts(self):
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

    def test_extremely_large_shifts(self):
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

if __name__ == "__main__":
    unittest.main()
