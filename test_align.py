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

    def test_insufficient_points_fallback(self):
        flat_img = np.zeros((10, 10), dtype=np.float32)
        origin, direction = find_peak_line(flat_img, 99.0)
        
        self.assertEqual(origin[0], 5.0)
        self.assertEqual(origin[1], 5.0)
        self.assertEqual(direction[0], 1.0)
        self.assertEqual(direction[1], 0.0)

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


if __name__ == "__main__":
    unittest.main()
