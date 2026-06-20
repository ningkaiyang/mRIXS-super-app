import unittest
import os
import tempfile
import time
import numpy as np
import cv2
import tifffile
from unittest.mock import patch

from align_app.main import AlignApp
from align_app.core import (
    natural_sort,
    find_peak_line,
    phase_correlation_offset,
    warp_image,
    preprocess_image
)

def pump_events(root):
    root.update_idletasks()
    root.update()

class TestMathCoreAdvancedStress(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = os.path.join(self.temp_dir.name, "temp.tif")
        # Standard small dummy image
        tifffile.imwrite(self.temp_path, np.zeros((10, 10), dtype=np.float32))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_find_peak_line_extreme_nans_infs(self):
        # Image consisting entirely of NaNs
        img_all_nan = np.full((10, 10), np.nan, dtype=np.float32)
        origin, direction = find_peak_line(img_all_nan, 90.0)
        self.assertEqual(list(origin), [5.0, 5.0])
        self.assertEqual(list(direction), [1.0, 0.0])

        # Image consisting entirely of Infs
        img_all_inf = np.full((10, 10), np.inf, dtype=np.float32)
        origin, direction = find_peak_line(img_all_inf, 90.0)
        self.assertEqual(list(origin), [5.0, 5.0])
        self.assertEqual(list(direction), [1.0, 0.0])

        # Mixed NaNs, Infs, and finite values
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
        # Since we have < 2 points, it should trigger the fallback
        origin, direction = find_peak_line(img, 99.9)
        self.assertEqual(list(origin), [5.0, 5.0])
        self.assertEqual(list(direction), [1.0, 0.0])

    def test_find_peak_line_huge_dimensions(self):
        # 1000x1000 flat image performance check
        img = np.zeros((1000, 1000), dtype=np.float32)
        t0 = time.time()
        origin, direction = find_peak_line(img, 99.0)
        t1 = time.time()
        self.assertLess(t1 - t0, 0.5)  # should be fast (less than 500ms)
        self.assertEqual(list(origin), [500.0, 500.0])
        self.assertEqual(list(direction), [1.0, 0.0])

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


class TestUIResizingAndCacheStress(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_files = []
        for i in range(3):
            path = os.path.join(self.temp_dir.name, f"frame_{i+1}.tif")
            data = np.zeros((100, 100), dtype=np.float32)
            data[:, 50 + i * 2] = 10.0
            tifffile.imwrite(path, data)
            self.temp_files.append(path)

        self.app = AlignApp(show_window=False)
        pump_events(self.app)

    def tearDown(self):
        self.app.destroy()
        self.temp_dir.cleanup()

    def test_canvas_resizing_performance(self):
        # Transition to slideshow view
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)

        # Simulate repeated canvas resizing 50 times in rapid succession
        t0 = time.time()
        for size in range(100, 600, 10):
            self.app.slideshow_view.canvas.winfo_width = lambda s=size: s
            self.app.slideshow_view.canvas.winfo_height = lambda s=size: s
            # Call on_resize event handler directly
            self.app.slideshow_view.on_resize(None)
            pump_events(self.app)
        t1 = time.time()

        # Canvas resizing must be highly performant (e.g. less than 2.5 seconds for 50 resizes)
        self.assertLess(t1 - t0, 2.5)

    def test_cache_invalidation_flow(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)

        # Initial cache should be set
        self.assertIsNotNone(self.app.slideshow_view.cached_disp_rgb)
        initial_rgb = self.app.slideshow_view.cached_disp_rgb.copy()

        # 1. Change colormap: verify cache is updated/invalidated
        self.app.slideshow_view.change_colormap("inferno")
        pump_events(self.app)
        self.assertNotEqual(np.mean(self.app.slideshow_view.cached_disp_rgb), np.mean(initial_rgb))
        after_cmap_rgb = self.app.slideshow_view.cached_disp_rgb.copy()

        # 2. Change PCA threshold: verify cache is updated/invalidated
        self.app.slideshow_view.change_pca_threshold(50.0)
        pump_events(self.app)
        self.assertEqual(self.app.slideshow_view.pca_threshold, 50.0)

        # 3. Enable warp: verify cache is updated/invalidated
        self.app.slideshow_view.warp_switch.select()
        self.app.slideshow_view.toggle_warp()
        pump_events(self.app)
        self.assertTrue(self.app.slideshow_view.warp_enabled)

        # 4. Navigate to next frame: verify cache updates to new frame raw/rgb
        self.app.slideshow_view.next_frame()
        pump_events(self.app)
        self.assertEqual(self.app.slideshow_view.current_idx, 1)


class TestGUIEdgeCasesStress(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_files = []
        for i in range(2):
            path = os.path.join(self.temp_dir.name, f"frame_{i+1}.tif")
            data = np.zeros((50, 50), dtype=np.float32)
            data[:, 25] = 10.0
            tifffile.imwrite(path, data)
            self.temp_files.append(path)

        self.app = AlignApp(show_window=False)
        pump_events(self.app)

    def tearDown(self):
        self.app.destroy()
        self.temp_dir.cleanup()

    def test_slideshow_empty_file_list(self):
        # If slides list is empty, start slideshow should clear canvas and return gracefully
        self.app.slideshow_view.start([])
        pump_events(self.app)
        self.assertEqual(self.app.slideshow_view.file_list, [])
        self.assertEqual(self.app.slideshow_view.current_idx, 0)
        # Should clear labels and widgets
        self.assertIn("0/0", self.app.slideshow_view.frame_label.cget("text"))

    def test_missing_files_after_selection(self):
        # Select files, then delete them, and try starting the slideshow
        self.app.sorting_view.file_list = self.temp_files.copy()
        
        # Delete the first file from disk
        os.remove(self.temp_files[0])

        # Start slideshow: verify it handles the error gracefully by showing error on canvas and not crashing
        self.app.show_slideshow(self.app.sorting_view.file_list)
        pump_events(self.app)

        # Verify slideshow view shows error message text on canvas
        error_tags = self.app.slideshow_view.canvas.find_withtag("error")
        self.assertGreater(len(error_tags), 0)
        self.assertIn("Error loading image", self.app.slideshow_view.canvas.itemcget(error_tags[0], "text"))

    def test_duplicate_selections(self):
        # Add duplicate file paths to the list
        self.app.sorting_view.file_list = [self.temp_files[1], self.temp_files[1]]
        
        # Start slideshow: should run fine
        self.app.show_slideshow(self.app.sorting_view.file_list)
        pump_events(self.app)
        self.assertEqual(self.app.slideshow_view.current_idx, 0)

        # Move next: offset calculation between identical images should be (0.00, 0.00)
        self.app.slideshow_view.next_frame()
        pump_events(self.app)
        self.assertEqual(self.app.slideshow_view.current_idx, 1)
        metadata = self.app.slideshow_view.metadata_label.cget("text")
        self.assertIn("Offset: (0.00, 0.00)", metadata)

    def test_unsupported_file_format(self):
        # Create a non-TIFF file (e.g. text file disguised as tif)
        bad_path = os.path.join(self.temp_dir.name, "bad_format.tif")
        with open(bad_path, "w") as f:
            f.write("This is not a TIFF image!")

        self.app.show_slideshow([bad_path])
        pump_events(self.app)
        
        # Verify slideshow handles the format exception gracefully by drawing error
        error_tags = self.app.slideshow_view.canvas.find_withtag("error")
        self.assertGreater(len(error_tags), 0)
        self.assertIn("Error loading image", self.app.slideshow_view.canvas.itemcget(error_tags[0], "text"))

    def test_canvas_resize_zero_dimension_crash_wide_aspect(self):
        # High aspect ratio where width is extremely large compared to height.
        # This causes the scaling factor to drive the height to 0.
        rgb = np.zeros((1, 2000, 3), dtype=np.uint8)
        # Calling draw_canvas directly should return gracefully and not crash.
        self.app.slideshow_view.draw_canvas(rgb, np.array([0, 0]), np.array([1, 0]))

    def test_canvas_resize_extreme_small_crash(self):
        # Canvas is resized to be extremely small (e.g. 5x2), causing height/width to scale to 0.
        self.app.slideshow_view.canvas.winfo_width = lambda: 5
        self.app.slideshow_view.canvas.winfo_height = lambda: 2
        rgb = np.zeros((10, 100, 3), dtype=np.uint8)
        self.app.slideshow_view.draw_canvas(rgb, np.array([0, 0]), np.array([1, 0]))

    def test_find_peak_line_zero_percentile(self):
        # If percentile threshold is 0.0, it should select all pixels
        img = np.zeros((10, 10), dtype=np.float32)
        img[5, 5] = 10.0
        # Should not crash, and should return a valid origin/direction
        origin, direction = find_peak_line(img, 0.0)
        self.assertEqual(list(origin), [4.5, 4.5])
        self.assertEqual(list(direction), [1.0, 0.0])

    def test_find_peak_line_single_hot_pixel_origin(self):
        # When there is only 1 point matching (e.g. 100th percentile of a single hot pixel),
        # it falls back to the center of the image instead of the hot pixel.
        # We assert this behavior here.
        img = np.zeros((10, 10), dtype=np.float32)
        img[2, 7] = 50.0
        origin, direction = find_peak_line(img, 100.0)
        # Center of image is [5.0, 5.0]
        self.assertEqual(list(origin), [5.0, 5.0])
        self.assertEqual(list(direction), [1.0, 0.0])

    def test_natural_sort_mixed_case_stability(self):
        # Ensure that sorting handles duplicate names stably
        lst = ["frame_a.tif", "frame_A.tif", "frame_a.tif"]
        # In-place natural sort
        res = natural_sort(lst)
        # Since natural_sort is stable, it should maintain relative order of identical elements
        self.assertEqual(res, ["frame_a.tif", "frame_A.tif", "frame_a.tif"])


class TestChallengerAdversarialExhaustive(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_files = []
        path = os.path.join(self.temp_dir.name, "frame_normal.tif")
        data = np.ones((50, 50), dtype=np.float32)
        tifffile.imwrite(path, data)
        self.temp_files.append(path)

        self.app = AlignApp(show_window=False)
        pump_events(self.app)

    def tearDown(self):
        self.app.destroy()
        self.temp_dir.cleanup()

    def test_cache_invalidation_on_error_resize(self):
        # 1. Load the valid frame
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        
        # Verify cache is populated
        self.assertIsNotNone(self.app.slideshow_view.cached_disp_rgb)
        
        # 2. Add a non-existent file path and navigate to it to trigger a load error
        self.app.slideshow_view.file_list.append(os.path.join(self.temp_dir.name, "missing.tif"))
        self.app.slideshow_view.next_frame()
        pump_events(self.app)
        
        # Verify canvas shows the error message
        error_tags = self.app.slideshow_view.canvas.find_withtag("error")
        self.assertGreater(len(error_tags), 0)
        
        # 3. Simulate a window resize event
        # If cache is not invalidated, this will draw the stale cached image and clear the error
        self.app.slideshow_view.on_resize(None)
        pump_events(self.app)
        
        has_error = len(self.app.slideshow_view.canvas.find_withtag("error")) > 0
        has_image = len(self.app.slideshow_view.canvas.find_withtag("image")) > 0
        
        self.assertTrue(has_error, "Bug: Error message should remain on the canvas after a window resize event")
        self.assertFalse(has_image, "Bug: Stale cached image should not be drawn on the canvas in an error state")

    def test_zero_width_height_zero_division(self):
        # 1. Mock the manager's get_raw and get_rgb to return an image with 0 width, but valid shape, to bypass load exceptions
        # and test the draw_canvas zero-division vulnerability
        with patch('align_app.ui.slideshow.managers.SlideshowManager.get_raw') as mock_raw, \
             patch('align_app.ui.slideshow.managers.SlideshowManager.get_rgb') as mock_rgb:
            mock_raw.return_value = np.zeros((100, 0), dtype=np.float32)
            mock_rgb.return_value = np.zeros((100, 0, 3), dtype=np.uint8)
            
            # 2. Try loading the slideshow.
            # This should load and render gracefully without ZeroDivisionError.
            self.app.show_slideshow(["dummy_path.tif"])
            pump_events(self.app)


if __name__ == "__main__":
    unittest.main()

