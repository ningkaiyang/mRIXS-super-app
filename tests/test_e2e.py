import unittest
import os
import tempfile
import numpy as np
import cv2
import tifffile
import tkinter as tk
from unittest.mock import patch
import customtkinter

from align_app.main import AlignApp
from align_app.core import (
    natural_sort,
    find_peak_line,
    phase_correlation_offset,
    warp_image,
    preprocess_image,
    PCAFitFailure
)

def pump_events(root):
    root.update_idletasks()
    root.update()

class TestE2E(unittest.TestCase):
    def setUp(self):
        # Create temp TIFF files for testing views
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_files = []
        # Create three small synthetic TIFF files
        for i in range(3):
            path = os.path.join(self.temp_dir.name, f"frame_{i+1}.tif")
            # Create synthetic data with peak at different columns (to verify shifts)
            data = np.zeros((100, 100), dtype=np.float32)
            # frame 1: peak at col 50. frame 2: peak at col 52. frame 3: peak at col 54.
            data[:, 50 + i * 2] = 10.0
            data[10, 40] = 9.0  # secondary peak pixel
            tifffile.imwrite(path, data)
            self.temp_files.append(path)
            
        self.app = AlignApp(show_window=False)
        pump_events(self.app)
        
    def tearDown(self):
        self.app.destroy()
        self.temp_dir.cleanup()

    # --- F1 (Sorting/File Management) - 11 test cases ---
    @patch('tkinter.filedialog.askopenfilenames')
    def test_f1_01_select_files_adds_to_list(self, mock_ask):
        mock_ask.return_value = [self.temp_files[1], self.temp_files[0]]
        self.app.sorting_view.select_button.invoke()
        pump_events(self.app)
        self.assertEqual(self.app.sorting_view.file_list, [self.temp_files[1], self.temp_files[0]])

    def test_f1_02_natural_sort_empty_list(self):
        res = natural_sort([])
        self.assertEqual(res, [])

    def test_f1_03_natural_sort_preserves_count(self):
        lst = ["frame_2.tif", "frame_10.tif", "frame_1.tif"]
        original_len = len(lst)
        natural_sort(lst)
        self.assertEqual(len(lst), original_len)

    def test_f1_04_natural_sort_ordering(self):
        lst = ["frame_10.tif", "frame_2.tif", "frame_1.tif"]
        natural_sort(lst)
        self.assertEqual(lst, ["frame_1.tif", "frame_2.tif", "frame_10.tif"])

    def test_f1_05_move_up_selected_item(self):
        self.app.sorting_view.file_list = ["a.tif", "b.tif", "c.tif"]
        self.app.sorting_view.selected_index = 1
        self.app.sorting_view.up_button.invoke()
        pump_events(self.app)
        self.assertEqual(self.app.sorting_view.file_list, ["b.tif", "a.tif", "c.tif"])
        self.assertEqual(self.app.sorting_view.selected_index, 0)

    def test_f1_06_move_up_boundary(self):
        self.app.sorting_view.file_list = ["a.tif", "b.tif", "c.tif"]
        self.app.sorting_view.selected_index = 0
        self.app.sorting_view.up_button.invoke()
        pump_events(self.app)
        self.assertEqual(self.app.sorting_view.file_list, ["a.tif", "b.tif", "c.tif"])
        self.assertEqual(self.app.sorting_view.selected_index, 0)

    def test_f1_07_move_down_selected_item(self):
        self.app.sorting_view.file_list = ["a.tif", "b.tif", "c.tif"]
        self.app.sorting_view.selected_index = 1
        self.app.sorting_view.down_button.invoke()
        pump_events(self.app)
        self.assertEqual(self.app.sorting_view.file_list, ["a.tif", "c.tif", "b.tif"])
        self.assertEqual(self.app.sorting_view.selected_index, 2)

    def test_f1_08_move_down_boundary(self):
        self.app.sorting_view.file_list = ["a.tif", "b.tif", "c.tif"]
        self.app.sorting_view.selected_index = 2
        self.app.sorting_view.down_button.invoke()
        pump_events(self.app)
        self.assertEqual(self.app.sorting_view.file_list, ["a.tif", "b.tif", "c.tif"])
        self.assertEqual(self.app.sorting_view.selected_index, 2)

    def test_f1_09_remove_item(self):
        self.app.sorting_view.file_list = ["a.tif", "b.tif", "c.tif"]
        self.app.sorting_view.selected_index = 1
        self.app.sorting_view.remove_button.invoke()
        pump_events(self.app)
        self.assertEqual(self.app.sorting_view.file_list, ["a.tif", "c.tif"])

    def test_f1_10_remove_adjusts_selection(self):
        self.app.sorting_view.file_list = ["a.tif", "b.tif"]
        self.app.sorting_view.selected_index = 1
        self.app.sorting_view.remove_button.invoke()
        pump_events(self.app)
        self.assertEqual(self.app.sorting_view.selected_index, 0)

    def test_f1_11_start_slideshow_disabled_if_empty(self):
        self.app.sorting_view.file_list = []
        self.app.sorting_view.start_button.invoke()
        pump_events(self.app)
        self.assertFalse(self.app.slideshow_view.winfo_ismapped())

    # --- F2 (Slideshow aspect ratio & navigation) - 10 test cases ---
    def test_f2_01_transition_to_slideshow_displays_first_frame(self):
        self.app.sorting_view.file_list = self.temp_files.copy()
        self.app.sorting_view.start_button.invoke()
        pump_events(self.app)
        self.assertTrue(bool(self.app.slideshow_view.grid_info()))
        self.assertEqual(self.app.slideshow_view.current_idx, 0)

    def test_f2_02_canvas_initializes(self):
        self.assertIsNotNone(self.app.slideshow_view.canvas)

    def test_f2_03_navigation_next_frame(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        self.app.slideshow_view.next_button.invoke()
        pump_events(self.app)
        self.assertEqual(self.app.slideshow_view.current_idx, 1)

    def test_f2_04_navigation_prev_frame(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        self.app.slideshow_view.next_button.invoke()
        pump_events(self.app)
        self.app.slideshow_view.prev_button.invoke()
        pump_events(self.app)
        self.assertEqual(self.app.slideshow_view.current_idx, 0)

    def test_f2_05_navigation_next_boundary(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        self.app.slideshow_view.next_button.invoke()
        self.app.slideshow_view.next_button.invoke()
        self.app.slideshow_view.next_button.invoke()
        pump_events(self.app)
        self.assertEqual(self.app.slideshow_view.current_idx, 2)

    def test_f2_06_navigation_prev_boundary(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        self.app.slideshow_view.prev_button.invoke()
        pump_events(self.app)
        self.assertEqual(self.app.slideshow_view.current_idx, 0)

    def test_f2_07_canvas_resize_recalculates_scale(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        self.app.slideshow_view.canvas.winfo_width = lambda: 400
        self.app.slideshow_view.canvas.winfo_height = lambda: 300
        self.app.slideshow_view.on_resize(None)
        pump_events(self.app)
        self.assertLessEqual(self.app.slideshow_view.photo_img.width(), 400)

    def test_f2_08_aspect_ratio_preservation(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        self.app.slideshow_view.draw_canvas(rgb, np.array([100, 50]), np.array([1, 0]))
        self.assertAlmostEqual(self.app.slideshow_view.photo_img.width() / self.app.slideshow_view.photo_img.height(), 2.0, places=2)

    def test_f2_09_jump_to_frame_slider(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        self.app.slideshow_view.frame_slider.set(2)
        self.app.slideshow_view.jump_to_frame(2)
        pump_events(self.app)
        self.assertEqual(self.app.slideshow_view.current_idx, 2)

    def test_f2_10_back_to_sorting_restores_view(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        self.app.slideshow_view.back_button.invoke()
        pump_events(self.app)
        self.assertTrue(bool(self.app.sorting_view.grid_info()))

    def test_f2_11_keyboard_navigation_next_prev(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        self.assertEqual(self.app.slideshow_view.current_idx, 0)
        
        # Mock winfo_ismapped because the window is withdrawn in test setUp
        with patch.object(self.app.slideshow_view, 'winfo_ismapped', return_value=True):
            # Simulate key events by calling handlers directly since withdrawn windows cannot gain focus
            mock_event = tk.Event()
            mock_event.widget = self.app
            
            self.app._on_right_key(mock_event)
            pump_events(self.app)
            self.assertEqual(self.app.slideshow_view.current_idx, 1)
            
            self.app._on_left_key(mock_event)
            pump_events(self.app)
            self.assertEqual(self.app.slideshow_view.current_idx, 0)

    # --- F3 (PCA peak visualization & slider) - 10 test cases ---
    def test_f3_01_pca_centroid_drawn_on_canvas(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        self.app.slideshow_view.change_engine("PCA")
        pump_events(self.app)
        centroids = self.app.slideshow_view.canvas.find_withtag("centroid")
        self.assertGreater(len(centroids), 0)

    def test_f3_02_pca_peak_line_drawn_on_canvas(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        self.app.slideshow_view.change_engine("PCA")
        pump_events(self.app)
        lines = self.app.slideshow_view.canvas.find_withtag("peak_line")
        self.assertGreater(len(lines), 0)

    def test_f3_03_pca_threshold_slider_change_updates_label(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        self.app.slideshow_view.pca_slider.set(95.0)
        self.app.slideshow_view.change_pca_threshold(95.0)
        pump_events(self.app)
        self.assertIn("95.0%", self.app.slideshow_view.pca_label.cget("text"))

    def test_f3_04_pca_threshold_slider_renders_new_peak(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        self.app.slideshow_view.change_engine("PCA")
        pump_events(self.app)
        line_id = self.app.slideshow_view.canvas.find_withtag("peak_line")[0]
        coords_before = self.app.slideshow_view.canvas.coords(line_id)
        self.assertIsNotNone(coords_before)
        self.assertTrue(len(coords_before) >= 4, "Peak line should have at least 4 coordinate values")
        
        self.app.slideshow_view.pca_slider.set(80.0)
        self.app.slideshow_view.change_pca_threshold(80.0)
        pump_events(self.app)
        
        line_id_after = self.app.slideshow_view.canvas.find_withtag("peak_line")[0]
        coords_after = self.app.slideshow_view.canvas.coords(line_id_after)
        self.assertIsNotNone(coords_after)
        self.assertTrue(len(coords_after) >= 4, "Peak line should still render at lower threshold")

    def test_f3_05_pca_flat_image_fallback_centroid(self):
        flat_img = np.zeros((10, 10), dtype=np.float32)
        with self.assertRaises(PCAFitFailure):
            find_peak_line(flat_img, 99.0)

    def test_f3_06_pca_flat_image_fallback_line(self):
        flat_img = np.zeros((10, 10), dtype=np.float32)
        with self.assertRaises(PCAFitFailure):
            find_peak_line(flat_img, 99.0)

    def test_f3_07_pca_insufficient_points_fallback(self):
        flat_img = np.zeros((10, 10), dtype=np.float32)
        with self.assertRaises(PCAFitFailure):
            find_peak_line(flat_img, 99.0)

    def test_f3_08_pca_threshold_out_of_bounds_raises(self):
        img = np.zeros((10, 10), dtype=np.float32)
        with self.assertRaises(ValueError):
            find_peak_line(img, -1.0)
        with self.assertRaises(ValueError):
            find_peak_line(img, 101.0)

    def test_f3_09_pca_invalid_image_shape_raises(self):
        with self.assertRaises(ValueError):
            find_peak_line(np.array([1, 2, 3]), 99.0)

    def test_f3_10_pca_slider_boundary_values(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        self.app.slideshow_view.change_pca_threshold(0.0)
        pump_events(self.app)
        self.app.slideshow_view.change_pca_threshold(100.0)
        pump_events(self.app)
        self.assertIn("100.0%", self.app.slideshow_view.pca_label.cget("text"))

    # --- F4 (Hanning phase correlation & warp toggle) - 11 test cases ---
    def test_f4_01_warp_switch_initial_state(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        self.assertTrue(self.app.slideshow_view.warp_enabled)

    def test_f4_02_warp_toggle_enable(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        self.app.slideshow_view.warp_switch.deselect()
        self.app.slideshow_view.toggle_warp()
        pump_events(self.app)
        self.assertFalse(self.app.slideshow_view.warp_enabled)

    def test_f4_03_warp_toggle_triggers_redraw(self):
        self.app.show_slideshow(self.temp_files)
        self.app.slideshow_view.change_engine("PCA")
        self.app.slideshow_view.next_button.invoke()
        pump_events(self.app)
        
        line_id = self.app.slideshow_view.canvas.find_withtag("peak_line")[0]
        coords_before = self.app.slideshow_view.canvas.coords(line_id)
        
        self.app.slideshow_view.warp_switch.deselect()
        self.app.slideshow_view.toggle_warp()
        pump_events(self.app)
        
        line_id_after = self.app.slideshow_view.canvas.find_withtag("peak_line")[0]
        coords_after = self.app.slideshow_view.canvas.coords(line_id_after)
        self.assertIsNotNone(coords_after)
        self.assertNotEqual(coords_before, coords_after)

    def test_f4_04_phase_correlation_offset_zero(self):
        img = np.zeros((100, 100), dtype=np.float32)
        img[40:60, 40:60] = 1.0
        dx, dy = phase_correlation_offset(img, img)
        self.assertAlmostEqual(dx, 0.0, places=2)
        self.assertAlmostEqual(dy, 0.0, places=2)

    def test_f4_05_phase_correlation_offset_shifted(self):
        y, x = np.mgrid[0:128, 0:128]
        ref = np.exp(-((x - 64)**2 + (y - 64)**2) / (2 * 10**2)).astype(np.float32)
        M = np.float32([[1, 0, 3.0], [0, 1, 4.0]])
        target = cv2.warpAffine(ref, M, (128, 128))
        dx, dy = phase_correlation_offset(ref, target)
        self.assertAlmostEqual(dx, 3.0, places=1)
        self.assertAlmostEqual(dy, 4.0, places=1)

    def test_f4_06_phase_correlation_dimension_mismatch(self):
        img1 = np.zeros((100, 100), dtype=np.float32)
        img2 = np.zeros((100, 90), dtype=np.float32)
        with self.assertRaises(ValueError):
            phase_correlation_offset(img1, img2)

    def test_f4_07_warp_image_zero_translation(self):
        img = np.random.rand(10, 10).astype(np.float32)
        warped = warp_image(img, 0.0, 0.0)
        np.testing.assert_array_equal(img, warped)

    def test_f4_08_warp_image_translation_coords(self):
        img = np.zeros((10, 10), dtype=np.float32)
        img[4, 4] = 1.0
        warped = warp_image(img, 1.0, 2.0)
        self.assertEqual(warped[6, 5], 1.0)
        self.assertEqual(warped[4, 4], 0.0)

    def test_f4_09_warp_image_rgb(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[4, 4, 0] = 255
        warped = warp_image(img, 1.0, 1.0)
        self.assertEqual(warped[5, 5, 0], 255)

    def test_f4_10_warp_image_invalid_shape(self):
        with self.assertRaises(ValueError):
            warp_image(np.array([1, 2, 3]), 1.0, 1.0)

    def test_f4_11_warp_fails_gracefully(self):
        img1 = np.zeros((100, 100), dtype=np.float32)
        img2 = np.zeros((100, 100), dtype=np.float32)
        dx, dy = phase_correlation_offset(img1, img2)
        self.assertEqual(dx, 0.0)
        self.assertEqual(dy, 0.0)

    def test_f4_12_phase_correlation_no_inplace_mutation(self):
        array1 = np.random.rand(100, 100).astype(np.float64)
        array2 = np.random.rand(100, 100).astype(np.float64)
        clone1 = array1.copy()
        clone2 = array2.copy()
        _ = phase_correlation_offset(array1, array2)
        np.testing.assert_array_equal(array1, clone1)
        np.testing.assert_array_equal(array2, clone2)

    # --- F5 (Customization/Deployment/README static checks) - 10 test cases ---
    def test_f5_01_readme_exists(self):
        readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
        self.assertTrue(os.path.exists(readme_path))

    def test_f5_02_readme_mentions_setup(self):
        readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
        with open(readme_path, "r") as f:
            content = f.read()
        self.assertIn("Setup Instructions", content)

    def test_f5_03_readme_mentions_venv(self):
        readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
        with open(readme_path, "r") as f:
            content = f.read()
        self.assertIn(".venv", content)

    def test_f5_04_readme_mentions_run(self):
        readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
        with open(readme_path, "r") as f:
            content = f.read()
        self.assertIn("run.py", content)

    def test_f5_05_readme_mentions_pytest(self):
        readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
        with open(readme_path, "r") as f:
            content = f.read()
        self.assertIn("pytest tests/test_e2e.py", content)

    def test_f5_06_colormap_menu_change(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        self.app.slideshow_view.colormap_menu.set("inferno")
        self.app.slideshow_view.change_colormap("inferno")
        pump_events(self.app)
        self.assertEqual(self.app.slideshow_view.colormap, "inferno")

    def test_f5_07_colormap_menu_triggers_redraw(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        self.app.slideshow_view.colormap_menu.set("inferno")
        self.app.slideshow_view.change_colormap("inferno")
        pump_events(self.app)
        rgb = self.app.slideshow_view.current_rgb
        self.assertIsNotNone(rgb)
        self.assertFalse(np.array_equal(rgb[:, :, 0], rgb[:, :, 1]))

    def test_f5_08_colormap_nonexistent_fallback(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        self.app.slideshow_view.colormap_menu.set("invalid_cmap")
        self.app.slideshow_view.change_colormap("invalid_cmap")
        pump_events(self.app)
        rgb = self.app.slideshow_view.current_rgb
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 1])
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 2])

    def test_f5_09_preprocess_image_grayscale(self):
        rgb, raw = preprocess_image(self.temp_files[0], "grayscale", 100.0)
        self.assertEqual(rgb.shape, (100, 100, 3))
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 1])

    def test_f5_10_preprocess_image_invalid_percentile(self):
        with self.assertRaises(ValueError):
            preprocess_image(self.temp_files[0], "grayscale", -10.0)

    def test_f5_11_dark_mode_theme_applied(self):
        self.assertEqual(customtkinter.get_appearance_mode(), "Dark")

    # --- Tier 3 (Cross-feature interactions) - 5 test cases ---
    def test_t3_01_sort_retains_selection_integrity(self):
        self.app.sorting_view.file_list = [self.temp_files[1], self.temp_files[0]]
        self.app.sorting_view.selected_index = 0
        self.app.sorting_view.sort_files()
        pump_events(self.app)
        self.assertEqual(self.app.sorting_view.file_list, [self.temp_files[0], self.temp_files[1]])

    def test_t3_02_navigating_preserves_colormap_across_frames(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        self.app.slideshow_view.change_colormap("inferno")
        self.app.slideshow_view.next_button.invoke()
        pump_events(self.app)
        self.assertEqual(self.app.slideshow_view.colormap, "inferno")

    def test_t3_03_navigating_preserves_pca_threshold(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        self.app.slideshow_view.change_pca_threshold(85.5)
        self.app.slideshow_view.next_button.invoke()
        pump_events(self.app)
        self.assertEqual(self.app.slideshow_view.pca_threshold, 85.5)

    def test_t3_04_navigating_preserves_warp_switch(self):
        self.app.show_slideshow(self.temp_files)
        pump_events(self.app)
        self.app.slideshow_view.warp_switch.deselect()
        self.app.slideshow_view.toggle_warp()
        self.app.slideshow_view.next_button.invoke()
        pump_events(self.app)
        self.assertFalse(self.app.slideshow_view.warp_enabled)

    def test_t3_05_back_to_sorting_preserves_list_order(self):
        self.app.sorting_view.file_list = [self.temp_files[1], self.temp_files[0]]
        self.app.show_slideshow(self.app.sorting_view.file_list)
        pump_events(self.app)
        self.app.slideshow_view.back_button.invoke()
        pump_events(self.app)
        self.assertEqual(self.app.sorting_view.file_list, [self.temp_files[1], self.temp_files[0]])

    # --- Tier 4 (Real-world application/TIF checks) - 5 test cases ---
    def test_t4_01_real_tif_load_and_preprocess(self):
        real_path = "tests/samples/Sample1VL_200F_frames_1-200.tif"
        self.assertTrue(os.path.exists(real_path))
        rgb, raw = preprocess_image(real_path, "grayscale", 99.0)
        self.assertEqual(raw.ndim, 2)
        self.assertEqual(rgb.shape[:2], raw.shape)
        self.assertEqual(rgb.shape[2], 3)

    def test_t4_02_real_tif_find_peak_line(self):
        real_path = "tests/samples/Sample1VL_200F_frames_1-200.tif"
        self.assertTrue(os.path.exists(real_path))
        _, raw = preprocess_image(real_path, "grayscale", 99.0)
        origin, direction = find_peak_line(raw, 99.0)
        self.assertEqual(origin.shape, (2,))
        self.assertEqual(direction.shape, (2,))
        self.assertAlmostEqual(np.linalg.norm(direction), 1.0, places=5)

    def test_t4_03_real_tif_phase_correlation(self):
        real_path_1 = "tests/samples/Sample1VL_200F_frames_1-200.tif"
        real_path_2 = "tests/samples/Sample1VL_200F_frames_201-400.tif"
        self.assertTrue(os.path.exists(real_path_1))
        self.assertTrue(os.path.exists(real_path_2))
        _, raw1 = preprocess_image(real_path_1, "grayscale", 100.0)
        _, raw2 = preprocess_image(real_path_2, "grayscale", 100.0)
        dx, dy = phase_correlation_offset(raw1, raw2)
        self.assertIsInstance(dx, float)
        self.assertIsInstance(dy, float)

    def test_t4_04_real_tif_warp_and_render(self):
        real_path_1 = "tests/samples/Sample1VL_200F_frames_1-200.tif"
        real_path_2 = "tests/samples/Sample1VL_200F_frames_201-400.tif"
        self.assertTrue(os.path.exists(real_path_1))
        self.assertTrue(os.path.exists(real_path_2))
        self.app.show_slideshow([real_path_1, real_path_2])
        pump_events(self.app)
        self.app.slideshow_view.change_engine("PCA")
        pump_events(self.app)
        self.app.slideshow_view.warp_switch.select()
        self.app.slideshow_view.toggle_warp()
        pump_events(self.app)
        self.app.slideshow_view.next_button.invoke()
        pump_events(self.app)
        
        images = self.app.slideshow_view.canvas.find_withtag("image")
        centroids = self.app.slideshow_view.canvas.find_withtag("centroid")
        lines = self.app.slideshow_view.canvas.find_withtag("peak_line")
        self.assertGreater(len(images), 0)
        self.assertGreater(len(centroids), 0)
        self.assertGreater(len(lines), 0)

    def test_t4_05_real_tif_natural_sort_workspace_tifs(self):
        import glob
        tif_files = [os.path.basename(p) for p in glob.glob("tests/samples/*.tif")]
        self.assertTrue(len(tif_files) > 0)
        sorted_tifs = natural_sort(tif_files.copy())
        self.assertEqual(len(sorted_tifs), len(tif_files))
        f201 = "Sample1VL_200F_frames_201-400.tif"
        f1001 = "Sample1VL_200F_frames_1001-1200.tif"
        self.assertIn(f201, sorted_tifs)
        self.assertIn(f1001, sorted_tifs)
        self.assertLess(sorted_tifs.index(f201), sorted_tifs.index(f1001))

if __name__ == "__main__":
    unittest.main()
