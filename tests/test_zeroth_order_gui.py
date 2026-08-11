import unittest
import os
import tempfile
import queue
import time
import numpy as np
import tifffile
from unittest.mock import patch, MagicMock
import customtkinter

from rixs_app.main import RixsApp
from rixs_app.ui.zeroth_order_slideshow.slideshow_view import ZerothOrderSlideshowView
from rixs_app.ui.zeroth_order_slideshow.manager import ZerothOrderManager
from rixs_app.core.dataset import ZarrSequenceManager

def pump_events(root):
    root.update_idletasks()
    root.update()

class TestZerothOrderGUI(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_files = []
        for i in range(3):
            path = os.path.join(self.temp_dir.name, f"frame_{i+1}.tif")
            data = np.zeros((100, 100), dtype=np.float32)
            data[40:60, 40:60] = 5.0  # mock a broad line
            tifffile.imwrite(path, data)
            self.temp_files.append(path)

        self.app = RixsApp(show_window=False)
        pump_events(self.app)

    def tearDown(self):
        try:
            if hasattr(self, "app") and self.app.zeroth_order_view is not None:
                if self.app.zeroth_order_view.manager is not None:
                    self.app.zeroth_order_view.manager.zarr_manager = None
        except Exception:
            pass
        try:
            self.app.destroy()
        except Exception:
            pass
        import gc
        gc.collect()
        self.temp_dir.cleanup()


    def test_zeroth_order_slideshow_instantiation(self):
        self.assertIsNotNone(self.app.zeroth_order_view)
        self.assertIsNotNone(self.app.zeroth_order_view.canvas_panel)
        self.assertIsNotNone(self.app.zeroth_order_view.navbar)
        self.assertIsNotNone(self.app.zeroth_order_view.control_panel)
        self.assertIsNotNone(self.app.zeroth_order_view.tools_panel)
        self.assertIsNotNone(self.app.zeroth_order_view.bottom_bar)

    def test_navigation_and_timeline_bounds(self):
        self.app.show_zeroth_order_calibration(self.temp_files)
        pump_events(self.app)
        view = self.app.zeroth_order_view

        self.assertEqual(view.manager.current_idx, 0)
        view.next_frame()
        self.assertEqual(view.manager.current_idx, 1)
        view.prev_frame()
        self.assertEqual(view.manager.current_idx, 0)

        # Autoplay toggle test
        self.assertFalse(view.manager.autoplay_active)
        view.navbar.autoplay_button.invoke()
        self.assertTrue(view.manager.autoplay_active)
        view.navbar.autoplay_button.invoke()
        self.assertFalse(view.manager.autoplay_active)

    def test_pipeline_stage_selection_updates_description(self):
        self.app.show_zeroth_order_calibration(self.temp_files)
        pump_events(self.app)
        view = self.app.zeroth_order_view

        view.navbar.stage_menu.set("Denoised (D)")
        view.change_pipeline_stage("Denoised (D)")

        self.assertEqual(view.manager.pipeline_stage, "Denoised (D)")

        view.navbar.stage_menu.set("Fitted-Line Strip")
        view.change_pipeline_stage("Fitted-Line Strip")
        self.assertEqual(view.manager.pipeline_stage, "Fitted-Line Strip")

    @patch("rixs_app.ui.zeroth_order_slideshow.manager.ZerothOrderManager.get_frame_pipeline_data")
    def test_load_and_render_calls_canvas_draw(self, mock_pipeline):
        mock_pipeline.return_value = {
            "raw_img": np.ones((100, 100)),
            "denoised_img": np.ones((100, 100)) * 2,
            "masked_img": np.ones((100, 100)) * 3,
            "score": 0.42,
            "centroid": np.array([50, 50]),
            "direction": np.array([1, 0]),
            "1d_profile": (np.ones(10), np.arange(10))
        }
        self.app.show_zeroth_order_calibration(self.temp_files)
        pump_events(self.app)
        view = self.app.zeroth_order_view
        view.load_and_render()
        self.assertIn("Score: 0.42", view.control_panel.score_label.cget("text"))

    def test_best_focus_badge_requires_all_frames_computed(self):
        self.app.show_zeroth_order_calibration(self.temp_files)
        pump_events(self.app)
        view = self.app.zeroth_order_view
        # Before all frames are computed, focus_badge should be empty on initial rendering
        view.load_and_render()
        self.assertEqual(view.control_panel.focus_badge.cget("text"), "")

    def test_slicing_changes(self):
        self.app.show_zeroth_order_calibration(self.temp_files)
        pump_events(self.app)
        view = self.app.zeroth_order_view

        # Test input submissions
        view.handle_floor_entry_submit("0.1")
        view.handle_ceiling_entry_submit("0.9")
        self.assertAlmostEqual(view.manager.slicing_floor, 0.1)
        self.assertAlmostEqual(view.manager.slicing_ceiling, 0.9)

        # Debouncing apply
        view._apply_slicing_change()
        self.assertIsNone(view._clamping_debounce_id)

    def test_colormap_changes(self):
        self.app.show_zeroth_order_calibration(self.temp_files)
        pump_events(self.app)
        view = self.app.zeroth_order_view

        view.change_colormap("plasma")
        self.assertEqual(view.manager.colormap, "plasma")

        view.change_colormap("grayscale")
        self.assertEqual(view.manager.colormap, "grayscale")

    def test_zoom_features(self):
        self.app.show_zeroth_order_calibration(self.temp_files)
        pump_events(self.app)
        view = self.app.zeroth_order_view

        self.assertEqual(view.zoom_factor, 1.0)
        view.zoom_in()
        self.assertTrue(view.zoom_mode)
        self.assertEqual(view.zoom_factor, 1.0)

        view.handle_canvas_click(50, 50)
        self.assertGreater(view.zoom_factor, 1.0)
        self.assertFalse(view.zoom_mode)
        self.assertEqual(view.zoom_center, (50, 50))

        view.zoom_out()
        self.assertEqual(view.zoom_factor, 1.0)
        self.assertIsNone(view.zoom_center)

        view.zoom_in()
        view.handle_canvas_click(50, 50)
        view.reset_view()
        self.assertEqual(view.zoom_factor, 1.0)
        self.assertIsNone(view.zoom_center)

    @patch("os.makedirs")
    def test_zarr_cache_fallback(self, mock_makedirs):
        # Force a PermissionError when creating the directory
        mock_makedirs.side_effect = PermissionError("Permission Denied")
        # Initialize a ZarrSequenceManager with the temp files
        manager = ZarrSequenceManager(self.temp_files)

        self.assertIsNotNone(manager.zarr_group)

        # Verify fallback path contains "rixs_cache_" in tempfile.gettempdir()
        import hashlib
        tif_dir = os.path.dirname(os.path.abspath(self.temp_files[0]))
        dir_hash = hashlib.md5(tif_dir.encode("utf-8")).hexdigest()
        expected_fallback_path = os.path.join(tempfile.gettempdir(), f"rixs_cache_{dir_hash}")
        self.assertTrue(os.path.exists(expected_fallback_path))

    def test_precompute_worker_execution(self):
        self.app.show_zeroth_order_calibration(self.temp_files)
        pump_events(self.app)
        view = self.app.zeroth_order_view

        # Test precompute trigger and progress updates
        view.trigger_precompute()
        # Drain the queue to apply callbacks
        start_time = time.time()
        while time.time() - start_time < 3.0:
            pump_events(self.app)
            try:
                callback = view._result_queue.get_nowait()
                callback()
            except queue.Empty:
                if view.navbar.precompute_button.cget("text") == "Precompute All":
                    break
                time.sleep(0.05)

        self.assertEqual(view.navbar.precompute_button.cget("text"), "Precompute All")

    @patch("tkinter.messagebox.showerror")
    def test_precompute_worker_handles_missing_file_gracefully(self, mock_showerror):
        # We pass a list of files where one is missing/corrupted
        bad_files = self.temp_files + [os.path.join(self.temp_dir.name, "missing_frame.tif")]
        self.app.show_zeroth_order_calibration(bad_files)
        pump_events(self.app)
        view = self.app.zeroth_order_view

        # Verify initial button state is normal
        self.assertEqual(view.navbar.prev_button.cget("state"), "normal")

        # Trigger precompute
        view.trigger_precompute()

        # Wait and pump events to let the background thread run and raise the exception
        start_time = time.time()
        error_handled = False
        while time.time() - start_time < 3.0:
            pump_events(self.app)
            try:
                callback = view._result_queue.get_nowait()
                callback()
            except queue.Empty:
                if mock_showerror.called:
                    error_handled = True
                    break
                time.sleep(0.05)

        # Verify that showerror was called to display the error
        self.assertTrue(error_handled)
        mock_showerror.assert_called_once()

        # Check that UI elements were re-enabled
        self.assertEqual(view.navbar.prev_button.cget("state"), "normal")
        self.assertEqual(view.navbar.next_button.cget("state"), "normal")
        self.assertEqual(view.navbar.precompute_button.cget("state"), "normal")
        self.assertEqual(view.navbar.precompute_button.cget("text"), "Precompute All")

    def test_first_file_missing_crashes_gui_on_load(self):
        """Verify that if the first file is missing, showing the slideshow loads gracefully without crashing."""
        bad_files = [os.path.join(self.temp_dir.name, "missing_first.tif")] + self.temp_files
        self.app.show_zeroth_order_calibration(bad_files)
        pump_events(self.app)

        view = self.app.zeroth_order_view
        self.assertIsNotNone(view)
        self.assertEqual(view.manager.current_idx, 0)

    def test_navigation_to_missing_file_raises_value_error(self):
        """Verify that navigating to a missing file is handled gracefully without crashing."""
        bad_files = self.temp_files + [os.path.join(self.temp_dir.name, "missing_last.tif")]
        self.app.show_zeroth_order_calibration(bad_files)
        pump_events(self.app)

        view = self.app.zeroth_order_view
        view.manager.current_idx = 3
        # Should not raise ValueError
        view.load_and_render()

    def test_destroy_view_during_precomputation(self):
        """Verify what happens if the view is destroyed while precomputation is in progress."""
        self.app.show_zeroth_order_calibration(self.temp_files)
        pump_events(self.app)
        view = self.app.zeroth_order_view

        # Trigger precompute
        view.trigger_precompute()

        # Immediately destroy the app/view
        self.app.destroy()

        # Wait a bit for the thread to run
        time.sleep(0.5)

    @patch("tkinter.messagebox.showerror")
    def test_export_worker_handles_write_failure_gracefully(self, mock_showerror):
        """Verify that export worker write failure is caught and reported gracefully."""
        self.app.show_zeroth_order_calibration(self.temp_files)
        pump_events(self.app)
        view = self.app.zeroth_order_view

        bad_export_dir = "/non_existent_directory_which_is_invalid"

        with patch("tkinter.filedialog.askdirectory", return_value=bad_export_dir):
            view.trigger_export()

        start_time = time.time()
        error_handled = False
        while time.time() - start_time < 3.0:
            pump_events(self.app)
            try:
                callback = view._result_queue.get_nowait()
                callback()
            except queue.Empty:
                if mock_showerror.called:
                    error_handled = True
                    break
                time.sleep(0.05)

        self.assertTrue(error_handled)
        mock_showerror.assert_called_once()
        self.assertEqual(view.bottom_bar.export_button.cget("state"), "normal")
        self.assertEqual(view.bottom_bar.progress_label.cget("text"), "")

    def test_manager_cache_preserves_all_metadata_keys(self):
        """Verify that get_frame_pipeline_data returns fit_ok and overlay metadata on cache hits."""
        self.app.show_zeroth_order_calibration(self.temp_files)
        pump_events(self.app)
        manager = self.app.zeroth_order_view.manager

        # First call (cache miss)
        data_miss = manager.get_frame_pipeline_data(0)
        self.assertIsNotNone(data_miss)
        self.assertIn("fit_ok", data_miss)

        # Second call (cache hit)
        data_hit = manager.get_frame_pipeline_data(0)
        self.assertIsNotNone(data_hit)
        self.assertIn("fit_ok", data_hit)
        self.assertEqual(data_miss["fit_ok"], data_hit["fit_ok"])
        self.assertIn("candidates_xy", data_hit)
        self.assertIn("inliers_xy", data_hit)
        self.assertIn("evaluator_result", data_hit)
