import unittest
from unittest.mock import patch
import queue
import numpy as np
from rixs_app.ui.slideshow.managers import SlideshowManager

class TestSlideshowManagerBugs(unittest.TestCase):
    def test_init_defines_manual_variables(self):
        """
        Verify that __init__ constructor declares manual_mode and manual_clicks.
        If this fails, manual mode state variable declaration is broken.
        """
        q = queue.Queue()
        mgr = SlideshowManager(q)
        self.assertFalse(mgr.manual_mode)
        self.assertEqual(mgr.manual_clicks, [])

    def test_start_resets_manual_variables(self):
        """
        Verify that start(file_list) resets manual_mode to False and clears manual_clicks.
        If this fails, manual clicks and mode are not properly cleared/reset on slideshow start.
        """
        q = queue.Queue()
        mgr = SlideshowManager(q)
        mgr.manual_mode = True
        mgr.manual_clicks = [(1, 2), (3, 4)]
        
        # Call start with empty list or dummy files
        mgr.start([])
        self.assertFalse(mgr.manual_mode)
        self.assertEqual(mgr.manual_clicks, [])

    def test_get_offset_with_none_ref_origin(self):
        """
        Verify that get_offset returns (0.0, 0.0) and does not raise TypeError
        when ref_origin is None and there is manual centroid data.
        If this fails, get_offset raises a TypeError when reference image fails to load.
        """
        q = queue.Queue()
        mgr = SlideshowManager(q)
        # Setup manual alignment for frame 0
        mgr.per_frame_manual[0] = np.array([10.0, 20.0])
        # Force ref_origin to be None
        mgr.ref_origin = None
        mgr.file_list = ["dummy.tif"]
        
        try:
            dx, dy = mgr.get_offset(0)
            self.assertEqual((dx, dy), (0.0, 0.0))
        except TypeError as e:
            self.fail(f"get_offset raised TypeError when ref_origin is None: {e}")

    @patch("rixs_app.ui.slideshow.managers.SlideshowManager.get_raw")
    def test_default_clamping_ceiling_percentile(self, mock_get_raw):
        """
        Verify that default clamping_ceiling starts at the 60th percentile
        of the reference image's intensities rather than the absolute maximum,
        while clamping_floor starts at the absolute minimum.
        """
        q = queue.Queue()
        mgr = SlideshowManager(q)
        
        # Create a mock 10x10 raw array with values from 0 to 99
        raw = np.arange(100, dtype=np.float32).reshape((10, 10))
        # Set an outlier to be very high to mimic outlier hot pixels
        raw[0, 0] = 1000.0  # outlier
        
        mock_get_raw.return_value = raw
        import os
        mock_path = os.path.join(os.path.dirname(__file__), "samples", "mock.tif")
        mgr.start([mock_path])
        
        self.assertEqual(mgr.intensity_min, 1.0)
        self.assertEqual(mgr.intensity_max, 1000.0)
        self.assertEqual(mgr.clamping_floor, 1.0)
        # 60th percentile of active pixels should be calculated
        active = raw[raw > mgr.intensity_min]
        expected_p60 = float(np.percentile(active, 60.0))
        self.assertEqual(mgr.clamping_ceiling, expected_p60)
        self.assertLess(mgr.clamping_ceiling, 1000.0)

    def test_manual_pca_line_does_not_affect_ecc_engine(self):
        """
        Verify that manual lines defined for PCA do not override or affect
        the offset calculations of the ECC engine.
        """
        q = queue.Queue()
        mgr = SlideshowManager(q)
        mgr.file_list = ["dummy1.tif", "dummy2.tif"]
        mgr.ref_raw = np.ones((10, 10), dtype=np.float32)
        
        # Setup manual alignment for frame 1
        mgr.per_frame_manual[1] = np.array([10.0, 20.0])
        mgr.ref_origin = np.array([5.0, 5.0])
        
        # Mock ecc_maximization_offset to return a specific offset
        with patch("rixs_app.ui.slideshow.managers.ecc_maximization_offset", return_value=(3.0, 4.0)) as mock_ecc, \
             patch("rixs_app.ui.slideshow.managers.SlideshowManager.get_raw", return_value=np.ones((10, 10), dtype=np.float32)):
            # ECC is the default engine — should ignore manual line and return ECC offset (3.0, 4.0)
            self.assertEqual(mgr.active_engine, "ECC")
            dx, dy = mgr.get_offset(1)
            self.assertEqual((dx, dy), (3.0, 4.0))
            mock_ecc.assert_called_once()
            
            # Switch to PCA — should return manual line offset (10 - 5 = 5, 20 - 5 = 15)
            mgr.active_engine = "PCA"
            mgr._invalidate_offset_cache(1)
            dx, dy = mgr.get_offset(1)
            self.assertEqual((dx, dy), (5.0, 15.0))

    def test_zoom_init_and_reset(self):
        """Verify zoom state properties are properly initialized and reset."""
        q = queue.Queue()
        mgr = SlideshowManager(q)
        self.assertFalse(mgr.zoom_mode)
        self.assertEqual(mgr.zoom_level, 0)
        self.assertEqual(mgr.pan_offset_x, 0)
        self.assertEqual(mgr.pan_offset_y, 0)
        
        mgr.zoom_mode = True
        mgr.zoom_level = 2
        mgr.pan_offset_x = 10
        mgr.pan_offset_y = 20
        mgr.reset_view()
        
        self.assertFalse(mgr.zoom_mode)
        self.assertEqual(mgr.zoom_level, 0)
        self.assertEqual(mgr.pan_offset_x, 0)
        self.assertEqual(mgr.pan_offset_y, 0)

    def test_zoom_in_on_point_and_zoom_out(self):
        """Verify zooming in on a point computes pan, and zoom_out recalculates correctly."""
        q = queue.Queue()
        mgr = SlideshowManager(q)
        # cw=1000, ch=500, image=1000x500 (fits perfectly, scale=1.0 at 1x)
        # Zoom level 1 is 2x, image is scaled to 2000x1000.
        # base_dx = (1000 - 2000) / 2 = -500.
        # ix=500, iy=250.
        # scale = 2.0.
        # raw_pan_x = 500 - (-500) - (500 * 2) = 1000 - 1000 = 0.
        # pan_offset_x = 0 (perfect center).
        
        # Test zoom in on (300, 200)
        mgr.zoom_in_on_point(cw=1000, ch=500, ix=300.0, iy=200.0, iw=1000, ih=500)
        self.assertEqual(mgr.zoom_level, 1)
        # scale = 2.0
        # raw_pan_x = 500 - (-500) - 300 * 2 = 1000 - 600 = 400.
        # bounds: base_dx = -500. max(base_dx, min(-base_dx, raw_pan_x)) -> max(-500, min(500, 400)) -> 400.
        self.assertEqual(mgr.pan_offset_x, 400)
        # raw_pan_y = 250 - (-250) - 200 * 2 = 500 - 400 = 100.
        # bounds: base_dy = -250. max(-250, min(250, 100)) -> 100.
        self.assertEqual(mgr.pan_offset_y, 100)

        # Test zoom out (keeps same center)
        mgr.zoom_out(cw=1000, ch=500, iw=1000, ih=500)
        self.assertEqual(mgr.zoom_level, 0)
        self.assertEqual(mgr.pan_offset_x, 0)
        self.assertEqual(mgr.pan_offset_y, 0)



