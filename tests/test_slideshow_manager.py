import unittest
from unittest.mock import patch
import queue
import numpy as np
from align_app.ui.slideshow.managers import SlideshowManager

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

    @patch("align_app.ui.slideshow.managers.SlideshowManager.get_raw")
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

