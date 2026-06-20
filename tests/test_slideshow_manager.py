import unittest
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
