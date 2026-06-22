"""Tests for ZarrSequenceManager path handling when path contains '#' characters."""

import unittest
import os
import shutil
import tempfile
import time
import numpy as np
import tifffile
from align_app.dataset import ZarrSequenceManager

class TestZarrHashPath(unittest.TestCase):
    """Test suite to verify ZarrSequenceManager's robustness with '#' in directories.

    This ensures that folder names containing '#' do not get truncated due to URI parsing.
    """

    def setUp(self) -> None:
        """Create a temporary directory structure for testing.

        Creates a base directory containing '#' (e.g., 'test_dir#1') and writes
        a mock TIFF file inside it to be managed by ZarrSequenceManager.
        """
        self.temp_parent = tempfile.mkdtemp()
        self.hash_dir = os.path.join(self.temp_parent, "spec#1")
        os.makedirs(self.hash_dir, exist_ok=True)

        self.mock_tif = os.path.join(self.hash_dir, "frame_001.tif")
        # Write dummy TIFF
        dummy_data = np.zeros((10, 10), dtype=np.float32)
        tifffile.imwrite(self.mock_tif, dummy_data)

    def tearDown(self) -> None:
        """Clean up all temporary directories and files created during the test."""
        if os.path.exists(self.temp_parent):
            # Wait briefly to ensure file locks are released
            time.sleep(0.1)
            shutil.rmtree(self.temp_parent)

    def test_zarr_cache_location_with_hash(self) -> None:
        """Verify that the Zarr cache is created inside the directory with '#'.

        The cache should reside at <hash_dir>/tif-cache/frames.zarr, and
        no truncated directory (like 'spec') should be created.
        """
        # Ensure only the expected hash directory exists at the beginning
        base_dir_contents = os.listdir(self.temp_parent)
        self.assertEqual(base_dir_contents, ["spec#1"])

        # Instantiate manager, which triggers _init_zarr()
        manager = ZarrSequenceManager([self.mock_tif])

        # Wait for the background thread to finish populating cache and computing median
        start_time = time.time()
        while manager.median_frame is None and time.time() - start_time < 5.0:
            time.sleep(0.05)

        # Verify that the zarr group is open and not None
        self.assertIsNotNone(manager.zarr_group)

        # Expected path of the cache
        expected_cache_dir = os.path.join(self.hash_dir, "tif-cache")
        expected_zarr_path = os.path.join(expected_cache_dir, "frames.zarr")

        # Verify the directory exists on disk literally
        self.assertTrue(os.path.exists(expected_cache_dir), "tif-cache subdirectory should exist")
        self.assertTrue(os.path.exists(expected_zarr_path), "frames.zarr directory should exist")

        # Verify no truncated directory was created in the temp parent
        current_contents = os.listdir(self.temp_parent)
        self.assertEqual(current_contents, ["spec#1"], "No truncated or extra directories should be created")
