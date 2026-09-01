"""Tests for ZarrSequenceManager path handling when path contains '#' characters."""

import unittest
import os
import shutil
import tempfile
import time
import numpy as np
import tifffile
from rixs_app.core.dataset import ZarrSequenceManager

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

    def test_zarr_cache_skipped_for_empty_or_missing_files(self) -> None:
        """Verify that no tif-cache is created if file_list is empty or files don't exist."""
        # Case 1: Empty file list
        empty_manager = ZarrSequenceManager([])
        empty_manager._loading_done.wait(timeout=1.0)
        self.assertIsNone(empty_manager.zarr_group)

        # Case 2: Non-existent file path
        missing_path = os.path.join(self.temp_parent, "non_existent", "frame_001.tif")
        manager = ZarrSequenceManager([missing_path])
        manager._loading_done.wait(timeout=1.0)

        cache_dir = os.path.join(os.path.dirname(missing_path), "tif-cache")
        self.assertFalse(os.path.exists(cache_dir), "tif-cache should not be created for non-existent files")
        self.assertIsNone(manager.zarr_group)

    def test_zarr_cache_readme_and_gitignore_created(self) -> None:
        """Verify that README_CACHE.txt and .gitignore are created inside tif-cache."""
        manager = ZarrSequenceManager([self.mock_tif])
        manager._loading_done.wait(timeout=5.0)

        cache_dir = os.path.join(self.hash_dir, "tif-cache")
        readme_path = os.path.join(cache_dir, "README_CACHE.txt")
        gitignore_path = os.path.join(cache_dir, ".gitignore")

        self.assertTrue(os.path.exists(readme_path), "README_CACHE.txt should be created")
        self.assertTrue(os.path.exists(gitignore_path), ".gitignore should be created")

        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("QERLIN Beamline 6.0.2 RIXS Super-App", content)
        self.assertIn("100% SAFE TO DELETE", content)
        self.assertIn("frame_001.tif", content)
        self.assertIn(self.hash_dir, content)
        self.assertIn("Source Directory :", content)
        self.assertIn("Last Updated     :", content)
        self.assertIn("Cached Files:", content)

        with open(gitignore_path, "r", encoding="utf-8") as f:
            gi_content = f.read()
        self.assertEqual(gi_content.strip(), "*")

    def test_zarr_cache_readme_updates_on_new_file(self) -> None:
        """Verify that README_CACHE.txt is updated with all files when a sequence is loaded."""
        mock_tif2 = os.path.join(self.hash_dir, "frame_002.tif")
        dummy_data2 = np.ones((10, 10), dtype=np.float32)
        tifffile.imwrite(mock_tif2, dummy_data2)

        manager = ZarrSequenceManager([self.mock_tif, mock_tif2])
        self.assertTrue(manager._loading_done.wait(timeout=5.0))

        cache_dir = os.path.join(self.hash_dir, "tif-cache")
        readme_path = os.path.join(cache_dir, "README_CACHE.txt")
        self.assertTrue(os.path.exists(readme_path), "README_CACHE.txt should exist")

        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("[001] frame_001.tif", content)
        self.assertIn("[002] frame_002.tif", content)


