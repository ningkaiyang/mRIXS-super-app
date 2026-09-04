"""Tests for SequenceManager in-memory frame caching and path safety."""

import os
import shutil
import tempfile
import numpy as np
import tifffile
import pytest

from rixs_app.core.dataset import SequenceManager


class TestSequenceManager:
    """Test suite verifying SequenceManager in-memory caching and path handling."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        self.temp_parent = tempfile.mkdtemp()
        self.hash_dir = os.path.join(self.temp_parent, "spec#1")
        os.makedirs(self.hash_dir, exist_ok=True)

        self.tif_0 = os.path.join(self.hash_dir, "frame_000.tif")
        self.tif_1 = os.path.join(self.hash_dir, "frame_001.tif")
        tifffile.imwrite(self.tif_0, np.ones((20, 20), dtype=np.float32) * 5.0)
        tifffile.imwrite(self.tif_1, np.ones((20, 20), dtype=np.float32) * 15.0)

        yield

        if os.path.exists(self.temp_parent):
            shutil.rmtree(self.temp_parent, ignore_errors=True)

    def test_sequence_manager_initialization(self) -> None:
        """Verify initialization sets file list, count, and ready flag."""
        manager = SequenceManager([self.tif_0, self.tif_1])
        assert manager.n_frames == 2
        assert manager.file_list == [self.tif_0, self.tif_1]
        assert manager._loading_done.is_set()
        assert manager.median_frame is None

    def test_get_frame_caching_and_isolation(self) -> None:
        """Verify get_frame loads from disk into memory cache and hits cache subsequently."""
        manager = SequenceManager([self.tif_0, self.tif_1])
        assert not manager._mem_cache.has(0)

        data = manager.get_frame(0)
        assert data is not None
        assert data.shape == (20, 20)
        assert data[0, 0] == 5.0
        assert manager._mem_cache.has(0)

        # Overwrite file on disk — subsequent read must hit memory cache
        tifffile.imwrite(self.tif_0, np.zeros((20, 20), dtype=np.float32))
        cached = manager.get_frame(0)
        assert cached is not None
        assert cached[0, 0] == 5.0

    def test_get_frame_out_of_bounds_and_missing(self) -> None:
        """Verify out-of-bounds indices and non-existent files return None safely."""
        missing = os.path.join(self.hash_dir, "missing.tif")
        manager = SequenceManager([self.tif_0, missing])
        assert manager.get_frame(-1) is None
        assert manager.get_frame(2) is None
        assert manager.get_frame(1) is None

    def test_set_frame_in_memory(self) -> None:
        """Verify set_frame populates memory cache."""
        manager = SequenceManager([self.tif_0])
        new_data = np.full((20, 20), 42.0, dtype=np.float32)
        manager.set_frame(0, new_data)
        np.testing.assert_allclose(manager.get_frame(0), new_data)

    def test_derived_frames_caching(self) -> None:
        """Verify get_derived_frame and set_derived_frame operate strictly in memory."""
        manager = SequenceManager([self.tif_0])
        assert manager.get_derived_frame(0, "denoised_img") is None

        denoised = np.full((20, 20), 99.0, dtype=np.float32)
        manager.set_derived_frame(0, "denoised_img", denoised)
        ret = manager.get_derived_frame(0, "denoised_img")
        assert ret is not None
        np.testing.assert_allclose(ret, denoised)

    def test_compute_median(self) -> None:
        """Verify compute_median computes pixel-wise median across frames."""
        manager = SequenceManager([self.tif_0, self.tif_1])
        manager.compute_median()
        assert manager.median_frame is not None
        assert manager.median_frame.shape == (20, 20)
        assert manager.median_frame[0, 0] == 10.0

    def test_zero_on_disk_cache_created(self) -> None:
        """Verify that NO tif-cache/ directory or .zarr directory is created on disk."""
        manager = SequenceManager([self.tif_0, self.tif_1])
        manager.get_frame(0)
        manager.get_frame(1)
        manager.compute_median()

        cache_dir = os.path.join(self.hash_dir, "tif-cache")
        assert not os.path.exists(cache_dir), "tif-cache/ must never be created on disk"
        for root, dirs, files in os.walk(self.temp_parent):
            for d in dirs:
                assert not d.endswith(".zarr"), f"Found zarr directory {d}"
                assert d != "tif-cache", f"Found tif-cache directory in {root}"

    def test_zero_zarr_imports_in_dataset(self) -> None:
        """Verify rixs_app.core.dataset has no zarr import."""
        import sys
        import rixs_app.core.dataset as ds_mod
        assert not hasattr(ds_mod, "zarr")
        with open(ds_mod.__file__, "r", encoding="utf-8") as f:
            content = f.read()
        forbidden = ["import " + "zarr", "from " + "zarr"]
        for token in forbidden:
            assert token not in content

    def test_sequence_manager_concurrency_stress(self) -> None:
        """Verify SequenceManager is thread-safe under concurrent frame reads and derived writes."""
        import threading
        manager = SequenceManager([self.tif_0, self.tif_1])
        errors = []

        def worker(tid: int):
            try:
                for op in range(15):
                    idx = op % 2
                    frame = manager.get_frame(idx)
                    if frame is None:
                        errors.append(f"Thread {tid} got None for frame {idx}")
                    derived = np.full((20, 20), float(tid * 10 + op), dtype=np.float32)
                    manager.set_derived_frame(idx, f"stage_{tid}", derived)
                    ret = manager.get_derived_frame(idx, f"stage_{tid}")
                    if ret is None or not np.isclose(ret[0, 0], float(tid * 10 + op)):
                        errors.append(f"Thread {tid} derived frame mismatch")
            except Exception as e:
                errors.append(f"Thread {tid} exception: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrency errors occurred: {errors}"

    def test_sequence_manager_readonly_directory(self) -> None:
        """Verify SequenceManager functions when source TIFF directory is strictly read-only (0o555)."""
        ro_dir = os.path.join(self.temp_parent, "readonly_dir")
        os.makedirs(ro_dir, exist_ok=True)
        ro_file = os.path.join(ro_dir, "ro_frame.tif")
        tifffile.imwrite(ro_file, np.ones((10, 10), dtype=np.float32) * 7.0)

        os.chmod(ro_dir, 0o555)
        try:
            manager = SequenceManager([ro_file])
            frame = manager.get_frame(0)
            assert frame is not None
            assert frame[0, 0] == 7.0
            manager.set_derived_frame(0, "filtered", frame)
            ret = manager.get_derived_frame(0, "filtered")
            assert ret is not None
            manager.compute_median()
            assert manager.median_frame is not None
        finally:
            os.chmod(ro_dir, 0o755)

    def test_sequence_manager_tricky_filenames_and_lru_eviction(self) -> None:
        """Verify handling of tricky filenames with spaces, brackets, hashes, and LRU eviction."""
        tricky_dir = os.path.join(self.temp_parent, "Tricky [ALS 6.0.2] {spec#test}")
        os.makedirs(tricky_dir, exist_ok=True)
        paths = []
        for i in range(4):
            fpath = os.path.join(tricky_dir, f"frame #{i:02d} (run.seq_{i}) [2026].tif")
            tifffile.imwrite(fpath, np.full((10, 10), float(i + 1), dtype=np.float32))
            paths.append(fpath)

        manager = SequenceManager(paths, capacity=2)
        assert manager.n_frames == 4
        # Load all frames to trigger LRU eviction (capacity=2)
        for i in range(4):
            f = manager.get_frame(i)
            assert f is not None
            assert np.isclose(f[0, 0], float(i + 1))

        # Reload frame 0 which was evicted
        f0 = manager.get_frame(0)
        assert f0 is not None
        assert np.isclose(f0[0, 0], 1.0)

