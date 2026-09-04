"""Tests for CompressedFrameCache pure-core component."""

import concurrent.futures
import sys
import numpy as np
import pytest

from rixs_app.core.frame_cache import CompressedFrameCache


class TestCompressedFrameCache:
    """Test suite for CompressedFrameCache."""

    def test_init_defaults_and_validation(self) -> None:
        """Test cache initialization defaults and capacity validation."""
        cache = CompressedFrameCache()
        assert cache.capacity == 128
        assert len(cache) == 0
        assert cache.memory_usage_mb() == 0.0

        cache_custom = CompressedFrameCache(capacity=10, compression_level=3)
        assert cache_custom.capacity == 10
        assert len(cache_custom) == 0

        with pytest.raises(ValueError, match="capacity must be greater than 0"):
            CompressedFrameCache(capacity=0)

        with pytest.raises(ValueError, match="capacity must be greater than 0"):
            CompressedFrameCache(capacity=-5)

    def test_put_get_basic(self) -> None:
        """Test putting and retrieving frames with float32 return type."""
        cache = CompressedFrameCache(capacity=5)
        frame = np.zeros((100, 200), dtype=np.float32)
        frame[10:20, 30:40] = 42.5

        assert not cache.has(0)
        cache.put(0, frame)

        assert cache.has(0)
        assert not cache.has(1)
        assert len(cache) == 1

        retrieved = cache.get(0)
        assert retrieved is not None
        assert retrieved.dtype == np.float32
        assert retrieved.shape == (100, 200)
        np.testing.assert_allclose(retrieved, frame, atol=1e-2)

    def test_get_nonexistent_returns_none(self) -> None:
        """Test retrieving non-existent index returns None."""
        cache = CompressedFrameCache(capacity=5)
        assert cache.get(999) is None

    def test_compression_efficiency_and_fidelity(self) -> None:
        """Verify sparse 2048x3840 frame achieves massive compression (~2000x) and < 5MB for 128 frames."""
        cache = CompressedFrameCache(capacity=128, compression_level=1)

        # 2048x3840 detector frame, 99.8% background zeros
        frame = np.zeros((2048, 3840), dtype=np.float32)
        np.random.seed(42)
        sparse_y = np.random.randint(0, 2048, size=1000)
        sparse_x = np.random.randint(0, 3840, size=1000)
        frame[sparse_y, sparse_x] = np.random.uniform(50.0, 500.0, size=1000)

        raw_size_mb = frame.nbytes / (1024 * 1024)
        assert raw_size_mb >= 30.0  # 30.0 MB uncompressed (2048x3840 float32)

        cache.put(0, frame)
        mem_mb = cache.memory_usage_mb()

        # Compressed frame should be < 0.05 MB (50 KB)
        assert mem_mb < 0.05
        # Compression ratio > 500x
        compression_ratio = raw_size_mb / mem_mb
        assert compression_ratio > 500.0

        # Verify fidelity
        restored = cache.get(0)
        assert restored is not None
        assert restored.shape == frame.shape
        assert restored.dtype == np.float32
        np.testing.assert_allclose(restored, frame, rtol=1e-3, atol=0.2)
        np.testing.assert_array_equal(restored, frame.astype(np.float16).astype(np.float32))

    def test_lru_eviction(self) -> None:
        """Test LRU eviction policy when capacity is reached."""
        cache = CompressedFrameCache(capacity=3)

        f0 = np.full((10, 10), 0.0, dtype=np.float32)
        f1 = np.full((10, 10), 1.0, dtype=np.float32)
        f2 = np.full((10, 10), 2.0, dtype=np.float32)
        f3 = np.full((10, 10), 3.0, dtype=np.float32)
        f4 = np.full((10, 10), 4.0, dtype=np.float32)

        cache.put(0, f0)
        cache.put(1, f1)
        cache.put(2, f2)
        assert len(cache) == 3

        # Access index 0 to mark it recently used (order now: 1, 2, 0)
        assert cache.get(0) is not None

        # Insert index 3 -> least recently used (index 1) must be evicted
        cache.put(3, f3)
        assert len(cache) == 3
        assert cache.has(0)
        assert not cache.has(1)
        assert cache.has(2)
        assert cache.has(3)

        # Insert index 4 -> least recently used (index 2) must be evicted
        cache.put(4, f4)
        assert len(cache) == 3
        assert cache.has(0)
        assert not cache.has(2)
        assert cache.has(3)
        assert cache.has(4)

    def test_overwrite_key(self) -> None:
        """Test overwriting an existing key updates content and maintains correct memory usage."""
        cache = CompressedFrameCache(capacity=3)
        f_small = np.zeros((10, 10), dtype=np.float32)
        f_large = np.ones((100, 100), dtype=np.float32)

        cache.put(0, f_small)
        mem1 = cache.memory_usage_mb()
        assert len(cache) == 1

        cache.put(0, f_large)
        mem2 = cache.memory_usage_mb()
        assert len(cache) == 1

        retrieved = cache.get(0)
        assert retrieved is not None
        assert retrieved.shape == (100, 100)
        np.testing.assert_allclose(retrieved, f_large, atol=1e-2)
        assert mem2 > 0.0

    def test_preload_batch(self) -> None:
        """Test batch preloading of multiple frames."""
        cache = CompressedFrameCache(capacity=5)
        frames = {
            10: np.full((20, 20), 10.0, dtype=np.float32),
            20: np.full((20, 20), 20.0, dtype=np.float32),
            30: np.full((20, 20), 30.0, dtype=np.float32),
        }
        cache.preload_batch(frames)

        assert len(cache) == 3
        for idx in (10, 20, 30):
            assert cache.has(idx)
            ret = cache.get(idx)
            assert ret is not None
            np.testing.assert_allclose(ret, frames[idx], atol=1e-2)

    def test_preload_batch_exceeds_capacity(self) -> None:
        """Test batch preloading when batch size exceeds cache capacity."""
        cache = CompressedFrameCache(capacity=2)
        frames = {
            1: np.full((10, 10), 1.0, dtype=np.float32),
            2: np.full((10, 10), 2.0, dtype=np.float32),
            3: np.full((10, 10), 3.0, dtype=np.float32),
        }
        cache.preload_batch(frames)
        assert len(cache) == 2
        assert not cache.has(1)
        assert cache.has(2)
        assert cache.has(3)

    def test_clear(self) -> None:
        """Test clear empties cache and resets memory tracking."""
        cache = CompressedFrameCache(capacity=5)
        cache.put(0, np.zeros((50, 50), dtype=np.float32))
        cache.put(1, np.ones((50, 50), dtype=np.float32))

        assert len(cache) == 2
        assert cache.memory_usage_mb() > 0.0

        cache.clear()
        assert len(cache) == 0
        assert cache.memory_usage_mb() == 0.0
        assert not cache.has(0)
        assert cache.get(0) is None

    def test_dtype_flexibility(self) -> None:
        """Test non-float32 input arrays (e.g. uint16, float64) are converted to float32 on get."""
        cache = CompressedFrameCache(capacity=5)

        u16_frame = np.array([[10, 20], [30, 40]], dtype=np.uint16)
        f64_frame = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)

        cache.put(1, u16_frame)
        cache.put(2, f64_frame)

        ret1 = cache.get(1)
        ret2 = cache.get(2)

        assert ret1 is not None and ret1.dtype == np.float32
        np.testing.assert_allclose(ret1, u16_frame.astype(np.float32), atol=1e-2)

        assert ret2 is not None and ret2.dtype == np.float32
        np.testing.assert_allclose(ret2, f64_frame.astype(np.float32), atol=1e-2)

    def test_thread_safety(self) -> None:
        """Test concurrent reads and writes from multiple threads."""
        cache = CompressedFrameCache(capacity=50)
        num_threads = 8
        ops_per_thread = 40

        def worker(thread_id: int) -> None:
            for i in range(ops_per_thread):
                idx = (thread_id * 10) + (i % 20)
                frame = np.full((32, 32), float(idx), dtype=np.float32)
                cache.put(idx, frame)
                assert cache.has(idx)
                ret = cache.get(idx)
                if ret is not None:
                    assert ret.shape == (32, 32)
                    assert ret.dtype == np.float32
                _ = cache.memory_usage_mb()
                _ = len(cache)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, t) for t in range(num_threads)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        assert len(cache) <= 50
        assert cache.memory_usage_mb() >= 0.0

    def test_zero_fan_out(self) -> None:
        """Verify rixs_app.core.frame_cache does not import PySide6 or Qt."""
        import rixs_app.core.frame_cache as fc_mod

        assert not hasattr(fc_mod, "PySide6")
        assert not hasattr(fc_mod, "QtCore")
        assert not hasattr(fc_mod, "QtWidgets")
        for mod_name in sys.modules:
            if "rixs_app.core.frame_cache" in mod_name:
                module = sys.modules[mod_name]
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    assert "PySide6" not in getattr(attr, "__module__", "")
                    assert "QtCore" not in getattr(attr, "__module__", "")
                    assert "QtWidgets" not in getattr(attr, "__module__", "")


class TestSequenceManagerCache:
    """Integration tests verifying SequenceManager uses CompressedFrameCache."""

    def test_dataset_manager_mem_cache_initialization(self, tmp_path) -> None:
        """Verify SequenceManager initializes _mem_cache and _derived_mem_cache."""
        from rixs_app.core.dataset import SequenceManager
        tif_path = tmp_path / "frame_0.tif"
        import tifffile
        tifffile.imwrite(str(tif_path), np.ones((20, 20), dtype=np.float32))

        manager = SequenceManager([str(tif_path)])
        assert hasattr(manager, "_mem_cache")
        assert isinstance(manager._mem_cache, CompressedFrameCache)
        assert manager._mem_cache.capacity == 128
        assert hasattr(manager, "_derived_mem_cache")
        assert isinstance(manager._derived_mem_cache, dict)

    def test_dataset_manager_get_frame_caches_in_mem_cache(self, tmp_path) -> None:
        """Verify get_frame stores result in _mem_cache and returns identical data on hit."""
        from rixs_app.core.dataset import SequenceManager
        tif_path = tmp_path / "frame_0.tif"
        import tifffile
        original = np.linspace(0, 100, 400, dtype=np.float32).reshape(20, 20)
        tifffile.imwrite(str(tif_path), original)

        manager = SequenceManager([str(tif_path)])
        # Before read, index 0 is not in memory cache
        assert not manager._mem_cache.has(0)

        # First read: populates _mem_cache
        data1 = manager.get_frame(0)
        assert data1 is not None
        assert manager._mem_cache.has(0)
        np.testing.assert_allclose(data1, original, atol=0.1)

        # Overwrite file on disk to ensure subsequent get_frame(0) hits memory cache
        tifffile.imwrite(str(tif_path), np.zeros((20, 20), dtype=np.float32))
        data2 = manager.get_frame(0)
        assert data2 is not None
        # Must still return original cached data from _mem_cache
        np.testing.assert_allclose(data2, original, atol=0.1)

    def test_dataset_manager_set_frame(self, tmp_path) -> None:
        """Verify set_frame populates _mem_cache strictly in memory."""
        from rixs_app.core.dataset import SequenceManager
        tif_path = tmp_path / "frame_0.tif"
        import tifffile
        tifffile.imwrite(str(tif_path), np.zeros((20, 20), dtype=np.float32))

        manager = SequenceManager([str(tif_path)])
        new_frame = np.full((20, 20), 42.0, dtype=np.float32)
        manager.set_frame(0, new_frame)

        assert manager._mem_cache.has(0)
        ret = manager.get_frame(0)
        assert ret is not None
        np.testing.assert_allclose(ret, new_frame, atol=0.1)

    def test_dataset_manager_derived_mem_cache(self, tmp_path) -> None:
        """Verify get_derived_frame and set_derived_frame utilize _derived_mem_cache."""
        from rixs_app.core.dataset import SequenceManager
        tif_path = tmp_path / "frame_0.tif"
        import tifffile
        tifffile.imwrite(str(tif_path), np.zeros((20, 20), dtype=np.float32))

        manager = SequenceManager([str(tif_path)])
        denoised = np.full((20, 20), 99.0, dtype=np.float32)

        # Before setting, derived cache for "denoised_img" is empty or not created
        assert manager.get_derived_frame(0, "denoised_img") is None

        manager.set_derived_frame(0, "denoised_img", denoised)
        assert "denoised_img" in manager._derived_mem_cache
        assert manager._derived_mem_cache["denoised_img"].has(0)

        ret = manager.get_derived_frame(0, "denoised_img")
        assert ret is not None
        np.testing.assert_allclose(ret, denoised, atol=0.1)

    def test_dataset_zero_fan_out(self) -> None:
        """Verify rixs_app.core.dataset does not import PySide6 or Qt."""
        import rixs_app.core.dataset as ds_mod

        assert not hasattr(ds_mod, "PySide6")
        assert not hasattr(ds_mod, "QtCore")
        assert not hasattr(ds_mod, "QtWidgets")
        for mod_name in sys.modules:
            if "rixs_app.core.dataset" in mod_name:
                module = sys.modules[mod_name]
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    assert "PySide6" not in getattr(attr, "__module__", "")
                    assert "QtCore" not in getattr(attr, "__module__", "")
                    assert "QtWidgets" not in getattr(attr, "__module__", "")
