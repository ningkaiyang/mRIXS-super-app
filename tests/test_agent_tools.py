"""Unit tests for RIXS Co-Pilot agent tools: CLI helper whitelist and 8-view stack GUI state mapping."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from rixs_app.agent.tools import ToolRegistry, create_default_registry


def test_get_cli_help_allowed_scripts():
    """Verify get_cli_help succeeds for cluster_cli.py and other allowed scripts, and rejects unknown scripts."""
    async def _run():
        registry = create_default_registry()

        # Verify cluster_cli.py succeeds and returns valid help text
        res = await registry.execute("get_cli_help", {"cli_script": "cluster_cli.py"})
        assert "usage: cluster_cli.py" in res or "Single-Photon Clustering" in res
        assert "dark-mask" in res or "cluster" in res

        # Verify align_cli.py
        res_align = await registry.execute("get_cli_help", {"cli_script": "align_cli.py"})
        assert "usage: align_cli.py" in res_align or "alignment" in res_align.lower()

        # Verify unknown script rejection
        res_unknown = await registry.execute("get_cli_help", {"cli_script": "malicious.py"})
        assert "Error: Unknown script 'malicious.py'" in res_unknown
        assert "cluster_cli.py" in res_unknown
        assert "align_cli.py" in res_unknown
        assert "zeroth_order_cli.py" in res_unknown
        assert "denoise_cli.py" in res_unknown

    asyncio.run(_run())


def test_cli_runner_allows_cluster_cli():
    """Verify cli_runner executes cluster_cli.py with --help."""
    async def _run():
        registry = create_default_registry()
        res = await registry.execute("cli_runner", {"command": "cluster_cli.py --help"})
        assert "cluster_cli.py" in res
        assert "Single-Photon Clustering" in res

    asyncio.run(_run())


def test_get_active_gui_state_all_8_views():
    """Verify get_active_gui_state correctly maps indices 0..7 to canonical view names and inspects view details."""
    async def _run():
        registry = create_default_registry()

        mock_main_window = MagicMock()
        registry.set_gui_context(mock_main_window)

        # 0: HomeLaunchpadView
        mock_main_window.home_view = MagicMock()
        # 1: DarkMaskingView
        mock_main_window.dark_mask_view = MagicMock()
        mock_main_window.dark_mask_view.dark_paths = ["/path/to/dark/frame_001.tif", "/path/to/dark/frame_002.tif"]
        mock_main_window.dark_mask_view.dark_frame_count = 2
        mock_main_window.dark_mask_view._stddev_thresh = 40.0
        mock_main_window.dark_mask_view._absdev_thresh = 60.0
        mock_main_window.dark_mask_view._diagnostics = MagicMock()
        # 2: ClusteringFileSelectionView
        mock_main_window.clustering_file_view = MagicMock()
        mock_main_window.clustering_file_view.signal_paths = ["/path/to/signal/frame_001.tif"]
        mock_main_window.clustering_file_view.chunk_size = 80
        mock_main_window.clustering_file_view._has_valid_cal = True
        # 3: ClusteringStudioView
        mock_main_window.clustering_studio_view = MagicMock()
        mock_main_window.clustering_studio_view.manager = MagicMock()
        mock_main_window.clustering_studio_view.manager.total_frames = 100
        mock_main_window.clustering_studio_view.manager.total_chunks = 2
        mock_main_window.clustering_studio_view.manager.state = MagicMock()
        mock_main_window.clustering_studio_view.manager.state.signal_paths = ["/path/to/signal/frame_001.tif"]
        mock_main_window.clustering_studio_view.manager.state.is_processing = False
        mock_main_window.clustering_studio_view.manager.state.df_clusters = [1, 2, 3]
        mock_main_window.clustering_studio_view._current_frame_idx = 10
        mock_main_window.clustering_studio_view._current_chunk_idx = 0
        mock_main_window.clustering_studio_view.active_mode = "dashboard"
        # 4: SortingView
        mock_main_window.sorting_view = MagicMock()
        mock_main_window.sorting_view.file_list = ["/path/to/align/file_001.tif", "/path/to/align/file_002.tif"]
        # 5: SlideshowView
        mock_main_window.slideshow_view = MagicMock()
        mock_main_window.slideshow_view._manager = MagicMock()
        mock_main_window.slideshow_view._manager._n_frames = 50
        mock_main_window.slideshow_view._manager._current_idx = 5
        mock_main_window.slideshow_view._manager._directory = "/path/to/align"
        # 6: ExportComparisonView
        mock_main_window.export_comparison_view = MagicMock()
        mock_main_window.export_comparison_view.aligned_sum = MagicMock()
        mock_main_window.export_comparison_view.default_save_dir = "/path/to/export"
        # 7: ZerothOrderSlideshowView
        mock_main_window.zeroth_order_view = MagicMock()
        mock_main_window.zeroth_order_view._manager = MagicMock()
        mock_main_window.zeroth_order_view._manager._n_frames = 20
        mock_main_window.zeroth_order_view._manager._current_idx = 2
        mock_main_window.zeroth_order_view._manager._directory = "/path/to/zeroth"
        mock_main_window.zeroth_order_view._manager._txt_path = "/path/to/scan.txt"

        expected_views = {
            0: "HomeLaunchpadView",
            1: "DarkMaskingView",
            2: "ClusteringFileSelectionView",
            3: "ClusteringStudioView",
            4: "SortingView",
            5: "SlideshowView",
            6: "ExportComparisonView",
            7: "ZerothOrderSlideshowView",
        }

        for idx, expected_name in expected_views.items():
            mock_main_window._stack.currentIndex.return_value = idx
            raw = await registry.execute("get_active_gui_state", {})
            state = json.loads(raw)
            assert state["active_view"] == expected_name, f"Index {idx} should be {expected_name}, got {state['active_view']}"

            # Verify view-specific details
            if idx == 1:
                assert state["details"]["file_count"] == 2
                assert state["details"]["stddev_thresh"] == 40.0
                assert state["details"]["absdev_thresh"] == 60.0
                assert state["details"]["has_diagnostics"] is True
                assert state["details"]["first_file"] == "/path/to/dark/frame_001.tif"
                assert state["details"]["directory"] == "/path/to/dark"
            elif idx == 2:
                assert state["details"]["file_count"] == 1
                assert state["details"]["chunk_size"] == 80
                assert state["details"]["has_valid_cal"] is True
                assert state["details"]["directory"] == "/path/to/signal"
            elif idx == 3:
                assert state["details"]["file_count"] == 100
                assert state["details"]["total_chunks"] == 2
                assert state["details"]["cluster_count"] == 3
                assert state["details"]["current_frame"] == 10
                assert state["details"]["active_mode"] == "dashboard"
            elif idx == 4:
                assert state["details"]["file_count"] == 2
                assert state["details"]["first_file"] == "/path/to/align/file_001.tif"
                assert state["details"]["directory"] == "/path/to/align"
            elif idx == 5:
                assert state["details"]["file_count"] == 50
                assert state["details"]["current_frame"] == 5
                assert state["details"]["directory"] == "/path/to/align"
            elif idx == 6:
                assert state["details"]["has_aligned_sum"] is True
                assert state["details"]["directory"] == "/path/to/export"
            elif idx == 7:
                assert state["details"]["file_count"] == 20
                assert state["details"]["current_frame"] == 2
                assert state["details"]["directory"] == "/path/to/zeroth"
                assert state["details"]["has_scan_log"] is True

    asyncio.run(_run())


def test_update_gui_parameter_8_view_mapping():
    """Verify update_gui_parameter correctly handles frame_index navigation across 8-view stack indices."""
    async def _run():
        registry = create_default_registry()

        mock_main_window = MagicMock()
        registry.set_gui_context(mock_main_window)

        # Index 5 is SlideshowView
        mock_main_window._stack.currentIndex.return_value = 5
        mock_main_window.slideshow_view.next_frame = MagicMock()
        res = await registry.execute("update_gui_parameter", {"parameter": "frame_index", "value": "15"})
        assert "Frame navigation to index 15 requested" in res

        # Index 7 is ZerothOrderSlideshowView
        mock_main_window._stack.currentIndex.return_value = 7
        mock_main_window.zeroth_order_view.next_frame = MagicMock()
        res = await registry.execute("update_gui_parameter", {"parameter": "frame_index", "value": "5"})
        assert "Frame navigation to index 5 requested" in res

        # Index 3 is ClusteringStudioView
        mock_main_window._stack.currentIndex.return_value = 3
        mock_main_window.clustering_studio_view = MagicMock()
        res = await registry.execute("update_gui_parameter", {"parameter": "frame_index", "value": "20"})
        assert "Frame navigation to index 20 requested" in res

        # Non-slideshow view: index 0 (HomeLaunchpadView)
        mock_main_window._stack.currentIndex.return_value = 0
        res = await registry.execute("update_gui_parameter", {"parameter": "frame_index", "value": "5"})
        assert "Error: No active slideshow view." in res

        # Non-slideshow view: index 4 (SortingView)
        mock_main_window._stack.currentIndex.return_value = 4
        res = await registry.execute("update_gui_parameter", {"parameter": "frame_index", "value": "5"})
        assert "Error: No active slideshow view." in res

        # Unknown parameter
        res = await registry.execute("update_gui_parameter", {"parameter": "invalid_param", "value": "1"})
        assert "Error: Unknown parameter 'invalid_param'" in res

    asyncio.run(_run())
