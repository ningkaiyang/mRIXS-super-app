"""PySide6 unit tests for SortingView and DragDropListWidget (Milestone 3).

Tests:
1. Empty-state overlay card visibility, centering, and transparent mouse events.
2. External OS drag & drop events (Finder / Explorer URLs, directory scanning via glob_tifs).
3. Row formatting: anchor reference frame row 0 formatting, padded numbering for row 1+,
   and preservation of Qt.UserRole filepath data.
4. Alternating zebra striping enabled on QListWidget.
5. Dynamic button state synchronization (start, zeroth order, clear, remove).
6. Dynamic remove button text based on selection count (Remove Selected (N)).
7. Batch file addition helper (add_files) with deduplication, natural sort, and scan log discovery.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import tifffile
from PySide6.QtCore import Qt, QUrl, QMimeData, QPoint, QRect
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent, QResizeEvent
from PySide6.QtWidgets import QApplication

from rixs_app.ui.sorting_view import SortingView, DragDropListWidget, find_matching_scan_txt


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def sorting_widget(qapp, qtbot):
    """Instantiate SortingView."""
    view = SortingView()
    qtbot.addWidget(view)
    view.show()
    return view


def test_empty_state_overlay_initialized(sorting_widget):
    """Empty-state label must be visible initially when 0 files are loaded."""
    lw = sorting_widget.list_widget
    assert hasattr(lw, "_empty_label")
    assert not lw._empty_label.isHidden()
    assert "No TIFF Images Loaded" in lw._empty_label.text()
    assert lw._empty_label.testAttribute(Qt.WA_TransparentForMouseEvents)


def test_empty_state_overlay_toggles_with_files(sorting_widget):
    """Empty-state label toggles visibility when files are added and cleared."""
    lw = sorting_widget.list_widget
    assert not lw._empty_label.isHidden()

    sorting_widget.add_files(["frame_01.tif", "frame_02.tif"])
    assert lw._empty_label.isHidden()

    sorting_widget.clear_all()
    assert not lw._empty_label.isHidden()


def test_empty_state_resize_event(sorting_widget):
    """Empty-state label resizes to match list widget geometry."""
    lw = sorting_widget.list_widget
    ev = QResizeEvent(lw.size(), lw.size())
    lw.resizeEvent(ev)
    assert lw._empty_label.geometry() == lw.rect()


def test_row_badging_and_user_role(sorting_widget):
    """Row 0 must be badged with anchor reference, and rows 1+ numbered, while UserRole holds path."""
    files = ["/path/to/frame_02.tif", "/path/to/frame_01.tif", "/path/to/frame_03.tif"]
    sorting_widget.add_files(files)

    lw = sorting_widget.list_widget
    assert lw.count() == 3

    # Check alternating row colors enabled
    assert lw.alternatingRowColors()

    # After natural sort: frame_01, frame_02, frame_03
    item0 = lw.item(0)
    assert "⭐ [REF] 01. frame_01.tif (Anchor Reference Frame)" in item0.text()
    assert item0.data(Qt.UserRole) == "/path/to/frame_01.tif"

    item1 = lw.item(1)
    assert "02. frame_02.tif" in item1.text()
    assert "⭐" not in item1.text()
    assert item1.data(Qt.UserRole) == "/path/to/frame_02.tif"

    item2 = lw.item(2)
    assert "03. frame_03.tif" in item2.text()
    assert item2.data(Qt.UserRole) == "/path/to/frame_03.tif"


def test_dynamic_button_states_empty(sorting_widget):
    """When list is empty, start, zeroth order, clear, and remove buttons are disabled."""
    assert not sorting_widget.start_button.isEnabled()
    assert not sorting_widget.zeroth_order_button.isEnabled()
    assert not sorting_widget.clear_button.isEnabled()
    assert not sorting_widget.remove_button.isEnabled()


def test_dynamic_button_states_single_file(sorting_widget):
    """With 1 file, clear is enabled, start/zeroth order need >= 2 files."""
    sorting_widget.add_files(["frame_01.tif"])
    assert not sorting_widget.start_button.isEnabled()
    assert not sorting_widget.zeroth_order_button.isEnabled()
    assert sorting_widget.clear_button.isEnabled()


def test_dynamic_button_states_multiple_files(sorting_widget):
    """With >= 2 files, start, zeroth order, and clear are enabled."""
    sorting_widget.add_files(["frame_01.tif", "frame_02.tif"])
    assert sorting_widget.start_button.isEnabled()
    assert sorting_widget.zeroth_order_button.isEnabled()
    assert sorting_widget.clear_button.isEnabled()


def test_remove_button_selection_sync(sorting_widget):
    """Remove button reflects selection state and count."""
    sorting_widget.add_files(["frame_01.tif", "frame_02.tif", "frame_03.tif"])
    lw = sorting_widget.list_widget

    # Clear selections
    lw.clearSelection()
    sorting_widget.selected_index = -1
    sorting_widget._sync_button_states()
    assert not sorting_widget.remove_button.isEnabled()
    assert sorting_widget.remove_button.text() == "✕ Remove Selected"

    # Select single item
    lw.item(0).setSelected(True)
    assert sorting_widget.remove_button.isEnabled()
    assert sorting_widget.remove_button.text() == "✕ Remove Selected"

    # Select multiple items
    lw.item(1).setSelected(True)
    assert sorting_widget.remove_button.isEnabled()
    assert sorting_widget.remove_button.text() == "✕ Remove Selected (2)"


def test_remove_multiple_selected_files(sorting_widget):
    """Removing multiple selected items pops them all correctly."""
    sorting_widget.add_files(["a.tif", "b.tif", "c.tif", "d.tif"])
    lw = sorting_widget.list_widget

    # Select 'b.tif' (idx 1) and 'd.tif' (idx 3)
    lw.item(1).setSelected(True)
    lw.item(3).setSelected(True)

    sorting_widget.remove_file()
    assert sorting_widget.file_list == ["a.tif", "c.tif"]
    assert lw.count() == 2
    assert "⭐ [REF] 01. a.tif" in lw.item(0).text()
    assert "02. c.tif" in lw.item(1).text()


def test_add_files_deduplication_and_sorting(sorting_widget):
    """add_files deduplicates entries and applies natural sorting."""
    sorting_widget.add_files(["frame_10.tif", "frame_2.tif"])
    sorting_widget.add_files(["frame_1.tif", "frame_2.tif", "frame_10.tif"])

    assert sorting_widget.file_list == ["frame_1.tif", "frame_2.tif", "frame_10.tif"]


def test_add_files_directory_scan(tmp_path, sorting_widget):
    """add_files expands directories into contained TIFF files."""
    d = tmp_path / "scan_dir"
    d.mkdir()
    f1 = d / "f_2.tif"
    f2 = d / "f_1.tif"
    f3 = d / "notes.txt"
    f1.write_text("data")
    f2.write_text("data")
    f3.write_text("scan log info")

    sorting_widget.add_files([str(d)])
    assert len(sorting_widget.file_list) == 2
    assert sorting_widget.file_list[0].endswith("f_1.tif")
    assert sorting_widget.file_list[1].endswith("f_2.tif")
    assert sorting_widget.detected_scan_txt == str(f3)


def test_find_matching_scan_txt(tmp_path):
    """find_matching_scan_txt locates .txt scan logs in file directories."""
    d = tmp_path / "dataset"
    d.mkdir()
    txt = d / "Scan_001.txt"
    txt.write_text("Header\n...")
    tif = d / "frame_01.tif"
    tif.write_text("img")

    found = find_matching_scan_txt([str(tif)])
    assert found == str(txt)

    # If no txt in directory
    d2 = tmp_path / "empty_dir"
    d2.mkdir()
    tif2 = d2 / "frame_01.tif"
    tif2.write_text("img")
    assert find_matching_scan_txt([str(tif2)]) is None


def test_external_drag_drop_event(tmp_path, sorting_widget):
    """DragDropListWidget accepts external URL drops and adds files."""
    d = tmp_path / "drop_dir"
    d.mkdir()
    tif1 = d / "img_1.tif"
    tif2 = d / "img_2.tif"
    tif1.write_text("1")
    tif2.write_text("2")

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(d))])

    lw = sorting_widget.list_widget

    # Test dragEnterEvent
    enter_event = QDragEnterEvent(
        QPoint(10, 10),
        Qt.CopyAction,
        mime,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    lw.dragEnterEvent(enter_event)
    assert enter_event.isAccepted()

    # Test dragMoveEvent
    move_event = QDragMoveEvent(
        QPoint(10, 10),
        Qt.CopyAction,
        mime,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    lw.dragMoveEvent(move_event)
    assert move_event.isAccepted()

    # Test dropEvent
    drop_event = QDropEvent(
        QPoint(10, 10),
        Qt.CopyAction,
        mime,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    lw.dropEvent(drop_event)
    assert drop_event.isAccepted()
    assert len(sorting_widget.file_list) == 2


def test_find_matching_scan_txt_numeric_scan_id_parent_dir(tmp_path):
    """find_matching_scan_txt discovers sibling .txt in parent directory using \\d{4,} numeric scan ID."""
    parent_dir = tmp_path / "beamline_data"
    parent_dir.mkdir(parents=True, exist_ok=True)

    scan_txt = parent_dir / "004202_Fe_L3_scan.txt"
    scan_txt.write_text("Scan log metadata...")

    data_dir = parent_dir / "004202_Fe_L3_scan"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame1 = data_dir / "frame_001.tif"
    frame1.write_bytes(b"dummy")

    discovered = find_matching_scan_txt([str(frame1)])
    assert discovered is not None
    assert os.path.abspath(discovered) == str(scan_txt.resolve())


def test_find_matching_scan_txt_multiple_parent_matches_sorted(tmp_path):
    """Multiple matching .txt files in parent dir return the first lexicographically sorted file."""
    parent = tmp_path / "parent_multi"
    parent.mkdir()
    txt_b = parent / "004202_scan_b.txt"
    txt_a = parent / "004202_scan_a.txt"
    txt_b.write_text("b")
    txt_a.write_text("a")

    data = parent / "004202_data"
    data.mkdir()
    frame = data / "f_01.tif"
    frame.write_bytes(b"data")

    discovered = find_matching_scan_txt([str(frame)])
    assert discovered is not None
    assert os.path.basename(discovered) == "004202_scan_a.txt"


def test_find_matching_scan_txt_empty_and_invalid():
    """Empty list or non-existent paths return None without error."""
    assert find_matching_scan_txt([]) is None
    assert find_matching_scan_txt(["/non/existent/path/004202_scan/f.tif"]) is None


def test_sorting_view_back_to_home_button(qapp, qtbot):
    """SortingView provides ❮ Back to Home button that invokes on_back callback."""
    mock_back = MagicMock()
    view = SortingView(on_back=mock_back)
    qtbot.addWidget(view)
    view.show()

    assert hasattr(view, "_back_btn")
    assert view._back_btn.text() == "❮ Back to Home"
    view._back_btn.click()
    assert mock_back.call_count == 1


def test_sorting_view_copilot_docking(qapp, qtbot):
    """SortingView docks Co-Pilot toggle button into header."""
    from PySide6.QtWidgets import QPushButton

    view = SortingView()
    qtbot.addWidget(view)
    view.show()

    btn = QPushButton("🤖 Co-Pilot")
    view.set_copilot_button(btn)
    assert view.isAncestorOf(btn)
    assert btn.parent() == view

