"""Sorting view — PySide6 port.

Allows users to select and manage a list of TIFF files before launching
the alignment slideshow or zeroth-order calibration. Files are auto-sorted
naturally on selection and support native drag-and-drop item reordering.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QFileDialog, QDialog,
    QScrollArea, QDialogButtonBox, QMessageBox, QAbstractItemView,
)

from rixs_app.core import natural_sort
from rixs_app.core.cli_utils import glob_tifs
from rixs_app.ui.theme import (
    set_sort_btn, set_danger_btn, set_danger_secondary_btn,
    set_success_btn, set_tool_btn, set_accent_btn,
)


def find_matching_scan_txt(file_list: list[str]) -> str | None:
    """Automatically discover scan log (.txt) file across dataset directories.

    Args:
        file_list: List of image file paths.

    Returns:
        Absolute path to the first .txt scan log found, or None.
    """
    seen_dirs = dict.fromkeys(os.path.dirname(f) for f in file_list if f)
    for directory in seen_dirs:
        if directory and os.path.isdir(directory):
            try:
                txt_files = [
                    os.path.join(directory, f)
                    for f in sorted(os.listdir(directory))
                    if f.lower().endswith(".txt")
                ]
                if txt_files:
                    return txt_files[0]
            except OSError:
                continue
    return None


class DragDropListWidget(QListWidget):
    """Custom QListWidget with native drag-and-drop item reordering and external OS drop."""

    def __init__(self, parent=None, on_reordered=None):
        super().__init__(parent)
        self.parent_view = parent
        self.on_reordered = on_reordered
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAcceptDrops(True)

        # Empty state overlay card
        self._empty_label = QLabel(
            "<div style='text-align: center; color: #6b7280;'>"
            "<div style='font-size: 38px; margin-bottom: 8px;'>📂</div>"
            "<div style='font-size: 15px; font-weight: bold; color: #9ca3af; margin-bottom: 6px;'>"
            "No TIFF Images Loaded"
            "</div>"
            "<div style='font-size: 13px; color: #6b7280;'>"
            "Drag &amp; drop TIFF files or scan folders here"
            "</div>"
            "<div style='font-size: 12px; color: #4b5563; margin-top: 4px;'>"
            "or click 'Select TIFF Files' above"
            "</div>"
            "</div>",
            self,
        )
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._empty_label.setGeometry(self.rect())
        self.update_empty_state()

    def resizeEvent(self, event) -> None:
        """Keep empty-state overlay centered across resizes."""
        super().resizeEvent(event)
        if hasattr(self, "_empty_label"):
            self._empty_label.setGeometry(self.rect())

    def update_empty_state(self) -> None:
        """Toggle visibility of empty-state overlay card based on item count."""
        if hasattr(self, "_empty_label"):
            self._empty_label.setGeometry(self.rect())
            self._empty_label.setVisible(self.count() == 0)

    def dragEnterEvent(self, event) -> None:
        """Accept external OS file/folder drag events or internal drags."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        """Accept external OS drag move events or internal drags."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        """Handle drops from external OS (Finder/Explorer) or internal item reorder."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            urls = event.mimeData().urls()
            paths = [url.toLocalFile() for url in urls if url.isLocalFile()]
            if paths:
                tif_files: list[str] = []
                for p in paths:
                    if os.path.isdir(p):
                        tif_files.extend(glob_tifs(p))
                    elif os.path.isfile(p) and p.lower().endswith(('.tif', '.tiff')):
                        tif_files.append(os.path.abspath(p))
                if tif_files:
                    target = self.parent_view if self.parent_view is not None else self.parent()
                    if hasattr(target, "add_files"):
                        target.add_files(tif_files)
        else:
            super().dropEvent(event)
            if self.on_reordered is not None:
                self.on_reordered()
        self.update_empty_state()


class SortingView(QWidget):
    """Main sorting/workspace view for the mRIXS Super-App.

    Provides file selection, automatic natural sorting, drag-and-drop reorder,
    file removal, and buttons to launch the alignment slideshow or
    zeroth-order calibration.

    Args:
        parent: Parent widget.
        on_start_slideshow: Callback invoked with a file list when the user
            clicks Start Alignment Slideshow.
        on_zeroth_order: Callback invoked with (file_list, txt_path=...) when
            the user clicks Zeroth-Order Calibration.
    """

    def __init__(
        self,
        parent=None,
        *,
        on_start_slideshow=None,
        on_zeroth_order=None,
    ):
        """Initialise the sorting view.

        Args:
            parent: Parent QWidget.
            on_start_slideshow: Callback for launching alignment slideshow.
            on_zeroth_order: Callback for launching zeroth-order calibration.
        """
        super().__init__(parent)
        self.on_start_slideshow = on_start_slideshow
        self.on_zeroth_order = on_zeroth_order
        self.file_list: list[str] = []
        self._selected_index: int = -1
        self.detected_scan_txt: str | None = None

        self._build_ui()

    @property
    def selected_index(self) -> int:
        """Current selected index property."""
        return self._selected_index

    @selected_index.setter
    def selected_index(self, idx: int) -> None:
        """Set current selected index and sync button states."""
        self._selected_index = idx
        if hasattr(self, "list_widget"):
            self._sync_button_states()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build and lay out all widgets."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(10)

        # Header row (title centered, Co-Pilot button docks at right)
        self._header_row = QHBoxLayout()
        self._header_row.setContentsMargins(0, 0, 0, 0)
        header_label = QLabel("mRIXS Super-App Workspace")
        header_label.setObjectName("header_title")
        header_label.setAlignment(Qt.AlignCenter)
        self._header_row.addWidget(header_label, stretch=1)
        outer.addLayout(self._header_row)

        # --- Row 1: Primary Select Files ---
        self.select_button = QPushButton("📂 Select TIFF Files")
        self.select_button.setFixedHeight(42)
        set_accent_btn(self.select_button)
        f = QFont(); f.setPointSize(14); f.setBold(True); self.select_button.setFont(f)
        self.select_button.clicked.connect(self.select_files)
        outer.addWidget(self.select_button)

        # --- Caption ---
        self.caption_label = QLabel("(Files auto-sorted naturally by name — drag items up/down to reorder)")
        self.caption_label.setObjectName("dim_label")
        self.caption_label.setAlignment(Qt.AlignCenter)
        outer.addWidget(self.caption_label)

        # --- Row 2: Remove / Clear All ---
        row2 = QHBoxLayout()
        self.remove_button = QPushButton("✕ Remove Selected")
        self.remove_button.setFixedHeight(36)
        set_danger_btn(self.remove_button)
        self.remove_button.clicked.connect(self.remove_file)
        row2.addWidget(self.remove_button)

        self.clear_button = QPushButton("🗑 Clear All")
        self.clear_button.setFixedHeight(36)
        set_danger_secondary_btn(self.clear_button)
        self.clear_button.clicked.connect(self.clear_all)
        row2.addWidget(self.clear_button)
        outer.addLayout(row2)

        # --- Launch buttons ---
        self.start_button = QPushButton("► Start Alignment Slideshow")
        self.start_button.setFixedHeight(40)
        set_success_btn(self.start_button)
        f = QFont(); f.setPointSize(14); f.setBold(True); self.start_button.setFont(f)
        self.start_button.clicked.connect(self.start_slideshow)
        outer.addWidget(self.start_button)

        self.zeroth_order_button = QPushButton(
            "🔬 Zeroth-Order Focus & FWHM Calibration"
        )
        self.zeroth_order_button.setFixedHeight(40)
        set_sort_btn(self.zeroth_order_button)
        f = QFont(); f.setPointSize(14); f.setBold(True); self.zeroth_order_button.setFont(f)
        self.zeroth_order_button.clicked.connect(self.start_zeroth_order)
        outer.addWidget(self.zeroth_order_button)

        self.help_button = QPushButton("❓ Guide + Credits")
        self.help_button.setFixedWidth(170)
        set_tool_btn(self.help_button)
        self.help_button.clicked.connect(self.show_help)
        outer.addWidget(self.help_button, alignment=Qt.AlignCenter)

        # --- File list (underneath all buttons) ---
        self.list_widget = DragDropListWidget(self, on_reordered=self._on_list_reordered)
        self.list_widget.itemClicked.connect(
            lambda item: self._select_item(self.list_widget.row(item))
        )
        self.list_widget.itemSelectionChanged.connect(self._sync_button_states)
        outer.addWidget(self.list_widget, stretch=1)

        self._sync_button_states()

    # ------------------------------------------------------------------
    # Actions & State Management
    # ------------------------------------------------------------------

    def _sync_button_states(self) -> None:
        """Dynamically synchronize action button enabled states and selection counts."""
        count = len(self.file_list)
        selected_items = self.list_widget.selectedItems()
        sel_count = len(selected_items)
        has_sel = (
            sel_count > 0
            or (count > 0 and 0 <= self._selected_index < count)
            or (count > 0 and 0 <= self.list_widget.currentRow() < count)
        )

        self.start_button.setEnabled(count >= 2)
        self.zeroth_order_button.setEnabled(count >= 2)
        self.clear_button.setEnabled(count > 0)
        self.remove_button.setEnabled(count > 0 and has_sel)

        if sel_count > 1:
            self.remove_button.setText(f"✕ Remove Selected ({sel_count})")
        else:
            self.remove_button.setText("✕ Remove Selected")

    def add_files(self, files: list[str]) -> None:
        """Batch append, deduplicate, naturally sort TIFF files, and update UI.

        Args:
            files: List of file or folder paths to add.
        """
        if not files:
            return

        expanded: list[str] = []
        for f in files:
            if not f or not isinstance(f, str):
                continue
            if os.path.isdir(f):
                expanded.extend(glob_tifs(f))
            else:
                expanded.append(f)

        if not expanded:
            return

        existing = set(self.file_list)
        for item in expanded:
            if item not in existing:
                self.file_list.append(item)
                existing.add(item)

        self.file_list = natural_sort(self.file_list)
        self.detected_scan_txt = find_matching_scan_txt(self.file_list)
        self.update_listbox()
        self._sync_button_states()

    def select_files(self) -> None:
        """Open a file dialog to choose multiple TIFF files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select TIFF files", "",
            "TIFF Files (*.tif *.tiff)"
        )
        if files:
            self.add_files(list(files))

    def _on_list_reordered(self) -> None:
        """Synchronise internal file_list order with visual drag-drop item order."""
        new_list = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            filepath = item.data(Qt.UserRole)
            if filepath:
                new_list.append(filepath)
        self.file_list = new_list
        self.update_listbox()

    def _select_item(self, idx: int) -> None:
        """Select an item by index.

        Args:
            idx: The item index.
        """
        self.selected_index = idx
        self._sync_button_states()

    def remove_file(self) -> None:
        """Remove the currently selected file(s)."""
        selected_items = self.list_widget.selectedItems()
        if selected_items:
            selected_rows = sorted(
                [self.list_widget.row(item) for item in selected_items],
                reverse=True,
            )
            for r in selected_rows:
                if 0 <= r < len(self.file_list):
                    self.file_list.pop(r)
            if self.file_list:
                min_row = min(selected_rows)
                self.selected_index = min(min_row, len(self.file_list) - 1)
            else:
                self.selected_index = -1
            self.update_listbox()
            return

        # Fallback to currentRow or selected_index
        idx = self.list_widget.currentRow()
        if idx < 0 and 0 <= self._selected_index < len(self.file_list):
            idx = self._selected_index
        if 0 <= idx < len(self.file_list):
            self.file_list.pop(idx)
            if self.file_list:
                self.selected_index = min(idx, len(self.file_list) - 1)
            else:
                self.selected_index = -1
            self.update_listbox()

    def remove_selected(self) -> None:
        """Alias for remove_file."""
        self.remove_file()

    def clear_all(self) -> None:
        """Clear all files from the list."""
        self.file_list.clear()
        self.selected_index = -1
        self.detected_scan_txt = None
        self.update_listbox()
        self._sync_button_states()

    def start_slideshow(self) -> None:
        """Trigger the alignment slideshow callback."""
        if self.on_start_slideshow and self.file_list:
            self.on_start_slideshow(self.file_list)

    def start_zeroth_order(self) -> None:
        """Automatically find scan log across all selected TIFF directories, then trigger zeroth-order callback."""
        if not self.on_zeroth_order or not self.file_list:
            return

        txt_path = self.detected_scan_txt or find_matching_scan_txt(self.file_list)

        if not txt_path:
            QMessageBox.warning(
                self,
                "No Scan Log Found",
                "No motor scan log (.txt) file was found in any of the dataset directories.\n\n"
                "Proceeding without motor scan metadata (using frame indices)."
            )

        self.on_zeroth_order(self.file_list, txt_path=txt_path)

    def update_listbox(self) -> None:
        """Refresh the visual file list."""
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        self.list_widget.setAlternatingRowColors(True)
        for idx, filepath in enumerate(self.file_list):
            filename = os.path.basename(filepath)
            if idx == 0:
                display_text = f"⭐ [REF] 01. {filename} (Anchor Reference Frame)"
            else:
                display_text = f"    {idx + 1:02d}. {filename}"
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, filepath)
            item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.list_widget.addItem(item)
        if 0 <= self._selected_index < self.list_widget.count():
            self.list_widget.setCurrentRow(self._selected_index)
        self.list_widget.blockSignals(False)
        self.list_widget.update_empty_state()
        self._sync_button_states()

    # ------------------------------------------------------------------
    # Help dialog
    # ------------------------------------------------------------------

    def show_help(self) -> None:
        """Open the help/guide dialog."""
        dlg = QDialog(self)
        dlg.setWindowTitle("mRIXS Super-App — Guide + Credits")
        dlg.resize(660, 580)

        dlg_layout = QVBoxLayout(dlg)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setSpacing(8)
        scroll.setWidget(scroll_content)
        dlg_layout.addWidget(scroll)

        guide_sections = [
            ("Step 1: Load Files",
             "Click '📂 Select TIFF Files' to choose your spectroscopy images.\n"
             "Files are automatically sorted naturally by name upon import.\n"
             "Use '🗑 Clear All' to remove all files and start over."),
            ("Step 2: Reorder Files (Drag & Drop)",
             "Click and drag any file up or down in the list to reorder.\n"
             "Frame 1 (top of list) is the REFERENCE frame — all other frames "
             "will be aligned to it."),
            ("Step 3: Start Slideshow",
             "Click '► Start Alignment Slideshow' to begin analysis.\n"
             "The default alignment engine is ECC (Enhanced Correlation "
             "Coefficient), which works well for most datasets. Warp is "
             "ON by default — frames are translated to align with Frame 1."),
            ("Zeroth-Order Calibration",
             "Click '🔬 Zeroth-Order Calibration' from the main menu to analyze mirror pitch.\n"
             "Optionally select a scan log TXT file to map motor pitch positions to frames,\n"
             "then Export to generate the focus curve (mirror pitch vs FWHM)."),
            ("Tips",
             "• ECC is recommended for most workflows\n"
             "• Use viridis colormap for best visibility of faint features\n"
             "• Drag files in the list to change frame alignment sequence"),
            ("Credits",
             "mRIXS Super-App developed at Lawrence Berkeley National Laboratory.\n"
             "App Credits:\n"
             "• Nickolas Yang — Computing Student Assistant"),
        ]

        for title, body in guide_sections:
            title_lbl = QLabel(title)
            f = QFont()
            f.setPointSize(12)
            f.setBold(True)
            title_lbl.setFont(f)
            content_layout.addWidget(title_lbl)

            body_lbl = QLabel(body)
            body_lbl.setWordWrap(True)
            content_layout.addWidget(body_lbl)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dlg.accept)
        dlg_layout.addWidget(buttons)

        dlg.exec()

    # ------------------------------------------------------------------
    # Compatibility shim (tests call this to refresh the displayed list)
    # ------------------------------------------------------------------

    def select_item(self, idx: int) -> None:
        """Select an item (public API shim used by unit tests).

        Args:
            idx: Item index to select.
        """
        self._select_item(idx)

    # ------------------------------------------------------------------
    # Co-Pilot button integration
    # ------------------------------------------------------------------

    def set_copilot_button(self, btn: QPushButton) -> None:
        """Insert the Co-Pilot toggle button into the header row.

        Args:
            btn: The Co-Pilot toggle QPushButton to reparent here.
        """
        self._header_row.addWidget(btn)
