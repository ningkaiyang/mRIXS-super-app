"""Unit tests verifying dropdown styling, SVG arrow assets, and QComboBox behavior."""

from pathlib import Path
import xml.etree.ElementTree as ET
import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QWidget

from rixs_app.ui.theme import (
    DARK_STYLE, FULL_QSS,
    _DROPDOWN_ARROW_SVG, _DROPDOWN_ARROW_HOVER_SVG, _DROPDOWN_ARROW_DISABLED_SVG,
)


@pytest.fixture(scope="session")
def qapp():
    """Ensure QApplication instance exists for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_dropdown_svg_assets_exist_and_valid():
    """Verify all 3 SVG arrow assets exist and parse as valid SVG/XML."""
    assets = [_DROPDOWN_ARROW_SVG, _DROPDOWN_ARROW_HOVER_SVG, _DROPDOWN_ARROW_DISABLED_SVG]
    for asset_path in assets:
        p = Path(asset_path)
        assert p.is_file(), f"Missing SVG asset: {asset_path}"
        assert p.stat().st_size > 0, f"Empty SVG asset: {asset_path}"
        
        # Validate XML structure
        tree = ET.parse(p)
        root = tree.getroot()
        assert "svg" in root.tag, f"Root element is not svg in {asset_path}"
        polyline = root.find("{http://www.w3.org/2000/svg}polyline")
        assert polyline is not None, f"Missing polyline in {asset_path}"


def test_theme_qss_references_svg_arrows():
    """Verify DARK_STYLE contains url(...) references for down-arrows."""
    assert "QComboBox::down-arrow" in DARK_STYLE
    assert f"image: url({_DROPDOWN_ARROW_SVG});" in DARK_STYLE
    assert f"image: url({_DROPDOWN_ARROW_HOVER_SVG});" in DARK_STYLE
    assert f"image: url({_DROPDOWN_ARROW_DISABLED_SVG});" in DARK_STYLE
    assert "image: none;" not in DARK_STYLE


def test_all_app_comboboxes_styled_cleanly(qapp):
    """Verify that all dropdowns across the application instantiate and apply FULL_QSS without error."""
    parent = QWidget()
    parent.setStyleSheet(FULL_QSS)

    # 1. Alignment slideshow Engine combobox
    engine_combo = QComboBox(parent)
    engine_combo.setObjectName("engine_menu")
    engine_combo.addItems(["PCA", "ECC", "Phase Correlation"])
    assert engine_combo.count() == 3

    # 2. Colormap combobox
    colormap_combo = QComboBox(parent)
    colormap_combo.setObjectName("colormap_menu")
    colormap_combo.addItems(["viridis", "inferno", "plasma", "magma", "grayscale"])
    assert colormap_combo.count() == 5

    # 3. Zeroth-order stage combobox
    stage_combo = QComboBox(parent)
    stage_combo.setObjectName("stage_menu")
    stage_combo.addItems(["Raw", "Denoised (D)", "Row-Smoothed (Dsm)", "Gradient (G)", "Fitted-Line Strip"])
    assert stage_combo.count() == 5

    # 4. Agent sidebar model selector
    model_combo = QComboBox(parent)
    model_combo.addItems(["lbl/cborg-deepthought:latest", "lbl/cborg-coder:latest"])
    assert model_combo.count() == 2
