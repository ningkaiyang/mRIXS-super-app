"""
Slideshow UI components module for the spectroscopy alignment application.

Architecture:
  - Controller-Manager-View (CMV) paradigm:
    - SlideshowView serves as the controller.
    - SlideshowManager serves as the state manager.
    - Panel classes (Canvas, Clamping, Control, Export, NavBar, Tools) serve as modular view panels.
"""

from rixs_app.ui.alignment_slideshow.alignment_manager import SlideshowManager
from rixs_app.ui.alignment_slideshow.navbar import SlideshowNavBar
from rixs_app.ui.alignment_slideshow.control_panel import SlideshowControlPanel
from rixs_app.ui.alignment_slideshow.tools_panel import SlideshowToolsPanel
from rixs_app.ui.alignment_slideshow.clamping_panel import SlideshowClampingPanel
from rixs_app.ui.alignment_slideshow.export_panel import SlideshowExportPanel
from rixs_app.ui.alignment_slideshow.canvas_panel import SlideshowCanvasPanel
from rixs_app.ui.alignment_slideshow.comparison_view import ExportComparisonView

__all__ = [
    "SlideshowManager",
    "SlideshowNavBar",
    "SlideshowCanvasPanel",
    "SlideshowToolsPanel",
    "SlideshowControlPanel",
    "SlideshowClampingPanel",
    "SlideshowExportPanel",
    "ExportComparisonView",
]
