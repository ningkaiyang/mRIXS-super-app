"""
Slideshow UI components module for the spectroscopy alignment application.

Architecture:
  - Controller-Manager-View (CMV) paradigm:
    - SlideshowView serves as the controller.
    - SlideshowManager serves as the state manager.
    - Panel classes (Canvas, Clamping, Control, Export, NavBar, Tools) serve as modular view panels.
"""

from align_app.ui.slideshow.managers import SlideshowManager
from align_app.ui.slideshow.navbar import SlideshowNavBar
from align_app.ui.slideshow.control_panel import SlideshowControlPanel
from align_app.ui.slideshow.tools_panel import SlideshowToolsPanel
from align_app.ui.slideshow.clamping_panel import SlideshowClampingPanel
from align_app.ui.slideshow.export_panel import SlideshowExportPanel
from align_app.ui.slideshow.canvas_panel import SlideshowCanvasPanel
