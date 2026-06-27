# rixs_app/ui/slideshow/tools_panel.py

import customtkinter

class SlideshowToolsPanel(customtkinter.CTkFrame):
    """
    Modular UI panel for manual alignment tools and viewport configurations.

    GUI Structure & Manual Alignment:
      - "Manual Line": Activates cursor-crosshairs, letting users click two points on the canvas.
      - Computes the sub-pixel midpoint of clicked points relative to letterbox dimensions to override PCA centroids.
      - Provides Zoom In, Zoom Out, and Reset buttons that update the canvas viewport scale and center.
      - Displays real-time frame details (e.g., custom thresholds or manual override notifications).
    """
    def __init__(self, parent, controller, **kwargs):
        """
        Initialize the SlideshowToolsPanel.

        Args:
            parent: The parent widget.
            controller: The controller managing the slideshow logic and state.
            **kwargs: Additional keyword arguments for the customtkinter.CTkFrame.
        """
        super().__init__(parent, **kwargs)
        self.controller = controller

        self.manual_line_button = customtkinter.CTkButton(
            self, text="✏ Manual Line", command=self.controller.toggle_manual_mode,
            width=110, fg_color="#555", hover_color="#777"
        )
        self.manual_line_button.pack(side="left", padx=5)

        self.clear_manual_button = customtkinter.CTkButton(
            self, text="Clear Manual", command=self.controller.clear_manual_line,
            width=100, fg_color="#555", hover_color="#777"
        )
        self.clear_manual_button.pack(side="left", padx=2)

        self.zoom_in_button = customtkinter.CTkButton(
            self, text="🔍+ Zoom In", command=self.controller.zoom_in,
            width=100, fg_color="#555", hover_color="#777"
        )
        self.zoom_in_button.pack(side="left", padx=5)

        self.zoom_out_button = customtkinter.CTkButton(
            self, text="🔍- Zoom Out", command=self.controller.zoom_out,
            width=100, fg_color="#555", hover_color="#777"
        )
        self.zoom_out_button.pack(side="left", padx=2)

        self.reset_view_button = customtkinter.CTkButton(
            self, text="⟲ Reset View", command=self.controller.reset_view,
            width=100, fg_color="#555", hover_color="#777"
        )
        self.reset_view_button.pack(side="left", padx=2)

        self.zoom_label = customtkinter.CTkLabel(self, text="Zoom: 1×")
        self.zoom_label.pack(side="left", padx=5)

        # Per-frame info label
        self.frame_info_label = customtkinter.CTkLabel(
            self, text="", text_color="#88aacc"
        )
        self.frame_info_label.pack(side="right", padx=10)

    def sync_zoom_label(self, factor):
        """
        Update the displayed zoom multiplier text on the interface.

        Args:
            factor (float or int): The zoom factor to display (e.g., 2 for 2x zoom).
        """
        self.zoom_label.configure(text=f"Zoom: {factor}×")
