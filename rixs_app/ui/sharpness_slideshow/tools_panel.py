"""Tools panel providing image zoom options and intensity clamping configurations."""

import customtkinter
from rixs_app.ui.widgets import RangeSlider

class SharpnessToolsPanel(customtkinter.CTkFrame):
    """Panel housing zoom helpers and custom RangeSlider elements."""

    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller

        # Left side: Zoom Controls
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

        self.zoom_label = customtkinter.CTkLabel(self, text="Zoom: 1.0×")
        self.zoom_label.pack(side="left", padx=5)

        # Right side: Slicing adjustments
        self.slicing_label = customtkinter.CTkLabel(self, text="Slicing:")
        self.slicing_label.pack(side="left", padx=(20, 5))

        self.floor_entry = customtkinter.CTkEntry(self, width=80)
        self.floor_entry.pack(side="left", padx=5)
        self.floor_entry.bind("<Return>", self._on_floor_submit)

        self.range_slider = RangeSlider(
            self, height=25, command=self.controller.handle_slicing_change
        )
        self.range_slider.pack(side="left", fill="x", expand=True, padx=5)

        self.ceiling_entry = customtkinter.CTkEntry(self, width=80)
        self.ceiling_entry.pack(side="left", padx=5)
        self.ceiling_entry.bind("<Return>", self._on_ceiling_submit)

    def sync_zoom_label(self, val):
        self.zoom_label.configure(text=f"Zoom: {val:.1f}×")

    def sync_slicing_inputs(self, floor, ceiling):
        self.floor_entry.delete(0, "end")
        self.floor_entry.insert(0, f"{floor:.4f}")
        self.ceiling_entry.delete(0, "end")
        self.ceiling_entry.insert(0, f"{ceiling:.4f}")
        self.range_slider.set_values(floor, ceiling)

    def _on_floor_submit(self, event=None):
        self.controller.handle_floor_entry_submit(self.floor_entry.get())

    def _on_ceiling_submit(self, event=None):
        self.controller.handle_ceiling_entry_submit(self.ceiling_entry.get())
