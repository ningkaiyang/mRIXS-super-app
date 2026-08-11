"""Tools panel providing image zoom options, intensity clamping, fitted-line toggle, and energy dispersion input."""

import customtkinter
from rixs_app.ui.widgets import RangeSlider

class ZerothOrderToolsPanel(customtkinter.CTkFrame):
    """Panel housing zoom helpers, custom RangeSlider, fitted-line toggle, and energy dispersion input."""

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

        # Options checkboxes and energy dispersion
        self.options_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.options_frame.pack(side="left", padx=20)

        self.show_support_points_var = customtkinter.BooleanVar(value=False)
        self.support_points_cb = customtkinter.CTkCheckBox(
            self.options_frame, text="Show support points",
            variable=self.show_support_points_var,
            command=self.controller.load_and_render
        )
        self.support_points_cb.pack(side="top", pady=2, anchor="w")

        self.show_extrapolation_var = customtkinter.BooleanVar(value=False)
        self.extrapolation_cb = customtkinter.CTkCheckBox(
            self.options_frame, text="Show fitted-line extrapolation",
            variable=self.show_extrapolation_var,
            command=self.controller.load_and_render
        )
        self.extrapolation_cb.pack(side="top", pady=2, anchor="w")

        # Show fitted line toggle (default ON)
        self.show_fitted_line_var = customtkinter.BooleanVar(value=True)
        self.fitted_line_cb = customtkinter.CTkCheckBox(
            self.options_frame, text="Show fitted line",
            variable=self.show_fitted_line_var,
            command=self.controller.load_and_render
        )
        self.fitted_line_cb.pack(side="top", pady=2, anchor="w")

        # Energy dispersion input
        self.disp_frame = customtkinter.CTkFrame(self.options_frame, fg_color="transparent")
        self.disp_frame.pack(side="top", pady=4, anchor="w")
        customtkinter.CTkLabel(self.disp_frame, text="meV/px:").pack(side="left")
        self.disp_entry = customtkinter.CTkEntry(self.disp_frame, width=60)
        self.disp_entry.insert(0, "0")
        self.disp_entry.pack(side="left", padx=3)
        self.disp_entry.bind("<Return>", self._on_dispersion_change)

    def _on_dispersion_change(self, event=None):
        """Update energy dispersion on the controller (cheap — no pipeline recomputation)."""
        try:
            val = float(self.disp_entry.get())
            self.controller.set_energy_dispersion(val)
        except ValueError:
            pass

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
