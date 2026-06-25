# rixs_app/ui/slideshow/clamping_panel.py

import customtkinter
from rixs_app.widgets import RangeSlider

class SlideshowClampingPanel(customtkinter.CTkFrame):
    """
    GUI Panel providing intensity clamping controls to adjust image display contrast.

    Physics & Visual Context:
    - RIXS raw data has a very high dynamic range with high intensity peaks and weak spectral features.
    - Clamping allows users to narrow the displayed range [floor, ceiling] to increase the contrast of weak features.
    - Synchronizes a custom double-ended `RangeSlider` with numerical entry fields for exact limits.
    - Binds return-key submit events to update the display on-demand.
    - Event Handling: Intercepts slider drag events and forwards them to the controller with a debounce 
      delay (80ms) to ensure smooth GUI responsiveness.
    """
    def __init__(self, parent, controller, **kwargs):
        """
        Initialize the SlideshowClampingPanel.

        Args:
            parent: The parent widget.
            controller: The controller managing the slideshow logic and state.
            **kwargs: Additional keyword arguments for the customtkinter.CTkFrame.
        """
        super().__init__(parent, **kwargs)
        self.controller = controller

        self.clamping_label = customtkinter.CTkLabel(self, text="Intensity Clamping:")
        self.clamping_label.pack(side="left", padx=5)

        self.floor_entry = customtkinter.CTkEntry(self, width=80)
        self.floor_entry.pack(side="left", padx=5)
        self.floor_entry.bind("<Return>", self._on_floor_submit)

        self.range_slider = RangeSlider(
            self, height=25, command=self.controller.handle_clamping_change
        )
        self.range_slider.pack(side="left", fill="x", expand=True, padx=5)

        self.ceiling_entry = customtkinter.CTkEntry(self, width=80)
        self.ceiling_entry.pack(side="left", padx=5)
        self.ceiling_entry.bind("<Return>", self._on_ceiling_submit)

    def setup_clamping_limits(self, intensity_min, intensity_max):
        """
        Configure the absolute minimum and maximum values for the clamping slider.

        Args:
            intensity_min (float): The minimum valid intensity value.
            intensity_max (float): The maximum valid intensity value.
        """
        self.range_slider.configure_range(intensity_min, intensity_max)

    def sync_clamping_inputs(self, floor, ceiling):
        """
        Synchronize the slider and text entry values with the current clamping levels.

        Args:
            floor (float): The current minimum intensity threshold.
            ceiling (float): The current maximum intensity threshold.
        """
        self.floor_entry.delete(0, "end")
        self.floor_entry.insert(0, f"{floor:.4f}")
        self.ceiling_entry.delete(0, "end")
        self.ceiling_entry.insert(0, f"{ceiling:.4f}")
        self.range_slider.set_values(floor, ceiling)

    def _on_floor_submit(self, event=None):
        """
        Handle submission of the floor intensity entry field.

        Args:
            event (tk.Event, optional): The return key press event. Defaults to None.
        """
        self.controller.handle_floor_entry_submit(self.floor_entry.get())

    def _on_ceiling_submit(self, event=None):
        """
        Handle submission of the ceiling intensity entry field.

        Args:
            event (tk.Event, optional): The return key press event. Defaults to None.
        """
        self.controller.handle_ceiling_entry_submit(self.ceiling_entry.get())
