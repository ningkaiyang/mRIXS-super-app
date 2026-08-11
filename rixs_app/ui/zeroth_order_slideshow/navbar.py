"""Navbar component for the zeroth-order calibration slideshow."""

import customtkinter

class ZerothOrderNavBar(customtkinter.CTkFrame):
    """Top navigation bar providing back-to-sorting, colormap, stage selection, and peak focus."""

    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller

        # Left side: Back to Sorting View
        self.back_button = customtkinter.CTkButton(
            self, text="◀ Back", command=self.controller.back_to_sorting, width=80
        )
        self.back_button.pack(side="left", padx=5)

        # Timeline steps
        self.prev_button = customtkinter.CTkButton(
            self, text="◀ Prev", command=self.controller.prev_frame, width=80
        )
        self.prev_button.pack(side="left", padx=5)

        self.next_button = customtkinter.CTkButton(
            self, text="Next ▶", command=self.controller.next_frame, width=80
        )
        self.next_button.pack(side="left", padx=5)

        # Precompute All button
        self.precompute_button = customtkinter.CTkButton(
            self, text="Precompute All", command=self.controller.trigger_precompute,
            width=120, fg_color="#1F6AA5", hover_color="#165a8a"
        )
        self.precompute_button.pack(side="left", padx=5)

        # Peak Focus button
        self.peak_focus_button = customtkinter.CTkButton(
            self, text="🎯 Peak Focus",
            command=self.controller.jump_to_peak_focus,
            width=120, fg_color="#A55D2F", hover_color="#8A4A22",
        )
        self.peak_focus_button.pack(side="left", padx=5)

        # Autoplay button
        self.autoplay_button = customtkinter.CTkButton(
            self, text="▶ Play", command=self.controller.toggle_autoplay,
            width=80, fg_color="#2FA572", hover_color="#238a5a"
        )
        self.autoplay_button.pack(side="left", padx=5)

        # Right side: colormap selection dropdown
        self.colormap_menu = customtkinter.CTkOptionMenu(
            self,
            values=["viridis", "inferno", "plasma", "magma", "grayscale"],
            command=self.controller.change_colormap
        )
        self.colormap_menu.set("viridis")
        self.colormap_menu.pack(side="right", padx=5)

        # Right side: pipeline stage toggle dropdown
        self.stage_menu = customtkinter.CTkOptionMenu(
            self,
            values=["Raw", "Denoised (D)", "Row-Smoothed (Dsm)", "Gradient (G)", "Fitted-Line Strip"],
            command=self.controller.change_pipeline_stage
        )
        self.stage_menu.set("Raw")
        self.stage_menu.pack(side="right", padx=5)
