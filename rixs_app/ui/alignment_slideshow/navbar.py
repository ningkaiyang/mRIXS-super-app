# rixs_app/ui/slideshow/navbar.py

import customtkinter

class SlideshowNavBar(customtkinter.CTkFrame):
    """
    Modular UI navigation bar positioned at the top of the slideshow view.

    GUI Structure & Operations:
      - Provides frame scrubbing buttons (Back, Prev, Next).
      - Provides an autoplay toggle button linked to a periodic timer job in the main UI loop.
      - Provides colormap selection menu (supporting viridis, inferno, plasma, magma, and grayscale).
      - Provides toggle switches to show/hide the reference line overlay and enable/disable real-time warping.
    """
    def __init__(self, parent, controller, **kwargs):
        """
        Initialize the SlideshowNavBar.

        Args:
            parent: The parent widget.
            controller: The controller managing the slideshow logic and state.
            **kwargs: Additional keyword arguments for the customtkinter.CTkFrame.
        """
        super().__init__(parent, **kwargs)
        self.controller = controller

        self.back_button = customtkinter.CTkButton(
            self, text="◀ Back", command=self.controller.back_to_sorting, width=80
        )
        self.back_button.pack(side="left", padx=5)

        self.prev_button = customtkinter.CTkButton(
            self, text="◀ Prev", command=self.controller.prev_frame, width=80
        )
        self.prev_button.pack(side="left", padx=5)

        self.next_button = customtkinter.CTkButton(
            self, text="Next ▶", command=self.controller.next_frame, width=80
        )
        self.next_button.pack(side="left", padx=5)

        self.autoplay_button = customtkinter.CTkButton(
            self, text="▶ Play", command=self.controller.toggle_autoplay,
            width=80, fg_color="#2FA572", hover_color="#238a5a"
        )
        self.autoplay_button.pack(side="left", padx=5)

        self.warp_switch = customtkinter.CTkSwitch(
            self, text="Warp Image", command=self.controller.toggle_warp
        )
        self.warp_switch.select()
        self.warp_switch.pack(side="right", padx=5)

        self.colormap_menu = customtkinter.CTkOptionMenu(
            self,
            values=["viridis", "inferno", "plasma", "magma", "grayscale"],
            command=self.controller.change_colormap
        )
        self.colormap_menu.set("viridis")
        self.colormap_menu.pack(side="right", padx=5)

        self.engine_menu = customtkinter.CTkOptionMenu(
            self,
            values=["PCA", "ECC", "Phase Correlation"],
            command=self.controller.change_engine
        )
        self.engine_menu.set("ECC")
        self.engine_menu.pack(side="right", padx=5)
