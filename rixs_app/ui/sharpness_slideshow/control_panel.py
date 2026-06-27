"""Control panel hosting navigation sliders, precomputation triggers, and descriptions."""

import customtkinter

STAGE_DESCRIPTIONS = {
    "Raw": "Raw CCD Frame: Displays original detector intensities, containing raw cosmic rays, electronic read noise, and Poisson fluctuations.",
    "Denoised": "Denoised Frame: Shows frame after MAD despiking, Anscombe VST, and Edge-Preserving Bilateral filtering. Centroid line fit is calculated here.",
    "Masked": "Masked Frame: Isolates the spectroscopic elastic line by applying a parallel mask strip. Extraneous background elements are mathematically zeroed."
}

class SharpnessControlPanel(customtkinter.CTkFrame):
    """Layout manager for frame scrubbing, background processing execution, and descriptions."""

    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller

        # Timeline row
        self.timeline_frame = customtkinter.CTkFrame(self)
        self.timeline_frame.pack(fill="x", pady=2)

        self.frame_label = customtkinter.CTkLabel(self.timeline_frame, text="Frame: 0/0")
        self.frame_label.pack(side="left", padx=5)

        self.frame_slider = customtkinter.CTkSlider(
            self.timeline_frame, from_=0, to=1,
            number_of_steps=1, command=self.controller.handle_frame_slider_move
        )
        self.frame_slider.set(0)
        self.frame_slider.pack(side="left", fill="x", expand=True, padx=5)

        # Precompute and Score row
        self.action_frame = customtkinter.CTkFrame(self)
        self.action_frame.pack(fill="x", pady=2)

        self.precompute_button = customtkinter.CTkButton(
            self.action_frame, text="Precompute All", command=self.controller.trigger_precompute,
            width=140, fg_color="#2F72A5", hover_color="#1F5A85"
        )
        self.precompute_button.pack(side="left", padx=5)

        self.score_label = customtkinter.CTkLabel(
            self.action_frame, text="Sharpness Score: -",
            font=customtkinter.CTkFont(size=14, weight="bold"),
            text_color="#88aacc"
        )
        self.score_label.pack(side="right", padx=15)

        # Description Label
        self.description_label = customtkinter.CTkLabel(
            self, text=STAGE_DESCRIPTIONS["Raw"],
            wraplength=700, justify="left", text_color="#aaaaaa"
        )
        self.description_label.pack(fill="x", pady=4, padx=10)

    def set_stage_description(self, stage):
        """Updates description text corresponding to stage selection."""
        self.description_label.configure(text=STAGE_DESCRIPTIONS.get(stage, ""))

    def sync_timeline_label(self, current, total):
        self.frame_label.configure(text=f"Frame: {current}/{total}")

    def sync_score(self, score):
        if score is not None:
            self.score_label.configure(text=f"Sharpness Score: {score:.6f}")
        else:
            self.score_label.configure(text="Sharpness Score: -")
