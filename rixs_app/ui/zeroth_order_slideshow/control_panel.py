"""Control panel hosting navigation sliders, precomputation triggers, FWHM metrics, and descriptions."""

import customtkinter

STAGE_DESCRIPTIONS = {
    "Raw": "Raw detector frame — includes noise, cosmic rays, and hot pixels. Uncropped original.",
    "Denoised (D)": "After MAD despiking, Anscombe VST, and bilateral filtering. Cropped 100px edges.",
    "Row-Smoothed (Dsm)": "Row-wise Gaussian smoothing (σ=2.5) of D, then rolling-min background subtraction. Input to the V8 row scanner.",
    "Gradient (G)": "Scharr gradient magnitude of D after Gaussian blur. Shows edge strength.",
    "Fitted-Line Strip": "Gradient masked to the detected support range of the fitted line.",
}

class ZerothOrderControlPanel(customtkinter.CTkFrame):
    """Layout manager for frame scrubbing, background processing execution, FWHM metrics, and descriptions."""

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

        self.score_frame = customtkinter.CTkFrame(self.action_frame, fg_color="transparent")
        self.score_frame.pack(side="right", padx=15)

        self.score_label = customtkinter.CTkLabel(
            self.score_frame, text="Score: -",
            font=customtkinter.CTkFont(size=14, weight="bold"),
            text_color="#88aacc"
        )
        self.score_label.pack(side="top", anchor="e")

        self.fwhm_label = customtkinter.CTkLabel(
            self.score_frame, text="FWHM: — px",
            font=customtkinter.CTkFont(size=16, weight="bold"),
            text_color="#cc8844",
        )
        self.fwhm_label.pack(side="top", anchor="e")

        self.fwhm_mev_label = customtkinter.CTkLabel(
            self.score_frame, text="FWHM: — meV",
            font=customtkinter.CTkFont(size=14), text_color="#88aacc",
        )
        self.fwhm_mev_label.pack(side="top", anchor="e")

        self.resolving_power_label = customtkinter.CTkLabel(
            self.score_frame, text="R: —",
            font=customtkinter.CTkFont(size=14), text_color="#88ccaa",
        )
        self.resolving_power_label.pack(side="top", anchor="e")

        self.focus_badge = customtkinter.CTkLabel(
            self.score_frame, text="",
            font=customtkinter.CTkFont(size=12, weight="bold"),
            text_color="#22cc66",
        )
        self.focus_badge.pack(side="top", anchor="e")

        self.detection_label = customtkinter.CTkLabel(
            self.score_frame, text="Detection: -",
            font=customtkinter.CTkFont(size=12),
            text_color="#aaaaaa"
        )
        self.detection_label.pack(side="top", anchor="e")

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
            self.score_label.configure(text=f"Score: {score:.6f}")
        else:
            self.score_label.configure(text="Score: -")

    def sync_score_with_evaluator(self, eval_result, fallback_score):
        if eval_result is not None and eval_result.score_valid:
            self.score_label.configure(text=f"⚗️ {eval_result.score:.6f} (1/FWHM px⁻¹)")
        elif fallback_score is not None:
            self.score_label.configure(text=f"Score: {fallback_score:.6f} (grad peak)")
        else:
            self.score_label.configure(text="Score: -")

    def sync_fwhm(self, eval_result, energy_dispersion=0.0, mono_energy_ev=0.0, is_best_focus=False):
        """Update FWHM (px), FWHM (meV), and resolving power R displays."""
        if eval_result is not None and eval_result.fwhm_px is not None and eval_result.score_valid:
            self.fwhm_label.configure(text=f"FWHM: {eval_result.fwhm_px:.2f} px")
            if energy_dispersion > 0:
                fwhm_mev = eval_result.fwhm_px * energy_dispersion
                self.fwhm_mev_label.configure(text=f"FWHM: {fwhm_mev:.1f} meV")
                if mono_energy_ev > 0:
                    delta_e_ev = fwhm_mev * 1e-3
                    R = mono_energy_ev / delta_e_ev
                    self.resolving_power_label.configure(text=f"R = {R:,.0f}")
                else:
                    self.resolving_power_label.configure(text="R: —")
            else:
                self.fwhm_mev_label.configure(text="FWHM: — meV")
                self.resolving_power_label.configure(text="R: —")
        else:
            self.fwhm_label.configure(text="FWHM: — px")
            self.fwhm_mev_label.configure(text="FWHM: — meV")
            self.resolving_power_label.configure(text="R: —")
        self.focus_badge.configure(text="✅ BEST FOCUS" if is_best_focus else "")

    def sync_detection_status(self, data):
        if not data.get("fit_ok", False):
            self.detection_label.configure(text="No valid line fit")
        else:
            angle = data.get("angle_deg")
            n_cand = data.get("n_candidates", 0)
            n_inl = data.get("n_inliers", 0)
            angle_str = f"{angle:.1f}°" if angle is not None else "N/A"
            self.detection_label.configure(text=f"Angle: {angle_str} | Cand: {n_cand} | Inl: {n_inl}")
