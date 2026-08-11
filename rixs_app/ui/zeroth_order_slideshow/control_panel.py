"""Control panel hosting navigation slider, motor info, FWHM metrics, and meV/px dispersion input in a sleek, thin horizontal bar."""

import os
import numpy as np
import customtkinter

STAGE_DESCRIPTIONS = {
    "Raw": "Raw detector frame — includes noise, cosmic rays, and hot pixels. Uncropped original.",
    "Denoised (D)": "After MAD despiking, Anscombe VST, and bilateral filtering. Cropped 100px edges.",
    "Row-Smoothed (Dsm)": "Row-wise Gaussian smoothing (σ=2.5) of D, then rolling-min background subtraction. Input to the V8 row scanner.",
    "Gradient (G)": "Scharr gradient magnitude of D after Gaussian blur. Shows edge strength.",
    "Fitted-Line Strip": "Gradient masked to the detected support range of the fitted line.",
}

class ZerothOrderControlPanel(customtkinter.CTkFrame):
    """Layout manager for frame scrubbing, motor scan metadata, FWHM metrics, and meV/px dispersion input."""

    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller

        # Row 1: Timeline scrub bar
        self.timeline_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.timeline_frame.pack(fill="x", pady=(2, 0), padx=5)

        self.frame_label = customtkinter.CTkLabel(self.timeline_frame, text="Frame: 0/0", font=customtkinter.CTkFont(size=12, weight="bold"))
        self.frame_label.pack(side="left", padx=(0, 5))

        self.frame_slider = customtkinter.CTkSlider(
            self.timeline_frame, from_=0, to=1,
            number_of_steps=1, command=self.controller.handle_frame_slider_move,
            height=16
        )
        self.frame_slider.set(0)
        self.frame_slider.pack(side="left", fill="x", expand=True, padx=5)

        # Row 2: Thin horizontal information bar
        self.info_bar = customtkinter.CTkFrame(self, fg_color="transparent")
        self.info_bar.pack(fill="x", pady=(2, 2), padx=5)

        # Left side elements (left to right)
        self.motor_label = customtkinter.CTkLabel(
            self.info_bar, text="Motor: —",
            font=customtkinter.CTkFont(size=13, weight="bold"),
            text_color="#88ccff"
        )
        self.motor_label.pack(side="left", padx=(0, 10))

        self.score_label = customtkinter.CTkLabel(
            self.info_bar, text="Score: —",
            font=customtkinter.CTkFont(size=12, weight="bold"),
            text_color="#88aacc"
        )
        self.score_label.pack(side="left", padx=8)

        self.fwhm_label = customtkinter.CTkLabel(
            self.info_bar, text="FWHM: — px",
            font=customtkinter.CTkFont(size=13, weight="bold"),
            text_color="#cc8844"
        )
        self.fwhm_label.pack(side="left", padx=8)

        # meV/px input frame
        self.disp_frame = customtkinter.CTkFrame(self.info_bar, fg_color="transparent")
        self.disp_frame.pack(side="left", padx=8)

        customtkinter.CTkLabel(self.disp_frame, text="meV/px:", font=customtkinter.CTkFont(size=12)).pack(side="left", padx=(0, 2))
        self.disp_entry = customtkinter.CTkEntry(self.disp_frame, width=55, height=24, font=customtkinter.CTkFont(size=12))
        self.disp_entry.insert(0, "0")
        self.disp_entry.pack(side="left")
        self.disp_entry.bind("<Return>", self._on_dispersion_change)

        self.fwhm_mev_label = customtkinter.CTkLabel(
            self.info_bar, text="— meV",
            font=customtkinter.CTkFont(size=13, weight="bold"),
            text_color="#88aacc"
        )
        self.fwhm_mev_label.pack(side="left", padx=6)

        self.resolving_power_label = customtkinter.CTkLabel(
            self.info_bar, text="R: —",
            font=customtkinter.CTkFont(size=13, weight="bold"),
            text_color="#88ccaa"
        )
        self.resolving_power_label.pack(side="left", padx=8)

        self.focus_badge = customtkinter.CTkLabel(
            self.info_bar, text="",
            font=customtkinter.CTkFont(size=12, weight="bold"),
            text_color="#22cc66"
        )
        self.focus_badge.pack(side="left", padx=8)

        # Right side element (aligned right)
        self.detection_label = customtkinter.CTkLabel(
            self.info_bar, text="Detection: —",
            font=customtkinter.CTkFont(size=11),
            text_color="#aaaaaa"
        )
        self.detection_label.pack(side="right", padx=(10, 0))

    def _on_dispersion_change(self, event=None):
        """Update energy dispersion on the controller."""
        try:
            val = float(self.disp_entry.get())
            self.controller.set_energy_dispersion(val)
        except ValueError:
            pass

    def sync_motor_info(self, txt_metadata, filename, frame_idx, total_frames):
        """Update motor variable name and value for current frame."""
        if txt_metadata and "frames" in txt_metadata:
            bn = os.path.basename(filename)
            finfo = txt_metadata["frames"].get(bn)
            motor_name = txt_metadata.get("motor_name", "Motor")
            if finfo and not np.isnan(finfo.get("motor_goal", np.nan)):
                val = finfo["motor_goal"]
                self.motor_label.configure(text=f"{motor_name}: {val:.4f}")
                return
        self.motor_label.configure(text=f"Frame {frame_idx + 1}/{total_frames}")

    def set_stage_description(self, stage):
        """No-op retained for API compatibility."""
        pass

    def sync_timeline_label(self, current, total):
        self.frame_label.configure(text=f"Frame: {current}/{total}")

    def sync_score(self, score):
        if score is not None:
            self.score_label.configure(text=f"Score: {score:.4f}")
        else:
            self.score_label.configure(text="Score: —")

    def sync_score_with_evaluator(self, eval_result, fallback_score):
        if eval_result is not None and eval_result.score_valid:
            self.score_label.configure(text=f"1/FWHM: {eval_result.score:.4f} px⁻¹")
        elif fallback_score is not None:
            self.score_label.configure(text=f"Score: {fallback_score:.4f}")
        else:
            self.score_label.configure(text="Score: —")

    def sync_fwhm(self, eval_result, energy_dispersion=0.0, mono_energy_ev=0.0, is_best_focus=False):
        """Update FWHM (px), FWHM (meV), and resolving power R displays."""
        if eval_result is not None and eval_result.fwhm_px is not None and eval_result.score_valid:
            self.fwhm_label.configure(text=f"FWHM: {eval_result.fwhm_px:.2f} px")
            if energy_dispersion > 0:
                fwhm_mev = eval_result.fwhm_px * energy_dispersion
                self.fwhm_mev_label.configure(text=f"{fwhm_mev:.1f} meV")
                if mono_energy_ev > 0:
                    delta_e_ev = fwhm_mev * 1e-3
                    R = mono_energy_ev / delta_e_ev
                    self.resolving_power_label.configure(text=f"R: {R:,.0f}")
                else:
                    self.resolving_power_label.configure(text="R: —")
            else:
                self.fwhm_mev_label.configure(text="— meV")
                self.resolving_power_label.configure(text="R: —")
        else:
            self.fwhm_label.configure(text="FWHM: — px")
            self.fwhm_mev_label.configure(text="— meV")
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
