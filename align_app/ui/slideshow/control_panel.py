# align_app/ui/slideshow/control_panel.py

import customtkinter

class PcaSettingsPanel(customtkinter.CTkFrame):
    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller

        self.pca_label = customtkinter.CTkLabel(self, text="PCA Threshold: 99.9000%")
        self.pca_label.pack(side="left", padx=5)

        self.pca_slider = customtkinter.CTkSlider(
            self, from_=95.0, to=99.9999,
            number_of_steps=4999, command=self.controller.handle_pca_slider_drag
        )
        self.pca_slider.set(99.9)
        self.pca_slider.pack(side="left", fill="x", expand=True, padx=5)

        self.pca_entry = customtkinter.CTkEntry(self, width=80, placeholder_text="99.9000")
        self.pca_entry.pack(side="left", padx=2)
        self.pca_entry.insert(0, "99.9000")
        self.pca_entry.bind("<Return>", self._on_pca_entry_submit)

        self.auto_snap_button = customtkinter.CTkButton(
            self, text="Auto", command=self.controller.trigger_auto_snap,
            width=50, fg_color="#555", hover_color="#777"
        )
        self.auto_snap_button.pack(side="left", padx=2)

        self.auto_all_button = customtkinter.CTkButton(
            self, text="Auto All", command=self.controller.trigger_auto_snap_all,
            width=65, fg_color="#2F72A5", hover_color="#1F5A85"
        )
        self.auto_all_button.pack(side="left", padx=2)

        self.show_line_switch = customtkinter.CTkSwitch(
            self, text="Show Ref Line", command=self.controller._render_display
        )
        self.show_line_switch.select()
        self.show_line_switch.pack(side="right", padx=5)

    def set_ui_state(self, state: str):
        self.pca_slider.configure(state=state)
        self.pca_entry.configure(state=state)
        self.auto_snap_button.configure(state=state)
        self.auto_all_button.configure(state=state)
        self.show_line_switch.configure(state=state)

    def _format_threshold(self, t):
        s = f"{t:.4f}".rstrip('0')
        if s.endswith('.'):
            s += '0'
        return s

    def sync_pca_elements(self, t):
        self.pca_slider.set(min(t, 99.9999))
        self.pca_label.configure(text=f"PCA Threshold: {self._format_threshold(t)}%")
        self.pca_entry.delete(0, "end")
        self.pca_entry.insert(0, f"{t:.4f}")

    def sync_pca_label_and_entry(self, t):
        self.pca_label.configure(text=f"PCA Threshold: {self._format_threshold(t)}%")
        self.pca_entry.delete(0, "end")
        self.pca_entry.insert(0, f"{t:.4f}")

    def _on_pca_entry_submit(self, event=None):
        self.controller.handle_pca_entry_submit(self.pca_entry.get())


class EccSettingsPanel(customtkinter.CTkFrame):
    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller
        
        self.info_label = customtkinter.CTkLabel(self, text="ECC uses automatic coarse-to-fine pyramiding.")
        self.info_label.pack(side="left", fill="x", expand=True, padx=5)

        self.precompute_button = customtkinter.CTkButton(
            self, text="Precompute All", command=self.controller.trigger_auto_snap_all,
            width=100, fg_color="#2F72A5", hover_color="#1F5A85"
        )
        self.precompute_button.pack(side="right", padx=2)

    def set_ui_state(self, state: str):
        self.precompute_button.configure(state=state)


class SlideshowControlPanel(customtkinter.CTkFrame):
    """
    GUI panel hosting Engine settings and timeline navigation elements.
    """
    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller

        self.pca_panel = PcaSettingsPanel(self, controller)
        self.ecc_panel = EccSettingsPanel(self, controller)
        
        # Start with PCA visible
        self.active_engine_panel = self.pca_panel
        self.active_engine_panel.pack(fill="x", pady=2)

        # Frame slider row
        self.frame_nav_frame = customtkinter.CTkFrame(self)
        self.frame_nav_frame.pack(fill="x", pady=2)

        self.frame_label = customtkinter.CTkLabel(self.frame_nav_frame, text="Frame: 0/0")
        self.frame_label.pack(side="left", padx=5)

        self.frame_slider = customtkinter.CTkSlider(
            self.frame_nav_frame, from_=0, to=1,
            number_of_steps=1, command=self.controller.handle_frame_slider_move
        )
        self.frame_slider.set(0)
        self.frame_slider.pack(side="left", fill="x", expand=True, padx=5)

    def switch_engine(self, engine_name: str):
        self.active_engine_panel.pack_forget()
        if engine_name == "ECC":
            self.active_engine_panel = self.ecc_panel
        else:
            self.active_engine_panel = self.pca_panel
        self.active_engine_panel.pack(fill="x", pady=2, before=self.frame_nav_frame)

    def set_ui_state(self, state: str):
        self.active_engine_panel.set_ui_state(state)
        self.frame_slider.configure(state=state)

    def sync_timeline_label(self, current, total):
        self.frame_label.configure(text=f"Frame: {current}/{total}")

    def sync_pca_elements(self, t):
        self.pca_panel.sync_pca_elements(t)

    def sync_pca_label_and_entry(self, t):
        self.pca_panel.sync_pca_label_and_entry(t)
