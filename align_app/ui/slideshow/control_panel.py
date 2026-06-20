# align_app/ui/slideshow/control_panel.py

import customtkinter

class SlideshowControlPanel(customtkinter.CTkFrame):
    """
    GUI panel hosting PCA threshold sliders and timeline navigation elements.

    Features:
    - PCA Threshold Slider: Adjusts the percentile threshold for line fitting. Supports entries and drags 
      between 95.0% and 99.9999% with debounced updates (80ms).
    - Auto Snap Button: Triggers an automated threshold sweep for the current frame to find the threshold 
      that minimizes perpendicular line spread.
    - Auto All Button: Triggers threshold sweeps across all frames, executing SVD evaluations in background 
      worker threads and feeding progress back to the GUI.
    - Timeline Navigation Slider: Navigates through frames in the active file list.
    """
    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller

        # PCA threshold row
        self.pca_frame = customtkinter.CTkFrame(self)
        self.pca_frame.pack(fill="x", pady=2)

        self.pca_label = customtkinter.CTkLabel(self.pca_frame, text="PCA Threshold: 99.9000%")
        self.pca_label.pack(side="left", padx=5)

        self.pca_slider = customtkinter.CTkSlider(
            self.pca_frame, from_=95.0, to=99.9999,
            number_of_steps=4999, command=self.controller.handle_pca_slider_drag
        )
        self.pca_slider.set(99.9)
        self.pca_slider.pack(side="left", fill="x", expand=True, padx=5)

        self.pca_entry = customtkinter.CTkEntry(self.pca_frame, width=80, placeholder_text="99.9000")
        self.pca_entry.pack(side="left", padx=2)
        self.pca_entry.insert(0, "99.9000")
        self.pca_entry.bind("<Return>", self._on_pca_entry_submit)

        self.auto_snap_button = customtkinter.CTkButton(
            self.pca_frame, text="Auto", command=self.controller.trigger_auto_snap,
            width=50, fg_color="#555", hover_color="#777"
        )
        self.auto_snap_button.pack(side="left", padx=2)

        self.auto_all_button = customtkinter.CTkButton(
            self.pca_frame, text="Auto All", command=self.controller.trigger_auto_snap_all,
            width=65, fg_color="#2F72A5", hover_color="#1F5A85"
        )
        self.auto_all_button.pack(side="left", padx=2)

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

    def _format_threshold(self, t):
        s = f"{t:.4f}".rstrip('0')
        if s.endswith('.'):
            s += '0'
        return s

    def sync_timeline_label(self, current, total):
        self.frame_label.configure(text=f"Frame: {current}/{total}")

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
