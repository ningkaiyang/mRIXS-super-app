# align_app/ui/slideshow/export_panel.py

import customtkinter

class SlideshowExportPanel(customtkinter.CTkFrame):
    """
    GUI panel displaying export progress and triggering the compilation of the aligned sum image.

    Features:
    - Progress Label: Displays status text during alignment estimation and frame accumulation.
    - Export Button: Disables the UI during export, prompts the user for a save location, and launches 
      the background export worker thread. Re-enables the UI upon completion.
    """
    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller

        self.progress_label = customtkinter.CTkLabel(self, text="", text_color="#aaaaaa")
        self.progress_label.pack(side="left", padx=5)

        self.export_button = customtkinter.CTkButton(
            self, text="💾 Export Aligned Sum",
            command=self.controller.trigger_export,
            fg_color="#2F72A5", hover_color="#1F5A85"
        )
        self.export_button.pack(side="left", padx=5)
