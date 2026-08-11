"""Export panel for exporting side-by-side diagnostic plots for the whole sequence."""

import customtkinter

class ZerothOrderExportPanel(customtkinter.CTkFrame):
    """UI bar containing bulk export triggers and background worker progress labels."""

    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller

        self.progress_label = customtkinter.CTkLabel(self, text="", text_color="#aaaaaa")
        self.progress_label.pack(side="left", padx=10)

        self.export_button = customtkinter.CTkButton(
            self, text="💾 Export Diagnostic PNGs",
            command=self.controller.trigger_export,
            width=200, height=35,
            fg_color="#2F72A5", hover_color="#1F5A85"
        )
        self.export_button.pack(side="right", padx=10)
