import os
import customtkinter
import tkinter.filedialog
from align_app.core import natural_sort

class SortingView(customtkinter.CTkFrame):
    def __init__(self, parent, on_start_slideshow=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.on_start_slideshow = on_start_slideshow
        self.file_list = []
        self.selected_index = -1

        # Stub widgets
        self.select_button = customtkinter.CTkButton(self, text="📁 Select Files", command=self.select_files)
        self.select_button.pack(pady=5)

        self.sort_button = customtkinter.CTkButton(
            self, text="↕ Sort Files", command=self.sort_files,
            fg_color="#1F6AA5", hover_color="#165a8a",
            font=customtkinter.CTkFont(size=14, weight="bold"),
            height=36
        )
        self.sort_button.pack(pady=5)

        self.up_button = customtkinter.CTkButton(self, text="▲ Up", command=self.move_up, width=80)
        self.up_button.pack(pady=3)

        self.down_button = customtkinter.CTkButton(self, text="▼ Down", command=self.move_down, width=80)
        self.down_button.pack(pady=3)

        self.remove_button = customtkinter.CTkButton(
            self, text="✕ Remove", command=self.remove_file,
            width=80, fg_color="#aa3333", hover_color="#882222"
        )
        self.remove_button.pack(pady=3)

        self.start_button = customtkinter.CTkButton(
            self, text="▶ Start Slideshow", command=self.start_slideshow,
            fg_color="#2FA572", hover_color="#238a5a",
            font=customtkinter.CTkFont(size=16, weight="bold"),
            height=44
        )
        self.start_button.pack(pady=10)

        self.scroll_frame = customtkinter.CTkScrollableFrame(self, label_text="Selected Files")
        self.scroll_frame.pack(pady=5, fill="both", expand=True)
        
        self.labels = []

    def select_files(self):
        # Allow mock override of file dialog
        files = tkinter.filedialog.askopenfilenames(
            title="Select TIFF files",
            filetypes=[("TIFF Files", "*.tif;*.tiff")]
        )
        if files:
            self.file_list.extend(list(files))
            self.update_listbox()

    def sort_files(self):
        self.file_list = natural_sort(self.file_list)
        self.update_listbox()

    def select_item(self, idx):
        self.selected_index = idx
        self.update_listbox()

    def move_up(self):
        idx = self.selected_index
        if 0 < idx < len(self.file_list):
            self.file_list[idx], self.file_list[idx-1] = self.file_list[idx-1], self.file_list[idx]
            self.selected_index = idx - 1
            self.update_listbox()

    def move_down(self):
        idx = self.selected_index
        if 0 <= idx < len(self.file_list) - 1:
            self.file_list[idx], self.file_list[idx+1] = self.file_list[idx+1], self.file_list[idx]
            self.selected_index = idx + 1
            self.update_listbox()

    def remove_file(self):
        idx = self.selected_index
        if 0 <= idx < len(self.file_list):
            self.file_list.pop(idx)
            if self.file_list:
                self.selected_index = min(idx, len(self.file_list) - 1)
            else:
                self.selected_index = -1
            self.update_listbox()

    def start_slideshow(self):
        if self.on_start_slideshow and self.file_list:
            self.on_start_slideshow(self.file_list)

    def update_listbox(self):
        for lbl in self.labels:
            lbl.destroy()
        self.labels.clear()

        for idx, filename in enumerate(self.file_list):
            bg = "blue" if idx == self.selected_index else "transparent"
            lbl = customtkinter.CTkLabel(
                self.scroll_frame,
                text=os.path.basename(filename),
                fg_color=bg
            )
            lbl.bind("<Button-1>", lambda event, i=idx: self.select_item(i))
            lbl.pack(fill="x", anchor="w")
            self.labels.append(lbl)
