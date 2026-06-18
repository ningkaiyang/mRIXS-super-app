import platform
import customtkinter
from align_app.sorting_view import SortingView
from align_app.slideshow_view import SlideshowView

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

class AlignApp(customtkinter.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.withdraw()  # Hide window initially to prevent flickering
        self.title("Spectroscopy Image Alignment GUI")
        self.geometry("800x600")

        self.container = customtkinter.CTkFrame(self)
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.sorting_view = SortingView(
            self.container,
            on_start_slideshow=self.show_slideshow
        )
        self.slideshow_view = SlideshowView(
            self.container,
            on_back_to_sorting=self.show_sorting
        )

        self.sorting_view.grid(row=0, column=0, sticky="nsew")
        self.slideshow_view.grid(row=0, column=0, sticky="nsew")

        # Global keyboard navigation binding (bind_all captures events
        # regardless of which widget has focus)
        self.bind_all("<Left>", self._on_left_key)
        self.bind_all("<Right>", self._on_right_key)

        self.show_sorting()

        # Maximize the geometry/state in the background while withdrawn
        self.maximize_window()
        # Force the OS window manager to apply the geometry changes before showing
        self.update()
        # Reveal the window directly in its maximized state
        self.deiconify()

    def _on_left_key(self, event):
        # Ignore if the event is targeting a slider widget (sliders use arrows for value adjustment)
        widget_class = event.widget.winfo_class()
        if widget_class in ("Scale", "TScale"):
            return
        if isinstance(event.widget, (customtkinter.CTkSlider,)):
            return
        # Also check parent — CTkSlider nests a tk canvas inside
        try:
            parent = event.widget.master
            if isinstance(parent, customtkinter.CTkSlider):
                return
        except Exception:
            pass
        if self.slideshow_view.winfo_ismapped():
            self.slideshow_view.prev_frame()

    def _on_right_key(self, event):
        widget_class = event.widget.winfo_class()
        if widget_class in ("Scale", "TScale"):
            return
        if isinstance(event.widget, (customtkinter.CTkSlider,)):
            return
        try:
            parent = event.widget.master
            if isinstance(parent, customtkinter.CTkSlider):
                return
        except Exception:
            pass
        if self.slideshow_view.winfo_ismapped():
            self.slideshow_view.next_frame()

    def show_sorting(self):
        self.slideshow_view.grid_remove()
        self.sorting_view.grid()
        self.sorting_view.update_listbox()

    def show_slideshow(self, file_list):
        self.sorting_view.grid_remove()
        self.slideshow_view.grid()
        self.slideshow_view.start(file_list)

    def maximize_window(self):
        os_name = platform.system()
        try:
            if os_name == "Windows":
                self.state("zoomed")
            elif os_name == "Linux":
                try:
                    self.attributes("-zoomed", True)
                except Exception:
                    self.state("zoomed")
            elif os_name == "Darwin":
                self.state("zoomed")
        except Exception:
            self.apply_geometry_fallback()

    def apply_geometry_fallback(self):
        scale = customtkinter.ScalingTracker.get_window_scaling(self)
        scaled_w = int(self.winfo_screenwidth() / scale)
        scaled_h = int(self.winfo_screenheight() / scale)
        self.geometry(f"{scaled_w}x{scaled_h}+0+0")

MainApplication = AlignApp
