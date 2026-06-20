import platform
import customtkinter
from align_app.sorting_view import SortingView
from align_app.slideshow_view import SlideshowView

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

class AlignApp(customtkinter.CTk):
    """
    Main application class for the Spectroscopy Image Alignment GUI.

    This class manages the main window, acts as a container for the different
    views (SortingView and SlideshowView), and handles global application state
    such as keyboard navigation and window maximization.
    """
    def __init__(self, *args, show_window=True, **kwargs):
        """
        Initialize the main application window and its components.

        Args:
            *args: Variable length argument list passed to the CTk constructor.
            show_window (bool): Whether to immediately display the window after initialization.
            **kwargs: Arbitrary keyword arguments passed to the CTk constructor.
        """
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

        if show_window:
            # Maximize the geometry/state in the background while withdrawn
            self.maximize_window()
            # Force the OS window manager to apply the geometry changes before showing
            self.update()
            # Reveal the window directly in its maximized state
            self.deiconify()

    def _on_left_key(self, event):
        """
        Handle the global left arrow key press event.

        If the slideshow view is active and the event didn't originate from a
        slider widget, this method navigates to the previous frame.

        Args:
            event: The Tkinter event object containing event details.
        """
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
        """
        Handle the global right arrow key press event.

        If the slideshow view is active and the event didn't originate from a
        slider widget, this method navigates to the next frame.

        Args:
            event: The Tkinter event object containing event details.
        """
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
        """
        Display the file sorting view and hide the slideshow view.

        This method also triggers an update of the file listbox in the sorting view.
        """
        self.slideshow_view.grid_remove()
        self.sorting_view.grid()
        self.sorting_view.update_listbox()

    def show_slideshow(self, file_list):
        """
        Display the slideshow view and start the slideshow with the provided files.

        Args:
            file_list (list of str): A list of file paths to be displayed in the slideshow.
        """
        self.sorting_view.grid_remove()
        self.slideshow_view.grid()
        self.slideshow_view.start(file_list)

    def maximize_window(self):
        """
        Maximize the application window based on the current operating system.

        Attempts to use the appropriate window state command for Windows, Linux,
        or macOS. Falls back to manual geometry calculation if maximization fails.
        """
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
        """
        Apply a manual window size based on the screen dimensions and UI scaling.

        Used as a fallback when native window maximization is not supported or fails.
        """
        scale = customtkinter.ScalingTracker.get_window_scaling(self)
        scaled_w = int(self.winfo_screenwidth() / scale)
        scaled_h = int(self.winfo_screenheight() / scale)
        self.geometry(f"{scaled_w}x{scaled_h}+0+0")

MainApplication = AlignApp
