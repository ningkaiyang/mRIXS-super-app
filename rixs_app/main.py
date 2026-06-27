import platform
import customtkinter
from rixs_app.ui.sorting_view import SortingView
from rixs_app.ui.alignment_slideshow.slideshow_view import SlideshowView
from rixs_app.ui.alignment_slideshow.comparison_view import ExportComparisonView
from rixs_app.ui.sharpness_slideshow.slideshow_view import SharpnessSlideshowView

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

class RixsApp(customtkinter.CTk):
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
            on_start_slideshow=self.show_slideshow,
            on_evaluate_sharpness=self.show_sharpness_slideshow
        )
        self.slideshow_view = SlideshowView(
            self.container,
            on_back_to_sorting=self.show_sorting,
            on_show_export_comparison=self.show_export_comparison
        )
        self.export_comparison_view = ExportComparisonView(
            self.container,
            on_back=self.show_slideshow_from_comparison
        )
        self.sharpness_view = SharpnessSlideshowView(
            self.container,
            on_back_to_sorting=self.show_sorting
        )

        self.sorting_view.grid(row=0, column=0, sticky="nsew")
        self.slideshow_view.grid(row=0, column=0, sticky="nsew")
        self.export_comparison_view.grid(row=0, column=0, sticky="nsew")
        self.sharpness_view.grid(row=0, column=0, sticky="nsew")

        # Global keyboard navigation binding (bind_all captures events
        # regardless of which widget has focus)
        self.bind_all("<Left>", self._on_left_key)
        self.bind_all("<Right>", self._on_right_key)

        # Bind window close protocol to ensure proper teardown of resources (e.g. Matplotlib)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

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
        elif self.sharpness_view.winfo_ismapped():
            self.sharpness_view.prev_frame()

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
        elif self.sharpness_view.winfo_ismapped():
            self.sharpness_view.next_frame()

    def show_sorting(self):
        """
        Display the file sorting view and hide the slideshow and comparison views.

        This method also triggers an update of the file listbox in the sorting view.
        """
        self.slideshow_view.grid_remove()
        self.export_comparison_view.grid_remove()
        self.sharpness_view.grid_remove()
        self.sorting_view.grid()
        self.sorting_view.update_listbox()

    def show_slideshow(self, file_list):
        """
        Display the slideshow view and start the slideshow with the provided files.

        Args:
            file_list (list of str): A list of file paths to be displayed in the slideshow.
        """
        self.sorting_view.grid_remove()
        self.export_comparison_view.grid_remove()
        self.sharpness_view.grid_remove()
        self.slideshow_view.grid()
        self.slideshow_view.start(file_list)

    def show_sharpness_slideshow(self, file_list):
        """
        Display the sharpness slideshow view and start the evaluation.

        Args:
            file_list (list of str): A list of file paths to be analyzed.
        """
        self.sorting_view.grid_remove()
        self.slideshow_view.grid_remove()
        self.export_comparison_view.grid_remove()
        self.sharpness_view.grid()
        self.sharpness_view.start(file_list)

    def show_export_comparison(self, aligned_sum, direct_sum, initial_dir):
        """Transition from the slideshow view to the in-app comparison view.

        Args:
            aligned_sum (np.ndarray): The drift-corrected summed image.
            direct_sum (np.ndarray): The naïve (unaligned) summed image.
            initial_dir (str): Default directory for the save-file dialog.
        """
        self.slideshow_view.grid_remove()
        self.export_comparison_view.load_comparison(aligned_sum, direct_sum, initial_dir)
        self.export_comparison_view.grid()

    def show_slideshow_from_comparison(self):
        """Return from the comparison view back to the slideshow view.

        The slideshow state (current frame, alignment parameters, etc.) is
        preserved because the SlideshowView is never destroyed.
        """
        self.export_comparison_view.grid_remove()
        self.slideshow_view.grid()

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

    def on_close(self):
        """
        Handle the window delete/close request (WM_DELETE_WINDOW protocol).

        This method acts as a custom exit handler. When the user closes the main
        window, it ensures that the active Matplotlib figures, canvases, and
        associated navigation toolbars are cleanly torn down before calling
        destroy() to destroy the widget hierarchy. This prevents PyEval thread
        GIL errors.
        """
        self.export_comparison_view._teardown_mpl()
        self.destroy()

MainApplication = RixsApp
