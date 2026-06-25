import tkinter as tk
import customtkinter

class RangeSlider(tk.Canvas):
    """
    A custom dual-handle range slider widget built on a Tkinter Canvas.

    This widget allows the user to select a range defined by a minimum and
    maximum value. It supports clicking and dragging the handles to adjust
    the selected range.
    """
    def __init__(self, parent, width=300, height=25, command=None, **kwargs):
        """
        Initialize the range slider widget.

        Args:
            parent: The parent widget.
            width (int): The width of the canvas in pixels. Default is 300.
            height (int): The height of the canvas in pixels. Default is 25.
            command (callable, optional): A callback function executed when the
                slider values change. Receives the left and right values.
            **kwargs: Additional keyword arguments passed to the tk.Canvas constructor.
        """
        # Determine background dynamically to match parent CTkFrame
        bg = "#2B2B2B"
        if hasattr(parent, "cget"):
            try:
                c = parent.cget("fg_color")
                if isinstance(c, (list, tuple)):
                    mode = customtkinter.get_appearance_mode().lower()
                    c = c[1] if mode == "dark" else c[0]
                if c and c != "transparent":
                    bg = c
            except Exception:
                pass

        super().__init__(parent, width=width, height=height, bg=bg, highlightthickness=0, **kwargs)
        self.command = command
        
        self.min_val = 0.0
        self.max_val = 1.0
        self.val_left = 0.0
        self.val_right = 1.0
        
        self.track_height = 4
        self.handle_radius = 8
        self.padding = 10  # Horizontal offset to prevent handles clipping at canvas bounds
        
        self.active_handle = None  # "left" or "right"
        
        # Colors
        self.track_bg = "#4A4A4A"
        self.track_active = "#1F72A5"
        self.handle_bg = "#D3D3D3"
        self.handle_active = "#1F72A5"
        
        self.bind("<Configure>", self._on_resize)
        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Motion>", self._on_mouse_move)
        
    def configure_range(self, min_val, max_val):
        """
        Set the absolute minimum and maximum boundaries for the slider.

        Args:
            min_val (float): The lower bound of the slider.
            max_val (float): The upper bound of the slider.
        """
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        if self.max_val <= self.min_val:
            self.max_val = self.min_val + 1.0
        
        self.val_left = max(self.min_val, min(self.max_val, self.val_left))
        self.val_right = max(self.min_val, min(self.max_val, self.val_right))
        if self.val_right < self.val_left:
            self.val_right = self.val_left
        self._draw()

    def set_values(self, val_left, val_right):
        """
        Set the current left and right handle values.

        The values are clamped to the configured minimum and maximum bounds.

        Args:
            val_left (float): The value for the left handle.
            val_right (float): The value for the right handle.
        """
        self.val_left = max(self.min_val, min(self.max_val, float(val_left)))
        self.val_right = max(self.min_val, min(self.max_val, float(val_right)))
        if self.val_right < self.val_left:
            self.val_right = self.val_left
        self._draw()

    def _get_coords(self):
        """
        Calculate and retrieve the current drawing coordinates for the slider.

        Returns:
            tuple: A tuple containing (x_min, x_max, y_center, x_left, x_right)
                representing the pixel coordinates of the slider bounds and handles.
        """
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1: w = 300
        if h <= 1: h = 25
        
        x_min = self.padding
        x_max = w - self.padding
        y_center = h / 2
        
        val_range = self.max_val - self.min_val
        x_left = x_min + ((self.val_left - self.min_val) / val_range) * (x_max - x_min)
        x_right = x_min + ((self.val_right - self.min_val) / val_range) * (x_max - x_min)
        return x_min, x_max, y_center, x_left, x_right

    def _draw(self):
        """
        Redraw the slider track, active range, and handles on the canvas.

        This is called whenever the values change or the widget is resized.
        """
        self.delete("all")
        x_min, x_max, y_center, x_left, x_right = self._get_coords()
        
        # Background track
        self.create_rectangle(
            x_min, y_center - self.track_height/2,
            x_max, y_center + self.track_height/2,
            fill=self.track_bg, outline=""
        )
        # Active track range
        self.create_rectangle(
            x_left, y_center - self.track_height/2,
            x_right, y_center + self.track_height/2,
            fill=self.track_active, outline=""
        )
        # Left handle
        l_color = self.handle_active if self.active_handle == "left" else self.handle_bg
        self.create_oval(
            x_left - self.handle_radius, y_center - self.handle_radius,
            x_left + self.handle_radius, y_center + self.handle_radius,
            fill=l_color, outline="#555555", width=1, tags="handle_left"
        )
        # Right handle
        r_color = self.handle_active if self.active_handle == "right" else self.handle_bg
        self.create_oval(
            x_right - self.handle_radius, y_center - self.handle_radius,
            x_right + self.handle_radius, y_center + self.handle_radius,
            fill=r_color, outline="#555555", width=1, tags="handle_right"
        )

    def _on_resize(self, event):
        """
        Handle the canvas resize event by redrawing the slider.

        Args:
            event: The Tkinter event object.
        """
        self._draw()

    def _on_click(self, event):
        """
        Handle mouse click events to determine which slider handle to activate.

        Args:
            event: The Tkinter mouse event containing the click coordinates.
        """
        x_min, x_max, y_center, x_left, x_right = self._get_coords()
        dist_l = abs(event.x - x_left)
        dist_r = abs(event.x - x_right)
        
        threshold = self.handle_radius + 4
        if dist_l < threshold or dist_r < threshold:
            if dist_l < dist_r:
                self.active_handle = "left"
            elif dist_r < dist_l:
                self.active_handle = "right"
            else:
                self.active_handle = "left" if event.x < x_left else "right"
            self._draw()

    def _on_drag(self, event):
        """
        Handle mouse drag events to update the position of the active handle.

        Enforces constraints such as preventing handles from crossing each other
        and clamps values within the allowed range.

        Args:
            event: The Tkinter mouse drag event containing the current coordinates.
        """
        if not self.active_handle:
            return
        x_min, x_max, y_center, x_left, x_right = self._get_coords()
        x = max(x_min, min(x_max, event.x))
        
        val_range = self.max_val - self.min_val
        val = self.min_val + ((x - x_min) / (x_max - x_min)) * val_range
        
        # Enforce that left handle cannot pass right handle
        if self.active_handle == "left":
            self.val_left = min(val, self.val_right - 1e-5)
        else:
            self.val_right = max(val, self.val_left + 1e-5)
            
        self._draw()
        if self.command:
            self.command(self.val_left, self.val_right)

    def _on_release(self, event):
        """
        Handle mouse release events to deactivate the currently dragged handle.

        Args:
            event: The Tkinter mouse release event.
        """
        self.active_handle = None
        self._draw()

    def _on_mouse_move(self, event):
        """
        Handle mouse motion events to update the cursor when hovering over handles.

        Args:
            event: The Tkinter mouse motion event.
        """
        x_min, x_max, y_center, x_left, x_right = self._get_coords()
        if abs(event.x - x_left) < self.handle_radius or abs(event.x - x_right) < self.handle_radius:
            self.configure(cursor="hand2")
        else:
            self.configure(cursor="")
