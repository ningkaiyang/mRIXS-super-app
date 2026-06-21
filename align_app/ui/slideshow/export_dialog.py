import tkinter as tk
import customtkinter as ctk
import numpy as np
import tifffile

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class ExportComparisonDialog(ctk.CTkToplevel):
    def __init__(self, parent, aligned_sum, direct_sum, default_save_dir, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.title("Alignment Comparison")
        self.geometry("1000x600")
        self.minsize(800, 500)
        
        self.aligned_sum = aligned_sum
        self.direct_sum = direct_sum
        self.default_save_dir = default_save_dir

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.focus()
        self.grab_set()  # Make dialog modal

        # Extract active (non-background) pixels for independent scaling
        aligned_min = float(np.min(aligned_sum)) if aligned_sum.size > 0 else 0.0
        aligned_active = aligned_sum[aligned_sum > aligned_min]
        aligned_p60 = np.percentile(aligned_active, 60.0) if aligned_active.size > 0 else 1.0

        direct_min = float(np.min(direct_sum)) if direct_sum.size > 0 else 0.0
        direct_active = direct_sum[direct_sum > direct_min]
        direct_p60 = np.percentile(direct_active, 60.0) if direct_active.size > 0 else 1.0

        # Main Layout
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Plot frame
        self.plot_frame = ctk.CTkFrame(self.main_frame)
        self.plot_frame.pack(fill="both", expand=True)

        self.figure = Figure(figsize=(10, 5), dpi=100)
        self.ax1 = self.figure.add_subplot(121)
        self.ax2 = self.figure.add_subplot(122, sharex=self.ax1, sharey=self.ax1)

        # Plot Direct Sum
        self.ax1.imshow(direct_sum, cmap="viridis", vmin=0, vmax=direct_p60)
        self.ax1.set_title("Direct Sum (Unaligned)")
        self.ax1.axis("off")

        # Plot Aligned Sum
        self.ax2.imshow(aligned_sum, cmap="viridis", vmin=0, vmax=aligned_p60)
        self.ax2.set_title("Aligned Sum")
        self.ax2.axis("off")

        self.figure.tight_layout()

        # Canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Toolbar
        self.toolbar_frame = ctk.CTkFrame(self.main_frame, height=40)
        self.toolbar_frame.pack(fill="x", pady=(5, 0))
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        
        # Action Button Frame
        self.action_frame = ctk.CTkFrame(self.main_frame)
        self.action_frame.pack(fill="x", pady=10)

        self.cancel_button = ctk.CTkButton(
            self.action_frame, text="Cancel", command=self.destroy,
            fg_color="#888", hover_color="#666"
        )
        self.cancel_button.pack(side="left", padx=10)

        self.export_button = ctk.CTkButton(
            self.action_frame, text="💾 Export Aligned Sum", command=self.export_and_close
        )
        self.export_button.pack(side="right", padx=10)

    def export_and_close(self):
        save_path = tk.filedialog.asksaveasfilename(
            initialdir=self.default_save_dir,
            initialfile="aligned_sum.tif",
            defaultextension=".tif",
            filetypes=[("TIFF Files", "*.tif;*.tiff")],
            parent=self
        )
        if save_path:
            try:
                tifffile.imwrite(save_path, self.aligned_sum)
                tk.messagebox.showinfo("Export Successful", f"Aligned sum saved to:\n{save_path}", parent=self)
                self.destroy()
            except Exception as e:
                tk.messagebox.showerror("Export Failed", f"Failed to save file:\n{e}", parent=self)
