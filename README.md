# RIXS Super-App

A comprehensive desktop application and headless CLI tool suite designed for LBNL RIXS beamlines (e.g., QERLIN). Originally an alignment GUI, this project is evolving into a unified platform to parse and handle all detector data, featuring offline alignment, zeroth-order line calibration (FWHM, resolving power R, mirror pitch focus curves), and automated batch export diagnostics.

## Setup Instructions

### Prerequisites
- Python 3.10 or higher
- System libraries for Tkinter/CustomTkinter

### Installation
1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   ```
2. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application
To launch the graphical user interface:
```bash
python run.py
```

### Creating a Desktop Shortcut
To create a simple, double-clickable Desktop shortcut for lab computers:

- **On Windows:**
  ```cmd
  python create_shortcut.py
  ```
- **On macOS:**
  ```bash
  python3 create_shortcut.py --terminal
  ```

This automatically generates a Desktop shortcut with the custom RIXS icon pointing to `run.py` and the project working directory.

## Key Features

1. **Zarr-Backed Frame Caching:** High-resolution TIFF loading is optimized by caching frame arrays inside a content-addressed `tif-cache/frames.zarr` database in the dataset directory. Cache keys are computed via MD5 hashes of filepaths and modification times, ensuring instantaneous sequence reloads even across different run sessions.
2. **Multi-Engine Alignment Architecture:**
   - **Iterative ECC Maximization (Default):** Uses a 3-level Gaussian Pyramid for robust sub-pixel alignment. It applies a Scharr gradient magnitude pre-filter after a Gaussian blur (sigma=4.0) to convert the image into a clean peak-edge intensity map, crops horizontally to the spectral band, and projects the 2D offset onto a 1D physical drift vector.
   - **PCA (SVD) Peak-Line Fitting:** Solves the intensity-weighted 2D coordinate covariance to align the critical cross-dispersion direction, utilizing Fourier-Domain Phase Correlation for the parallel component. Ideal for datasets with a sharp, prominent spectral line.
   - **Phase Correlation:** DoG bandpass-filtered Fourier-domain translation estimation with a Tukey (tapered cosine) window that preserves spectral line features near detector edges.
   - **Background Precomputation:** The UI supports batch precomputation of alignment offsets and sharpness scores on background threads, keeping the GUI responsive while progress is updated via thread-safe queues.
3. **Interactive UI Panels:**
   - **File Selection View:** Add and order files dynamically, with a **"Clear All"** button to quickly empty the queue, and natural sorting to sort by filename.
   - **Zeroth-Order Calibration Slideshow:** A specialized view (`ZerothOrderSlideshowView`) for mirror pitch calibration. It visualizes pipeline stages (Raw → Denoised → Row-Smoothed → Gradient → Fitted-Line Strip) alongside a 1D Gaussian profile fit, reporting FWHM in pixels and meV, and resolving power R. Supports TXT scan log import to map motor pitch to frames, peak-focus jump, and bulk diagnostic PNG export including a mirror pitch vs FWHM focus curve.
   - **Clamping Controls:** Adjust intensity floor and ceiling sliders in real-time to strip hot pixels or boost low-intensity structural boundaries.
   - **Auto-Snap Threshold Sweeps:** Sweep PCA thresholds dynamically on a background thread to locate the minimum perpendicular spread and align cleanly.
   - **Manual Line Alignment (PCA only):** Draw custom reference lines and click-to-override alignment offsets.
   - **Zooming & Panning:** Smooth multi-level zoom to target fine sub-pixel features.
4. **Inline Export Comparison & Multi-Plot Export:** 
   - Side-by-side comparison of the direct (unaligned) and aligned summation arrays with independent contrast scaling via the isolated `ExportComparisonView`, allowing visual validation of alignment efficacy prior to saving.
   - The Zeroth-Order Calibration Slideshow supports bulk exporting of multi-plot diagnostic PNGs summarizing the entire pipeline for each frame, plus a `focus_curve.png` focus curve with parabolic fit and resolving power R annotation. When a TXT scan log is loaded, the X-axis shows motor goal position; when absent, it falls back to Frame Index.

## Running the Tests
To run the entire test suite (including E2E, manager, stress, CLI, and core algorithmic tests):
```bash
pytest -v
```
To run specific test modules, run them individually:
```bash
pytest tests/test_e2e.py -v
pytest tests/test_align.py -v
pytest tests/test_cli.py -v
```

## Headless CLI Tool

A standalone command-line interface (`align_cli.py`) is provided for batch-processing TIFF sequences without a GUI. This is ideal for remote servers, HPC clusters, or automated workflows.

### CLI Usage

**Process a single directory:**
```bash
python3 align_cli.py -d "tif-files/Fe L" -e ECC --png
```

**Recursively process all subdirectories:**
```bash
python3 align_cli.py -d tif-files/ -r -e ECC --png
```

**Run multiple engines:**
```bash
python3 align_cli.py -d tif-files/ -r -e ECC PhaseCorrelation PCA --png
```

**Use automatic PCA threshold optimization:**
```bash
python3 align_cli.py -d tif-files/ -r -e PCA -t auto --png
```

### CLI Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `-d`, `--dir` | Root directory containing TIFF files | *(required)* |
| `-r`, `--recursive` | Recurse into subdirectories | `False` |
| `-e`, `--engines` | Alignment engines to run (`ECC`, `PCA`, `PhaseCorrelation`, `all`) | `ECC` |
| `-t`, `--threshold` | PCA percentile threshold (float or `auto` for auto-optimization) | `99.9` |
| `--png` | Save side-by-side comparison PNG (Direct Sum vs Aligned Sum) | `False` |
| `--json` | Save per-frame offset log as a JSON file | `False` |
| `--ephemeral-cache` | Delete `tif-cache/` after processing | `False` |
| `--overwrite` | Overwrite existing `sum/` output files | `False` |

### CLI Output Structure

For each processed directory, the CLI creates a `sum/` subdirectory containing:
```
dataset_folder/
└── sum/
    ├── base_sum.tif                     # Unaligned direct sum
    ├── aligned_sum_ECC.tif              # Aligned sum (per engine)
    ├── aligned_offsets_ECC.json         # Per-frame (dx, dy) offsets + metadata (only if --json)
    └── comparison_ECC.png               # Side-by-side comparison (if --png)
```

## Zeroth-Order Calibration CLI

A dedicated headless CLI (`zeroth_order_cli.py`) batch-processes zeroth-order mirror-pitch scan directories, computing FWHM for every frame and identifying the optimal mirror position. It mirrors the feature set of the GUI's Export panel, including focus curve generation, diagnostic PNGs, and tabular summary reports.

### CLI Usage

**Process a single scan directory (with TXT scan log and physical parameters):**
```bash
python3 zeroth_order_cli.py -d "RIXS_ZeroOrderScan/Single Motor Scan 004202 Images" \
    --dispersion 2.5 --mono-energy 850.0
```

**Recursive batch across all scan subdirectories:**
```bash
python3 zeroth_order_cli.py -d RIXS_ZeroOrderScan -r \
    --dispersion 2.5 --mono-energy 850.0 --format all --export-plots all
```

**Minimal run — terminal table only, no plots:**
```bash
python3 zeroth_order_cli.py -d path/to/scan --export-plots none --no-focus-curve
```

**Export JSON summary and all diagnostic PNGs:**
```bash
python3 zeroth_order_cli.py -d RIXS_ZeroOrderScan -r --format json --export-plots all --overwrite
```

### CLI Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `-d`, `--dir` | Root directory containing TIFF images (or parent when combined with `-r`) | *(required)* |
| `-r`, `--recursive` | Recursively scan subdirectories for TIFF datasets | `False` |
| `-t`, `--txt` | Explicit path to a `.txt` scan log file (overrides auto-discovery) | auto-discover |
| `-o`, `--output-dir` | Custom output directory | `<scan_dir>/zeroth_order_analysis/` |
| `--dispersion` | Energy dispersion in meV/px (e.g. `2.5`) | *(optional)* |
| `--mono-energy` | Monochromator energy E_mono in eV (e.g. `850.0`). Enables resolving power R | *(optional)* |
| `--plot-focus-curve` / `--no-focus-curve` | Toggle `focus_curve.png` generation | enabled |
| `--export-plots` | Diagnostic PNG mode: `best` (optimal frame only), `all`, or `none` | `best` |
| `--format` | Summary report format: `table` (terminal), `csv`, `json`, or `all` | `table` |
| `--overwrite` | Overwrite existing `zeroth_order_analysis/` output directory | `False` |
| `-q`, `--quiet` | Suppress terminal output | `False` |

### CLI Output Structure

For each processed directory, the CLI creates a `zeroth_order_analysis/` subdirectory containing:
```
scan_directory/
└── zeroth_order_analysis/
    ├── focus_curve.png                  # Motor position (or Frame Index) vs FWHM with parabolic fit
    ├── frame_NNN_diagnostic.png         # 2×2 diagnostic PNG for best (or all) frames
    ├── summary.csv                      # Per-frame table (if --format csv or all)
    └── summary.json                     # Full metadata + per-frame records (if --format json or all)
```

**`summary.csv` columns**: `frame_index`, `filename`, `motor_position`, `fwhm_px`, `fwhm_mev`, `resolving_power`, `score`, `fit_ok`.

**`summary.json` top-level fields**: `scan_dir`, `txt_log`, `total_frames`, `valid_fwhm_count`, `best_frame_index`, `best_fwhm_px`, `best_fwhm_mev`, `best_resolving_power`, `optimal_motor_position`, `energy_dispersion_mev_per_px`, `mono_energy_ev`, `frames` (array of per-frame records).

**TXT scan log auto-discovery**: The CLI searches for a `.txt` file inside each scan directory (the same location as the TIF images). When multiple `.txt` files exist, the first sorted file is used. The `-t` flag overrides auto-discovery.

**Focus curve fallback**: When no `.txt` scan log is found, `focus_curve.png` is still generated with **Frame Index** on the X-axis, so you always get a visual overview of the FWHM trend across the sequence.

# QERLIN Beamline Spectrometer Scan Alignment Project

## 1. Context & Experimental Physics
This project supports the QERLIN beamline (Beamline 6.0.2) at the Advanced Light Source (ALS), Lawrence Berkeley National Laboratory (LBNL). QERLIN performs high-resolution, momentum-resolved (q-resolved) Resonant Inelastic X-ray Scattering (RIXS) in the soft X-ray regime. 

To resolve fine-grained electronic features (such as phonons, magnons, and d-d excitations), the spectrometer requires extreme energy resolution and uses long path-length disperser arms. Consequently, the instrument is highly sensitive to sub-pixel spatial drift at the detector face. This drift is induced by mechanical relaxation and micro-thermal fluctuations over the course of long experimental runs.

## 2. The Core Problem
To prevent spatial drift from broadening and blurring the spectral features, long scans are acquired as a sequence of shorter, sliced exposures (e.g., 200 frames per scan merged together in TIFF format). However, because the camera and optics shift dynamically during the process, these individual slices undergo 2D translations (Δx, Δy) relative to one another. If summed directly without alignment, the resulting spectrum suffers from severe resolution loss.

## 3. Current Implementation & Limitations
The alignment application corrects these shifts using three auto-alignment engines and manual override modes:
- **Iterative ECC Engine (Default):** 3-level Gaussian Pyramid for robust sub-pixel alignment. It applies a Scharr gradient magnitude pre-filter after a Gaussian blur to convert the image into a clean peak-edge intensity map, cropping horizontally to the valid spectral band.
- **PCA Engine:** Locates the spectroscopic line via intensity-weighted PCA (SVD) on pixels that exceed an intensity percentile threshold. Best for sharp, prominent lines. Supports automatic threshold optimization.
- **Phase Correlation Engine:** Computes translation offsets in the Fourier domain with DoG bandpass pre-filtering and Tukey (tapered cosine) windowing for improved accuracy under low SNR.
- **Manual Mode (PCA only):** The user manually draws a line corresponding to a visible spectral feature to guide or override PCA alignment offsets.

### The PCA Limitation:
The intensity-weighted PCA auto-alignment engine assumes the **elastic scattering line** (zero energy loss) is the brightest feature on the detector. However, in many experimental regimes (such as certain transition metal L-edges or highly absorbing samples), elastic scattering is weak. As a result, the elastic line is dim, sparse, or completely absent. 

Under these conditions, the intensity-weighted PCA is dominated by Poisson noise, cosmic rays, or the diffuse 2D inelastic scattering cloud in the center of the detector. This leads to erratic centroid determinations on noise spikes, failed line fits, and incorrect alignment offsets. To bypass this limitation, users should use the default **ECC Engine**, which precomputes robust sub-pixel alignment over diffuse, line-less spectral clouds.

## 4. Technical Goals & Roadmap
The ultimate objective is to develop a robust, fully automated super-app that handles the entire lifecycle of RIXS detector data, from live acquisition to offline alignment and analysis.

### A. Core Tool Suite & Diagnostics
1. **Zeroth-Order Calibration (Implemented):** A robust programmatic zeroth-order line detection and FWHM evaluation pipeline (`zeroth_order.py`, `zeroth_order_evaluator.py`) paired with a full GUI (`ZerothOrderSlideshowView`). Reports FWHM in px and meV, resolving power R, and generates a mirror-pitch focus curve from imported scan log TXT files. Headless batch CLI available via `zeroth_order_cli.py` — supports single and recursive batch modes, focus curve generation (motor position or Frame Index fallback), diagnostic PNGs, CSV/JSON summary reports, and resolving power R calculation.
2. **Live Data Streaming & Cluster Analysis:** Integrate real-time processing pipelines (from legacy scripts) to monitor live data collection. This includes dark background masking, connected-component cluster analysis (identifying single-photon events), and generating live 2D spatial event maps and IntDen histograms.
3. **Multi-Panel UI:** Expand the GUI beyond the alignment slideshow to host dedicated workspaces for sharpness checking, live streaming dashboards, and histogram visualization.

### B. Advanced Algorithmic Registration
1. **Iterative ECC Maximization (Implemented):** A 3-level Gaussian Pyramid that pre-filters frames using a Scharr gradient magnitude filter after a Gaussian blur (sigma=4.0).
2. **Fourier-Domain Phase Correlation with Spatial Pre-filtering (Implemented):** DoG bandpass filtering isolates structural boundaries while a Tukey window preserves signal near detector edges.
3. **Headless CLI Batch Processing (Implemented):** Standalone `align_cli.py` script for batch-processing.

### C. Machine Learning & Deep Learning Methods
1. **Self-Supervised Deep Denoising (Noise2Noise / Noise2Void):** Leverage the temporal correlation of successive frames to train a U-Net to denoise sparse, photon-starved frames before registration.
2. **Supervised Siamese Regression CNNs:** Train a convolutional network to predict (Δx, Δy) offsets directly from a stacked pair of frames.
3. **Unsupervised Spatial Transformer Networks (STNs):** Train an end-to-end registration network that optimizes similarity metrics (e.g., SSIM) between the reference frame and the warped target frame.
