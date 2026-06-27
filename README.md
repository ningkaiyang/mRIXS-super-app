# RIXS Beamline Super-App

A comprehensive desktop application and headless CLI tool suite designed for LBNL map-RIXS beamlines (e.g., QERLIN). Originally an alignment GUI, this project is evolving into a unified platform to parse and handle all detector data, featuring offline alignment, real-time data streaming, cluster analysis, and sharpness detection.

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

## Key Features

1. **Zarr-Backed Frame Caching:** High-resolution TIFF loading is optimized by caching frame arrays inside a content-addressed `tif-cache/frames.zarr` database in the dataset directory. Cache keys are computed via MD5 hashes of filepaths and modification times, ensuring instantaneous sequence reloads even across different run sessions.
2. **Multi-Engine Alignment Architecture:**
   - **Iterative ECC Maximization (Default):** Uses a 3-level Gaussian Pyramid for robust sub-pixel alignment. It applies a Scharr gradient magnitude pre-filter after a Gaussian blur (sigma=4.0) to convert the image into a clean peak-edge intensity map, crops horizontally to the spectral band, and projects the 2D offset onto a 1D physical drift vector.
   - **PCA (SVD) Peak-Line Fitting:** Solves the intensity-weighted 2D coordinate covariance to align the critical cross-dispersion direction, utilizing Fourier-Domain Phase Correlation for the parallel component. Ideal for datasets with a sharp, prominent spectral line.
   - **Phase Correlation:** DoG bandpass-filtered Fourier-domain translation estimation with a Tukey (tapered cosine) window that preserves spectral line features near detector edges.
   - **Background Precomputation:** The UI supports batch precomputation of alignment offsets and sharpness scores on background threads, keeping the GUI responsive while progress is updated via thread-safe queues.
3. **Interactive UI Panels:**
   - **File Selection View:** Add and order files dynamically, with a **"Clear All"** button to quickly empty the queue, and natural sorting to sort by filename.
   - **Sharpness Evaluation Slideshow:** A specialized view (`SharpnessSlideshowView`) for evaluating mirror angle sharpness. It visualizes the internal states of the sharpness pipeline (Raw, Denoised, Masked) alongside 1D profile correlation metrics, allowing visual inspection against human-ranked ground truth.
   - **Clamping Controls:** Adjust intensity floor and ceiling sliders in real-time to strip hot pixels or boost low-intensity structural boundaries.
   - **Auto-Snap Threshold Sweeps:** Sweep PCA thresholds dynamically on a background thread to locate the minimum perpendicular spread and align cleanly.
   - **Manual Line Alignment (PCA only):** Draw custom reference lines and click-to-override alignment offsets.
   - **Zooming & Panning:** Smooth multi-level zoom to target fine sub-pixel features.
4. **Inline Export Comparison & Multi-Plot Export:** 
   - Side-by-side comparison of the direct (unaligned) and aligned summation arrays with independent contrast scaling via the isolated `ExportComparisonView`, allowing visual validation of alignment efficacy prior to saving.
   - The Sharpness Slideshow supports bulk exporting of multi-plot diagnostic PNGs summarizing the entire sharpness pipeline for each frame.

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
1. **Sharpness Evaluation Metrics (Implemented):** A robust programmatic testing loop (`sharpness_cli.py`) has been implemented alongside a state-of-the-art denoising pipeline (`test_denoise.py`) to evaluate mirror angles. Currently iterating on mathematical isolation techniques (e.g., PCA filtering) to stabilize high-frequency metric performance across highly-noisy raw CCD data.
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
