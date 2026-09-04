# RIXS Super-App

[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![UI Framework: PySide6 (Qt 6)](https://img.shields.io/badge/GUI-PySide6%20(Qt%206)-41CD52.svg)](https://wiki.qt.io/Qt_for_Python)
[![Core: NumPy / SciPy / OpenCV / Zstd](https://img.shields.io/badge/core-NumPy%20%7C%20SciPy%20%7C%20OpenCV%20%7C%20Zstd-orange.svg)](https://numpy.org/)
[![LLM Agent: LBNL CBORG](https://img.shields.io/badge/AI%20Agent-LBNL%20CBORG-purple.svg)](https://cborg.lbl.gov/)
[![Tests: 650+ passing](https://img.shields.io/badge/tests-650%2B%20passing%20(34%20suites)-brightgreen.svg)](https://pytest.org/)

**RIXS Super-App** is a high-performance desktop application and headless CLI toolkit engineered for the **QERLIN Spectrometer** at **Beamline 6.0.2** of the **Advanced Light Source (ALS), Lawrence Berkeley National Laboratory (LBNL)**.

The software provides an end-to-end data processing workflow:
1. **Dark Frame Masking & Baseline Calibration** — Hot-pixel filtering, RTS noise rejection, and baseline correction
2. **Single-Photon Clustering & Event Reconstruction** — Single-photon identification, energy filtering, and sub-pixel event map generation
3. **Spatial Drift Alignment** — Sub-pixel frame registration to correct drift across long acquisitions
4. **Zeroth-Order Focus Calibration** — Spectrometer mirror pitch focus optimization via FWHM line analysis

Additionally, it integrates the **RIXS Co-Pilot Agent Sidebar**—an LLM-powered assistant connected to LBNL CBORG for natural-language automation, tool execution, and dataset diagnostics.

---

## Table of Contents

- [The 4 Core Pillars](#the-4-core-pillars)
  - [Pillar 1: Dark Frame Masking & Baseline Calibration](#pillar-1-dark-frame-masking--baseline-calibration)
  - [Pillar 2: Single-Photon Clustering & Super-Resolution Event Reconstruction](#pillar-2-single-photon-clustering--super-resolution-event-reconstruction)
  - [Pillar 3: Spatial Drift Alignment](#pillar-3-spatial-drift-alignment)
  - [Pillar 4: Zeroth-Order Focus Calibration (Mirror Pitch)](#pillar-4-zeroth-order-focus-calibration-mirror-pitch)
- [RIXS Co-Pilot AI Agent Sidebar](#rixs-co-pilot-ai-agent-sidebar)
- [Architecture & Performance Highlights](#architecture--performance-highlights)
- [Setup & Installation](#setup--installation)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
  - [Desktop Shortcut Creation](#desktop-shortcut-creation)
  - [GUI Framework Notice (PySide6 Only)](#gui-framework-notice-pyside6-only)
- [Execution & CLI Command Reference](#execution--cli-command-reference)
  - [1. Desktop GUI Launcher](#1-desktop-gui-launcher)
  - [2. Spatial Drift Alignment CLI (`align_cli.py`)](#2-spatial-drift-alignment-cli-align_clipy)
  - [3. Photon Clustering CLI (`cluster_cli.py`)](#3-photon-clustering-cli-cluster_clipy)
  - [4. Zeroth-Order Calibration CLI (`zeroth_order_cli.py`)](#4-zeroth-order-calibration-cli-zeroth_order_clipy)
  - [5. Image Denoising CLI (`denoise_cli.py`)](#5-image-denoising-cli-denoise_clipy)
- [Automated Testing Suite](#automated-testing-suite)
- [Project Directory Structure](#project-directory-structure)
- [License & Acknowledgements](#license--acknowledgements)

---

## The 4 Core Pillars

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                    RIXS Super-App                                      │
├────────────────────┬────────────────────┬────────────────────┬─────────────────────────┤
│    Pillar 1        │    Pillar 2        │    Pillar 3        │    Pillar 4             │
│    Dark Masking    │ Photon Clustering  │ Spatial Drift      │ Zeroth-Order Focus      │
│   & Baseline Cal   │ & Event Recon      │    Alignment       │ (Mirror Pitch Cal)      │
├────────────────────┼────────────────────┼────────────────────┼─────────────────────────┤
│ • Temporal Median  │ • 8-Conn Clusters  │ • Iterative ECC    │ • 6-Stage Pipeline      │
│ • StdDev Cutoff    │ • IntDen Filtering │ • PCA Line Fit     │ • FWHM (px & meV)       │
│ • RTS Noise Reject │ • Centroiding      │ • Phase Correlation│ • Resolving Power R     │
│ • CDF Overlays     │ • Event Map Recon  │ • Side-by-Side View│ • Parabolic Focus Curve │
└────────────────────┴────────────────────┴────────────────────┴─────────────────────────┘
```

---

### Pillar 1: Dark Frame Masking & Baseline Calibration

**UI View:** `DarkMaskingView` (Nav Index `1`) · **CLI Subcommand:** `python cluster_cli.py dark-mask`

Establishes a clean detector baseline across the sCMOS array, filtering out hot pixels, telegraph (RTS) noise, and spatial offset variations prior to signal processing.

#### Features & Workflow
- **Temporal Median Baseline:** Computes a per-pixel median across dark frames, rejecting transient cosmic rays and baseline fluctuations.
- **Two-Tier Noise Masking:**
  - *Standard Deviation Cutoff:* Identifies and masks persistently noisy or unstable pixels.
  - *Tail Excursion Cutoff:* Flags random telegraph signal (RTS) switching pixels that fluctuate between discrete energy levels.
- **Interactive Dual Histograms:** Matplotlib visualization showing StdDev and Tail Residual distributions with draggable threshold sliders and twin-axis cumulative distribution (CDF) overlays.
- **Real-Time KPI Cards:** Displays surviving pixel counts, active detector area percentage, rejection rates, mean baseline ADU, and outlier counts.
- **Publication-Ready Export:** Generates high-resolution dual-panel PNG diagnostic figures.
- **Atomic Disk Persistence:** Saves calibration outputs (`MED_Dark.tif`, `Final_Mask.tif`, and metadata JSON) via safe temporary file swapping.

---

### Pillar 2: Single-Photon Clustering & Super-Resolution Event Reconstruction

**UI View:** `ClusteringStudioView` (Nav Index `3`) · **CLI Subcommands:** `cluster_cli.py cluster`, `reconstruct`, `full`

Isolates individual photon events on sCMOS frames in the soft X-ray regime (e.g., Ni L3, Fe L, O K), discriminates against cosmic rays, and reconstructs super-resolution event maps.

```
Raw Signal TIFFs ──► Baseline Conditioning & Noise Masking ──► Threshold Cutoff
                          │
                          ▼
                 8-Connected Components Cluster Extraction
                          │
                          ▼
             Cluster Features (IntDen, Centroid XM/YM, Area, Circularity)
                          │
                          ▼
             Energy & Geometry Filtering (IntDen Window, Max Area, Min Circularity)
                          │
                          ▼
             Super-Resolution Sub-Pixel Accumulation
                          │
                          ▼
             Output: Photon_Event_Map.tif & IntDen_histogram.png
```

#### Features & Workflow
- **Signal Conditioning:** Applies the dark baseline and noise mask, zeroing out sub-threshold background noise.
- **Cluster Extraction:** Groups adjacent lit pixels using 8-connected component labeling and computes cluster properties: integrated density (IntDen), area, circularity, and sub-pixel intensity centroid coordinates.
- **Energy & Geometry Filtering:**
  - *Integrated Density (IntDen):* Selects photon events matching specific absorption edge energy windows.
  - *Morphology Filter:* Rejects elongated cosmic ray tracks and electrical discharge events based on cluster size and circularity thresholds.
- **Super-Resolution Event Map:** Accumulates sub-pixel centroid coordinates into a 2D event map (at 1× or 2× scale).
- **Three Workspace Modes:**
  - *Mode A: Dashboard Overview* — Total photon events, average flux rate, interactive IntDen histogram with cutoff sliders, and event map preview.
  - *Mode B: Frame Inspection* — Synchronized raw vs. conditioned frame viewer with overlaid cluster bounding boxes, centroids, and cluster property tooltips.
  - *Mode C: Chunk Analysis* — Sequence chunking to track photon event rates, beam decay, and stability over time.

---

### Pillar 3: Spatial Drift Alignment

**UI View:** `SlideshowView` (Nav Index `5`) · `ExportComparisonView` (Nav Index `6`) · **CLI:** `python align_cli.py`

Corrects sub-pixel spatial drift between sequential frames across long exposure runs to prevent spectral broadening when summing frames.

#### Multi-Engine Registration Architecture

| Engine | Description | Best Suited For |
|---|---|---|
| **Iterative ECC Maximization** *(Default)* | Multi-level Gaussian Pyramid correlation with Scharr gradient magnitude filtering and physical drift projection. | Diffuse inelastic scattering clouds, line-less spectra, noisy or low-contrast acquisitions. |
| **PCA (SVD) Peak-Line Fitting** | Intensity-weighted spatial covariance analysis along prominent spectral lines with automated threshold optimization (`-t auto`). | Sharp, prominent elastic scattering lines with high SNR. |
| **Fourier Phase Correlation** | Translation estimation in the Fourier domain with bandpass pre-filtering and Tukey windowing. | Datasets with distributed or periodic spectral features. |

#### Features & GUI Controls
- **Interactive Slideshow & Zoom:** Sub-pixel navigation with smooth multi-level zoom, intensity ceiling/floor sliders, and hot pixel suppression.
- **Auto-Snap Threshold Sweeps:** Sweeps over intensity percentiles to find the optimal perpendicular line spread.
- **Manual Line Drawing Override:** User-defined guide lines to anchor or override alignment vectors.
- **Side-by-Side Comparison:** Direct visual comparison of unaligned vs. aligned sums with independent contrast scaling before saving.

---

### Pillar 4: Zeroth-Order Focus Calibration (Mirror Pitch)

**UI View:** `ZerothOrderSlideshowView` (Nav Index `7`) · **CLI:** `python zeroth_order_cli.py`

Analyzes specular reflection (zeroth order, m=0) grating sweeps to determine the optimal spectrometer mirror pitch angle (SM3) by measuring line sharpness.

```
Raw Frame ──► Border Crop ──► Multistage Denoise ──► Background Subtraction
                                                            │
                                                            ▼
Focus Curve & Motor Position ◄── 1D Gaussian Fit ◄── Robust Line Fit ◄── Scharr Edge
```

#### The 6-Stage Algorithmic Pipeline
1. **Boundary Edge Cropping:** Strips dead detector borders and non-illuminated pixels.
2. **Multistage Denoising:** Anscombe transform, Median Absolute Deviation (MAD) despiking, and edge-preserving bilateral filtering.
3. **Background Subtraction:** Row-wise smoothing and baseline offset removal.
4. **Edge Enhancement:** Scharr gradient magnitude filtering perpendicular to the spectral line.
5. **Robust Line Extraction:** Identifies line orientation via RANSAC and SVD.
6. **1D Gaussian Profile Fitting:** Extracts a perpendicular profile and fits a Gaussian curve to measure sub-pixel Full Width at Half Maximum (FWHM).

#### Physical Metrics & Focus Analysis
- **Energy Conversion:** Converts FWHM from pixels to meV using detector dispersion.
- **Resolving Power (R):** Calculates spectrometer resolving power (R = E_mono / FWHM_eV) when incident monochromator energy is supplied.
- **Parabolic Focus Curve:** Fits a parabola to motor position vs. FWHM to determine the optimal focus position.
- **GUI & CLI Diagnostics:** 5-stage intermediate pipeline strip viewer, automatic beamline `.txt` scan log pairing, and `focus_curve.png` export.

---

## RIXS Co-Pilot AI Agent Sidebar

The application includes an integrated, asynchronous AI Co-Pilot agent connected to the **LBNL CBORG OpenAI-compatible endpoint** (`cborg.lbl.gov`).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             PySide6 GUI Thread                              │
│  ┌───────────────────────┐                    ┌───────────────────────────┐ │
│  │   RixsStackedWidget   │                    │   SafeFigureCanvasQTAgg   │ │
│  │   (8 Views)           │                    │   (Matplotlib)            │ │
│  └───────────┬───────────┘                    └───────────────────────────┘ │
│              │                                              ▲               │
│              │ Qt Signals / Slots                           │               │
│  ┌───────────▼──────────────────────────────────────────────┴─────────────┐ │
│  │               AgentSidebarWidget (Collapsible Preloaded Panel)          │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │ QWebEngineView (Chromium) + QWebChannel Bridge (Markdown & UI)   │  │ │
│  │  └─────────────────────────────────┬────────────────────────────────┘  │ │
│  └────────────────────────────────────┼───────────────────────────────────┘ │
└───────────────────────────────────────┼─────────────────────────────────────┘
                                        │ run_coroutine_threadsafe
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    _AsyncLoopThread (Dedicated asyncio Event Loop)          │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ CborgAgentEngine (AsyncOpenAI / CBORG API)                            │  │
│  │  • Multi-turn streaming chat                                          │  │
│  │  • Multi-tier API key resolution (Env, appdata/cborg-auth/.env, UI)   │  │
│  │  • Approval Barriers (asyncio.Event for destructive / CLI actions)    │  │
│  │  • ToolRegistry dispatch (Auto JSON schemas from Python type hints)   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Capabilities
- **Streaming Chat:** Renders real-time responses with syntax-highlighted code.
- **Approval-Gated Tool Calling:** Destructive actions (CLI runs, calibrations, disk modifications) present an interactive UI approval card (`Approve` / `Reject` with feedback) before executing.
- **Subprocess Log Streaming:** Subprocess stdout/stderr streams directly to the Co-Pilot chat transcript.
- **Multi-Tier API Key Resolution:**
  1. Shell environment variable (`export CBORG_API_KEY=...`)
  2. Application persistent data file (`rixs_app/appdata/cborg-auth/.env`)
  3. In-app Setup Wizard modal

### Built-in Agent Tools

| Tool Name | Type | Description | Requires Approval |
|---|:---:|---|:---:|
| `list_directory` | Read-only | Lists files and subdirectories with file size summaries. | No |
| `check_alignment_readiness` | Read-only | Verifies TIFF file counts and dataset suitability for spatial alignment. | No |
| `check_calibration_readiness`| Read-only | Verifies TIFF sequence and scan log `.txt` metadata for mirror calibration. | No |
| `read_file_contents` | Read-only | Safely inspects configuration, scan logs, JSON summaries, and scripts. | No |
| `get_cli_help` | Read-only | Queries `--help` options for project CLI utilities. | No |
| `get_active_gui_state` | Read-only | Queries active view index, loaded dataset path, frame index, and metadata. | No |
| `run_spatial_alignment` | Execution | Triggers `align_cli.py` with selected engine and parameters. | **Yes** |
| `run_zeroth_order_calibration`| Execution | Triggers `zeroth_order_cli.py` with dispersion and mono energy. | **Yes** |
| `run_image_denoising` | Execution | Runs `denoise_cli.py` on single frames or batch directories. | **Yes** |
| `cli_runner` | Execution | Executes custom arguments against project CLI tools with streaming logs. | **Yes** |
| `update_gui_parameter` | GUI Control | Updates GUI state (such as jumping to a specific frame index). | **Yes** |
| `execute_terminal_command` | Execution | Arbitrary shell execution (available when Full Terminal Access is enabled). | **Yes** |

---

## Architecture & Performance Highlights

- **Pure CMV (Controller-Manager-View) Architecture:**
  - **Managers** (`SlideshowManager`, `ZerothOrderManager`, `ClusteringManager`): Handle all state machines, caching, and algorithmic workflows with **zero PySide6/Qt imports**.
  - **Controllers** (`*View` classes): Wire UI panels, handle Qt signals, and manage worker thread lifecycles.
  - **Panels**: Modular Qt view widgets communicating exclusively via controller references.
- **In-Memory Compressed Frame Cache (`frame_cache.py`, `dataset.py`):**
  - High-resolution TIFF sequences and derived stage filter frames are cached strictly in RAM using `CompressedFrameCache` (`float16` + Zstd).
  - Subsequent sequence loads and scrubbing operations are instantaneous (< 0.5 ms per frame) with zero disk clutter or cache files.
- **Thread Safety & GC Retention:**
  - Background computations run via `QThreadPool` and `QRunnable` workers.
  - Explicit GC retention pattern (`self._workers.add(worker)`) prevents Python garbage collection from tearing down active workers before Qt signal emission.
- **Performance Budgets:**
  - `apply_dark_thresholds()`: < 10 ms for 2048×2048 arrays.
  - `get_reconstruction()`: < 50 ms for real-time slider updates.
- **Fusion Dark Theme:** High-contrast palette (`#1a1a2e` base, `#16213e` panels, `#2196f3` accent) with native font stack fallbacks and dark tooltip palette enforcement.

---

## Setup & Installation

### Prerequisites
- **Python 3.10, 3.11, or 3.12**
- Standard C compiler / build tools (for NumPy, SciPy, and OpenCV)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/als-beamline-602/Each200Frames.git
   cd Each200Frames
   ```

2. **Create and activate a virtual environment:**
   - **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows:**
     ```cmd
     python -m venv .venv
     .venv\Scripts\activate
     ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Desktop Shortcut Creation

A cross-platform shortcut creator (`create_shortcut.py`) creates a desktop launcher with the application icon:

- **macOS (Recommended):**
  ```bash
  python3 create_shortcut.py --terminal
  ```
  *(The `--terminal` flag launches via Terminal.app, avoiding macOS Gatekeeper Automator stub restrictions).*

- **Windows:**
  ```cmd
  python create_shortcut.py
  ```

---

### GUI Framework Notice (PySide6 Only)

> [!IMPORTANT]
> **PySide6 (Qt 6) is the sole graphical user interface framework.**
> All legacy Tkinter and CustomTkinter code has been completely deprecated and purged from the repository.

---

## Execution & CLI Command Reference

### 1. Desktop GUI Launcher

Launch the primary desktop interface:
```bash
python run.py
```

---

### 2. Spatial Drift Alignment CLI (`align_cli.py`)

Headless batch alignment tool for TIFF sequences.

```bash
# Align single directory with default ECC engine and save comparison PNG
python -u align_cli.py -d "path/to/dataset" -e ECC --png

# Recursively align all subdirectories with multiple engines and JSON offset logging
python -u align_cli.py -d "path/to/root_dir" -r -e ECC PCA PhaseCorrelation --png --json

# Automatic PCA threshold optimization with overwrite enabled
python -u align_cli.py -d "path/to/dataset" -e PCA -t auto --overwrite --png
```

#### CLI Options (`align_cli.py`)
| Option | Description | Default |
|---|---|:---:|
| `-d`, `--dir` | Target directory containing TIFF images *(required)* | — |
| `-r`, `--recursive` | Recursively scan and process subdirectories | `False` |
| `-e`, `--engines` | Alignment engine(s): `ECC`, `PCA`, `PhaseCorrelation`, `all` | `ECC` |
| `-t`, `--threshold` | PCA intensity percentile cutoff (`float` or `auto`) | `99.9` |
| `--png` | Save side-by-side comparison PNG (`comparison_<engine>.png`) | `False` |
| `--json` | Export per-frame (dx, dy) drift offsets to JSON | `False` |
| `--overwrite` | Overwrite existing files in `sum/` output directory | `False` |

---

### 3. Photon Clustering CLI (`cluster_cli.py`)

Headless toolkit for single-photon event extraction across Stages 1, 2, and 3.

```bash
# Stage 1: Generate Dark Baseline & 2-Tier Noise Mask
python -u cluster_cli.py dark-mask -d "path/to/dark_tifs" -o "path/to/output_cal"

# Stage 2: Cluster Extraction on Raw Signal Frames
python -u cluster_cli.py cluster -s "path/to/signal_tifs" \
    --dark-tif "path/to/MED_Dark.tif" --mask-tif "path/to/Final_Mask.tif"

# Stage 3: Super-Resolution Event Map Reconstruction from Cluster Table
python -u cluster_cli.py reconstruct -c "path/to/Results_clusters.parquet" \
    --intden-low 175 --intden-high 500 --scale-factor 2

# End-to-End Pipeline (Stages 1, 2, and 3 Chained)
python -u cluster_cli.py full -d "path/to/dark_tifs" -s "path/to/signal_tifs" \
    --intden-low 175 --intden-high 500 --scale-factor 1
```

#### Subcommand Options (`cluster_cli.py`)
| Subcommand | Key Options | Description |
|---|---|---|
| `dark-mask` | `-d`, `--stddev-thresh 40.0`, `--tail-thresh-ratio 0.9333` | Computes temporal median and noise rejection mask. |
| `cluster` | `-s`, `--dark-tif`, `--mask-tif`, `--pixel-thresh 45.0` | Extracts 8-connected photon clusters and coordinates. |
| `reconstruct` | `-c`, `--intden-low`, `--intden-high`, `--max-area 9`, `--min-circ 0.3`, `--scale-factor` | Reconstructs super-resolution 2D event map. |
| `full` | `-d`, `-s`, `--intden-low`, `--intden-high`, `-o` | Executes full calibration -> clustering -> map generation. |

---

### 4. Zeroth-Order Calibration CLI (`zeroth_order_cli.py`)

Mirror pitch focus calibration and resolving power evaluation tool.

```bash
# Process scan directory with dispersion and mono energy (auto-discovers .txt scan log)
python -u zeroth_order_cli.py -d "path/to/Single_Motor_Scan_004202" \
    --dispersion 2.5 --mono-energy 850.0

# Batch process all scan runs with full CSV/JSON reporting and all diagnostic PNGs
python -u zeroth_order_cli.py -d "path/to/scans_root" -r \
    --dispersion 2.5 --mono-energy 850.0 --format all --export-plots all

# Minimal terminal summary table without saving plot images
python -u zeroth_order_cli.py -d "path/to/scan" --export-plots none --no-focus-curve
```

#### CLI Options (`zeroth_order_cli.py`)
| Option | Description | Default |
|---|---|:---:|
| `-d`, `--dir` | Target scan directory or parent folder *(required)* | — |
| `-r`, `--recursive` | Recursively scan subdirectories for TIFF sequences | `False` |
| `-t`, `--txt` | Explicit path to `.txt` scan log (overrides auto-discovery) | Auto |
| `-o`, `--output-dir` | Output folder for plots and tables | `<dir>/zeroth_order_analysis/` |
| `--dispersion` | Detector energy dispersion in meV/pixel (e.g. `2.5`) | `0.0` |
| `--mono-energy` | Incident monochromator energy in eV (e.g. `850.0`) | `0.0` |
| `--plot-focus-curve` | Generate parabolic `focus_curve.png` plot | `True` |
| `--export-plots` | Diagnostic PNG export mode: `best`, `all`, `none` | `best` |
| `--format` | Summary format: `table`, `csv`, `json`, `all` | `table` |
| `--overwrite` | Overwrite existing output analysis folder | `False` |

---

### 5. Image Denoising CLI (`denoise_cli.py`)

Stand-alone preprocessing utility for 2D spectroscopic frames.

```bash
# Denoise all TIFF frames in a directory using full pipeline
python -u denoise_cli.py -d "path/to/raw_tifs" --output-dir "path/to/denoised" \
    --clip --despike --anscombe --bilateral

# Denoise a single image file with custom MAD despiking threshold
python -u denoise_cli.py --input frame_001.tif --output frame_001_clean.tif \
    --despike --mad-threshold 4.5
```

---

## Automated Testing Suite

The repository features an automated test suite comprising **650+ tests across 34 test modules**, covering core algorithms, PySide6 CMV architecture, GUI widgets, asynchronous agent flows, and CLI interfaces.

### Running Tests

Execute the full test suite headlessly:
```bash
# Fast test run with quiet output
pytest tests/ -x -q

# Full verbose test execution
pytest -v

# Run specific subsystem test suites
pytest tests/test_photon_clustering.py -v       # Photon clustering & event reconstruction
pytest tests/test_dark_diagnostics.py -v        # Dark masking & noise thresholding
pytest tests/test_align_core.py -v              # Registration algorithms (ECC, PCA, PhaseCorr)
pytest tests/test_zeroth_order.py -v            # Zeroth-order 6-stage calibration
pytest tests/test_agent_chat_flow.py -v         # AI Co-Pilot agent bridge & tool calling
pytest tests/e2e/ -v                            # End-to-end integration workflows
```

---

## Project Directory Structure

```
Each200Frames/
├── run.py                          # GUI application entry point
├── align_cli.py                    # Headless spatial drift alignment CLI
├── cluster_cli.py                  # Single-photon clustering & dark masking CLI
├── zeroth_order_cli.py             # Mirror pitch zeroth-order focus calibration CLI
├── denoise_cli.py                  # Spectroscopic image preprocessing CLI
├── create_shortcut.py              # Cross-platform Desktop shortcut creator
├── requirements.txt                # Python package dependencies
├── rixs_app/
│   ├── core/                       # Pure algorithmic layer (ZERO UI dependencies)
│   │   ├── alignment.py            # ECC, PCA, Phase Correlation registration
│   │   ├── photon_clustering.py    # Connected components, IntDen, event map recon
│   │   ├── zeroth_order.py         # 6-stage mirror pitch calibration
│   │   ├── preprocessing.py        # Anscombe VST, MAD despiking, bilateral filter
│   │   ├── dataset.py              # In-memory sequence manager & cache
│   │   ├── dark_mask_store.py      # Atomic dark mask persistence
│   │   ├── dark_mask_export.py     # Publication-grade histogram figures
│   │   ├── txt_metadata_parser.py  # Beamline scan log parsing
│   │   └── cli_utils.py            # Directory discovery & CLI helpers
│   ├── agent/                      # RIXS Co-Pilot LLM Subsystem
│   │   ├── auth.py                 # Multi-tier API key resolution
│   │   ├── engine.py               # AsyncOpenAI / CBORG engine & streaming
│   │   ├── bridge.py               # Qt Signals <-> Asyncio bridge
│   │   ├── tools.py                # Tool registry & auto-schema generators
│   │   └── system_prompt.py        # Co-Pilot domain system prompt
│   └── ui/                         # PySide6 (Qt 6) Graphical Interface
│       ├── home_launchpad.py       # 2×2 squircle card home view
│       ├── dark_masking/           # Pillar 1: Dark masking studio
│       ├── clustering_slideshow/   # Pillar 2: 3-mode photon clustering studio
│       ├── alignment_slideshow/    # Pillar 3: Drift slideshow & comparison view
│       ├── zeroth_order_slideshow/ # Pillar 4: Mirror pitch focus slideshow
│       ├── agent_sidebar/          # QWebEngineView Chromium agent chat panel
│       ├── sorting_view.py         # File queue manager & drag-and-drop
│       ├── theme.py                # Fusion dark palette & button styling
│       └── widgets.py              # Custom RangeSlider & SafeFigureCanvasQTAgg
└── tests/                          # 34 test suites (650+ unit/GUI/E2E tests)
```

---

## License & Acknowledgements

Developed for **Beamline 6.0.2 (QERLIN)** at the **Advanced Light Source (ALS), Lawrence Berkeley National Laboratory (LBNL)**.

Built with Python, PySide6 (Qt 6), NumPy, SciPy, OpenCV, Matplotlib, and LBNL CBORG.
