# Spectroscopy Image Alignment GUI

A desktop application designed to load, sort, and align a sequence of spectroscopy images (TIFF format) using peak-line fitting (via PCA/SVD) and Hanning-windowed phase correlation offset computation.

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

## Running the Tests
To run the E2E test suite:
```bash
pytest tests/test_e2e.py -v
```
To run the core algorithmic tests:
```bash
python test_align.py
```
